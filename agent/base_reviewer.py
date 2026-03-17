"""
base_reviewer.py
~~~~~~~~~~~~~~~~
提供 LLM 图表分析的基础架构：
- 加载配置和 prompt
- 读取候选股票列表
- 查找本地 K 线图
- 遍历调用子类实现的单股评分模型
- 结果汇总和输出
- 并发执行支持
"""

import json
import logging
import re
import time
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 配置模块级日志
logger = logging.getLogger(__name__)


class ReviewerError(Exception):
    """Reviewer 基础异常类"""
    pass


class APIKeyError(ReviewerError):
    """API Key 相关错误"""
    pass


class JSONExtractError(ReviewerError):
    """JSON 提取失败错误"""
    pass


class BaseReviewer:
    """LLM 图表分析基础类
    
    子类需要实现:
    - _review_single_stock(): 单股票分析的具体实现
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.prompt = self.load_prompt(Path(config["prompt_path"]))
        self.kline_dir = Path(config["kline_dir"])
        self.output_dir = Path(config["output_dir"])
        self.max_workers = config.get("max_workers", 5)
        self.request_delay = config.get("request_delay", 5)
        self.skip_existing = config.get("skip_existing", False)
        self.suggest_min_score = config.get("suggest_min_score", 4.0)

    @staticmethod
    def load_prompt(prompt_path: Path) -> str:
        """加载 prompt 文件"""
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt 文件不存在: {prompt_path}")
        return prompt_path.read_text(encoding="utf-8")

    @staticmethod
    def load_candidates(path: Path) -> Dict[str, Any]:
        """加载候选股票列表"""
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def find_chart_images(self, pick_date: str, code: str) -> Optional[Path]:
        """查找股票的日线图文件"""
        date_dir = self.kline_dir / pick_date
        day_chart = date_dir / f"{code}_day.jpg"
        if not day_chart.exists():
            day_chart_png = date_dir / f"{code}_day.png"
            day_chart = day_chart_png if day_chart_png.exists() else None
        return day_chart

    @staticmethod
    def extract_json(text: str) -> Dict[str, Any]:
        """从 LLM 响应中提取 JSON 对象
        
        支持多种格式:
        - Markdown 代码块: ```json {...} ```
        - 纯 JSON: {...}
        - 带前缀的 JSON: 前缀文字 {...}
        """
        if not text or not text.strip():
            raise JSONExtractError("响应文本为空")
        
        text = text.strip()
        
        # 尝试提取 Markdown 代码块
        code_block_patterns = [
            r"```(?:json)?\s*([\s\S]*?)```",  # ```json {...} ```
            r"`([\s\S]*?)`",  # ` {...} `
        ]
        
        for pattern in code_block_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                text = match.group(1).strip()
                break
        
        # 查找 JSON 对象边界
        start = text.find("{")
        end = text.rfind("}") + 1
        
        if start == -1 or end <= 0:
            raise JSONExtractError(f"未能在响应中找到 JSON 对象，响应内容:\n{text[:500]}")
        
        json_str = text[start:end]
        
        # 尝试解析
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # 尝试修复常见的 JSON 格式问题
            try:
                # 处理尾部逗号
                fixed = re.sub(r",(\s*[}\]])", r"\1", json_str)
                return json.loads(fixed)
            except json.JSONDecodeError:
                raise JSONExtractError(f"JSON 解析失败: {e}\n内容片段: {json_str[:200]}")

    @abstractmethod
    def _review_single_stock(
        self, code: str, day_chart: Path, prompt: str, out_file: Path
    ) -> Tuple[str, Optional[Dict[str, Any]], str]:
        """单股票分析实现
        
        Args:
            code: 股票代码
            day_chart: 日线图路径
            prompt: 系统提示词
            out_file: 输出文件路径
            
        Returns:
            (code, result_dict, status_message)
        """
        raise NotImplementedError("子类必须实现 _review_single_stock 方法")

    def generate_suggestion(
        self, pick_date: str, all_results: List[Dict[str, Any]], min_score: float
    ) -> Dict[str, Any]:
        """生成汇总推荐建议"""
        passed = [r for r in all_results if r.get("total_score", 0) >= min_score]
        excluded = [r["code"] for r in all_results if r.get("total_score", 0) < min_score]

        passed.sort(key=lambda r: r.get("total_score", 0), reverse=True)

        recommendations = [
            {
                "rank": i + 1,
                "code": r["code"],
                "verdict": r.get("verdict", ""),
                "total_score": r.get("total_score", 0),
                "signal_type": r.get("signal_type", ""),
                "comment": r.get("comment", ""),
            }
            for i, r in enumerate(passed)
        ]

        return {
            "date": pick_date,
            "min_score_threshold": min_score,
            "total_reviewed": len(all_results),
            "recommendations": recommendations,
            "excluded": excluded,
        }

    def _prepare_tasks(self, candidates_data: Dict[str, Any]) -> Tuple[str, List[Tuple[str, Path, Path]], List[Dict[str, Any]], List[str]]:
        """准备分析任务
        
        Returns:
            (pick_date, tasks, existing_results, failed_codes)
        """
        pick_date: str = candidates_data["pick_date"]
        candidates: List[Dict[str, Any]] = candidates_data["candidates"]
        
        logger.info("pick_date=%s，候选股票数=%d", pick_date, len(candidates))

        out_dir = self.output_dir / pick_date
        out_dir.mkdir(parents=True, exist_ok=True)

        all_results: List[Dict[str, Any]] = []
        failed_codes: List[str] = []
        tasks: List[Tuple[str, Path, Path]] = []

        for candidate in candidates:
            code: str = candidate["code"]
            out_file = out_dir / f"{code}.json"

            if self.skip_existing and out_file.exists():
                logger.debug("[%s] 已存在，跳过", code)
                try:
                    with open(out_file, encoding="utf-8") as f:
                        result = json.load(f)
                    all_results.append(result)
                except json.JSONDecodeError as e:
                    logger.warning("[%s] 现有结果文件损坏，将重新分析: %s", code, e)
                    day_chart = self.find_chart_images(pick_date, code)
                    if day_chart is None:
                        logger.warning("[%s] 缺少日线图，跳过", code)
                        failed_codes.append(code)
                        continue
                    tasks.append((code, day_chart, out_file))
                continue

            day_chart = self.find_chart_images(pick_date, code)
            if day_chart is None:
                logger.warning("[%s] 缺少日线图，跳过", code)
                failed_codes.append(code)
                continue

            tasks.append((code, day_chart, out_file))

        return pick_date, tasks, all_results, failed_codes

    def _execute_concurrent(
        self, tasks: List[Tuple[str, Path, Path]]
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """并发执行任务
        
        Returns:
            (results, failed_codes)
        """
        all_results: List[Dict[str, Any]] = []
        failed_codes: List[str] = []

        if not tasks:
            logger.info("所有任务均已跳过或无可用股票")
            return all_results, failed_codes

        logger.info("开始并发处理 %d 支股票，最大并发数: %d", len(tasks), self.max_workers)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._review_single_stock,
                    code,
                    day_chart,
                    self.prompt,
                    out_file,
                ): code
                for code, day_chart, out_file in tasks
            }

            completed = 0
            for future in as_completed(futures):
                code = futures[future]
                completed += 1
                try:
                    _, result, status = future.result()
                    logger.info("[%d/%d] %s — %s", completed, len(tasks), code, status)
                    if result is not None:
                        all_results.append(result)
                    else:
                        failed_codes.append(code)
                except Exception as e:
                    logger.error("[%d/%d] %s — 异常: %s", completed, len(tasks), code, e)
                    failed_codes.append(code)

                # 控制请求速率
                if completed < len(tasks):
                    delay = self.request_delay / self.max_workers
                    if delay > 0:
                        time.sleep(delay)

        return all_results, failed_codes

    def run(self) -> None:
        """运行完整的分析流程"""
        candidates_data = self.load_candidates(Path(self.config["candidates"]))
        
        pick_date, tasks, existing_results, pre_failed = self._prepare_tasks(candidates_data)
        
        # 执行并发分析
        new_results, new_failed = self._execute_concurrent(tasks)
        
        all_results = existing_results + new_results
        failed_codes = pre_failed + new_failed

        logger.info("评分完成：成功 %d 支，失败/跳过 %d 支", len(all_results), len(failed_codes))
        if failed_codes:
            logger.warning("未处理股票: %s", failed_codes)

        if not all_results:
            logger.error("没有可用的评分结果，跳过汇总")
            return

        logger.info("正在生成汇总推荐建议...")
        suggestion = self.generate_suggestion(
            pick_date=pick_date,
            all_results=all_results,
            min_score=self.suggest_min_score,
        )
        
        out_dir = self.output_dir / pick_date
        suggestion_file = out_dir / "suggestion.json"
        with open(suggestion_file, "w", encoding="utf-8") as f:
            json.dump(suggestion, f, ensure_ascii=False, indent=2)
        
        logger.info("汇总推荐已写入: %s", suggestion_file)
        logger.info("推荐股票数（score≥%.1f）: %d", self.suggest_min_score, len(suggestion["recommendations"]))
        logger.info("全部完成，输出目录: %s", out_dir)
