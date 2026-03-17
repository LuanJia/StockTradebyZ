"""
openai_review.py
~~~~~~~~~~~~~~~~
使用 OpenAI 及兼容接口（DeepSeek、Qwen 等）对候选股票进行图表分析评分。
继承自 BaseReviewer 基础架构。

用法：
    python agent/openai_review.py
    python agent/openai_review.py --config config/openai_review.yaml

配置：
    默认读取 config/openai_review.yaml。
    max_workers: 多线程并发数（默认 5，可根据 API 限流情况调整）
    base_url: OpenAI 兼容接口地址（可选，默认使用 OpenAI 官方接口）
    model: 模型名称

环境变量：
    OPENAI_API_KEY  —— OpenAI API Key 或兼容接口的 Key（必填）

输出：
    ./data/review/{pick_date}/{code}.json   每支股票的评分 JSON
    ./data/review/{pick_date}/suggestion.json  汇总推荐建议
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] 未安装 openai 包，请先安装：pip install openai", file=sys.stderr)
    sys.exit(1)

import yaml
from dotenv import load_dotenv


from base_reviewer import BaseReviewer

# 加载环境变量
load_dotenv()
# ────────────────────────────────────────────────
# 配置加载
# ────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _ROOT / "config" / "openai_review.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    # 路径参数（相对路径默认基于项目根目录）
    "candidates": "data/candidates/candidates_latest.json",
    "kline_dir": "data/kline",
    "output_dir": "data/review",
    "prompt_path": "agent/prompt.md",
    # OpenAI 模型参数
    "model": "kimi-k2.5",
    "base_url": "https://api.moonshot.cn/v1",  # OpenAI 兼容接口地址，如 DeepSeek、Qwen 等
    "request_delay": 5,
    "skip_existing": True,
    "suggest_min_score": 4.0,
}


def _resolve_cfg_path(path_like: str | Path, base_dir: Path = _ROOT) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (base_dir / p)


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    cfg_path = config_path or _DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"找不到配置文件：{cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = {**DEFAULT_CONFIG, **raw}

    # BaseReviewer 依赖这些路径字段为 Path 对象
    cfg["candidates"] = _resolve_cfg_path(cfg["candidates"])
    cfg["kline_dir"] = _resolve_cfg_path(cfg["kline_dir"])
    cfg["output_dir"] = _resolve_cfg_path(cfg["output_dir"])
    cfg["prompt_path"] = _resolve_cfg_path(cfg["prompt_path"])

    return cfg


class OpenAIReviewer(BaseReviewer):
    def __init__(self, config):
        super().__init__(config)

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print(
                "[ERROR] 未找到环境变量 OPENAI_API_KEY，请先设置后重试。",
                file=sys.stderr,
            )
            sys.exit(1)

        # 初始化 OpenAI 客户端（支持兼容接口）
        client_kwargs = {"api_key": api_key}
        if config.get("base_url"):
            client_kwargs["base_url"] = config["base_url"]

        self.client = OpenAI(**client_kwargs)
        self.max_workers = config.get("max_workers", 5)

    @staticmethod
    def image_to_base64(path: Path) -> str:
        """将图片文件转为 base64 编码。"""
        suffix = path.suffix.lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
        mime_type = mime_map.get(suffix, "image/jpeg")
        data = path.read_bytes()
        base64_data = base64.b64encode(data).decode("utf-8")
        return f"data:{mime_type};base64,{base64_data}"

    def _review_single_stock(
        self, code: str, day_chart: Path, prompt: str, out_file: Path
    ) -> tuple[str, dict | None, str]:
        """
        单支股票分析的内部方法，返回 (code, result, status)。
        """
        try:
            image_base64 = self.image_to_base64(day_chart)

            user_content = [
                {
                    "type": "text",
                    "text": f"股票代码：{code}\n\n以下是该股票的 **日线图**，请按照系统提示中的框架进行分析，并严格按照要求输出 JSON。",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_base64, "detail": "high"},
                },
            ]

            response = self.client.chat.completions.create(
                model=self.config.get("model", "kimi-k2.5"),
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.2,
            )

            response_text = response.choices[0].message.content
            if response_text is None:
                return (code, None, f"OpenAI 返回空响应")

            result = self.extract_json(response_text)
            result["code"] = code

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            verdict = result.get("verdict", "?")
            score = result.get("total_score", "?")
            return (code, result, f"完成 — verdict={verdict}, score={score}")

        except Exception as e:
            return (code, None, f"失败 — {e}")

    def review_stock(self, code: str, day_chart: Path, prompt: str) -> dict:
        """
        调用 OpenAI API，对单支股票进行图表分析，返回解析后的 JSON 结果。
        保留此方法以兼容基类接口。
        """
        image_base64 = self.image_to_base64(day_chart)

        user_content = [
            {
                "type": "text",
                "text": f"股票代码：{code}\n\n以下是该股票的 **日线图**，请按照系统提示中的框架进行分析，并严格按照要求输出 JSON。",
            },
            {"type": "image_url", "image_url": {"url": image_base64, "detail": "high"}},
        ]

        response = self.client.chat.completions.create(
            model=self.config.get("model", "gpt-4o"),
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
        )

        response_text = response.choices[0].message.content
        if response_text is None:
            raise RuntimeError(f"OpenAI 返回空响应，无法解析 JSON（code={code}）")

        result = self.extract_json(response_text)
        result["code"] = code
        return result

    def run(self):
        """
        使用多线程并发执行股票分析任务。
        """
        candidates_data = self.load_candidates(Path(self.config["candidates"]))
        pick_date: str = candidates_data["pick_date"]
        candidates: List[dict] = candidates_data["candidates"]
        print(f"[INFO] pick_date={pick_date}，候选股票数={len(candidates)}")
        print(f"[INFO] 使用多线程模式，最大并发数：{self.max_workers}")

        # 显示当前使用的接口信息
        base_url = self.config.get("base_url")
        model = self.config.get("model", "gpt-4o")
        if base_url:
            print(f"[INFO] 使用 OpenAI 兼容接口：{base_url}")
        print(f"[INFO] 使用模型：{model}")

        out_dir = self.output_dir / pick_date
        out_dir.mkdir(parents=True, exist_ok=True)

        all_results: List[dict] = []
        failed_codes: List[str] = []

        tasks = []
        for candidate in candidates:
            code: str = candidate["code"]
            out_file = out_dir / f"{code}.json"

            if self.config.get("skip_existing", False) and out_file.exists():
                print(f"[SKIP] {code} — 已存在，跳过。")
                with open(out_file, encoding="utf-8") as f:
                    result = json.load(f)
                all_results.append(result)
                continue

            day_chart = self.find_chart_images(pick_date, code)
            if day_chart is None:
                print(f"[SKIP] {code} — 缺少日线图，跳过。")
                failed_codes.append(code)
                continue

            tasks.append((code, day_chart, out_file))

        if not tasks:
            print("[INFO] 所有任务均已跳过或无可用股票。")
        else:
            print(f"[INFO] 开始并发处理 {len(tasks)} 支股票...")

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self._review_single_stock,
                        code,
                        day_chart,
                        self.prompt,
                        out_file,
                    ): (code, idx)
                    for idx, (code, day_chart, out_file) in enumerate(tasks)
                }

                completed = 0
                for future in as_completed(futures):
                    code, idx = futures[future]
                    completed += 1
                    try:
                        result_code, result, status = future.result()
                        print(f"[{completed}/{len(tasks)}] {code} — {status}")
                        if result is not None:
                            all_results.append(result)
                        else:
                            failed_codes.append(code)
                    except Exception as e:
                        print(f"[{completed}/{len(tasks)}] {code} — 异常：{e}")
                        failed_codes.append(code)

                    if completed < len(tasks):
                        time.sleep(
                            self.config.get("request_delay", 5) / self.max_workers
                        )

        print(
            f"\n[INFO] 评分完成：成功 {len(all_results)} 支，失败/跳过 {len(failed_codes)} 支"
        )
        if failed_codes:
            print(f"[WARN] 未处理股票：{failed_codes}")

        if not all_results:
            print("[ERROR] 没有可用的评分结果，跳过汇总。")
            return

        print("\n[INFO] 正在生成汇总推荐建议 ...")
        min_score = self.config.get("suggest_min_score", 4.0)
        suggestion = self.generate_suggestion(
            pick_date=pick_date,
            all_results=all_results,
            min_score=min_score,
        )
        suggestion_file = out_dir / "suggestion.json"
        with open(suggestion_file, "w", encoding="utf-8") as f:
            json.dump(suggestion, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 汇总推荐已写入：{suggestion_file}")
        print(
            f"       推荐股票数（score≥{min_score}）: {len(suggestion['recommendations'])}"
        )

        print("\n✅ 全部完成。")
        print(f"   输出目录：{out_dir}")


def main():
    parser = argparse.ArgumentParser(description="OpenAI 图表复评")
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG_PATH),
        help="配置文件路径（默认 config/openai_review.yaml）",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    reviewer = OpenAIReviewer(config)
    reviewer.run()


if __name__ == "__main__":
    main()
