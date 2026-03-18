"""
openai_review.py
~~~~~~~~~~~~~~~~
使用 OpenAI 及兼容接口（DeepSeek、Qwen、Kimi 等）对候选股票进行图表分析评分。
继承自 BaseReviewer 基础架构。

用法：
    python agent/openai_review.py
    python agent/openai_review.py --config config/openai_review.yaml

环境变量：
    OPENAI_APIKEY  —— OpenAI API Key 或兼容接口的 Key（必填）

输出：
    ./data/review/{pick_date}/{code}.json   每支股票的评分 JSON
    ./data/review/{pick_date}/suggestion.json  汇总推荐建议
"""

import argparse
import base64
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] 未安装 openai 包，请先安装：pip install openai", file=sys.stderr)
    sys.exit(1)

import yaml
from dotenv import load_dotenv

from base_reviewer import BaseReviewer, APIKeyError, JSONExtractError

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────
# 配置加载
# ────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _ROOT / "config" / "openai_review.yaml"

DEFAULT_CONFIG: Dict[str, Any] = {
    # 路径参数（相对路径默认基于项目根目录）
    "candidates": "data/candidates/candidates_latest.json",
    "kline_dir": "data/kline",
    "output_dir": "data/review",
    "prompt_path": "agent/prompt.md",
    # OpenAI 模型参数
    "model": "kimi-k2.5",
    "base_url": "https://api.moonshot.cn/v1",
    "request_delay": 5,
    "max_workers": 5,
    "skip_existing": True,
    "suggest_min_score": 4.0,
}


def _resolve_cfg_path(path_like: str | Path, base_dir: Path = _ROOT) -> Path:
    """将配置中的路径统一解析为绝对路径"""
    p = Path(path_like)
    return p if p.is_absolute() else (base_dir / p)


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """加载配置文件，合并默认值"""
    cfg_path = config_path or _DEFAULT_CONFIG_PATH
    
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        logger.info("已加载配置文件: %s", cfg_path)
    else:
        logger.warning("配置文件不存在，使用默认配置: %s", cfg_path)
        raw = {}

    cfg = {**DEFAULT_CONFIG, **raw}

    # 路径字段转换为 Path 对象
    cfg["candidates"] = _resolve_cfg_path(cfg["candidates"])
    cfg["kline_dir"] = _resolve_cfg_path(cfg["kline_dir"])
    cfg["output_dir"] = _resolve_cfg_path(cfg["output_dir"])
    cfg["prompt_path"] = _resolve_cfg_path(cfg["prompt_path"])

    return cfg


class OpenAIReviewer(BaseReviewer):
    """OpenAI 兼容接口的股票图表分析器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = self._init_client()
        self.model = config.get("model", "kimi-k2.5")
        logger.info("OpenAIReviewer 初始化完成，模型: %s", self.model)

    def _init_client(self) -> OpenAI:
        """初始化 OpenAI 客户端"""
        api_key = os.environ.get("OPENAI_APIKEY", "").strip()
        if not api_key:
            raise APIKeyError(
                "未找到环境变量 OPENAI_APIKEY，请先设置后重试。"
                "例如: export OPENAI_APIKEY=your_api_key"
            )

        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        base_url = self.config.get("base_url")
        if base_url:
            client_kwargs["base_url"] = base_url
            logger.info("使用自定义 base_url: %s", base_url)

        return OpenAI(**client_kwargs)

    @staticmethod
    def image_to_base64(path: Path) -> str:
        """将图片文件转为 base64 编码"""
        suffix = path.suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_map.get(suffix, "image/jpeg")
        
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            raise FileNotFoundError(f"图片文件不存在: {path}")
        except IOError as e:
            raise IOError(f"读取图片文件失败 {path}: {e}")
            
        base64_data = base64.b64encode(data).decode("utf-8")
        return f"data:{mime_type};base64,{base64_data}"

    def _review_single_stock(
        self, code: str, day_chart: Path, prompt: str, out_file: Path
    ) -> Tuple[str, Optional[Dict[str, Any]], str]:
        """单支股票分析实现"""
        max_retries = 3
        base_delay = 3
        max_delay = 5
        
        for retry in range(max_retries):
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
                    model=self.model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=1,
                )

                response_text = response.choices[0].message.content
                if response_text is None:
                    return (code, None, "API 返回空响应")

                result = self.extract_json(response_text)
                result["code"] = code

                # 保存结果
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

                verdict = result.get("verdict", "?")
                score = result.get("total_score", "?")
                return (code, result, f"完成 — verdict={verdict}, score={score}")

            except APIKeyError:
                raise
            except Exception as e:
                if retry < max_retries - 1 and "429" in str(e):
                    # 指数延迟，每次重试延迟时间翻倍
                    delay = min(base_delay * (2 ** retry), max_delay)
                    logger.warning("[%s] 遇到 429 错误，将在 %.1f 秒后重试 (尝试 %d/%d)", code, delay, retry + 2, max_retries)
                    time.sleep(delay)
                else:
                    return (code, None, f"失败 — {type(e).__name__}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI 图表复评")
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG_PATH),
        help="配置文件路径（默认 config/openai_review.yaml）",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="启用详细日志输出",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        config = load_config(Path(args.config))
        reviewer = OpenAIReviewer(config)
        reviewer.run()
    except APIKeyError as e:
        logger.error("API Key 错误: %s", e)
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error("文件错误: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("运行失败: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
