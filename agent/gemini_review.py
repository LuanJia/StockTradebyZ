"""
gemini_review.py
~~~~~~~~~~~~~~~~
使用 Google Gemini 对候选股票进行图表分析评分。
继承自 BaseReviewer 基础架构。

用法：
    python agent/gemini_review.py
    python agent/gemini_review.py --config config/gemini_review.yaml

环境变量：
    GEMINI_API_KEY  —— Google Gemini API Key（必填）

输出：
    ./data/review/{pick_date}/{code}.json   每支股票的评分 JSON
    ./data/review/{pick_date}/suggestion.json  汇总推荐建议
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    from google import genai
    from google.genai import types
except ImportError:
    print(
        "[ERROR] 未安装 google-genai 包，请先安装：pip install google-genai",
        file=sys.stderr,
    )
    sys.exit(1)

import yaml
from dotenv import load_dotenv

from base_reviewer import BaseReviewer, APIKeyError

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
_DEFAULT_CONFIG_PATH = _ROOT / "config" / "gemini_review.yaml"

DEFAULT_CONFIG: Dict[str, Any] = {
    # 路径参数（相对路径默认基于项目根目录）
    "candidates": "data/candidates/candidates_latest.json",
    "kline_dir": "data/kline",
    "output_dir": "data/review",
    "prompt_path": "agent/prompt.md",
    # Gemini 模型参数
    "model": "gemini-3.1-pro-preview",
    "location": "global",
    "project": "gen-lang-client-0688895389",
    "request_delay": 5,
    "max_workers": 5,
    "skip_existing": False,
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


class GeminiReviewer(BaseReviewer):
    """Google Gemini 股票图表分析器"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = self._init_client()
        self.model = config.get("model", "gemini-1.5-pro")
        logger.info("GeminiReviewer 初始化完成，模型: %s", self.model)

    def _init_client(self) -> genai.Client:
        """初始化 Gemini 客户端（优先使用 ADC，其次使用 API Key）"""
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        project = self.config.get("project", "gen-lang-client-0688895389")
        location = self.config.get("location", "global")

        # 优先使用 ADC（Service Account），无 API Key 时自动切换
        if not api_key:
            adc_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
            if adc_creds and not os.path.isfile(adc_creds):
                raise APIKeyError(
                    f"GOOGLE_APPLICATION_CREDENTIALS 指向的文件不存在: {adc_creds}"
                )
            logger.info("使用 ADC 认证（Service Account），项目: %s, 位置: %s", project, location)
            return genai.Client(vertexai=True, project=project, location=location)
        else:
            logger.info("使用 API Key 认证")
            return genai.Client(api_key=api_key)

    @staticmethod
    def image_to_part(path: Path) -> types.Part:
        """将图片文件转为 Gemini Part 对象"""
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

        return types.Part.from_bytes(data=data, mime_type=mime_type)

    def _review_single_stock(
        self, code: str, day_chart: Path, prompt: str, out_file: Path
    ) -> Tuple[str, Optional[Dict[str, Any]], str]:
        """单支股票分析实现"""
        try:
            user_text = (
                f"股票代码：{code}\n\n"
                "以下是该股票的 **日线图**，请按照系统提示中的框架进行分析，"
                "并严格按照要求输出 JSON。"
            )

            parts: list[types.Part] = [
                types.Part.from_text(text="【日线图】"),
                self.image_to_part(day_chart),
                types.Part.from_text(text=user_text),
            ]

            response = self.client.models.generate_content(
                model=self.model,
                contents=[types.Content(role="user", parts=parts)],
                config=types.GenerateContentConfig(
                    system_instruction=prompt,
                    temperature=0.2,
                ),
            )

            response_text = response.text
            if response_text is None:
                return (code, None, "Gemini 返回空响应")

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
            return (code, None, f"失败 — {type(e).__name__}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemini 图表复评")
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG_PATH),
        help="配置文件路径（默认 config/gemini_review.yaml）",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="启用详细日志输出",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        config = load_config(Path(args.config))
        reviewer = GeminiReviewer(config)
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
