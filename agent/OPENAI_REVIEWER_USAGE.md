# OpenAI Reviewer 使用指南

`OpenAIReviewer` 提供了与 `GeminiReviewer` 相同的功能，但使用 OpenAI 及兼容接口进行 LLM 调用。

## 特性

- ✅ 支持 OpenAI 官方接口（GPT-4o、GPT-4-Turbo 等）
- ✅ 支持 DeepSeek 接口（deepseek-chat、deepseek-vision）
- ✅ 支持阿里云通义千问（Qwen）接口
- ✅ 支持 Moonshot（Kimi）接口
- ✅ 支持 Ollama 本地部署
- ✅ 多线程并发处理
- ✅ 与 GeminiReviewer 相同的输出格式

## 快速开始

### 1. 安装依赖

```bash
poetry add openai
# 或
pip install openai
```

### 2. 配置环境变量

在 `.env` 文件中设置 API Key：

```bash
# OpenAI 官方
OPENAI_APIKEY=sk-your-openai-api-key

# DeepSeek
OPENAI_APIKEY=sk-your-deepseek-api-key

# 阿里云通义千问
OPENAI_APIKEY=sk-your-dashscope-api-key

# Moonshot (Kimi)
OPENAI_APIKEY=sk-your-moonshot-api-key
```

### 3. 配置文件

编辑 `config/openai_review.yaml`：

```yaml
# 模型选择
model: gpt-4o  # 或 deepseek-chat, qwen-vl-max, moonshot-v1-32k

# API 接口地址（可选）
# OpenAI 官方：留空或注释
# DeepSeek: https://api.deepseek.com/v1
# 阿里云：https://dashscope.aliyuncs.com/compatible-mode/v1
# Moonshot: https://api.moonshot.cn/v1
# Ollama 本地：http://localhost:11434/v1
base_url: null

# 并发数（根据 API 限流调整）
max_workers: 5

# 请求延迟（秒）
request_delay: 5
```

## 不同服务商的配置示例

### OpenAI 官方

```yaml
model: gpt-4o
base_url: null  # 或注释掉
```

环境变量：
```bash
OPENAI_APIKEY=sk-...
```

### DeepSeek

```yaml
model: deepseek-chat  # 或 deepseek-vision
base_url: https://api.deepseek.com/v1
```

环境变量：
```bash
OPENAI_APIKEY=sk-your-deepseek-key
```

### 阿里云通义千问（Qwen）

```yaml
model: qwen-vl-max  # 或 qwen-vl-plus
base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
```

环境变量：
```bash
OPENAI_APIKEY=sk-your-dashscope-key
```

### Moonshot（Kimi）

```yaml
model: moonshot-v1-32k
base_url: https://api.moonshot.cn/v1
```

环境变量：
```bash
OPENAI_APIKEY=sk-your-moonshot-key
```

### Ollama 本地部署

```yaml
model: llava  # 或其他视觉模型
base_url: http://localhost:11434/v1
```

环境变量：
```bash
OPENAI_APIKEY=ollama  # Ollama 通常不需要 API Key
```

## 运行方式

### 单独运行

```bash
# 使用默认配置
poetry run python agent/openai_review.py

# 指定配置文件
poetry run python agent/openai_review.py --config config/openai_review.yaml
```

### 集成到 run_all.py

修改 `run_all.py` 中的步骤 4：

```python
# 原代码（使用 Gemini）
_run(
    "4/4  Gemini 图表分析（gemini_review）",
    [PYTHON, str(ROOT / "agent" / "gemini_review.py")],
)

# 修改为（使用 OpenAI）
_run(
    "4/4  OpenAI 图表分析（openai_review）",
    [PYTHON, str(ROOT / "agent" / "openai_review.py")],
)
```

然后运行：
```bash
poetry run python run_all.py --start-from 4
```

## 性能调优

### 并发数调整

根据 API 服务商的限流策略调整 `max_workers`：

- **OpenAI 官方**：建议 3-5（根据套餐限制）
- **DeepSeek**：建议 5-10
- **阿里云**：建议 5-8
- **Moonshot**：建议 3-5
- **Ollama 本地**：建议 1-3（受本地资源限制）

### 请求延迟调整

```yaml
# 快速模式（API 不限流时）
request_delay: 1
max_workers: 10

# 保守模式（避免触发限流）
request_delay: 10
max_workers: 3
```

## 输出格式

与 `GeminiReviewer` 完全一致：

- 单股结果：`data/review/{pick_date}/{code}.json`
- 汇总结果：`data/review/{pick_date}/suggestion.json`

## 故障排查

### 问题：导入错误

```
[ERROR] 未安装 openai 包，请先安装：pip install openai
```

解决：
```bash
poetry add openai
```

### 问题：API Key 错误

```
[ERROR] 未找到环境变量 OPENAI_APIKEY，请先设置后重试。
```

解决：在 `.env` 文件中设置 `OPENAI_APIKEY`

### 问题：接口地址错误

检查 `base_url` 配置是否正确，确保：
- URL 格式正确（包含 `http://` 或 `https://`）
- 接口支持 OpenAI 兼容格式
- 网络可达

### 问题：模型不支持视觉输入

确保选择的模型支持图像输入：
- OpenAI: gpt-4o, gpt-4-turbo
- DeepSeek: deepseek-vision
- Qwen: qwen-vl-max, qwen-vl-plus
- Moonshot: 需要确认是否支持视觉

## 与 GeminiReviewer 的对比

| 特性 | GeminiReviewer | OpenAIReviewer |
|------|----------------|----------------|
| 接口 | Google Gemini | OpenAI 及兼容接口 |
| 多线程 | ✅ | ✅ |
| 图片处理 | Part 对象 | Base64 编码 |
| 输出格式 | JSON | JSON |
| 配置方式 | YAML | YAML |
| 支持服务商 | Google | OpenAI/DeepSeek/Qwen/Moonshot 等 |

## 最佳实践

1. **首次使用**：先用少量股票测试（修改候选列表）
2. **调整并发**：根据 API 响应速度和限流策略调整 `max_workers`
3. **断点续跑**：启用 `skip_existing: true` 避免重复处理
4. **监控成本**：注意 API 调用费用，特别是使用商业 API 时
5. **本地测试**：使用 Ollama 本地部署进行开发和测试
