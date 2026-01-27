[English](../../get_started/minicpm4.md)

# 部署 MiniCPM4.1-8B 模型

本指南介绍如何使用 FastDeploy 部署 OpenBMB MiniCPM4.1-8B 模型，实现高性能推理。

## 模型概述

MiniCPM4.1-8B 是 OpenBMB 开发的强大语言模型，具有以下主要特点：

- **80亿参数**，采用高效的分组查询注意力机制（GQA）
- **65536 上下文长度**，通过 LongRoPE 位置编码实现
- **SiLU 激活函数**用于 MLP 层
- **训练稳定性**通过 scale_emb 和 scale_depth 机制保证

更多详情请访问[官方模型页面](https://huggingface.co/openbmb/MiniCPM4.1-8B)。

## 环境要求

部署前，请确保您的环境满足以下要求：

- GPU 驱动 ≥ 535
- CUDA ≥ 12.3
- cuDNN ≥ 9.5
- Linux X86_64
- Python ≥ 3.10
- GPU 显存 ≥ 16GB（BF16 推理）

有关如何安装 FastDeploy 的更多信息，请参阅[安装文档](installation/README.md)。

## 1. 启动服务

### 基础部署（BF16）

安装 FastDeploy 后，在终端执行以下命令启动服务：

```bash
export ENABLE_V1_KVCACHE_SCHEDULER=1
python -m fastdeploy.entrypoints.openai.api_server \
       --model openbmb/MiniCPM4.1-8B \
       --port 8180 \
       --metrics-port 8181 \
       --engine-worker-queue-port 8182 \
       --max-model-len 32768 \
       --max-num-seqs 32
```

### 量化部署（WINT8）

如需减少显存占用，可以使用 INT8 权重量化：

```bash
export ENABLE_V1_KVCACHE_SCHEDULER=1
python -m fastdeploy.entrypoints.openai.api_server \
       --model openbmb/MiniCPM4.1-8B \
       --port 8180 \
       --metrics-port 8181 \
       --engine-worker-queue-port 8182 \
       --max-model-len 32768 \
       --max-num-seqs 32 \
       --quantization wint8
```

### 多 GPU 部署（张量并行）

如需更大的批处理大小或更长的上下文长度，请使用张量并行：

```bash
export ENABLE_V1_KVCACHE_SCHEDULER=1
python -m fastdeploy.entrypoints.openai.api_server \
       --model openbmb/MiniCPM4.1-8B \
       --port 8180 \
       --metrics-port 8181 \
       --engine-worker-queue-port 8182 \
       --max-model-len 65536 \
       --max-num-seqs 64 \
       --tensor-parallel-size 2
```

> 💡 **说明：**
> - `--model`：模型路径或 HuggingFace 模型 ID。如果本地未找到，FastDeploy 会自动从配置的模型源下载。
> - `--max-model-len`：服务的最大序列长度。MiniCPM4.1-8B 最大支持 65536 个 token。
> - `--max-num-seqs`：服务可处理的最大并发请求数。
> - `--tensor-parallel-size`：用于张量并行的 GPU 数量。

**相关文档：**
- [服务部署](../online_serving/README.md)
- [服务监控](../online_serving/metrics.md)
- [量化指南](../quantization/README.md)

## 2. 请求服务

服务启动后，以下输出表示初始化成功：

```shell
api_server.py[line:91] Launching metrics service at http://0.0.0.0:8181/metrics
api_server.py[line:94] Launching chat completion service at http://0.0.0.0:8180/v1/chat/completions
api_server.py[line:97] Launching completion service at http://0.0.0.0:8180/v1/completions
INFO:     Started server process [13909]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8180 (Press CTRL+C to quit)
```

### 健康检查

验证服务状态（HTTP 200 表示成功）：

```shell
curl -i http://0.0.0.0:8180/health
```

### cURL 请求

使用以下命令向服务发送请求：

```shell
curl -X POST "http://0.0.0.0:8180/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "法国的首都是哪里？"}
  ],
  "stream": true
}'
```

### Python 客户端（OpenAI 兼容 API）

FastDeploy 的 API 与 OpenAI 兼容。您也可以使用 Python 发送请求：

```python
import openai

host = "0.0.0.0"
port = "8180"
client = openai.Client(base_url=f"http://{host}:{port}/v1", api_key="null")

response = client.chat.completions.create(
    model="null",
    messages=[
        {"role": "system", "content": "你是一个有帮助的AI助手。"},
        {"role": "user", "content": "用简单的语言解释量子计算。"},
    ],
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta:
        print(chunk.choices[0].delta.content, end='')
print('\n')
```

## 3. 性能优化

### 启用图优化

为提高推理速度，启用 CUDA 图优化：

```bash
export ENABLE_V1_KVCACHE_SCHEDULER=1
export FD_ENABLE_GRAPH_OPTIMIZATION=1
python -m fastdeploy.entrypoints.openai.api_server \
       --model openbmb/MiniCPM4.1-8B \
       --port 8180 \
       --metrics-port 8181 \
       --engine-worker-queue-port 8182 \
       --max-model-len 32768 \
       --max-num-seqs 32
```

### FP8 量化（推荐用于 H100/H800）

对于 NVIDIA H100/H800 GPU，FP8 量化可提供出色的性能：

```bash
export ENABLE_V1_KVCACHE_SCHEDULER=1
python -m fastdeploy.entrypoints.openai.api_server \
       --model openbmb/MiniCPM4.1-8B \
       --port 8180 \
       --metrics-port 8181 \
       --engine-worker-queue-port 8182 \
       --max-model-len 32768 \
       --max-num-seqs 32 \
       --quantization wfp8afp8
```

## 4. 支持的量化方法

MiniCPM4.1-8B 在 FastDeploy 中支持以下量化方法：

| 方法 | 描述 | 显存降低 | 推荐 GPU |
|------|------|----------|----------|
| BF16 | 全精度（默认） | - | 所有 ≥16GB 显存的 GPU |
| WINT8 | INT8 权重量化 | ~50% | 所有 GPU |
| WINT4 | INT4 权重量化 | ~75% | 所有 GPU |
| FP8 | FP8 权重和激活 | ~50% | H100/H800 |

## 5. 故障排除

### 显存不足

如果遇到 OOM 错误，请尝试：

1. 减小 `--max-model-len`
2. 减小 `--max-num-seqs`
3. 使用量化（例如 `--quantization wint8`）
4. 增加张量并行度

### 首次请求较慢

由于模型加载和 CUDA 内核编译，首次请求可能较慢。后续请求会更快。

### 模型下载问题

设置模型源环境变量：

```bash
export FD_MODEL_SOURCE=HUGGINGFACE  # 或 MODELSCOPE, AISTUDIO
export FD_MODEL_CACHE=/path/to/cache
```
