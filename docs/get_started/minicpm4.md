[简体中文](../zh/get_started/minicpm4.md)

# Deploy MiniCPM4.1-8B Model

This guide explains how to deploy the OpenBMB MiniCPM4.1-8B model using FastDeploy for high-performance inference.

## Model Overview

MiniCPM4.1-8B is a powerful language model developed by OpenBMB with the following key features:

- **8B parameters** with efficient Grouped Query Attention (GQA)
- **65536 context length** via LongRoPE position encoding
- **SiLU activation** in MLP layers
- **Training stability** through scale_emb and scale_depth mechanisms

For more details, visit the [official model page](https://huggingface.co/openbmb/MiniCPM4.1-8B).

## Environment Requirements

Before deployment, ensure your environment meets the following requirements:

- GPU Driver ≥ 535
- CUDA ≥ 12.3
- cuDNN ≥ 9.5
- Linux X86_64
- Python ≥ 3.10
- GPU Memory ≥ 16GB (for BF16 inference)

For more information about how to install FastDeploy, refer to the [installation document](installation/README.md).

## 1. Launch Service

### Basic Deployment (BF16)

After installing FastDeploy, execute the following command in the terminal to start the service:

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

### Deployment with Quantization (WINT8)

For reduced memory usage, you can use INT8 weight quantization:

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

### Multi-GPU Deployment with Tensor Parallelism

For larger batch sizes or longer context lengths, use tensor parallelism:

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

> 💡 **Notes:**
> - `--model`: Model path or HuggingFace model ID. If not found locally, FastDeploy will automatically download from the configured model source.
> - `--max-model-len`: Maximum sequence length for the service. MiniCPM4.1-8B supports up to 65536 tokens.
> - `--max-num-seqs`: Maximum concurrent requests the service can handle.
> - `--tensor-parallel-size`: Number of GPUs for tensor parallelism.

**Related Documents:**
- [Service Deployment](../online_serving/README.md)
- [Service Monitoring](../online_serving/metrics.md)
- [Quantization Guide](../quantization/README.md)

## 2. Request the Service

After starting the service, the following output indicates successful initialization:

```shell
api_server.py[line:91] Launching metrics service at http://0.0.0.0:8181/metrics
api_server.py[line:94] Launching chat completion service at http://0.0.0.0:8180/v1/chat/completions
api_server.py[line:97] Launching completion service at http://0.0.0.0:8180/v1/completions
INFO:     Started server process [13909]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8180 (Press CTRL+C to quit)
```

### Health Check

Verify service status (HTTP 200 indicates success):

```shell
curl -i http://0.0.0.0:8180/health
```

### cURL Request

Send requests to the service with the following command:

```shell
curl -X POST "http://0.0.0.0:8180/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "stream": true
}'
```

### Python Client (OpenAI-compatible API)

FastDeploy's API is OpenAI-compatible. You can also use Python for requests:

```python
import openai

host = "0.0.0.0"
port = "8180"
client = openai.Client(base_url=f"http://{host}:{port}/v1", api_key="null")

response = client.chat.completions.create(
    model="null",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."},
    ],
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta:
        print(chunk.choices[0].delta.content, end='')
print('\n')
```

## 3. Performance Optimization

### Enable Graph Optimization

For improved inference speed, enable CUDA graph optimization:

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

### FP8 Quantization (Recommended for H100/H800)

For NVIDIA H100/H800 GPUs, FP8 quantization provides excellent performance:

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

## 4. Supported Quantization Methods

MiniCPM4.1-8B supports the following quantization methods in FastDeploy:

| Method | Description | GPU Memory Reduction | Recommended GPU |
|--------|-------------|---------------------|-----------------|
| BF16 | Full precision (default) | - | All GPUs with ≥16GB |
| WINT8 | INT8 weight quantization | ~50% | All GPUs |
| WINT4 | INT4 weight quantization | ~75% | All GPUs |
| FP8 | FP8 weight and activation | ~50% | H100/H800 |

## 5. Troubleshooting

### Out of Memory

If you encounter OOM errors, try:

1. Reduce `--max-model-len`
2. Reduce `--max-num-seqs`
3. Use quantization (e.g., `--quantization wint8`)
4. Increase tensor parallelism

### Slow First Request

The first request may be slower due to model loading and CUDA kernel compilation. Subsequent requests will be faster.

### Model Download Issues

Set the model source environment variable:

```bash
export FD_MODEL_SOURCE=HUGGINGFACE  # or MODELSCOPE, AISTUDIO
export FD_MODEL_CACHE=/path/to/cache
```
