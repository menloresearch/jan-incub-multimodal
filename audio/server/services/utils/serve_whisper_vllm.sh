#!/bin/bash
# Start ASR service with vLLM - MAX PERFORMANCE with 0.5 GPU memory
CUDA_VISIBLE_DEVICES=1 vllm serve openai/whisper-large-v3 \
    --trust-remote-code \
    --max-model-len 448 \
    --gpu-memory-utilization 0.5 \
    --served-model-name openai/whisper-large-v3 \
    --port 3349 \
    --task transcription \
    --max-num-seqs 32 \
    --max-num-batched-tokens 4096 \
    --kv-cache-dtype fp8 \
    --dtype bfloat16 \
    --limit-mm-per-prompt '{"audio": 1}' \
    --enable-prefix-caching \
    --disable-log-stats