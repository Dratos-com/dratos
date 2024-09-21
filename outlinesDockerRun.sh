docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=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" \
    -e VLLM_GPU_MEMORY_UTILIZATION=0.9 \
    -p 8000:8000 \
    --network=host \
    --name vllm \
    outlinesdev/outlines:latest \
    --model="meta-llama/Meta-Llama-3-8B-Instruct"