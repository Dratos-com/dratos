docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=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" \
    -p 8000:8000 \
    --ipc=host \
    --name vllm \
    vllm/vllm-openai:latest \
    --model="meta-llama/Meta-Llama-3-8B-Instruct"