import ray
from ray import serve
from beta.models.serve.engines.vllm_engine import build_app

if __name__ == "__main__":
    ray.init()
    serve.start()

    app = build_app(
        {
            "model": "NousResearch/Meta-Llama-3-8B-Instruct",
            "tensor_parallel_size": 1,
            "max_num_batched_tokens": 4096,
            "trust_remote_code": True,
        }
    )
    serve.run(app)

    # Example usage (can be moved to a separate client file)
    from openai import OpenAI

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="NOT_NEEDED",
    )

    chat_completion = client.chat.completions.create(
        model="NousResearch/Meta-Llama-3-8B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "What are some highly rated restaurants in San Francisco?",
            },
        ],
        temperature=0.01,
        stream=True,
    )

    for chunk in chat_completion:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)

    print("\n")

    serve.shutdown()
    ray.shutdown()
