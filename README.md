
# Install environement & dependencies

>Python 3.12 or later
```
poetry config virtualenvs.in-project true
poetry install
source .venv/bin/activate
```

# Get strated
Try examples in the `examples` folder


# Using VLLM for production
VLLM needs Nvidea GPUs, often best to run in the cloud
Reach out to get this setup for you in an hour

# Observability
`mlflow ui`

# Tests
Test within dratos use doctest. Doctests are here to make sure your code does what it says in the docstring. With our setup, testing is efficient, and you’re not constantly hitting external APIs—saving you time, money, and sanity..

- **First Run:** When you first run a test, it’ll hit the real API, grab the response, and store it in `testdata.json`.
- **Later Runs:** After that, tests use the local API with the saved responses—no more API calls.

## Running Tests

```bash
pytest -v --doctest-modules examples/test*
```
- Add `-s` flag captures all the output

Ensure the correct Python interpreter is being used:
```bash
python3 -m pytest
```

## Writing Tests

- **Location:** All test files go in the `examples` folder. Files must start with `test_` so PyTest knows what’s up.
- **Function Naming:** Don’t start function names with `test_`, or PyTest will treat them as regular PyTest tests. You can still write PyTest tests separately if you want.
- **Verbose=False:** Always set verbose to `False` to prevent anything printed in the terminal from being interpreted as part of the expected response in the doctest.

```python
def example_function():
    """
    >>> example_function()
    result
    """
    # Function logic here
```

🛑 **PRO TIP** 🛑 **First Run:** Run the test, get the actual response from the terminal, copy-paste it into the docstring.

# Streaming responses
Note that when streaming responses, two streams cannot be simultaneously streamed in the terminal. And so in that event, streams will be misformatted. However a single stream can be streamed simultanously by two distinct interface (e.g. terminal and app, etc)

Two functions within the Agent class: `sync_gen()` returns the complete response once it's generated and `async_gen()`. async_gen() support response streaming.


# Qdrant
Setup and Run Qdrant Locally:
The easiest way to run Qdrant locally is using Docker. Here are the steps:
```bash
# Pull the Qdrant image
docker pull qdrant/qdrant

# Run Qdrant
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

This will start Qdrant and make it accessible at:
REST API: localhost:6333
Dashboard: http://localhost:6333/dashboard
GRPC API: localhost:63341

To stop Qdrant, you can use Ctrl+C in the terminal where it's running, or stop using docker ui or use:
```bash
docker stop <container_id>
```

