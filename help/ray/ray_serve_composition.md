Deploy Compositions of Models
With this guide, you can:

Compose multiple deployments containing ML models or business logic into a single application

Independently scale and configure each of your ML models and business logic steps

Note

The deprecated RayServeHandle and RayServeSyncHandle APIs have been fully removed as of Ray 2.10.

Compose deployments using DeploymentHandles
When building an application, you can .bind() multiple deployments and pass them to each other’s constructors. At runtime, inside the deployment code Ray Serve substitutes the bound deployments with DeploymentHandles that you can use to call methods of other deployments. This capability lets you divide your application’s steps, such as preprocessing, model inference, and post-processing, into independent deployments that you can independently scale and configure.

Use handle.remote to send requests to a deployment. These requests can contain ordinary Python args and kwargs, which DeploymentHandles can pass directly to the method. The method call returns a DeploymentResponse that represents a future to the output. You can await the response to retrieve its result or pass it to another downstream DeploymentHandle call.

Basic DeploymentHandle example
This example has two deployments:

# File name: hello.py
from ray import serve
from ray.serve.handle import DeploymentHandle


@serve.deployment
class LanguageClassifer:
    def __init__(
        self, spanish_responder: DeploymentHandle, french_responder: DeploymentHandle
    ):
        self.spanish_responder = spanish_responder
        self.french_responder = french_responder

    async def __call__(self, http_request):
        request = await http_request.json()
        language, name = request["language"], request["name"]

        if language == "spanish":
            response = self.spanish_responder.say_hello.remote(name)
        elif language == "french":
            response = self.french_responder.say_hello.remote(name)
        else:
            return "Please try again."

        return await response


@serve.deployment
class SpanishResponder:
    def say_hello(self, name: str):
        return f"Hola {name}"


@serve.deployment
class FrenchResponder:
    def say_hello(self, name: str):
        return f"Bonjour {name}"


spanish_responder = SpanishResponder.bind()
french_responder = FrenchResponder.bind()
language_classifier = LanguageClassifer.bind(spanish_responder, french_responder)
In line 42, the LanguageClassifier deployment takes in the spanish_responder and french_responder as constructor arguments. At runtime, Ray Serve converts these arguments into DeploymentHandles. LanguageClassifier can then call the spanish_responder and french_responder’s deployment methods using this handle.

For example, the LanguageClassifier’s __call__ method uses the HTTP request’s values to decide whether to respond in Spanish or French. It then forwards the request’s name to the spanish_responder or the french_responder on lines 19 and 21 using the DeploymentHandles. The format of the calls is as follows:

response: DeploymentResponse = self.spanish_responder.say_hello.remote(name)
This call has a few parts:

self.spanish_responder is the SpanishResponder handle taken in through the constructor.

say_hello is the SpanishResponder method to invoke.

remote indicates that this is a DeploymentHandle call to another deployment.

name is the argument for say_hello. You can pass any number of arguments or keyword arguments here.

This call returns a DeploymentResponse object, which is a reference to the result, rather than the result itself. This pattern allows the call to execute asynchronously. To get the actual result, await the response. await blocks until the asynchronous call executes and then returns the result. In this example, line 25 calls await response and returns the resulting string.

Warning

You can use the response.result() method to get the return value of remote DeploymentHandle calls. However, avoid calling .result() from inside a deployment because it blocks the deployment from executing any other code until the remote method call finishes. Using await lets the deployment process other requests while waiting for the remote method call to finish. You should use await instead of .result() inside deployments.

You can copy the preceding hello.py script and run it with serve run. Make sure to run the command from a directory containing hello.py, so it can locate the script:

serve run hello:language_classifier
You can use this client script to interact with the example:

# File name: hello_client.py
import requests

response = requests.post(
    "http://localhost:8000", json={"language": "spanish", "name": "Dora"}
)
greeting = response.text
print(greeting)
While the serve run command is running, open a separate terminal window and run the script:

python hello_client.py

Hola Dora
Note

Composition lets you break apart your application and independently scale each part. For instance, suppose this LanguageClassifier application’s requests were 75% Spanish and 25% French. You could scale your SpanishResponder to have 3 replicas and your FrenchResponder to have 1 replica, so you can meet your workload’s demand. This flexibility also applies to reserving resources like CPUs and GPUs, as well as any other configurations you can set for each deployment.

With composition, you can avoid application-level bottlenecks when serving models and business logic steps that use different types and amounts of resources.

Chaining DeploymentHandle calls
Ray Serve can directly pass the DeploymentResponse object that a DeploymentHandle returns, to another DeploymentHandle call to chain together multiple stages of a pipeline. You don’t need to await the first response, Ray Serve manages the await behavior under the hood. When the first call finishes, Ray Serve passes the output of the first call, instead of the DeploymentResponse object, directly to the second call.

For example, the code sample below defines three deployments in an application:

An Adder deployment that increments a value by its configured increment.

A Multiplier deployment that multiplies a value by its configured multiple.

An Ingress deployment that chains calls to the adder and multiplier together and returns the final response.

Note how the response from the Adder handle passes directly to the Multiplier handle, but inside the multiplier, the input argument resolves to the output of the Adder call.

# File name: chain.py
from ray import serve
from ray.serve.handle import DeploymentHandle, DeploymentResponse


@serve.deployment
class Adder:
    def __init__(self, increment: int):
        self._increment = increment

    def __call__(self, val: int) -> int:
        return val + self._increment


@serve.deployment
class Multiplier:
    def __init__(self, multiple: int):
        self._multiple = multiple

    def __call__(self, val: int) -> int:
        return val * self._multiple


@serve.deployment
class Ingress:
    def __init__(self, adder: DeploymentHandle, multiplier: DeploymentHandle):
        self._adder = adder
        self._multiplier = multiplier

    async def __call__(self, input: int) -> int:
        adder_response: DeploymentResponse = self._adder.remote(input)
        # Pass the adder response directly into the multipler (no `await` needed).
        multiplier_response: DeploymentResponse = self._multiplier.remote(
            adder_response
        )
        # `await` the final chained response.
        return await multiplier_response


app = Ingress.bind(
    Adder.bind(increment=1),
    Multiplier.bind(multiple=2),
)

handle: DeploymentHandle = serve.run(app)
response = handle.remote(5)
assert response.result() == 12, "(5 + 1) * 2 = 12"
Streaming DeploymentHandle calls
You can also use DeploymentHandles to make streaming method calls that return multiple outputs. To make a streaming call, the method must be a generator and you must set handle.options(stream=True). Then, the handle call returns a DeploymentResponseGenerator instead of a unary DeploymentResponse. You can use DeploymentResponseGenerators as a sync or async generator, like in an async for code block. Similar to DeploymentResponse.result(), avoid using a DeploymentResponseGenerator as a sync generator within a deployment, as that blocks other requests from executing concurrently on that replica. Note that you can’t pass DeploymentResponseGenerators to other handle calls.

Example:

# File name: stream.py
from typing import AsyncGenerator, Generator

from ray import serve
from ray.serve.handle import DeploymentHandle, DeploymentResponseGenerator


@serve.deployment
class Streamer:
    def __call__(self, limit: int) -> Generator[int, None, None]:
        for i in range(limit):
            yield i


@serve.deployment
class Caller:
    def __init__(self, streamer: DeploymentHandle):
        self._streamer = streamer.options(
            # Must set `stream=True` on the handle, then the output will be a
            # response generator.
            stream=True,
        )

    async def __call__(self, limit: int) -> AsyncGenerator[int, None]:
        # Response generator can be used in an `async for` block.
        r: DeploymentResponseGenerator = self._streamer.remote(limit)
        async for i in r:
            yield i


app = Caller.bind(Streamer.bind())

handle: DeploymentHandle = serve.run(app).options(
    stream=True,
)

# Response generator can also be used as a regular generator in a sync context.
r: DeploymentResponseGenerator = handle.remote(10)
assert list(r) == list(range(10))
Advanced: Pass a DeploymentResponse in a nested object [DEPRECATED]
Warning

Passing a DeploymentResponse to downstream handle calls in nested objects is deprecated and will be removed in the next release. Ray Serve will no longer handle converting them to Ray ObjectRefs for you. Please manually use DeploymentResponse._to_object_ref() instead to pass the corresponding object reference in nested objects.

Passing a DeploymentResponse object as a top-level argument or keyword argument is still supported.

By default, when you pass a DeploymentResponse to another DeploymentHandle call, Ray Serve passes the result of the DeploymentResponse directly to the downstream method once it’s ready. However, in some cases you might want to start executing the downstream call before the result is ready. For example, to do some preprocessing or fetch a file from remote storage. To accomplish this behavior, pass the DeploymentResponse embedded in another Python object, such as a list or dictionary. When you pass responses in a nested object, Ray Serve replaces them with Ray ObjectRefs instead of the resulting value and they can start executing before the result is ready.

The example below has two deployments: a preprocessor and a downstream model that takes the output of the preprocessor. The downstream model has two methods:

pass_by_value takes the output of the preprocessor “by value,” so it doesn’t execute until the preprocessor finishes.

pass_by_reference takes the output “by reference,” so it gets an ObjectRef and executes eagerly.

# File name: response_to_object_ref.py
import asyncio
from typing import List

import ray
from ray import ObjectRef

from ray import serve
from ray.serve.handle import DeploymentHandle, DeploymentResponse


@serve.deployment
class Preprocessor:
    def __call__(self) -> str:
        return "Hello from preprocessor!"


@serve.deployment
class DownstreamModel:
    def pass_by_value(self, result: str) -> str:
        return f"Got result by value: '{result}'"

    async def pass_by_reference(self, wrapped_result: List[ObjectRef]) -> str:
        # Do some preprocessing work before fetching the result.
        await asyncio.sleep(0.1)

        result: str = await wrapped_result[0]
        return f"Got result by reference: '{result}'"


@serve.deployment
class Ingress:
    def __init__(self, preprocessor: DeploymentHandle, downstream: DeploymentHandle):
        self._preprocessor = preprocessor
        self._downstream = downstream

    async def __call__(self, mode: str) -> str:
        preprocessor_response: DeploymentResponse = self._preprocessor.remote()
        if mode == "by_value":
            return await self._downstream.pass_by_value.remote(preprocessor_response)
        else:
            return await self._downstream.pass_by_reference.remote(
                [preprocessor_response]
            )


app = Ingress.bind(Preprocessor.bind(), DownstreamModel.bind())
handle: DeploymentHandle = serve.run(app)

by_value_result = handle.remote(mode="by_value").result()
assert by_value_result == "Got result by value: 'Hello from preprocessor!'"

by_reference_result = handle.remote(mode="by_reference").result()
assert by_reference_result == "Got result by reference: 'Hello from preprocessor!'"
Advanced: Convert a DeploymentResponse to a Ray ObjectRef
Under the hood, each DeploymentResponse corresponds to a Ray ObjectRef, or an ObjectRefGenerator for streaming calls. To compose DeploymentHandle calls with Ray Actors or Tasks, you may want to resolve the response to its ObjectRef. For this purpose, you can use the DeploymentResponse._to_object_ref and DeploymentResponse._to_object_ref_sync developer APIs.

Example:

# File name: response_to_object_ref.py
import ray
from ray import serve
from ray.serve.handle import DeploymentHandle, DeploymentResponse


@ray.remote
def say_hi_task(inp: str):
    return f"Ray task got message: '{inp}'"


@serve.deployment
class SayHi:
    def __call__(self) -> str:
        return "Hi from Serve deployment"


@serve.deployment
class Ingress:
    def __init__(self, say_hi: DeploymentHandle):
        self._say_hi = say_hi

    async def __call__(self):
        # Make a call to the SayHi deployment and pass the result ref to
        # a downstream Ray task.
        response: DeploymentResponse = self._say_hi.remote()
        response_obj_ref: ray.ObjectRef = await response._to_object_ref()
        final_obj_ref: ray.ObjectRef = say_hi_task.remote(response_obj_ref)
        return await final_obj_ref


app = Ingress.bind(SayHi.bind())
handle: DeploymentHandle = serve.run(app)
assert handle.remote().result() == "Ray task got message: 'Hi from Serve deployment'"