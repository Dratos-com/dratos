from abc import abstractmethod
from typing import TYPE_CHECKING, List, Protocol, Type, Union

import numpy as np
import torch
from numpy.typing import NDArray

if TYPE_CHECKING:
    import mlx.core as mx


Array = Union[NDArray, torch.Tensor, List, "mx.array"]


def is_mlx_array_type(array_type):
    try:
        import mlx.core as mx
    except ImportError:
        return False
    return issubclass(array_type, mx.array)


class OutlinesLogitsProcessor(Protocol):
    """
    Base class for logits processors which normalizes types of logits:
    - ndarray (used by llama-cpp-python), converted to torch.Tensor
    - mlx.core.array (used by mlx-lm), converted to torch.Tensor
    - torch.Tensor (used by everything else)

    Normalization of types and conversion to torch.Tensor
    doesn't move memory, it just casts the type.

    Normalizing the types allows all logits processors inheriting from this class
    to implement a single method for all the business logit: `process_logits()`
    """

    @abstractmethod
    def process_logits(
        self, input_ids: List[List[int]], logits: torch.Tensor
    ) -> torch.Tensor:
        """
        input_ids and logits are always 2D tensors for handling a batch of sequences.

        - input_ids -> List[List[tokens]]
        - logits -> 2D_Tensor[logit floats]

        Important to keep in mind when designing universal logits processors
        - logits processors are only used once and never re-applied for a new sequence generator
        - Some models only pass output_ids, some models such as llamacpp and transformers prefix with input_ids
        - Some sampling methods, such as beam search, result in unstable sequence ordering in models like vLLM
        """
        pass

    @torch.no_grad()
    def __call__(
        self,
        input_ids: Array,
        logits: Array,
    ) -> Array:
        """
        Apply logits processor

        1) Unify type
        - convert input_ids: either ndarray, mlx array, List[int], or Tensor -> List[List[int]]
        - convert logits: either ndarray, mlx array, or Tensor -> 2D float Tensor
        2) Unify shape, ensure logits and input_ids are 2D
        3) Call self.process_logits() to perform business logic
        4) Cast logits back to original array library type
        """
        # ensure logits are torch Tensors
        torch_logits = self._to_torch(logits)
        input_ids = self._to_torch(input_ids)

        assert torch_logits.shape[:-1] == input_ids.shape[:-1]

        # Guarantee passed as 2D Tensors, then covert back to original (1D or 2D) shape
        if len(torch_logits.shape) == 2:
            processed_logits = self.process_logits(input_ids.tolist(), torch_logits)
        elif len(torch_logits.shape) == 1:
            processed_logits = self.process_logits(
                [input_ids.tolist()], torch_logits.unsqueeze(0)
            ).squeeze(0)

        # return logits as passed array type
        return self._from_torch(processed_logits, type(logits))

    @staticmethod
    def _to_torch(tensor_like: Array) -> torch.Tensor:
        """Convert various types to torch.Tensor."""
        if isinstance(tensor_like, torch.Tensor):
            return tensor_like

        elif isinstance(tensor_like, np.ndarray):
            return torch.from_numpy(tensor_like)

        elif isinstance(tensor_like, (list, tuple)):
            return torch.tensor(tensor_like)

        elif is_mlx_array_type(type(tensor_like)):
            # mlx -> torch -> mlx conversion docs:
            # https://ml-explore.github.io/mlx/build/html/usage/numpy.html
            return torch.from_dlpack(tensor_like)

        else:
            raise TypeError(
                "LogitsProcessor must be called with either np.NDArray, "
                "torch.Tensor, list, or mlx.core.array typed logits. "
                f"Logits type: `{type(tensor_like)}`"
            )

    @staticmethod
    def _from_torch(tensor: torch.Tensor, target_type: Type) -> Array:
        """Convert torch.Tensor to the specified target type."""
        if target_type == torch.Tensor:
            return tensor

        elif target_type == np.ndarray:
            return tensor.detach().numpy()

        elif target_type == list:
            return tensor.detach().tolist()

        elif target_type == tuple:
            return tuple(tensor.detach().tolist())

        elif is_mlx_array_type(target_type):
            import mlx.core as mx

            # numpy doesn't support bfloat16, mlx doesn't support direct conversion from torch
            return mx.array(tensor.float().numpy())

        else:
            raise TypeError(
                f"Failed to convert torch tensors to target_type `{target_type}`"
            )





import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

import torch
from pydantic import BaseModel

from outlines.fsm.guide import CFGGuide, Guide, RegexGuide
from outlines.fsm.json_schema import build_regex_from_schema, convert_json_schema_to_str

from .base_logits_processor import OutlinesLogitsProcessor

if TYPE_CHECKING:
    from outlines.models.tokenizer import Tokenizer


class GuideLogitsProcessor(OutlinesLogitsProcessor):
    """Bias generation using a finite

    Attributes
    ----------
    tokenizer
        The tokenizer used to convert tokens to ids.
    guide
        The `outlines.fsm.Guide` which is used to bias the logits.
    """

    tokenizer: "Tokenizer"
    guide: Guide
    _guide_states: Dict[int, Any]
    _seq_start_idx: Optional[int]

    def __init__(self, tokenizer: "Tokenizer", guide: Guide):
        """A Guide-based logits processor.

        Parameters
        ----------
        tokenizer
            The tokenizer used to convert tokens to ids.
        guide
            The `outlines.fsm.Guide. which is used to bias the logits.
        """
        self.tokenizer = tokenizer
        self.guide = guide
        self._guide_states = {hash(tuple([])): self.guide.initial_state}
        self._seq_start_idx = None

    def process_logits(
        self, input_ids: List[List[int]], logits: torch.Tensor
    ) -> torch.Tensor:
        """Use the Guide to bias the logits before sampling the next token.

        Parameters
        ----------
        input_ids
            The input token ids.
        logits
            The logits.

        Returns
        -------
        torch.Tensor
            The biased logits.
        """
        if self._seq_start_idx is None:
            self._seq_start_idx = len(input_ids[0])

        sequence_states: List[int] = []  # vector of states corresponding to `input_ids`

        for seq_ids in input_ids:
            gen_ids = seq_ids[self._seq_start_idx :]
            curr_state_key = hash(tuple(gen_ids))

            if curr_state_key not in self._guide_states:
                prev_state = self._guide_states[hash(tuple(gen_ids[:-1]))]
                curr_state = self.guide.get_next_state(prev_state, gen_ids[-1])
                self._guide_states[curr_state_key] = curr_state

            sequence_states.append(self._guide_states[curr_state_key])

        mask = torch.full_like(logits, -math.inf)
        for i, guide_state in enumerate(sequence_states):
            allowed_tokens = self.guide.get_next_instruction(guide_state).tokens
            mask[i, allowed_tokens] = logits[i, allowed_tokens]

        return mask

    def copy(self) -> "GuideLogitsProcessor":
        """Return a copy of the logits processor."""
        return GuideLogitsProcessor(tokenizer=self.tokenizer, guide=self.guide.copy())


class RegexLogitsProcessor(GuideLogitsProcessor):
    """Bias generation based on a regular expression.

    Attributes
    ----------
    tokenizer
        The tokenizer used to convert tokens to ids.
    guide
        The `outlines.fsm.RegexGuide. which is used to bias the logits.
    """

    def __init__(self, regex_string: str, tokenizer: "Tokenizer"):
        """Compile the RegexGuide that drives the regex-guided generation.

        Parameters
        ----------
        regex_string
            A string that represents a regular expression
        tokenizer
            An Outlines tokenizer
        """
        guide = RegexGuide(regex_string, tokenizer)
        super().__init__(tokenizer=tokenizer, guide=guide)


class JSONLogitsProcessor(RegexLogitsProcessor):
    """Bias generation based on a JSON schema.

    Attributes
    ----------
    tokenizer
        The tokenizer used to convert tokens to ids.
    guide
        The `outlines.fsm.RegexGuide. which is used to bias the logits.
    """

    def __init__(
        self,
        schema: Union[dict, Type[BaseModel], str],
        tokenizer: "Tokenizer",
        whitespace_pattern: Optional[str] = None,
    ):
        """Compile the Guide that drives the JSON-guided generation.

        Parameters
        ----------
        schema
            A JSON schema that encodes the structure we want the model to generate.
        tokenizer
            The tokenizer used to convert tokens to ids.
        whitespace_pattern
            Pattern to use for JSON syntactic whitespace (doesn't impact string
            literals). For example, to allow only a single space or newline with
            `whitespace_pattern=r"[\n ]?"`
        """
        schema_str = convert_json_schema_to_str(json_schema=schema)
        regex_string = build_regex_from_schema(schema_str, whitespace_pattern)
        super().__init__(regex_string=regex_string, tokenizer=tokenizer)


class CFGLogitsProcessor(GuideLogitsProcessor):
    """Bias generation based on a context-free grammar.

    Attributes
    ----------
    tokenizer
        The tokenizer used to convert tokens to ids.
    guide
        The `outlines.fsm.CFGGuide. which is used to bias the logits.
    """

    guide: CFGGuide

    def __init__(self, cfg_str: str, tokenizer: "Tokenizer"):
        """Compile the CFGGuide that drives the CFG-guided generation.

        Parameters
        ----------
        cfg_str
            A string that represents a grammar
        tokenizer
            The tokenizer used to convert tokens to ids.
        """
        cfg_guide = CFGGuide(cfg_string=cfg_str, tokenizer=tokenizer)
        super().__init__(tokenizer=tokenizer, guide=cfg_guide)

    def process_logits(
        self, input_ids: List[List[int]], logits: torch.Tensor
    ) -> torch.Tensor:
        """Same behavior as GuideLogitsProcessor, but uses rejection sampling"""
        if self._seq_start_idx is None:
            self._seq_start_idx = len(input_ids[0])

        sequence_states: List = []  # vector of states corresponding to `input_ids`

        for seq_ids in input_ids:
            gen_ids = seq_ids[self._seq_start_idx :]
            curr_state_key = hash(tuple(gen_ids))

            if curr_state_key not in self._guide_states:
                prev_state = self._guide_states[hash(tuple(gen_ids[:-1]))]
                curr_state = self.guide.get_next_state(prev_state, gen_ids[-1])
                self._guide_states[curr_state_key] = curr_state

            sequence_states.append(self._guide_states[curr_state_key])

        mask = torch.full_like(logits, -math.inf)
        for i, guide_state in enumerate(sequence_states):
            first_legal_token = next(
                self.guide.iter_valid_token_ids(
                    guide_state, torch.argsort(logits[i], descending=True)
                )
            )
            mask[i, [first_legal_token]] = logits[i, [first_legal_token]]

        return mask




#  _______________________________
# / Don't want to self-host?      \
# \ Try .json at http://dottxt.co /
#  -------------------------------
#        \   ^__^
#         \  (oo)\_______
#            (__)\       )\/\
#                ||----w |
#                ||     ||
#
#
# Copyright 2024- the Outlines developers
# Copyright 2023 the vLLM developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

from outlines.models.vllm import adapt_tokenizer
from outlines.processors import JSONLogitsProcessor, RegexLogitsProcessor

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()
engine = None
tokenizer = None


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - schema: the JSON schema to use for the generation (if regex is not provided).
    - regex: the regex to use for the generation (if schema is not provided).
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    assert engine is not None

    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)

    json_schema = request_dict.pop("schema", None)
    regex_string = request_dict.pop("regex", None)
    if json_schema is not None:
        logits_processors = [JSONLogitsProcessor(json_schema, tokenizer)]
    elif regex_string is not None:
        logits_processors = [RegexLogitsProcessor(regex_string, tokenizer)]
    else:
        logits_processors = []

    sampling_params = SamplingParams(
        **request_dict, logits_processors=logits_processors  # type: ignore
    )
    request_id = random_uuid()

    results_generator = engine.generate(prompt, sampling_params, request_id)  # type: ignore

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            text_outputs = [prompt + output.text for output in request_output.outputs]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)  # type: ignore
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    text_outputs = [prompt + output.text for output in final_output.outputs]
    ret = {"text": text_outputs}
    return JSONResponse(ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    # Adds the `engine_use_ray`,  `disable_log_requests` and `max_log_len`
    # arguments
    engine_args: AsyncEngineArgs = AsyncEngineArgs.from_cli_args(args)  # type: ignore

    # Sets default for the model (`facebook/opt-125m`)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = adapt_tokenizer(tokenizer=engine.engine.tokenizer.tokenizer)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
    )


    import datetime
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Union

from outlines.generate.generator import sequence_generator
from outlines.samplers import BeamSearchSampler, GreedySampler, MultinomialSampler

if TYPE_CHECKING:
    import torch

FormattedOutput = Union[
    str, int, float, bool, datetime.date, datetime.time, datetime.datetime
]


class SequenceGenerator:
    def __init__(
        self,
        fsm,
        model,
        sampler,
        device,
    ):
        self.fsm = fsm
        self.model = model
        self.sampler = sampler
        self.tokenizer = model.tokenizer
        self.device = device
        self.num_samples = sampler.samples

    def get_generated_token_ids(
        self,
        prompt_token_ids: "torch.Tensor",
        token_ids: "torch.Tensor",
    ) -> List["torch.Tensor"]:
        """Get the tokens generated so far.

        Parameters
        ----------
        prompt_token_ids
            Tensor that contains the token ids of the sequences' prompts.
        token_ids
            The generated token ids.

        Returns
        -------
        A tensor that contains the token ids that have been generated so far.

        """
        prompt_lengths = [len(prompt) for prompt in prompt_token_ids]
        token_ids = [
            cur_token_ids[length:]
            for cur_token_ids, length in zip(token_ids, prompt_lengths)
        ]

        return token_ids

    def is_stop_sequence_found(
        self, generated_sequences: List[str], stop_sequences: List[str]
    ) -> bool:
        """Determine whether one of the stop sequences has been generated.

        Parameters
        ----------
        generated_sequences
            The list of sequences generated so far.
        stop_sequences
            The list that contains the sequence which stop the generation when
            found.

        Returns
        -------
        True if at least one of the stop sequences has been found in each generated
        sequence.

        """
        return all(
            [
                any([seq in generated for seq in stop_sequences])
                for generated in generated_sequences
            ]
        )

    def strip_stop_sequences(
        self, sequence: str, stop_sequences: Optional[List[str]]
    ) -> str:
        """Remove the stop sequences from the generated sequences.

        Parameters
        ----------
        sequence
            One of the generated sequences.
        stop_sequences
            The list that contains the sequence which stop the generation when
            found.

        """
        if stop_sequences:
            match_indexes = [sequence.find(seq) for seq in stop_sequences]
            if any([index != -1 for index in match_indexes]):
                # select the stop_sequence that is found first in the sequence
                min_match_index_value = min([i for i in match_indexes if i != -1])
                min_match_index_pos = match_indexes.index(min_match_index_value)
                sequence = sequence[
                    : match_indexes[min_match_index_pos]
                    + len(stop_sequences[min_match_index_pos])
                ]

        return sequence

    def format_sequence(self, sequence: str) -> FormattedOutput:
        """Translate the generated sequence to another type.

        This method is for instance overridden when generating JSON to either
        return a dictionnary or a Pydantic model.

        Parameters
        ----------
        sequence
            A generated sequences.

        Returns
        -------
        The formatted sequence.

        """
        return sequence

    def __call__(
        self,
        prompts: Union[str, List[str]],
        max_tokens: Optional[int] = None,
        stop_at: Optional[Union[str, List[str]]] = None,
        rng: Optional["torch.Generator"] = None,
    ) -> Union[FormattedOutput, List[FormattedOutput], List[List[FormattedOutput]]]:
        """Generate the full text sequence.

        Since `SequenceGenerator.stream` calls the tokenizer at every step this
        method loops over the generator returned by `sequence_generator` itself
        so the tokenizer is called only once after all token ids have been
        generated.

        Parameters
        ----------
        prompts
            A string or list of strings that are passed to the model before
            generating the first token.
        max_tokens
            An integer representing maximum number of tokens that will be generated
            (per prompt)
        stop_at
            A string or list of strings at which the text generated will stop
        rng
            The random number generator. Defaults to a non-seeded `torch.Generator`
            instance.

        Returns
        -------
        The generation(s), potentially cast to another type.
        """
        import torch

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(stop_at, str):
            stop_at = [stop_at]

        stop_sequences = stop_at
        num_samples = self.num_samples

        if rng is None:
            rng = torch.Generator(device=self.device)
            rng.seed()

        prompt_token_ids, attention_masks = self.tokenizer.encode(prompts)
        prompt_token_ids = prompt_token_ids.to(self.device)
        attention_masks = attention_masks.to(self.device)

        # To draw multiple samples we repeat the prompt as many times
        # as there are samples. We copy the FSMs and initialize the
        # FSM states.
        num_samples = self.num_samples
        batch_size = len(prompts)

        prompt_token_ids = torch.repeat_interleave(prompt_token_ids, num_samples, dim=0)
        attention_masks = torch.repeat_interleave(attention_masks, num_samples, dim=0)
        fsm_states = [0 for _ in range(batch_size * num_samples)]
        fsms = [self.fsm.copy() for _ in range(batch_size * num_samples)]
        weights = torch.zeros(
            (batch_size * num_samples), dtype=torch.float, device=self.device
        )

        states = sequence_generator(
            self.model,
            self.sampler,
            fsms,
            prompt_token_ids,
            weights,
            attention_masks,
            fsm_states,
            rng=rng,
        )

        while True:
            try:
                last_state = next(states)
                if max_tokens or stop_sequences:
                    token_ids = last_state.token_ids
                    generated_token_ids = self.get_generated_token_ids(
                        prompt_token_ids, token_ids
                    )
                    if max_tokens and len(generated_token_ids[0]) >= max_tokens:
                        break
                    if stop_sequences and self.is_stop_sequence_found(
                        self.tokenizer.decode(generated_token_ids), stop_sequences
                    ):
                        break
            except StopIteration:
                break

        token_ids = last_state.token_ids
        generated_token_ids = self.get_generated_token_ids(prompt_token_ids, token_ids)

        generated = self.tokenizer.decode(generated_token_ids)
        stripped = [
            self.strip_stop_sequences(sequence, stop_sequences)
            for sequence in generated
        ]
        formatted = [self.format_sequence(sequence) for sequence in stripped]

        # We reshape the output to (batch_size, sample_size)
        output: List[List[FormattedOutput]] = list()
        for i in range(0, batch_size * num_samples, num_samples):
            output.append(formatted[i : i + num_samples])

        # We remove leading dimensions for the output
        if batch_size == 1 and num_samples == 1:
            return output[0][0]
        elif batch_size == 1:
            return output[0]
        elif num_samples == 1:
            return [samples[0] for samples in output]
        else:
            return output

    def stream(
        self,
        prompts: Union[str, List[str]],
        max_tokens: Optional[int] = None,
        stop_at: Optional[Union[str, List[str]]] = None,
        rng: Optional["torch.Generator"] = None,
    ) -> Iterator[Union[List[str], str, List[List[str]]]]:
        """Generate the text sequence one token at a time.

        Since `Tokenizer.decode` strips the whitespaces from the tokens we have no
        choice but to decode the generated token ids at each step and compare the
        current decoded strings to the previously decoded strings.

        Parameters
        ----------
        prompts
            A string or list of strings that are passed to the model before
            generating the first token.
        max_tokens
            An integer representing maximum number of tokens that will be generated
            (per prompt)
        stop_at
            A string or list of strings at which the text generated will stop
        rng
            The random number generator. Defaults to a non-seeded `torch.Generator`
            instance.

        Returns
        -------
        A string or list of strings that contain the generated text.

        """
        import torch

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(stop_at, str):
            stop_at = [stop_at]

        stop_sequences = stop_at
        num_samples = self.num_samples

        prompt_token_ids, attention_masks = self.tokenizer.encode(prompts)
        prompt_token_ids = prompt_token_ids.to(self.device)
        attention_masks = attention_masks.to(prompt_token_ids.device)

        # To draw multiple samples we repeat the prompt as many times
        # as there are samples. We copy the FSMs and initialize the
        # FSM states.
        num_samples = self.num_samples
        batch_size = len(prompts)

        prompt_token_ids = torch.repeat_interleave(prompt_token_ids, num_samples, dim=0)
        attention_masks = torch.repeat_interleave(attention_masks, num_samples, dim=0)
        fsm_states = [0 for _ in range(batch_size * num_samples)]
        fsms = [self.fsm.copy() for _ in range(batch_size * num_samples)]
        weights = torch.zeros(
            (batch_size * num_samples),
            dtype=torch.float,
            device=prompt_token_ids.device,
        )

        if rng is None:
            rng = torch.Generator(device=prompt_token_ids.device)
            rng.seed()

        states = sequence_generator(
            self.model,
            self.sampler,
            fsms,
            prompt_token_ids,
            weights,
            attention_masks,
            fsm_states,
            rng=rng,
        )

        def token_generator() -> Iterator[Union[List[str], str, List[List[str]]]]:
            previously_generated_sequences = [
                "" for _ in range(batch_size)
            ] * num_samples
            num_generated = 0
            is_stop_at_reached = [False for _ in range(batch_size)] * num_samples
            while True:
                if (max_tokens and num_generated >= max_tokens) or all(
                    is_stop_at_reached
                ):
                    return
                try:
                    sequence = next(states)
                    num_generated += 1
                except StopIteration:
                    return
                generated_token_ids = sequence.token_ids[:, -num_generated:]
                generated_sequences = self.tokenizer.decode(generated_token_ids)
                if stop_sequences:
                    is_stop_at_reached = [
                        stop
                        or self.is_stop_sequence_found(
                            [generated_sequence], stop_sequences
                        )
                        for generated_sequence, stop in zip(
                            generated_sequences, is_stop_at_reached
                        )
                    ]

                    generated_sequences = [
                        self.format_sequence(
                            self.strip_stop_sequences(sequence, stop_sequences)
                        )
                        if stop
                        else sequence
                        for sequence, stop in zip(
                            generated_sequences, is_stop_at_reached
                        )
                    ]
                next_tokens = [
                    token[len(sequence) :]
                    for token, sequence, stop in zip(
                        generated_sequences,
                        previously_generated_sequences,
                        is_stop_at_reached,
                    )
                ]
                previously_generated_sequences = generated_sequences
                # We reshape the output to (batch_size, sample_size)
                output: List[List[str]] = list()
                for i in range(0, batch_size * num_samples, num_samples):
                    output.append(next_tokens[i : i + num_samples])

                # We remove leading dimensions for the output
                if batch_size == 1 and num_samples == 1:
                    yield output[0][0]
                elif batch_size == 1:
                    yield output[0]
                elif num_samples == 1:
                    yield [samples[0] for samples in output]
                else:
                    yield output

        return token_generator()


@dataclass(frozen=True)
class GenerationParameters:
    """Generation parameters used in Outlines' public API."""

    max_tokens: Optional[int]
    stop_at: Optional[Union[str, List[str]]]
    seed: Optional[int]


@dataclass(frozen=True)
class SamplingParameters:
    """Sampling parameters available in Outlines."""

    sampler: str
    num_samples: int = 1
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    temperature: Optional[float] = None


class SequenceGeneratorAdapter:
    """Class used to unify the interface to the model providers'
    generation functions.

    Attributes
    ----------
    model
        The wrapped model.
    logits_processor
        The logits processor to use to generate text.
    sampler
        The sampler to use to generate text.

    """

    def __init__(self, model, logits_processor, sampler):
        self.model = model
        self.logits_processor = logits_processor

        if isinstance(sampler, MultinomialSampler):
            self.sampling_params = SamplingParameters(
                "multinomial",
                sampler.samples,
                sampler.top_p,
                sampler.top_k,
                sampler.temperature,
            )
        elif isinstance(sampler, GreedySampler):
            self.sampling_params = SamplingParameters(
                "greedy", sampler.samples, None, None, 0.0
            )
        elif isinstance(sampler, BeamSearchSampler):
            self.sampling_params = SamplingParameters(
                "beam_search", sampler.samples, None, None, 1.0
            )

    def prepare_generation_parameters(
        self,
        max_tokens: Optional[int],
        stop_at: Optional[Union[str, List[str]]],
        seed: Optional[int],
    ):
        if isinstance(stop_at, str):
            stop_at = [stop_at]

        generation_params = GenerationParameters(
            max_tokens,
            stop_at,
            seed,
        )

        return generation_params

    def format_sequence(self, sequence: str) -> FormattedOutput:
        """Translate the generated sequence to another type.

        This method is for instance overridden when generating JSON to either
        return a dictionnary or a Pydantic model.

        Parameters
        ----------
        sequence
            A generated sequences.

        Returns
        -------
        The formatted sequence.

        """
        return sequence

    def _format(self, sequences):
        """Apply formatting to every string in a completion."""
        if isinstance(sequences, list):
            return [self._format(sequence) for sequence in sequences]
        else:
            return self.format_sequence(sequences)

    def __call__(
        self,
        prompts: Union[str, List[str]],
        max_tokens: Optional[int] = None,
        stop_at: Optional[Union[str, List[str]]] = None,
        seed: Optional[int] = None,
        **model_specific_params,
    ):
        """Generate text from a prompt of list of prompts."""

        generation_params = self.prepare_generation_parameters(
            max_tokens, stop_at, seed
        )

        completions = self.model.generate(
            prompts,
            generation_params,
            self.logits_processor,
            self.sampling_params,
            **model_specific_params,
        )

        return self._format(completions)

    def stream(
        self,
        prompts: Union[str, List[str]],
        max_tokens: Optional[int] = None,
        stop_at: Optional[Union[str, List[str]]] = None,
        seed: Optional[int] = None,
        **model_specific_params,
    ):
        """Return a text generator from a prompt or a list of prompts."""
        generation_params = self.prepare_generation_parameters(
            max_tokens, stop_at, seed
        )
        return self.model.stream(
            prompts,
            generation_params,
            self.logits_processor,
            self.sampling_params,
            **model_specific_params,
        )


class VisionSequenceGeneratorAdapter(SequenceGeneratorAdapter):
    def __call__(  # type: ignore
        self,
        prompts: Union[str, List[str]],
        media: Union[str, Any],
        max_tokens: Optional[int] = None,
        stop_at: Optional[Union[str, List[str]]] = None,
        seed: Optional[int] = None,
        **model_specific_params,
    ):
        """
        Generate text from a prompt of list of prompts.

        Media: A URI to construct media or media object itself. Used as AutoProcessor argument.
        """
        prompts, media = self._validate_prompt_media_types(prompts, media)

        generation_params = self.prepare_generation_parameters(
            max_tokens, stop_at, seed
        )

        completions = self.model.generate(
            prompts,
            media,
            generation_params,
            self.logits_processor,
            self.sampling_params,
            **model_specific_params,
        )

        return self._format(completions)

    def stream(  # type: ignore
        self,
        prompts: Union[str, List[str]],
        media: List[Union[str, Any, List[Union[str, Any]]]],
        max_tokens: Optional[int] = None,
        stop_at: Optional[Union[str, List[str]]] = None,
        seed: Optional[int] = None,
        **model_specific_params,
    ):
        """Return a text generator from a prompt or a list of prompts."""
        prompts, media = self._validate_prompt_media_types(prompts, media)
        generation_params = self.prepare_generation_parameters(
            max_tokens, stop_at, seed
        )
        return self.model.stream(
            prompts,
            media,
            generation_params,
            self.logits_processor,
            self.sampling_params,
            **model_specific_params,
        )

    @classmethod
    def _validate_prompt_media_types(
        cls,
        prompts: Union[str, List[str]],
        media: Union[str, Any, List[Union[str, Any]]],
    ) -> Union[Any, List[Any]]:
        """
        Prepare media as PIL.Image and ensure for every prompt str there is one List[PIL.Image]
        """

        def valid_types(prompts, media):
            from PIL import Image  # type: ignore

            if isinstance(prompts, list):
                if not isinstance(media, list) or len(prompts) != len(media):
                    return False
                for subprompt, submedia in zip(prompts, media):
                    if not isinstance(subprompt, str) or not all(
                        isinstance(m, Image.Image) for m in submedia
                    ):
                        return False
            elif isinstance(prompts, str):
                if not all(isinstance(m, Image.Image) for m in media):
                    return False
            return True

        if not valid_types(prompts, media):
            raise TypeError(
                "Expected (prompts, media) to be of type "
                "(str, List[Image])), or (List[str], List[List[Image]]) "
                f"instead got prompts={prompts}, media={media}"
            )

        return prompts, media


        from functools import singledispatch

from outlines.generate.api import (
    SequenceGeneratorAdapter,
    VisionSequenceGeneratorAdapter,
)
from outlines.models import ExLlamaV2Model, LlamaCpp, OpenAI, TransformersVision
from outlines.samplers import Sampler, multinomial


@singledispatch
def cfg(
    model, cfg_str: str, sampler: Sampler = multinomial()
) -> SequenceGeneratorAdapter:
    """Generate text in the language of a Context-Free Grammar

    Arguments
    ---------
    model:
        An `outlines.model` instance.
    sampler:
        The sampling algorithm to use to generate token ids from the logits
        distribution.

    Returns
    -------
    A `SequenceGeneratorAdapter` instance that generates text.

    """
    from outlines.processors import CFGLogitsProcessor

    logits_processor = CFGLogitsProcessor(cfg_str, tokenizer=model.tokenizer)
    return SequenceGeneratorAdapter(model, logits_processor, sampler)


@cfg.register(TransformersVision)
def cfg_vision(model, cfg_str: str, sampler: Sampler = multinomial()):
    from outlines.processors import CFGLogitsProcessor

    logits_processor = CFGLogitsProcessor(cfg_str, tokenizer=model.tokenizer)
    return VisionSequenceGeneratorAdapter(model, logits_processor, sampler)


@cfg.register(ExLlamaV2Model)
def cfg_exllamav2(model, cfg_str: str, sampler: Sampler = multinomial()):
    raise NotImplementedError(
        "Not yet available, track progress in https://github.com/outlines-dev/outlines/pull/1010"
    )


@cfg.register(LlamaCpp)
def cfg_llamacpp(model, cfg_str: str, sampler: Sampler = multinomial()):
    raise NotImplementedError("Not yet available due to bug in llama_cpp tokenizer")


@cfg.register(OpenAI)
def cfg_openai(model, cfg_str: str, sampler: Sampler = multinomial()):
    raise NotImplementedError(
        "Cannot use grammar-structured generation with an OpenAI model"
        + "due to the limitations of the OpenAI API."
    )

    from functools import singledispatch
from typing import Callable, List

from outlines.generate.api import SequenceGeneratorAdapter
from outlines.models import OpenAI
from outlines.samplers import Sampler, multinomial

from .regex import regex


@singledispatch
def choice(
    model, choices: List[str], sampler: Sampler = multinomial()
) -> SequenceGeneratorAdapter:
    regex_str = r"(" + r"|".join(choices) + r")"

    generator = regex(model, regex_str, sampler)
    generator.format_sequence = lambda x: x

    return generator


@choice.register(OpenAI)
def choice_openai(
    model: OpenAI, choices: List[str], sampler: Sampler = multinomial()
) -> Callable:
    if not isinstance(sampler, multinomial):
        raise NotImplementedError(
            r"The OpenAI API does not support any other sampling algorithm "
            + "that the multinomial sampler."
        )

    def generate_choice(prompt: str, max_tokens: int = 1):
        return model.generate_choice(prompt, choices, max_tokens)

    return generate_choice


    from functools import singledispatch

from outlines.fsm.types import python_types_to_regex
from outlines.generate.api import SequenceGeneratorAdapter
from outlines.models import OpenAI
from outlines.samplers import Sampler, multinomial

from .regex import regex


@singledispatch
def format(
    model, python_type, sampler: Sampler = multinomial()
) -> SequenceGeneratorAdapter:
    """Generate structured data that can be parsed as a Python type.

    Parameters
    ----------
    model:
        An instance of `Transformer` that represents a model from the
        `transformers` library.
    python_type:
        A Python type. The output of the generator must be parseable into
        this type.
    sampler:
        The sampling algorithm to use to generate token ids from the logits
        distribution.

    Returns
    -------
    A `SequenceGenerator` instance that generates text constrained by the Python type
    and translates this text into the corresponding type.

    """
    regex_str, format_fn = python_types_to_regex(python_type)
    generator = regex(model, regex_str, sampler)
    generator.format_sequence = format_fn

    return generator


@format.register(OpenAI)
def format_openai(model, python_type, sampler: Sampler = multinomial()):
    raise NotImplementedError(
        "Cannot use Python type-structured generation with an OpenAI model"
        + " due to the limitations of the OpenAI API."
    )


    import dataclasses
import math
from typing import TYPE_CHECKING, Callable, Iterable, Iterator, List, Optional, Tuple

if TYPE_CHECKING:
    import torch

    from outlines.fsm.guide import Guide


class ContextLengthExceededError(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class GenerationState:
    token_ids: "torch.Tensor"
    kv_cache: "torch.Tensor"
    logits: "torch.Tensor"
    weights: "torch.Tensor"
    fsm_states: List[int]


def sequence_generator(
    model: Callable,
    sampler: Callable,
    fsms: List["Guide"],
    token_ids: "torch.Tensor",
    sequence_weights: "torch.Tensor",
    attention_masks: "torch.Tensor",
    fsm_states: List[int],
    rng: "torch.Generator",
) -> Iterator[GenerationState]:
    """Generates sequences of tokens.

    Parameters
    ----------
    model
        A callable that generates a probability distribution over the
        vocabulary when passed a tensor of token ids.
    sampler
        A callable that returns the next token ids, their ancestor sequence and
        the updated sequence weights when passed a distribution over the
        vocabulary.
    token_ids
        A tensor of token ids on which the sequence distribution is conditioned, of
        shape ``(n_seqs, n_prompt_tokens)``
    sequence_weights
        A tensor that contains the initial weights of the sequences, of shape
        ``(n_seqs,)``
    attention_masks
        A tensor of tensors that represent the tokens considered at the attention
        layer, of shape ``(n_seqs, n_prompt_tokens)``.
    fsms
        List of finite-state machines that drive the text generation,
        one for each sequence in the batch.
    fsm_states
        The initial states of the finite-state machine for each sequence in the batch.

    Yields
    ------
    A new sequence.

    """
    import torch

    if rng is None:
        rng = torch.Generator()

    kv_cache = None

    while True:
        try:
            logits, kv_cache = model(token_ids, attention_masks, kv_cache)
        except IndexError:  # Exceeding the context length
            raise ContextLengthExceededError(
                "The input length exceeds the context length of the model."
            )

        allowed_tokens = get_allowed_tokens(fsms, fsm_states)
        biased_logits = bias_logits(logits, allowed_tokens)
        next_token_ids, ancestors, sequence_weights = sampler(
            biased_logits, sequence_weights, rng
        )

        token_ids = update_token_ids(token_ids, next_token_ids, ancestors)
        attention_masks = update_attention_masks(attention_masks, ancestors)
        kv_cache = reorder_kv_cache(kv_cache, ancestors)
        if len(ancestors) > 1:
            fsms = reorder_fsms(fsms, ancestors)
            fsm_states = reorder_fsm_states(fsm_states, ancestors)

        fsm_states = get_next_fsm_states(fsms, fsm_states, next_token_ids)
        is_finished = is_generation_finished(fsms, fsm_states)

        if is_finished:
            yield GenerationState(
                token_ids,
                kv_cache,
                logits,
                sequence_weights,
                fsm_states,
            )
            return

        yield GenerationState(
            token_ids,
            kv_cache,
            logits,
            sequence_weights,
            fsm_states,
        )


def get_next_fsm_states(
    fsms: List["Guide"], fsm_states: List[int], next_token_ids: "torch.Tensor"
) -> List[int]:
    """

    Parameters
    ----------
    fsm
        The finite-state machine used to monitor this batch.
    next_token_ids
        The tokens that were just generated.

    Returns
    -------
    A `torch.Tensor` object that represents the next logit mask.

    """
    return [
        fsm.get_next_state(fsm_state, int(token_id[0]))
        for fsm, fsm_state, token_id in zip(fsms, fsm_states, next_token_ids)
    ]


def get_allowed_tokens(
    fsms: List["Guide"], fsm_states: List[int]
) -> List[Optional[Iterable[int]]]:
    """Get the new instructions for each sequence from the finite-state machine.

    Parameters
    ----------
    fsm
        The finite-state machine used to monitor this batch.
    fsm_states
        The FSM states corresponding to each sequence in the batch.

    Returns
    -------
    A nested list that contains the ids of the logits to keep.

    """
    return [
        fsm.get_next_instruction(state).tokens for fsm, state in zip(fsms, fsm_states)
    ]


def is_generation_finished(fsms: List["Guide"], fsm_states: List[int]) -> bool:
    """Determine if the generation is finished.

    A generation is considered finished if the FSM of every sequence in the
    batch is in a final state.

    A better solution is to return finished sequences as soon as their FSM
    is in a final state.

    Parameters
    ----------
    fsm
        The finite-state machine used to monitor this batch.
    fsm_states
        The FSM states corresponding to each sequence in the batch.

    Returns
    -------
    Whether all sequences are finished sampling.

    """
    return all([fsm.is_final_state(state) for fsm, state in zip(fsms, fsm_states)])


def update_token_ids(
    token_ids: "torch.Tensor", next_token_ids: "torch.Tensor", ancestors: "torch.Tensor"
) -> "torch.Tensor":
    """Append the sampled tokens to the running sequence of tokens.

    Parameters
    ----------
    token_ids
        The current token sequences
    next_token_ids
        The tokens that were just generated and that we need to append
        to the existing sequences.
    ancestors
        The sequences to which the token ids need to be added.

    Returns
    -------
    A new sequence of token ids that contains the tokens that were
    just generated.

    """
    import torch

    token_ids = torch.index_select(token_ids, 0, ancestors)
    return torch.concatenate([token_ids, next_token_ids], dim=-1)


def update_attention_masks(
    attention_masks: "torch.Tensor", ancestors: "torch.Tensor"
) -> "torch.Tensor":
    """Expand the attention masks.

    Parameters
    ----------
    attention_masks
        The attention masks for each sequence in the batch.
    ancestors
        The sequences to which the token ids need to be added.

    Returns
    -------
    The attention masks padded with 1s.

    """
    import torch

    attention_masks = torch.index_select(attention_masks, 0, ancestors)
    return torch.concatenate(
        [
            attention_masks,
            torch.ones(
                attention_masks.shape[:-1] + (1,), device=attention_masks.device
            ),
        ],
        axis=-1,
    )


def reorder_fsms(fsms: List["Guide"], ancestors: "torch.Tensor") -> List["Guide"]:
    reordered_fsms = []
    for ancestor in ancestors:
        reordered_fsms.append(fsms[ancestor].copy())

    return reordered_fsms


def reorder_fsm_states(fsm_states: List[int], ancestors: "torch.Tensor") -> List[int]:
    reordered_states = []
    for ancestor in ancestors:
        reordered_states.append(fsm_states[ancestor])

    return reordered_states


def reorder_kv_cache(
    kv_cache: Optional[Tuple], ancestors: "torch.Tensor"
) -> Optional[Tuple]:
    """Re-order the KV-cache based on the ancestors.

    In transformers, the object that stores the KV-cache is a tuple who elements
    are the key cache and the value cache. Each of these caches are tuples where
    each element correpond to a layer. To each layer corresponds a tensor whose
    first dimension is the batch size.

    """
    import torch

    if kv_cache is None:
        return None

    new_kv_cache: Tuple = tuple()
    for cache_item in kv_cache:
        new_cache_item: Tuple = tuple()
        for layer in cache_item:
            layer = torch.index_select(layer, 0, ancestors.to(layer.device))
            new_cache_item += (layer,)
        new_kv_cache += (new_cache_item,)

    return new_kv_cache


def bias_logits(logits: "torch.Tensor", allowed_token_ids: List) -> "torch.Tensor":
    """Mask the logits.

    The function iterates over a nested list where each list corresponds to the
    indices that need to be masked for each row in the array.

    Parameters
    ----------
    logits
        Two dimensional tensor that contains the next-token probability
        distribution.
    allowed_token_ids
        A list that contains the tokens that can be generated by the model.

    Returns
    -------
    A view of the original logits tensor where some values are masked.

    """
    import torch

    biased_logits = torch.full_like(logits, -math.inf, device=logits.device)
    for i, ids in enumerate(allowed_token_ids):
        if ids is not None:
            biased_logits[i, ids] = logits[i, ids]
        else:
            biased_logits[i] = logits[i]
    return biased_logits


    import json as pyjson
from functools import singledispatch
from typing import Callable, Optional, Union

from pydantic import BaseModel

from outlines.fsm.json_schema import build_regex_from_schema, get_schema_from_signature
from outlines.generate.api import SequenceGeneratorAdapter
from outlines.models import OpenAI
from outlines.samplers import Sampler, multinomial

from .regex import regex


@singledispatch
def json(
    model,
    schema_object: Union[str, object, Callable],
    sampler: Sampler = multinomial(),
    whitespace_pattern: Optional[str] = None,
) -> SequenceGeneratorAdapter:
    """
    Generate structured JSON data with a `Transformer` model based on a specified JSON Schema.

    Parameters
    ----------
    model:
        An instance of `Transformer` that represents a model from the
        `transformers` library.
    schema_object:
        The JSON Schema to generate data for. Can be a JSON string, a Pydantic model, or a callable
        that returns a JSON schema.
    sampler:
        The sampling algorithm to use to generate token ids from the logits
        distribution.
    whitespace_pattern
        Pattern to use for JSON syntactic whitespace (doesn't impact string literals)
        Example: allow only a single space or newline with `whitespace_pattern=r"[\n ]?"`

    Returns
    -------
    A `SequenceGenerator` instance that generates text constrained by the schema_object and
    transforms the result if BaseModel is used.

    """
    if isinstance(schema_object, type(BaseModel)):
        schema = pyjson.dumps(schema_object.model_json_schema())
        regex_str = build_regex_from_schema(schema, whitespace_pattern)
        generator = regex(model, regex_str, sampler)
        generator.format_sequence = lambda x: schema_object.parse_raw(x)
    elif callable(schema_object):
        schema = pyjson.dumps(get_schema_from_signature(schema_object))
        regex_str = build_regex_from_schema(schema, whitespace_pattern)
        generator = regex(model, regex_str, sampler)
        generator.format_sequence = lambda x: pyjson.loads(x)
    elif isinstance(schema_object, str):
        schema = schema_object
        regex_str = build_regex_from_schema(schema, whitespace_pattern)
        generator = regex(model, regex_str, sampler)
        generator.format_sequence = lambda x: pyjson.loads(x)
    else:
        raise ValueError(
            f"Cannot parse schema {schema_object}. The schema must be either "
            + "a Pydantic object, a function or a string that contains the JSON "
            + "Schema specification"
        )

    return generator


@json.register(OpenAI)
def json_openai(
    model, schema_object: Union[str, object, Callable], sampler: Sampler = multinomial()
):
    raise NotImplementedError(
        "Cannot use JSON Schema-structure generation with an OpenAI model "
        + "due to the limitations of the OpenAI API"
    )




    from functools import singledispatch

from outlines.fsm.guide import RegexGuide
from outlines.generate.api import (
    SequenceGenerator,
    SequenceGeneratorAdapter,
    VisionSequenceGeneratorAdapter,
)
from outlines.models import ExLlamaV2Model, OpenAI, TransformersVision
from outlines.samplers import Sampler, multinomial


@singledispatch
def regex(model, regex_str: str, sampler: Sampler = multinomial()):
    """Generate structured text in the language of a regular expression.

    Parameters
    ----------
    model:
        An instance of `Transformer` that represents a model from the
        `transformers` library.
    regex_str:
        The regular expression that the output must follow.
    sampler:
        The sampling algorithm to use to generate token ids from the logits
        distribution.

    Returns
    -------
    A `SequenceGeneratorAdapter` instance that generates text constrained by the
    regular expression.

    """
    from outlines.processors import RegexLogitsProcessor

    logits_processor = RegexLogitsProcessor(regex_str, tokenizer=model.tokenizer)
    return SequenceGeneratorAdapter(model, logits_processor, sampler)


@regex.register(TransformersVision)
def regex_vision(
    model,
    regex_str: str,
    sampler: Sampler = multinomial(),
):
    from outlines.processors import RegexLogitsProcessor

    logits_processor = RegexLogitsProcessor(regex_str, tokenizer=model.tokenizer)
    return VisionSequenceGeneratorAdapter(model, logits_processor, sampler)


@regex.register(ExLlamaV2Model)
def regex_exllamav2(
    model,
    regex_str: str,
    sampler: Sampler = multinomial(),
) -> SequenceGenerator:
    fsm = RegexGuide(regex_str, model.tokenizer)

    device = model.device
    generator = SequenceGenerator(fsm, model, sampler, device)

    return generator


@regex.register(OpenAI)
def regex_openai(
    model: OpenAI,
    regex_str: str,
    sampler: Sampler = multinomial(),
):
    raise NotImplementedError(
        "Cannot use regex-structured generation with an OpenAI model"
        + "due to the limitations of the OpenAI API."
    )



    from functools import singledispatch

from outlines.fsm.guide import StopAtEOSGuide
from outlines.generate.api import (
    SequenceGenerator,
    SequenceGeneratorAdapter,
    VisionSequenceGeneratorAdapter,
)
from outlines.models import ExLlamaV2Model, OpenAI, TransformersVision
from outlines.samplers import Sampler, multinomial


@singledispatch
def text(model, sampler: Sampler = multinomial()) -> SequenceGeneratorAdapter:
    """Generate text with a `Transformer` model.

    Note
    ----
    Python 3.11 allows dispatching on Union types and
    this should greatly simplify the code.

    Arguments
    ---------
    model:
        An instance of `Transformer` that represents a model from the
        `transformers` library.
    sampler:
        The sampling algorithm to use to generate token ids from the logits
        distribution.

    Returns
    -------
    A `SequenceGeneratorAdapter` instance that generates text.

    """
    return SequenceGeneratorAdapter(model, None, sampler)


@text.register(ExLlamaV2Model)
def text_exllamav2(model, sampler: Sampler = multinomial()) -> SequenceGenerator:
    fsm = StopAtEOSGuide(model.tokenizer)
    device = model.device
    return SequenceGenerator(fsm, model, sampler, device)


@text.register(TransformersVision)
def text_vision(model, sampler: Sampler = multinomial()):
    return VisionSequenceGeneratorAdapter(model, None, sampler)


@text.register(OpenAI)
def text_openai(model: OpenAI, sampler: Sampler = multinomial()) -> OpenAI:
    if not isinstance(sampler, multinomial):
        raise NotImplementedError(
            r"The OpenAI API does not support any other sampling algorithm "
            + "than the multinomial sampler."
        )

    return model

    import asyncio
import builtins
import functools
import inspect
from typing import Callable, Optional

import numpy as np
from numpy.lib.function_base import (
    _calculate_shapes,
    _parse_gufunc_signature,
    _parse_input_dimensions,
    _update_dim_sizes,
)

# Allow nested loops for running in notebook. We don't enable it globally as it
# may interfere with other libraries that use asyncio.
if hasattr(builtins, "__IPYTHON__"):
    try:
        import nest_asyncio

        nest_asyncio.apply()
    except ImportError:
        print(
            "Couldn't patch nest_asyncio because it's not installed. Running in the notebook might be have issues"
        )


class vectorize:
    """Returns an object that acts like a function but takes arrays as an input.

    The vectorized function evaluates `func` over successive tuples of the input
    chararrays and returns a single NumPy chararrays or a tuple of NumPy chararrays.

    Its behavior is similar to NumPy's `vectorize` for Python functions: the function
    being vectorized is executed in a `for` loop. Coroutines, however, are executed
    concurrently.

    Part of the code was adapted from `numpy.lib.function_base`.

    """

    def __init__(self, func: Callable, signature: Optional[str] = None):
        self.func = func
        self.signature = signature
        self.is_coroutine_fn = inspect.iscoroutinefunction(func)

        functools.update_wrapper(self, func)

        if signature is not None:
            # Parse the signature string into a Python data structure.
            # For instance "(m),(s)->(s,m)" becomes `([(m,),(s,)],[(s,m)])`.
            self._in_and_out_core_dimensions = _parse_gufunc_signature(signature)
        else:
            self._in_and_out_core_dimensions = None

    def __call__(self, *args, **kwargs):
        """Call the vectorized function."""
        if not args and not kwargs:
            return self.call_thunk()
        elif self.signature is not None:
            return self.call_with_signature(*args, **kwargs)
        else:
            return self.call_no_signature(*args, **kwargs)

    def call_thunk(self):
        """Call a vectorized thunk.

        Thunks have no arguments and can thus be called directly.

        """
        if self.is_coroutine_fn:
            loop = asyncio.new_event_loop()
            try:
                outputs = loop.run_until_complete(self.func())
            finally:
                loop.close()
        else:
            outputs = self.func()

        return outputs

    def call_no_signature(self, *args, **kwargs):
        """Call functions and coroutines when no signature is specified.

        When no signature is specified we assume that all of the function's
        inputs and outputs are scalars (core dimension of zero). We first
        broadcast the input arrays, then iteratively apply the function over the
        elements of the broadcasted arrays and finally reshape the results to
        match the input shape.

        Functions are executed in a for loop, coroutines are executed
        concurrently.

        """
        # Convert args and kwargs to arrays
        args = [np.array(arg) for arg in args]
        kwargs = {key: np.array(value) for key, value in kwargs.items()}

        # Broadcast args and kwargs
        broadcast_shape = np.broadcast(*args, *list(kwargs.values())).shape
        args = [np.broadcast_to(arg, broadcast_shape) for arg in args]
        kwargs = {
            key: np.broadcast_to(value, broadcast_shape)
            for key, value in kwargs.items()
        }

        # Execute functions in a loop, and coroutines concurrently
        if self.is_coroutine_fn:
            outputs = self.vectorize_call_coroutine(broadcast_shape, args, kwargs)
        else:
            outputs = self.vectorize_call(broadcast_shape, args, kwargs)

        # `outputs` is a flat array or a tuple of flat arrays. We reshape the arrays
        # to match the input shape.
        outputs = [
            results if isinstance(results, tuple) else (results,) for results in outputs
        ]
        outputs = tuple(
            [np.asarray(x).reshape(broadcast_shape).squeeze() for x in zip(*outputs)]
        )
        outputs = tuple([x.item() if np.ndim(x) == 0 else x for x in outputs])

        n_results = len(list(outputs))

        return outputs[0] if n_results == 1 else outputs

    def call_with_signature(self, *args, **kwargs):
        """Call functions and coroutines when a signature is specified."""
        input_core_dims, output_core_dims = self._in_and_out_core_dimensions

        # Make sure that the numbers of arguments passed is compatible with
        # the signature.
        num_args = len(args) + len(kwargs)
        if num_args != len(input_core_dims):
            raise TypeError(
                "wrong number of positional arguments: "
                "expected %r, got %r" % (len(input_core_dims), len(args))
            )

        # Convert args and kwargs to arrays
        args = [np.asarray(arg) for arg in args]
        kwargs = {key: np.array(value) for key, value in kwargs.items()}

        # Find the arguments' broadcast shape, and map placeholder
        # variables in the signature to the number of dimensions
        # they correspond to given the arguments.
        broadcast_shape, dim_sizes = _parse_input_dimensions(
            args + list(kwargs.values()), input_core_dims
        )

        # Calculate the shape to which each of the arguments should be broadcasted
        # and reshape them accordingly.
        input_shapes = _calculate_shapes(broadcast_shape, dim_sizes, input_core_dims)
        args = [
            np.broadcast_to(arg, shape, subok=True)
            for arg, shape in zip(args, input_shapes)
        ]
        kwargs = {
            key: np.broadcast_to(value, broadcast_shape)
            for key, value in kwargs.items()
        }

        n_out = len(output_core_dims)

        if self.is_coroutine_fn:
            outputs = self.vectorize_call_coroutine(broadcast_shape, args, kwargs)
        else:
            outputs = self.vectorize_call(broadcast_shape, args, kwargs)

        outputs = [
            results if isinstance(results, tuple) else (results,) for results in outputs
        ]

        flat_outputs = list(zip(*outputs))
        n_results = len(flat_outputs)

        if n_out != n_results:
            raise ValueError(
                f"wrong number of outputs from the function, expected {n_out}, got {n_results}"
            )

        # The number of dimensions of the outputs are not necessarily known in
        # advance. The following iterates over the results and updates the
        # number of dimensions of the outputs accordingly.
        for results, core_dims in zip(flat_outputs, output_core_dims):
            for result in results:
                _update_dim_sizes(dim_sizes, result, core_dims)

        # Calculate the shape to which each of the outputs should be broadcasted
        # and reshape them.
        shapes = _calculate_shapes(broadcast_shape, dim_sizes, output_core_dims)
        outputs = tuple(
            [
                np.hstack(results).reshape(shape).squeeze()
                for shape, results in zip(shapes, zip(*outputs))
            ]
        )
        outputs = tuple([x.item() if np.ndim(x) == 0 else x for x in outputs])

        return outputs[0] if n_results == 1 else outputs

    def vectorize_call(self, broadcast_shape, args, kwargs):
        """Run the function in a for loop.

        A possible extension would be to parallelize the calls.

        Parameters
        ----------
        broadcast_shape
            The brodcast shape of the input arrays.
        args
            The function's broadcasted arguments.
        kwargs
            The function's broadcasted keyword arguments.

        """
        outputs = []
        for index in np.ndindex(*broadcast_shape):
            current_args = tuple(arg[index] for arg in args)
            current_kwargs = {key: value[index] for key, value in kwargs.items()}
            outputs.append(self.func(*current_args, **current_kwargs))

        return outputs

    def vectorize_call_coroutine(self, broadcast_shape, args, kwargs):
        """Run coroutines concurrently.

        Creates as many tasks as needed and executes them in a new event
        loop.

        Parameters
        ----------
        broadcast_shape
            The brodcast shape of the input arrays.
        args
            The function's broadcasted arguments.
        kwargs
            The function's broadcasted keyword arguments.

        """

        async def create_and_gather_tasks():
            tasks = []
            for index in np.ndindex(*broadcast_shape):
                current_args = tuple(arg[index] for arg in args)
                current_kwargs = {key: value[index] for key, value in kwargs.items()}
                tasks.append(self.func(*current_args, **current_kwargs))

            outputs = await asyncio.gather(*tasks)

            return outputs

        loop = asyncio.new_event_loop()
        try:
            outputs = loop.run_until_complete(create_and_gather_tasks())
        finally:
            loop.close()

        return outputs


def _update_arrays_type(arrays, results):
    """Update the dtype of arrays.

    String arrays contain strings of fixed length. Here they are initialized with
    the type of the first results, so that if the next results contain longer
    strings they will be truncated when added to the output arrays. Here we
    update the type if the current results contain longer strings than in the
    current output array.

    Parameters
    ----------
    arrays
        Arrays that contain the vectorized function's results.
    results
        The current output of the function being vectorized.

    """

    updated_arrays = []
    for array, result in zip(arrays, results):
        if array.dtype.type == np.str_:
            if array.dtype < np.array(result).dtype:
                array = array.astype(np.array(result).dtype)

        updated_arrays.append(array)

    return tuple(updated_arrays)

    import asyncio
import contextlib
import functools
import os
from typing import Callable, Optional

import cloudpickle
from diskcache import Cache, Disk
from diskcache.core import ENOVAL, UNKNOWN, args_to_key, full_name

_caching_enabled = True


class CloudpickleDisk(Disk):
    def __init__(self, directory, compress_level=1, **kwargs):
        self.compress_level = compress_level
        super().__init__(directory, **kwargs)

    def put(self, key):
        data = cloudpickle.dumps(key)
        return super().put(data)

    def get(self, key, raw):
        data = super().get(key, raw)
        return cloudpickle.loads(data)

    def store(self, value, read, key=UNKNOWN):
        if not read:
            value = cloudpickle.dumps(value)
        return super().store(value, read, key=key)

    def fetch(self, mode, filename, value, read):
        data = super().fetch(mode, filename, value, read)
        if not read:
            data = cloudpickle.loads(data)
        return data


@functools.lru_cache(1)
def get_cache():
    """Get the context object that contains previously-computed return values.

    The cache is used to avoid unnecessary computations and API calls, which can
    be long and expensive for large models.

    The cache directory defaults to `HOMEDIR/.cache/outlines`, but this choice
    can be overridden by the user by setting the value of the `OUTLINES_CACHE_DIR`
    environment variable.

    """
    from outlines._version import __version__ as outlines_version  # type: ignore

    home_dir = os.path.expanduser("~")
    cache_dir = os.environ.get("OUTLINES_CACHE_DIR", f"{home_dir}/.cache/outlines")
    memory = Cache(
        cache_dir,
        eviction_policy="none",
        cull_limit=0,
        disk=CloudpickleDisk,
    )

    # ensure if version upgrade occurs, old cache is pruned
    if outlines_version != memory.get("__version__"):
        memory.clear()
    memory["__version__"] = outlines_version

    return memory


def cache(expire: Optional[float] = None, typed=False, ignore=()):
    """Caching decorator for memoizing function calls.

    The cache key is created based on the values returned by the key_function callable
    if provided or based on the arguments of the decorated function directly otherwise

    This is based on `diskcache`'s `memoize`.

    Parameters
    ----------
    expire
        Seconds until arguments expire.
    typed
        Cache different types separately.
    ignore
        Positional or keyword arguments to ignore.

    Returns
    -------
        A decorator function that can be applied to other functions.
    """

    def decorator(cached_function: Callable):
        memory = get_cache()

        base = (full_name(cached_function),)

        if asyncio.iscoroutinefunction(cached_function):

            async def wrapper(*args, **kwargs):
                if not _caching_enabled:
                    return await cached_function(*args, **kwargs)

                cache_key = wrapper.__cache_key__(*args, **kwargs)
                result = wrapper.__memory__.get(cache_key, default=ENOVAL, retry=True)

                if result is ENOVAL:
                    result = await cached_function(*args, **kwargs)
                    wrapper.__memory__.set(cache_key, result, expire, retry=True)

                return result

        else:

            def wrapper(*args, **kwargs):
                if not _caching_enabled:
                    return cached_function(*args, **kwargs)

                cache_key = wrapper.__cache_key__(*args, **kwargs)
                result = wrapper.__memory__.get(cache_key, default=ENOVAL, retry=True)

                if result is ENOVAL:
                    result = cached_function(*args, **kwargs)
                    wrapper.__memory__.set(cache_key, result, expire, retry=True)

                return result

        def __cache_key__(*args, **kwargs):
            """Make key for cache given function arguments."""
            return args_to_key(base, args, kwargs, typed, ignore)

        wrapper.__cache_key__ = __cache_key__  # type: ignore
        wrapper.__memory__ = memory  # type: ignore
        wrapper.__wrapped__ = cached_function  # type: ignore

        return wrapper

    return decorator


def disable_cache():
    """Disable the cache for this session.

    Generative models output different results each time they are called when
    sampling. This can be a desirable property for some workflows, in which case
    one can call `outlines.call.disable` to disable the cache for the session.

    This function does not delete the cache, call `outlines.cache.clear`
    instead. It also does not overwrite the cache with the values returned
    during the session.

    Example
    -------

    `outlines.cache.disable` should be called right after importing outlines:

    >>> import outlines.caching as cache
    >>> cache.disable_cache()

    """
    global _caching_enabled
    _caching_enabled = False


def clear_cache():
    """Erase the cache completely."""
    memory = get_cache()
    memory.clear()


@contextlib.contextmanager
def cache_disabled():
    # outlines.caching._caching_enabled
    global _caching_enabled
    original_state = _caching_enabled
    _caching_enabled = False
    try:
        yield
    finally:
        _caching_enabled = original_state

        import importlib.util
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union

import requests

from outlines import generate, models

if TYPE_CHECKING:
    from outlines.generate.api import SequenceGenerator
    from outlines.prompts import Prompt


@dataclass
class Function:
    """Represents an Outlines function.

    Functions are a convenient way to encapsulate a prompt template, a language
    model and a Pydantic model that define the output structure. Once defined,
    the function can be called with arguments that will be used to render the
    prompt template.

    """

    prompt_template: "Prompt"
    schema: Union[str, Callable, object]
    model_name: str
    generator: Optional["SequenceGenerator"] = None

    @classmethod
    def from_github(cls, program_path: str, function_name: str = "fn"):
        """Load a function stored on GitHub"""
        program_content = download_from_github(program_path)
        function = extract_function_from_file(program_content, function_name)

        return function

    def init_generator(self):
        """Load the model and initialize the generator."""
        model = models.transformers(self.model_name)
        self.generator = generate.json(model, self.schema)

    def __call__(self, *args, **kwargs):
        """Call the function.

        .. warning::

           This currently does not support batching.

        Parameters
        ----------
        args
            Values to pass to the prompt template as positional arguments.
        kwargs
            Values to pass to the prompt template as keyword arguments.

        """
        if self.generator is None:
            self.init_generator()

        prompt = self.prompt_template(*args, **kwargs)
        return self.generator(prompt)


def download_from_github(short_path: str):
    """Download the file in which the function is stored on GitHub."""
    GITHUB_BASE_URL = "https://raw.githubusercontent.com"
    BRANCH = "main"

    path = short_path.split("/")
    if len(path) < 3:
        raise ValueError(
            "Please provide a valid path in the form {USERNAME}/{REPO_NAME}/{PATH_TO_FILE}."
        )
    elif short_path[-3:] == ".py":
        raise ValueError("Do not append the `.py` extension to the program name.")

    username = path[0]
    repo = path[1]
    path_to_file = path[2:]

    url = "/".join([GITHUB_BASE_URL, username, repo, BRANCH] + path_to_file) + ".py"
    result = requests.get(url)

    if result.status_code == 200:
        return result.text
    elif result.status_code == 404:
        raise ValueError(
            f"Program could not be found at {url}. Please make sure you entered the GitHub username, repository name and path to the program correctly."
        )
    else:
        result.raise_for_status()


def extract_function_from_file(content: str, function_name: str) -> Tuple[Callable]:
    """Extract a function object from a downloaded file."""

    spec = importlib.util.spec_from_loader(
        "outlines_function", loader=None, origin="github"
    )
    if spec is not None:
        module = importlib.util.module_from_spec(spec)
        exec(content, module.__dict__)

        try:
            fn = getattr(module, function_name)
        except AttributeError:
            raise AttributeError(
                "Could not find an `outlines.Function` instance in the remote file. Make sure that the path you specified is correct."
            )

        if not isinstance(fn, module.outlines.Function):
            raise TypeError(
                f"The `{function_name}` variable in the program must be an instance of `outlines.Function`"
            )

    return fn

from pathlib import Path

GRAMMAR_PATH = Path(__file__).parent / "grammars"


def read_grammar(grammar_file_name, base_grammar_path=GRAMMAR_PATH):
    """Read grammar file from default grammar path"""
    full_path = base_grammar_path / grammar_file_name
    with open(full_path) as file:
        return file.read()


arithmetic = read_grammar("arithmetic.lark")
json = read_grammar("json.lark")

import functools
import inspect
import json
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, cast

from jinja2 import Environment, StrictUndefined
from pydantic import BaseModel


@dataclass
class Prompt:
    """Represents a prompt function.

    We return a `Prompt` class instead of a simple function so the
    template defined in prompt functions can be accessed.

    """

    template: str
    signature: inspect.Signature

    def __post_init__(self):
        self.parameters: List[str] = list(self.signature.parameters.keys())

    def __call__(self, *args, **kwargs) -> str:
        """Render and return the template.

        Returns
        -------
        The rendered template as a Python ``str``.

        """
        bound_arguments = self.signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        return render(self.template, **bound_arguments.arguments)

    def __str__(self):
        return self.template


def prompt(fn: Callable) -> Prompt:
    """Decorate a function that contains a prompt template.

    This allows to define prompts in the docstring of a function and simplify their
    manipulation by providing some degree of encapsulation. It uses the `render`
    function internally to render templates.

    >>> import outlines
    >>>
    >>> @outlines.prompt
    >>> def build_prompt(question):
    ...    "I have a ${question}"
    ...
    >>> prompt = build_prompt("How are you?")

    This API can also be helpful in an "agent" context where parts of the prompt
    are set when the agent is initialized and never modified later. In this situation
    we can partially apply the prompt function at initialization.

    >>> import outlines
    >>> import functools as ft
    ...
    >>> @outlines.prompt
    ... def solve_task(name: str, objective: str, task: str):
    ...     '''Your name is {{name}}.
    ..      Your overall objective is to {{objective}}.
    ...     Please solve the following task: {{task}}
    ...     '''
    ...
    >>> hal = ft.partial(solve_task, "HAL", "Travel to Jupiter")

    Returns
    -------
    A `Prompt` callable class which will render the template when called.

    """

    signature = inspect.signature(fn)

    # The docstring contains the template that will be rendered to be used
    # as a prompt to the language model.
    docstring = fn.__doc__
    if docstring is None:
        raise TypeError("Could not find a template in the function's docstring.")

    template = cast(str, docstring)

    return Prompt(template, signature)


def render(template: str, **values: Optional[Dict[str, Any]]) -> str:
    r"""Parse a Jinaj2 template and translate it into an Outlines graph.

    This function removes extra whitespaces and linebreaks from templates to
    allow users to enter prompts more naturally than if they used Python's
    constructs directly. See the examples for a detailed explanation.

    Examples
    --------

    Outlines follow Jinja2's syntax

    >>> import outlines
    >>> outline = outlines.render("I like {{food}} and {{sport}}", food="tomatoes", sport="tennis")
    I like tomatoes and tennis

    If the first line of the template is empty, `render` removes it

    >>> from outlines import render
    >>>
    >>> tpl = '''
    ... A new string'''
    >>> tpl
    ... '\nA new string'
    >>> render(tpl)
    ... 'a new string'

    Similarly, `render` ignores linebreaks introduced by placing the closing quotes
    underneath the text:

    >>> tpl = '''
    ... A new string
    ... '''
    >>> tpl
    ... '\nA new string\n'
    >>> render(tpl)
    ... 'A new string'

    If you want to insert a linebreak at the end of the rendered template, you will
    need to leave an empty line at the end of the template:

    >>> tpl = '''
    ... A new string
    ...
    ... '''
    >>> tpl
    ... '\nA new string\n\n'
    >>> render(tpl)
    ... 'A new string\n'

    `render` removes the identation in docstrings. This is particularly important
    when using prompt functions

    >>> tpl = '''
    ...    a string
    ...    and another string'''
    >>> tpl
    ... '\n   a string\n   and another string'
    >>> render(tpl)
    ... 'a string\nand another string'

    The indentation of the first line is assumed to be the same as the second line's

    >>> tpl = '''a string
    ...     and another'''
    >>> tpl
    ... 'a string\n    and another'
    >>> render(tpl)
    ... 'a string\nand another'

    To get a different indentation for the first and the second line, we can start the
    prompt on the string's second line:

    >>> tpl = '''
    ... First line
    ...   Second line'''
    >>> render(tpl)
    ... 'First Line\n  Second Line'

    Parameters
    ----------
    template
        A string that contains a template written with the Jinja2 syntax.
    **values
        Map from the variables in the template to their value.

    Returns
    -------
    A string that contains the rendered template.

    """
    # Dedent, and remove extra linebreak
    cleaned_template = inspect.cleandoc(template)

    # Add linebreak if there were any extra linebreaks that
    # `cleandoc` would have removed
    ends_with_linebreak = template.replace(" ", "").endswith("\n\n")
    if ends_with_linebreak:
        cleaned_template += "\n"

    # Remove extra whitespaces, except those that immediately follow a newline symbol.
    # This is necessary to avoid introducing whitespaces after backslash `\` characters
    # used to continue to the next line without linebreak.
    cleaned_template = re.sub(r"(?![\r\n])(\b\s+)", " ", cleaned_template)

    env = Environment(
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        undefined=StrictUndefined,
    )
    env.filters["name"] = get_fn_name
    env.filters["description"] = get_fn_description
    env.filters["source"] = get_fn_source
    env.filters["signature"] = get_fn_signature
    env.filters["schema"] = get_schema
    env.filters["args"] = get_fn_args

    jinja_template = env.from_string(cleaned_template)

    return jinja_template.render(**values)


def get_fn_name(fn: Callable):
    """Returns the name of a callable."""
    if not callable(fn):
        raise TypeError("The `name` filter only applies to callables.")

    if not hasattr(fn, "__name__"):
        name = type(fn).__name__
    else:
        name = fn.__name__

    return name


def get_fn_args(fn: Callable):
    """Returns the arguments of a function with annotations and default values if provided."""
    if not callable(fn):
        raise TypeError("The `args` filter only applies to callables.")

    arg_str_list = []
    signature = inspect.signature(fn)
    arg_str_list = [str(param) for param in signature.parameters.values()]
    arg_str = ", ".join(arg_str_list)
    return arg_str


def get_fn_description(fn: Callable):
    """Returns the first line of a callable's docstring."""
    if not callable(fn):
        raise TypeError("The `description` filter only applies to callables.")

    docstring = inspect.getdoc(fn)
    if docstring is None:
        description = ""
    else:
        description = docstring.split("\n")[0].strip()

    return description


def get_fn_source(fn: Callable):
    """Return the source code of a callable."""
    if not callable(fn):
        raise TypeError("The `source` filter only applies to callables.")

    source = textwrap.dedent(inspect.getsource(fn))
    re_search = re.search(re.compile(r"(\bdef\b.*)", re.DOTALL), source)
    if re_search is not None:
        source = re_search.group(0)
    else:
        raise TypeError("Could not read the function's source code")

    return source


def get_fn_signature(fn: Callable):
    """Return the signature of a callable."""
    if not callable(fn):
        raise TypeError("The `source` filter only applies to callables.")

    source = textwrap.dedent(inspect.getsource(fn))
    re_search = re.search(re.compile(r"\(([^)]+)\)"), source)
    if re_search is None:
        signature = ""
    else:
        signature = re_search.group(1)

    return signature


@functools.singledispatch
def get_schema(model: Any):
    raise NotImplementedError(
        f"No schema rendering function defined for type {type(model)}."
    )


@get_schema.register(dict)
def get_schema_dict(model: Dict):
    """Return a pretty-printed dictionary"""
    return json.dumps(model, indent=2)


@get_schema.register(type(BaseModel))
def get_schema_pydantic(model: Type[BaseModel]):
    """Return the schema of a Pydantic model."""
    if not type(model) == type(BaseModel):
        raise TypeError("The `schema` filter only applies to Pydantic models.")

    if hasattr(model, "model_json_schema"):
        def_key = "$defs"
        raw_schema = model.model_json_schema()
    else:  # pragma: no cover
        def_key = "definitions"
        raw_schema = model.schema()

    definitions = raw_schema.get(def_key, None)
    schema = parse_pydantic_schema(raw_schema, definitions)

    return json.dumps(schema, indent=2)


def parse_pydantic_schema(raw_schema, definitions):
    """Parse the output of `Basemodel.[schema|model_json_schema]()`.

    This recursively follows the references to other schemas in case
    of nested models. Other schemas are stored under the "definitions"
    key in the schema of the top-level model.

    """
    simple_schema = {}
    for name, value in raw_schema["properties"].items():
        if "description" in value:
            simple_schema[name] = value["description"]
        elif "$ref" in value:
            refs = value["$ref"].split("/")
            simple_schema[name] = parse_pydantic_schema(
                definitions[refs[2]], definitions
            )
        else:
            simple_schema[name] = f"<{name}>"

    return simple_schema

    import math
from typing import TYPE_CHECKING, Callable, Optional, Protocol, Tuple

if TYPE_CHECKING:
    import torch


class Sampler(Protocol):
    samples: int

    def __call__(
        self,
        next_token_logits: "torch.DoubleTensor",
        sequence_weights: "torch.DoubleTensor",
        rng: "torch.Generator",
    ) -> "torch.DoubleTensor":
        ...


class GreedySampler:
    """Greedy Sampling algorithm.

    Greedy sampling consists in choosing the token with the largest
    likelihood at every step.

    We don't allow more than one sample. We could attribute this a meaning, for
    instance the k-th sample represents the k-th most likely token. In which
    case it would be equivalent to beam search without the sequence weights.

    Attributes
    ----------
    samples
        The number of samples taken for each input sequence.

    """

    def __init__(self):
        self.samples = 1

    def __call__(
        self,
        next_token_logits: "torch.DoubleTensor",
        sequence_weights: "torch.DoubleTensor",
        _,
    ) -> "torch.DoubleTensor":
        """Call the greedy sampler.

        Parameters
        ----------
        next_token_logits
            A tensor of shape ``(n_seqs, vocab_size,)`` that represents the
            probability distribution of the next token over the vocabulary.
        sequence_weights
            A tensor of shape ``(n_seqs,)`` that represents the cumulative
            weight of each sequence.
        rng
            A random number generator.

        Returns
        -------
        A tuple with an array that contains the ids of the sampled tokens of
        shape ``(n_seqs, 1)``, an array that contains the ancestors of each
        sampled id of shape ``(n_seqs,)`` and an array that contains the updated
        cumulative weights of each sequence of shape ``(n_seqs,)``.

        """
        import torch

        logprobs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
        next_token_ids = torch.argmax(logprobs, dim=-1, keepdim=True)

        ancestors = torch.arange(
            next_token_logits.shape[0], device=next_token_logits.device
        )
        weights = sequence_weights + torch.gather(logprobs, 1, next_token_ids).squeeze()

        return next_token_ids, ancestors, weights


greedy = GreedySampler


class MultinomialSampler:
    """Multinomial sampling algorithm.

    Multinomial sampling consists in randomly sampling the next token assuming
    its distribution is a Categorical distribution parametrized by the
    next-token logits.


    Attributes
    ----------
    samples
        The number of samples taken for each input sequence.

    """

    def __init__(
        self,
        samples: int = 1,
        *,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
    ):
        self.samples = samples
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature

        self.logits_processors = []
        if top_k is not None:
            self.logits_processors.append(keep_top_k_logits(top_k))
        elif top_p is not None:
            self.logits_processors.append(keep_top_p_logits(top_p))

        if temperature is not None:
            self.logits_processors.append(rescale_logits(temperature))

    def __call__(
        self,
        next_token_logits: "torch.DoubleTensor",
        sequence_weights: "torch.DoubleTensor",
        rng: "torch.Generator",
    ) -> Tuple["torch.DoubleTensor", "torch.DoubleTensor", "torch.DoubleTensor"]:
        """Call the multinomial sampler.

        Parameters
        ----------
        next_token_logits
            A tensor of shape ``(n_seqs, vocab_size,)`` that represents the
            probability distribution of the next token over the vocabulary.
        sequence_weights
            A tensor of shape ``(n_seqs,)`` that represents the cumulative
            weight of each sequence.
        rng
            A random number generator.

        Returns
        -------
        A tuple with an array that contains the ids of the sampled tokens of
        shape ``(n_seqs, 1)``, an array that contains the ancestors of each
        sampled id of shape ``(n_seqs,)`` and an array that contains the updated
        cumulative weights of each sequence of shape ``(n_seqs,)``.

        """
        import torch

        altered_next_token_logits = next_token_logits
        for logit_processor in self.logits_processors:
            altered_next_token_logits = logit_processor(next_token_logits)

        probs = torch.nn.functional.softmax(altered_next_token_logits, dim=-1)
        next_token_ids = torch.multinomial(probs, num_samples=1, generator=rng)

        logprobs = torch.nn.functional.log_softmax(altered_next_token_logits, dim=-1)
        ancestors = torch.arange(
            altered_next_token_logits.shape[0], device=next_token_logits.device
        )
        weights = sequence_weights + torch.gather(logprobs, 1, next_token_ids).squeeze()

        return next_token_ids, ancestors, weights


multinomial = MultinomialSampler


def keep_top_k_logits(k: int) -> Callable[["torch.Tensor"], "torch.Tensor"]:
    """Build a function that masks logits values smaller than the top `k` ones.

    Parameters
    ----------
    k
        The ranking below which logit values are replaced by `-math.inf`.

    """
    import torch

    if not isinstance(k, int) or k < 1:
        raise ValueError(f"`k` must be a strictly positive integers, got {k} instead.")

    def logits_processor(logits: torch.Tensor) -> torch.Tensor:
        num_to_keep = min(k, logits.size(-1))
        mask_idx = logits < torch.topk(logits, num_to_keep)[0][..., -1, None]
        return logits.masked_fill(mask_idx, -math.inf)

    return logits_processor


def keep_top_p_logits(p: float) -> Callable[["torch.Tensor"], "torch.Tensor"]:
    """Build a function that masks the lowest probability tokens whose
    cumulative probability is below a certain threshold.

    Parameters
    ----------
    p
        The value of the threshold. We keep the highest probability tokens whose
        cumulative distribution is greater than or equal to `p` and mask the
        others. Its value must be between 0 (excluded) and 1 (included).

    """
    import torch

    if p <= 0.0 or p > 1.0:
        raise ValueError(
            f"`p` must be a floating point number between 0 (excluded) and 1 (included), got {p} instead."
        )

    def logits_processor(logits: torch.Tensor) -> torch.Tensor:
        sorted_logits, sorted_idx = torch.sort(logits, descending=False)
        cumulative_probabilties = torch.nn.functional.softmax(
            sorted_logits, dim=-1
        ).cumsum(dim=-1)

        sorted_masked_idx = cumulative_probabilties <= (1 - p)
        mask_idx = torch.scatter(sorted_masked_idx, 1, sorted_idx, sorted_masked_idx)
        return logits.masked_fill(mask_idx, -math.inf)

    return logits_processor


def rescale_logits(temperature: float) -> Callable[["torch.Tensor"], "torch.Tensor"]:
    """Build a function that rescales the token probabilities exponentially.

    Parameters
    ----------
    temperature
        The value by which we rescale the logits.

    """

    if not isinstance(temperature, float) or temperature < 0.0:
        raise ValueError(
            f"`temperature` must be a strictly positive floating point number, got {temperature} instead."
        )
    elif temperature == 0.0:
        raise ValueError(
            "Please use the greedy sampler instead of setting the temperature to 0."
        )

    def logits_processor(logits: "torch.Tensor") -> "torch.Tensor":
        return logits / temperature

    return logits_processor


class BeamSearchSampler:
    """Beam Search sampling algorithm.

    Attributes
    ----------
    samples
        The number of samples taken for each input sequence.

    """

    def __init__(self, beams: int = 1):
        self.samples = beams

    def __call__(
        self,
        next_token_logits: "torch.DoubleTensor",
        sequence_weights: "torch.DoubleTensor",
        _,
    ) -> Tuple["torch.DoubleTensor", "torch.DoubleTensor", "torch.DoubleTensor"]:
        """Call the beam search sampler.

        Parameters
        ----------
        next_token_logits
            A tensor of shape ``(n_seqs, vocab_size,)`` that represents the
            probability distribution of the next token over the vocabulary.
        sequence_weights
            A tensor of shape ``(n_seqs,)`` that represents the cumulative
            weight of each sequence.
        rng
            A random number generator.

        Returns
        -------
        A tuple with an array that contains the ids of the sampled tokens of
        shape ``(n_seqs, 1)``, an array that contains the ancestors of each
        sampled id of shape ``(n_seqs,)`` and an array that contains the updated
        cumulative weights of each sequence of shape ``(n_seqs,)``.

        """
        import torch

        logprobs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
        weights = logprobs + sequence_weights.unsqueeze(1).expand_as(next_token_logits)

        # Flatten scores to (n_batch, n_samples * vocab_size)
        # and find the top-k weights for each batch.
        batch_size = next_token_logits.shape[0] // self.samples
        vocab_size = next_token_logits.shape[-1]
        weights = weights.view(batch_size, self.samples * vocab_size)

        # If the weights are all equal to 0 we are at the beginning of the search
        # and thus only need to sample from one set of token logits for each
        # batch.
        if torch.all(sequence_weights == 0):
            weights = weights[:, :vocab_size]

        weights, indices = torch.topk(
            weights, self.samples, dim=1, largest=True, sorted=True
        )

        ancestors = torch.div(indices, vocab_size, rounding_mode="floor")
        next_token_ids = indices % vocab_size

        # Re-shape the weights, next_token_ids and ancestors to (n_batch * n_samples, 1)
        first_batch_idx = torch.arange(
            0, batch_size * self.samples, self.samples, device=next_token_logits.device
        ).unsqueeze(1)
        ancestors = ancestors + first_batch_idx

        ancestors = ancestors.view(self.samples * batch_size)
        weights = weights.view(self.samples * batch_size)
        next_token_ids = next_token_ids.view(self.samples * batch_size, 1)

        return next_token_ids, ancestors, weights


beam_search = BeamSearchSampler




import dataclasses
from typing import TYPE_CHECKING, List, Optional, Union

from transformers import SPIECE_UNDERLINE, PreTrainedTokenizerBase

from outlines.generate.api import GenerationParameters, SamplingParameters

if TYPE_CHECKING:
    from vllm import LLM
    from vllm.sampling_params import SamplingParams


class VLLM:
    """Represents a vLLM model.

    We wrap models from model providing libraries in order to give all of
    them the same interface in Outlines and allow users to easily switch
    between providers. This class wraps the `vllm.LLM` class from the
    `vllm` library.

    """

    def __init__(self, model: "LLM"):
        self.model = model
        self.lora_request = None

        self.tokenizer = self._get_tokenizer()

    def _get_tokenizer(self):
        if hasattr(self.model, "get_tokenizer"):
            tokenizer = self.model.get_tokenizer()
        elif hasattr(self.model, "tokenizer"):
            if hasattr(self.model.tokenizer, "tokenizer"):
                tokenizer = self.model.tokenizer.tokenizer
            else:
                tokenizer = self.model.tokenizer
        else:
            raise ValueError(
                "The provided LLM instance neither has a "
                "`tokenizer` attribute or a `get_tokenizer` method."
            )
        return adapt_tokenizer(tokenizer=tokenizer)

    def generate(
        self,
        prompts: Union[str, List[str]],
        generation_parameters: GenerationParameters,
        logits_processor,
        sampling_parameters: SamplingParameters,
        *,
        sampling_params: Optional["SamplingParams"] = None,
        use_tqdm: bool = True,
    ):
        """Generate text using vLLM.

        Arguments
        ---------
        prompts
            A prompt or list of prompts.
        generation_parameters
            An instance of `GenerationParameters` that contains the prompt,
            the maximum number of tokens, stop sequences and seed. All the
            arguments to `SequenceGeneratorAdapter`'s `__cal__` method.
        logits_processor
            The logits processor to use when generating text.
        sampling_parameters
            An instance of `SamplingParameters`, a dataclass that contains
            the name of the sampler to use and related parameters as available
            in Outlines.
        sampling_params
            An instance of `vllm.sampling_params.SamplingParams`. The values
            passed via this dataclass supersede the values of the parameters
            in `generation_parameters` and `sampling_parameters`. See the
            vLLM documentation for more details: https://docs.vllm.ai/en/latest/dev/sampling_params.html.
        use_tqdm
            A boolean in order to display progress bar while inferencing

        Returns
        -------
        The generated text, of shape `(n_batch, n_samples)`. If there are only
        one batch and several samples, the list is of shape `(n_samples)`. If
        this is a batch with several sequences but only one sample the list is
        of shape `(n_batch)`. If there is only one sequence and one sample, a
        string is returned.

        """
        from vllm.sampling_params import SamplingParams

        if sampling_params is None:
            sampling_params = SamplingParams()

        max_tokens, stop_at, seed = dataclasses.astuple(generation_parameters)

        # We only update the values in `sampling_params` if they
        # are specified by the user when calling the generator.
        if max_tokens is not None:
            sampling_params.max_tokens = max_tokens
        if stop_at is not None:
            if isinstance(stop_at, str):
                stop_at = [stop_at]
            sampling_params.stop = stop_at
        if seed is not None:
            sampling_params.seed = seed

        sampling_params.logits_processors = (
            [logits_processor] if logits_processor is not None else []
        )

        sampler, num_samples, top_p, top_k, temperature = dataclasses.astuple(
            sampling_parameters
        )

        # We only update the values in `sampling_params` that
        # were not specified by the user.
        if sampling_params.n == 1:
            sampling_params.n = num_samples
            sampling_params.best_of = num_samples
        if top_p is not None and sampling_params.top_p == 1.0:
            sampling_params.top_p = top_p
        if top_k is not None and sampling_params.top_k == -1:
            sampling_params.top_k = top_k
            # TODO: remove this if statement once fixed
            # https://github.com/vllm-project/vllm/issues/5404#issuecomment-2175972897
            if top_k == 1:
                sampling_params.repetition_penalty = 0
        if temperature is not None and sampling_params.temperature == 1.0:
            sampling_params.temperature = temperature
        if sampler == "beam_search":
            sampling_params.use_beam_search = True

        results = self.model.generate(
            prompts,
            sampling_params=sampling_params,
            lora_request=self.lora_request,
            use_tqdm=use_tqdm,
        )
        results = [[sample.text for sample in batch.outputs] for batch in results]

        batch_size = len(results)
        sample_size = len(results[0])

        if batch_size == 1 and sample_size == 1:
            return results[0][0]
        elif batch_size == 1:
            return results[0]
        elif sample_size == 1:
            return [batch[0] for batch in results]

        return results

    def stream(self, *args, **kwargs):
        """Return a text generator.

        Streaming is not yet available for `vllm.LLM`.

        TODO: Implement the streaming functionality ourselves.

        """
        raise NotImplementedError(
            "Streaming is not available for the vLLM integration."
        )

    def load_lora(self, adapter_path: Optional[str]):
        from vllm.lora.request import LoRARequest

        if adapter_path is None:
            self.lora_request = None
        else:
            self.lora_request = LoRARequest(adapter_path, 1, adapter_path)


def vllm(model_name: str, **vllm_model_params):
    """Load a vLLM model.

    Arguments
    ---------
    model_name
        The name of the model to load from the HuggingFace hub.
    vllm_model_params
        vLLM-specific model parameters. See the vLLM code for the full list:
        https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py

    """
    from vllm import LLM

    model = LLM(model_name, **vllm_model_params)

    return VLLM(model)


def adapt_tokenizer(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """Adapt a tokenizer to use to compile the FSM.

    The API of Outlines tokenizers is slightly different to that of `transformers`. In
    addition we need to handle the missing spaces to Llama's tokenizer to be able to
    compile FSMs for this model.

    Parameters
    ----------
    tokenizer
        The tokenizer of the model.

    Returns
    -------
    PreTrainedTokenizerBase
        The adapted tokenizer.
    """
    tokenizer.vocabulary = tokenizer.get_vocab()
    tokenizer.special_tokens = set(tokenizer.all_special_tokens)

    def convert_token_to_string(token: Union[str, bytes]) -> str:
        string = tokenizer.convert_tokens_to_string([token])

        # A hack to handle missing spaces to HF's Llama tokenizers
        if (
            type(token) is str
            and token.startswith(SPIECE_UNDERLINE)
            or token == "<0x20>"
        ):
            return " " + string

        return string

    tokenizer.convert_token_to_string = convert_token_to_string

    return tokenizer


    import dataclasses
import pickle
import warnings
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypedDict,
    Union,
)

from typing_extensions import Unpack

from outlines.generate.api import GenerationParameters, SamplingParameters
from outlines.models.tokenizer import Tokenizer

if TYPE_CHECKING:
    from llama_cpp import Llama, LogitsProcessorList


class LlamaCppTokenizer(Tokenizer):
    def __init__(self, model: "Llama"):
        self.eos_token_id = model.token_eos()
        self.eos_token = model.tokenizer().decode([self.eos_token_id])
        self.pad_token_id = self.eos_token_id
        self.special_tokens: Set[str] = set()

        self.vocabulary: Dict[str, int] = dict()

        self.tokenizer = model.tokenizer()

        # TODO: Remove when https://github.com/ggerganov/llama.cpp/pull/5613 is resolved
        self._hf_tokenizer = None
        try:
            self.vocabulary = model.tokenizer_.hf_tokenizer.get_vocab()
            self._hf_tokenizer = model.tokenizer_.hf_tokenizer
        except AttributeError:
            # ###
            for t in range(model.n_vocab()):
                token_piece = model.tokenizer().decode([t])
                self.vocabulary[token_piece] = t

        # ensure stable ordering of vocabulary
        self.vocabulary = {
            tok: tok_id
            for tok, tok_id in sorted(self.vocabulary.items(), key=lambda x: x[1])
        }

        self._hash = None

    def decode(self, token_ids: List[int]) -> List[str]:
        decoded_bytes = self.tokenizer.detokenize(token_ids)
        return [decoded_bytes.decode("utf-8", errors="ignore")]

    def encode(
        self, prompt: Union[str, List[str]], add_bos: bool = True, special: bool = True
    ) -> Tuple[List[int], List[int]]:
        if isinstance(prompt, list):
            raise NotImplementedError(
                "llama-cpp-python tokenizer doesn't support batch tokenization"
            )
        token_ids = self.tokenizer.tokenize(
            prompt.encode("utf-8", errors="ignore"), add_bos=add_bos, special=special
        )
        # generate attention mask, missing from llama-cpp-python
        attention_mask = [
            1 if token_id != self.pad_token_id else 0 for token_id in token_ids
        ]
        return token_ids, attention_mask

    def convert_token_to_string(self, token: str) -> str:
        if self._hf_tokenizer is not None:
            from transformers.file_utils import SPIECE_UNDERLINE

            token_str = self._hf_tokenizer.convert_tokens_to_string([token])
            if token.startswith(SPIECE_UNDERLINE) or token == "<0x20>":
                token_str = " " + token_str
            return token_str
        else:
            return token

    def __eq__(self, other):
        if not isinstance(other, LlamaCppTokenizer):
            return False
        return self.__getstate__() == other.__getstate__()

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(pickle.dumps(self))
        return self._hash

    def __getstate__(self):
        """Create a stable representation for outlines.caching"""
        return (
            self.vocabulary,
            self.eos_token_id,
            self.eos_token,
            self.pad_token_id,
            sorted(self.special_tokens),
        )

    def __setstate__(self, state):
        raise NotImplementedError("Cannot load a pickled llamacpp tokenizer")


class LlamaCppParams(TypedDict, total=False):
    suffix: Optional[str]
    temperature: float
    top_p: float
    min_p: float
    typical_p: float
    seed: int
    max_tokens: int
    logits_processor: "LogitsProcessorList"
    stop: Optional[Union[str, List[str]]]
    frequence_penalty: float
    presence_penalty: float
    repeat_penalty: float
    top_k: int
    tfs_z: float
    mirostat_mode: int
    mirostat_tau: float
    mirostat_eta: float
    stream: bool


class LlamaCpp:
    """Represents a model provided by the `llama-cpp-python` library.

    We wrap models from model providing libraries in order to give all of
    them the same interface in Outlines and allow users to easily switch
    between providers. This class wraps the `llama_cpp.Llama` class from the
    `llama-cpp-python` library.

    """

    def __init__(self, model: "Llama"):
        self.model = model

    @property
    def tokenizer(self):
        return LlamaCppTokenizer(self.model)

    def prepare_generation_parameters(
        self,
        generation_parameters: GenerationParameters,
        sampling_parameters: SamplingParameters,
        structure_logits_processor,
        **llama_cpp_params: Unpack[LlamaCppParams],
    ):
        """Prepare the generation parameters.

        `llama-cpp-python` uses different default values

        """
        from llama_cpp import LogitsProcessorList

        max_tokens, stop_at, seed = dataclasses.astuple(generation_parameters)

        # We update `llama_cpp_params` with the values the user passed to the
        # generator.
        if "stop" not in llama_cpp_params:
            llama_cpp_params["stop"] = stop_at
        if "seed" not in llama_cpp_params:
            llama_cpp_params["seed"] = seed

        # Somehow `llama-cpp-python` generates `max_tokens + 1`  tokens
        if "max_tokens" not in llama_cpp_params:
            if max_tokens is None:
                llama_cpp_params["max_tokens"] = -1  # indicates unlimited tokens
            else:
                llama_cpp_params["max_tokens"] = max_tokens - 1
        else:
            llama_cpp_params["max_tokens"] = llama_cpp_params["max_tokens"] - 1

        sampler, num_samples, top_p, top_k, temperature = dataclasses.astuple(
            sampling_parameters
        )

        # We update the `llama_cpp_params` with the sampling values that
        # were specified by the user via the `Sampler` class, unless they
        # are also specified in `llama_cpp_params`. We also disable other
        # sampling methods that are enabled by default and reset the temperature
        # value.
        #
        # See https://github.com/ggerganov/llama.cpp/blob/e11a8999b5690f810c2c99c14347f0834e68c524/common/sampling.h#L22
        # for the default values in `llama.cpp` and indications to disable the sampling modes.
        # Mirostat sampling, tail-free sampling and all penalties are disabled by default.
        #
        # See https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__call__
        # for default values in `llama-cpp-python`
        if sampler == "beam_search":
            raise NotImplementedError(
                "The `llama_cpp_python` library does not support Beam Search."
            )
        if num_samples != 1:
            raise NotImplementedError(
                "The `llama_cpp_python` library does not allow to take several samples."
            )
        if "top_p" not in llama_cpp_params:
            if top_p is not None:
                llama_cpp_params["top_p"] = top_p
            else:
                llama_cpp_params["top_p"] = 1.0

        if "min_p" not in llama_cpp_params:
            llama_cpp_params["min_p"] = 0.0

        if "top_k" not in llama_cpp_params:
            if top_k is not None:
                llama_cpp_params["top_k"] = top_k
            else:
                llama_cpp_params["top_k"] = -1

        if "temperature" not in llama_cpp_params:
            if temperature is not None:
                llama_cpp_params["temperature"] = temperature
            else:
                llama_cpp_params["temperature"] = 1.0

        if "repeat_penalty" not in llama_cpp_params:
            llama_cpp_params["repeat_penalty"] = 1.0

        # The choice to stream or not should happen via the high-level API
        llama_cpp_params["stream"] = False

        if structure_logits_processor is not None:
            if "logits_processor" in llama_cpp_params:
                llama_cpp_params["logits_processor"].append(structure_logits_processor)
            else:
                llama_cpp_params["logits_processor"] = LogitsProcessorList(
                    [structure_logits_processor]
                )

        return llama_cpp_params

    def generate(
        self,
        prompts: Union[str, List[str]],
        generation_parameters: GenerationParameters,
        structure_logits_processor,
        sampling_parameters: SamplingParameters,
        **llama_cpp_params: Unpack[LlamaCppParams],
    ) -> str:
        """Generate text using `llama-cpp-python`.

        Arguments
        ---------
        prompts
            A prompt or list of prompts.
        generation_parameters
            An instance of `GenerationParameters` that contains the prompt,
            the maximum number of tokens, stop sequences and seed. All the
            arguments to `SequenceGeneratorAdapter`'s `__cal__` method.
        logits_processor
            The logits processor to use when generating text.
        sampling_parameters
            An instance of `SamplingParameters`, a dataclass that contains
            the name of the sampler to use and related parameters as available
            in Outlines.
        llama_cpp_params
            Keyword arguments that can be passed to
            `llama_cpp_python.Llama.__call__`.  The values in `llama_cpp_params`
            supersede the values of the parameters in `generation_parameters` and
            `sampling_parameters`.  See the `llama_cpp_python` documentation for
            a list of possible values: https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__call__

        Returns
        -------
        The generated text.

        """
        if not isinstance(prompts, str):
            raise NotImplementedError(
                "The `llama-cpp-python` library does not support batch inference."
            )

        llama_cpp_params = self.prepare_generation_parameters(
            generation_parameters,
            sampling_parameters,
            structure_logits_processor,
            **llama_cpp_params,
        )
        completion = self.model(prompts, **llama_cpp_params)
        result = completion["choices"][0]["text"]

        self.model.reset()

        return result

    def stream(
        self,
        prompts: Union[str, List[str]],
        generation_parameters: GenerationParameters,
        structure_logits_processor,
        sampling_parameters: SamplingParameters,
        **llama_cpp_params: Unpack[LlamaCppParams],
    ) -> Iterator[str]:
        """Stream text using `llama-cpp-python`.

        Arguments
        ---------
        prompts
            A prompt or list of prompts.
        generation_parameters
            An instance of `GenerationParameters` that contains the prompt,
            the maximum number of tokens, stop sequences and seed. All the
            arguments to `SequenceGeneratorAdapter`'s `__cal__` method.
        logits_processor
            The logits processor to use when generating text.
        sampling_parameters
            An instance of `SamplingParameters`, a dataclass that contains
            the name of the sampler to use and related parameters as available
            in Outlines.
        llama_cpp_params
            Keyword arguments that can be passed to
            `llama_cpp_python.Llama.__call__`.  The values in `llama_cpp_params`
            supersede the values of the parameters in `generation_parameters` and
            `sampling_parameters`.  See the `llama_cpp_python` documentation for
            a list of possible values: https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__call__

        Returns
        -------
        A generator that return strings.

        """

        if not isinstance(prompts, str):
            raise NotImplementedError(
                "The `llama-cpp-python` library does not support batch inference."
            )

        llama_cpp_params = self.prepare_generation_parameters(
            generation_parameters,
            sampling_parameters,
            structure_logits_processor,
            **llama_cpp_params,
        )
        llama_cpp_params["stream"] = True
        generator = self.model(prompts, **llama_cpp_params)

        def token_generator() -> Iterator[str]:
            while True:
                try:
                    result = next(generator)
                    yield result["choices"][0]["text"]
                except StopIteration:
                    self.model.reset()
                    return

        return token_generator()

    def load_lora(self, adapter_path: str):
        if self.model._model.apply_lora_from_file(
            adapter_path,
            1.0,
        ):
            raise RuntimeError(f"Failed to apply LoRA from lora path: {adapter_path}")


def llamacpp(
    repo_id: str, filename: Optional[str] = None, **llamacpp_model_params
) -> LlamaCpp:
    """Load a model from the `llama-cpp-python` library.

    We use the `Llama.from_pretrained` classmethod that downloads models
    directly from the HuggingFace hub, instead of asking users to specify
    a path to the downloaded model. One can still load a local model
    by initializing `llama_cpp.Llama` directly.

    Arguments
    ---------
    repo_id
        The name of the model repository.
    filename:
        A filename of glob pattern to match the model file in the repo.
    llama_cpp_model_params
        Llama-specific model parameters. See the `llama-cpp-python` documentation
        for the full list: https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__init__

    """
    from llama_cpp import Llama

    # Default to using the model's full context length
    if "n_ctx" not in llamacpp_model_params:
        llamacpp_model_params["n_ctx"] = 0

    if "verbose" not in llamacpp_model_params:
        llamacpp_model_params["verbose"] = False

    # TODO: Remove when https://github.com/ggerganov/llama.cpp/pull/5613 is resolved
    if "tokenizer" not in llamacpp_model_params:
        warnings.warn(
            "The pre-tokenizer in `llama.cpp` handles unicode improperly "
            + "(https://github.com/ggerganov/llama.cpp/pull/5613)\n"
            + "Outlines may raise a `RuntimeError` when building the regex index.\n"
            + "To circumvent this error when using `models.llamacpp()` you may pass the argument"
            + "`tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained(<hf_repo_id>)`\n"
        )

    model = Llama.from_pretrained(repo_id, filename, **llamacpp_model_params)

    return LlamaCpp(model)




    """Integration with OpenAI's API."""
import functools
import warnings
from dataclasses import asdict, dataclass, field, replace
from itertools import zip_longest
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from outlines.base import vectorize
from outlines.caching import cache

__all__ = ["OpenAI", "openai", "azure_openai"]


@dataclass(frozen=True)
class OpenAIConfig:
    """Represents the parameters of the OpenAI API.

    The information was last fetched on 2023/11/20. We document below the
    properties that are specific to the OpenAI API. Not all these properties are
    supported by Outlines.

    Properties
    ----------
    model
        The name of the model. Available models can be found on OpenAI's website.
    frequence_penalty
        Number between 2.0 and -2.0. Positive values penalize new tokens based on
        their existing frequency in the text,
    logit_bias
        Modifies the likelihood of specified tokens to appear in the completion.
        Number between -100 (forbid) and +100 (only allows).
    n
        The number of completions to return for each prompt.
    presence_penalty
        Similar to frequency penalty.
    response_format
        Specifies the format the model must output. `{"type": "json_object"}`
        enables JSON mode.
    seed
        Two completions with the same `seed` value should return the same
        completion. This is however not guaranteed.
    stop
        Up to 4 words where the API will stop the completion.
    temperature
        Number between 0 and 2. Higher values make the output more random, while
        lower values make it more deterministic.
    top_p
        Number between 0 and 1. Parameter for nucleus sampling.
    user
        A unique identifier for the end-user.

    """

    model: str = ""
    frequency_penalty: float = 0
    logit_bias: Dict[int, int] = field(default_factory=dict)
    max_tokens: Optional[int] = None
    n: int = 1
    presence_penalty: float = 0
    response_format: Optional[Dict[str, str]] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    temperature: float = 1.0
    top_p: int = 1
    user: str = field(default_factory=str)


class OpenAI:
    """An object that represents the OpenAI API."""

    def __init__(
        self,
        client,
        config,
        tokenizer=None,
        system_prompt: Optional[str] = None,
    ):
        """Create an `OpenAI` instance.

        This class supports the standard OpenAI API, the Azure OpeanAI API as
        well as compatible APIs that rely on the OpenAI client.

        Parameters
        ----------
        client
            An instance of the API's async client.
        config
            An instance of `OpenAIConfig`. Can be useful to specify some
            parameters that cannot be set by calling this class' methods.
        tokenizer
            The tokenizer associated with the model the client connects to.

        """

        self.client = client
        self.tokenizer = tokenizer
        self.config = config

        # We count the total number of prompt and generated tokens as returned
        # by the OpenAI API, summed over all the requests performed with this
        # model instance.
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def __call__(
        self,
        prompt: Union[str, List[str]],
        max_tokens: Optional[int] = None,
        stop_at: Optional[Union[List[str], str]] = None,
        *,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        samples: Optional[int] = None,
    ) -> np.ndarray:
        """Call the OpenAI API to generate text.

        Parameters
        ----------
        prompt
            A string or list of strings that will be used to prompt the model
        max_tokens
            The maximum number of tokens to generate
        stop_at
            A string or array of strings which, such that the generation stops
            when they are generated.
        system_prompt
            The content of the system message that precedes the user's prompt.
        temperature
            The value of the temperature used to sample tokens
        samples
            The number of completions to generate for each prompt
        stop_at
            Up to 4 words where the API will stop the completion.

        """
        if max_tokens is None:
            max_tokens = self.config.max_tokens
        if stop_at is None:
            stop_at = self.config.stop
        if temperature is None:
            temperature = self.config.temperature
        if samples is None:
            samples = self.config.n

        config = replace(self.config, max_tokens=max_tokens, temperature=temperature, n=samples, stop=stop_at)  # type: ignore

        response, prompt_tokens, completion_tokens = generate_chat(
            prompt, system_prompt, self.client, config
        )
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens

        return response

    def stream(self, *args, **kwargs):
        raise NotImplementedError(
            "Streaming is currently not supported for the OpenAI API"
        )

    def generate_choice(
        self,
        prompt: str,
        choices: List[str],
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Call the OpenAI API to generate one of several choices.

        Parameters
        ----------
        prompt
            A string or list of strings that will be used to prompt the model
        choices
            The list of strings between which we ask the model to choose
        max_tokens
            The maximum number of tokens to generate
        system_prompt
            The content of the system message that precedes the user's prompt.

        """
        if self.tokenizer is None:
            raise ValueError(
                "You must initialize the `OpenAI` class with a tokenizer to use `outlines.generate.choice`"
            )

        config = replace(self.config, max_tokens=max_tokens)

        greedy = False
        decoded: List[str] = []
        encoded_choices_left: List[List[int]] = [
            self.tokenizer.encode(word) for word in choices
        ]

        while len(encoded_choices_left) > 0:
            max_tokens_left = max([len(tokens) for tokens in encoded_choices_left])
            transposed_choices_left: List[Set] = [
                {item for item in subset if item is not None}
                for subset in zip_longest(*encoded_choices_left)
            ]

            if not greedy:
                mask = build_optimistic_mask(transposed_choices_left)
            else:
                mask = {}
                for token in transposed_choices_left[0]:  # build greedy mask
                    mask[token] = 100

            if len(mask) == 0:
                break

            config = replace(config, logit_bias=mask, max_tokens=max_tokens_left)

            response, prompt_tokens, completion_tokens = generate_chat(
                prompt, system_prompt, self.client, config
            )
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens

            encoded_response = self.tokenizer.encode(response)

            if encoded_response in encoded_choices_left:
                decoded.append(response)
                break
            else:
                (
                    encoded_response,
                    encoded_choices_left,
                ) = find_response_choices_intersection(
                    encoded_response, encoded_choices_left
                )

                if len(encoded_response) == 0:
                    greedy = True  # next iteration will be "greedy"
                    continue
                else:
                    decoded.append("".join(self.tokenizer.decode(encoded_response)))

                    if len(encoded_choices_left) == 1:  # only one choice left
                        choice_left = self.tokenizer.decode(encoded_choices_left[0])
                        decoded.append(choice_left)
                        break

                    greedy = False  # after each success, stay with (or switch to) "optimistic" approach

                prompt = prompt + "".join(decoded)

        choice = "".join(decoded)

        return choice

    def generate_json(self):
        """Call the OpenAI API to generate a JSON object."""
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__ + " API"

    def __repr__(self):
        return str(self.config)


@functools.partial(vectorize, signature="(),(),(),()->(s),(),()")
async def generate_chat(
    prompt: str,
    system_prompt: Union[str, None],
    client,
    config: OpenAIConfig,
) -> Tuple[np.ndarray, int, int]:
    """Call OpenAI's Chat Completion API.

    Parameters
    ----------
    prompt
        The prompt we use to start the generation. Passed to the model
        with the "user" role.
    system_prompt
        The system prompt, passed to the model with the "system" role
        before the prompt.
    client
        The API client
    config
        An `OpenAIConfig` instance.

    Returns
    -------
    A tuple that contains the model's response(s) and usage statistics.

    """

    @error_handler
    @cache()
    async def call_api(prompt, system_prompt, config):
        responses = await client.chat.completions.create(
            messages=system_message + user_message,
            **asdict(config),  # type: ignore
        )
        return responses.model_dump()

    system_message = (
        [{"role": "system", "content": system_prompt}] if system_prompt else []
    )
    user_message = [{"role": "user", "content": prompt}]

    responses = await call_api(prompt, system_prompt, config)

    results = np.array(
        [responses["choices"][i]["message"]["content"] for i in range(config.n)]
    )
    usage = responses["usage"]

    return results, usage["prompt_tokens"], usage["completion_tokens"]


def find_longest_intersection(response: List[int], choice: List[int]) -> List[int]:
    """Find the longest intersection between the response and the choice."""
    for i, (token_r, token_c) in enumerate(zip_longest(response, choice)):
        if token_r != token_c:
            return response[:i]

    return response


def find_response_choices_intersection(
    response: List[int], choices: List[List[int]]
) -> Tuple[List[int], List[List[int]]]:
    """Find the longest intersection between the response and the different
    choices.

    Say the response is of the form `[1, 2, 3, 4, 5]` and we have the choices
    `[[1, 2], [1, 2, 3], [6, 7, 8]` then the function will return `[1, 2, 3]` as the
    intersection, and `[[]]` as the list of choices left.

    Parameters
    ----------
    response
        The model's response
    choices
        The remaining possible choices

    Returns
    -------
    A tuple that contains the longest intersection between the response and the
    different choices, and the choices which start with this intersection, with the
    intersection removed.

    """
    max_len_prefix = 0
    choices_left = []
    longest_prefix = []
    for i, choice in enumerate(choices):
        # Find the longest intersection between the response and the choice.
        prefix = find_longest_intersection(response, choice)

        if len(prefix) > max_len_prefix:
            max_len_prefix = len(prefix)
            choices_left = [choice[len(prefix) :]]
            longest_prefix = prefix

        elif len(prefix) == max_len_prefix:
            choices_left.append(choice[len(prefix) :])

    return longest_prefix, choices_left


def build_optimistic_mask(
    transposed: List[Set[int]], max_mask_size: int = 300
) -> Dict[int, int]:
    """We build the largest mask possible.

    Tokens are added from left to right, so if the encoded choices are e.g.
    `[[1,2], [3,4]]`, `1` and `3` will be added before `2` and `4`.

    Parameters
    ----------
    transposed
        A list of lists that contain the nth token of each choice.

    """
    mask: Dict[int, int] = {}
    for tokens in transposed:
        for token in tokens:
            if len(mask) == max_mask_size:
                return mask
            mask[token] = 100

    return mask


def error_handler(api_call_fn: Callable) -> Callable:
    """Handle OpenAI API errors and missing API key."""

    def call(*args, **kwargs):
        import openai

        try:
            return api_call_fn(*args, **kwargs)
        except (
            openai.APITimeoutError,
            openai.InternalServerError,
            openai.RateLimitError,
        ) as e:
            raise OSError(f"Could not connect to the OpenAI API: {e}")
        except (
            openai.AuthenticationError,
            openai.BadRequestError,
            openai.ConflictError,
            openai.PermissionDeniedError,
            openai.NotFoundError,
            openai.UnprocessableEntityError,
        ) as e:
            raise e

    return call


@functools.singledispatch
def openai(model_or_client, *args, **kwargs):
    return OpenAI(model_or_client, *args, **kwargs)


@openai.register(str)
def openai_model(
    model_name: str,
    config: Optional[OpenAIConfig] = None,
    **openai_client_params,
):
    try:
        import tiktoken
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError(
            "The `openai` and `tiktoken` libraries needs to be installed in order to use Outlines' OpenAI integration."
        )

    if config is not None:
        config = replace(config, model=model_name)  # type: ignore
    else:
        config = OpenAIConfig(model=model_name)

    client = AsyncOpenAI(**openai_client_params)

    try:
        tokenizer = tiktoken.encoding_for_model(model_name)
    except KeyError:
        warnings.warn(
            f"Could not find a tokenizer for model {model_name}. Using default cl100k_base."
        )
        tokenizer = tiktoken.get_encoding("cl100k_base")

    return OpenAI(client, config, tokenizer)


def azure_openai(
    deployment_name: str,
    model_name: Optional[str] = None,
    config: Optional[OpenAIConfig] = None,
    **azure_openai_client_params,
):
    try:
        import tiktoken
        from openai import AsyncAzureOpenAI
    except ImportError:
        raise ImportError(
            "The `openai` and `tiktoken` libraries needs to be installed in order to use Outlines' Azure OpenAI integration."
        )

    if config is not None:
        config = replace(config, model=deployment_name)  # type: ignore
    if config is None:
        config = OpenAIConfig(model=deployment_name)

    client = AsyncAzureOpenAI(**azure_openai_client_params)

    try:
        tokenizer = tiktoken.encoding_for_model(model_name or deployment_name)
    except KeyError:
        warnings.warn(
            f"Could not find a tokenizer for model {model_name or deployment_name}. Using default cl100k_base."
        )
        tokenizer = tiktoken.get_encoding("cl100k_base")

    return OpenAI(client, config, tokenizer)



    import dataclasses
import inspect
from typing import TYPE_CHECKING, Iterator, List, Optional, Tuple, Union

from datasets.fingerprint import Hasher

from outlines.generate.api import GenerationParameters, SamplingParameters
from outlines.models.tokenizer import Tokenizer

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizer

    from outlines.processors import OutlinesLogitsProcessor

__all__ = ["transformers"]


KVCacheType = Tuple[Tuple["torch.DoubleTensor", "torch.DoubleTensor"], ...]


def get_llama_tokenizer_types():
    """Get all the Llama tokenizer types/classes that need work-arounds.

    When they can't be imported, a dummy class is created.

    """
    try:
        from transformers.models.llama import LlamaTokenizer
    except ImportError:

        class LlamaTokenizer:  # type: ignore
            pass

    try:
        from transformers.models.llama import LlamaTokenizerFast
    except ImportError:

        class LlamaTokenizerFast:  # type: ignore
            pass

    try:
        from transformers.models.code_llama import CodeLlamaTokenizer
    except ImportError:

        class CodeLlamaTokenizer:  # type: ignore
            pass

    try:
        from transformers.models.code_llama import CodeLlamaTokenizerFast
    except ImportError:

        class CodeLlamaTokenizerFast:  # type: ignore
            pass

    return (
        LlamaTokenizer,
        LlamaTokenizerFast,
        CodeLlamaTokenizer,
        CodeLlamaTokenizerFast,
    )


class TransformerTokenizer(Tokenizer):
    """Represents a tokenizer for models in the `transformers` library."""

    def __init__(self, tokenizer: "PreTrainedTokenizer", **kwargs):
        self.tokenizer = tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id
        self.eos_token = self.tokenizer.eos_token

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.pad_token_id = self.eos_token_id
        else:
            self.pad_token_id = self.tokenizer.pad_token_id
            self.pad_token = self.tokenizer.pad_token

        self.special_tokens = set(self.tokenizer.all_special_tokens)

        self.vocabulary = self.tokenizer.get_vocab()
        self.is_llama = isinstance(self.tokenizer, get_llama_tokenizer_types())

    def encode(
        self, prompt: Union[str, List[str]], **kwargs
    ) -> Tuple["torch.LongTensor", "torch.LongTensor"]:
        kwargs["padding"] = True
        kwargs["return_tensors"] = "pt"
        output = self.tokenizer(prompt, **kwargs)
        return output["input_ids"], output["attention_mask"]

    def decode(self, token_ids: "torch.LongTensor") -> List[str]:
        text = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        return text

    def convert_token_to_string(self, token: str) -> str:
        from transformers.file_utils import SPIECE_UNDERLINE

        string = self.tokenizer.convert_tokens_to_string([token])

        if self.is_llama:
            # A hack to handle missing spaces to HF's Llama tokenizers
            if token.startswith(SPIECE_UNDERLINE) or token == "<0x20>":
                return " " + string

        return string

    def __eq__(self, other):
        if isinstance(other, type(self)):
            if hasattr(self, "model_name") and hasattr(self, "kwargs"):
                return (
                    other.model_name == self.model_name and other.kwargs == self.kwargs
                )
            else:
                return other.tokenizer == self.tokenizer
        return NotImplemented

    def __hash__(self):
        return hash(Hasher.hash(self.tokenizer))

    def __getstate__(self):
        state = {"tokenizer": self.tokenizer}
        return state

    def __setstate__(self, state):
        self.__init__(state["tokenizer"])


class Transformers:
    """Represents a `transformers` model."""

    def __init__(
        self,
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizer",
    ):
        self.model = model
        self.tokenizer = TransformerTokenizer(tokenizer)

    def forward(
        self,
        input_ids: "torch.LongTensor",
        attention_mask: "torch.LongTensor",
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple["torch.FloatTensor", Optional[KVCacheType]]:
        """Compute a forward pass through the transformer model.

        Parameters
        ----------
        input_ids
            The input token ids.  Must be one or two dimensional.
        attention_mask
            The attention mask.  Must be one or two dimensional.
        past_key_values
            A tuple of tuples containing the cached key and value tensors for each
            attention head.

        Returns
        -------
        The computed logits and the new cached key and value tensors.

        """
        try:
            import torch
        except ImportError:
            ImportError(
                "The `torch` library needs to be installed to use `transformers` models."
            )
        assert 0 < input_ids.ndim < 3

        if past_key_values:
            input_ids = input_ids[..., -1].unsqueeze(-1)

        with torch.inference_mode():
            output = self.model(
                input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
                past_key_values=past_key_values,
            )

        return output.logits, output.past_key_values

    def __call__(
        self,
        input_ids: "torch.LongTensor",
        attention_mask: "torch.LongTensor",
        past_key_values: Optional[Tuple] = None,
    ) -> "torch.FloatTensor":
        logits, kv_cache = self.forward(input_ids, attention_mask, past_key_values)
        next_token_logits = logits[..., -1, :]

        return next_token_logits, kv_cache

    def generate(
        self,
        prompts: Union[str, List[str]],
        generation_parameters: GenerationParameters,
        logits_processor: Optional["OutlinesLogitsProcessor"],
        sampling_parameters: SamplingParameters,
    ) -> Union[str, List[str], List[List[str]]]:
        """Generate text using `transformers`.

        Arguments
        ---------
        prompts
            A prompt or list of prompts.
        generation_parameters
            An instance of `GenerationParameters` that contains the prompt,
            the maximum number of tokens, stop sequences and seed. All the
            arguments to `SequenceGeneratorAdapter`'s `__cal__` method.
        logits_processor
            The logits processor to use when generating text.
        sampling_parameters
            An instance of `SamplingParameters`, a dataclass that contains
            the name of the sampler to use and related parameters as available
            in Outlines.

        Returns
        -------
        The generated text
        """
        if isinstance(prompts, str):
            # convert to 2d
            input_ids, attention_mask = self.tokenizer.encode([prompts])
        else:
            input_ids, attention_mask = self.tokenizer.encode(prompts)

        inputs = {
            "input_ids": input_ids.to(self.model.device),
            "attention_mask": attention_mask.to(self.model.device),
        }
        if (
            "attention_mask"
            not in inspect.signature(self.model.forward).parameters.keys()
        ):
            del inputs["attention_mask"]

        generation_kwargs = self._get_generation_kwargs(
            prompts,
            generation_parameters,
            logits_processor,
            sampling_parameters,
        )
        generated_ids = self._generate_output_seq(prompts, inputs, **generation_kwargs)

        # if single str input and single sample per input, convert to a 1D output
        if isinstance(prompts, str):
            generated_ids = generated_ids.squeeze(0)

        return self._decode_generation(generated_ids)

    def stream(
        self,
        prompts: Union[str, List[str]],
        generation_parameters: GenerationParameters,
        logits_processor: Optional["OutlinesLogitsProcessor"],
        sampling_parameters: SamplingParameters,
    ) -> Iterator[Union[str, List[str]]]:
        """
        Temporary stream stand-in which implements stream() signature
        and equivalent behaviour but isn't yielded until generation completes.

        TODO: implement following completion of https://github.com/huggingface/transformers/issues/30810
        """
        if isinstance(prompts, str):
            # convert to 2d
            input_ids, attention_mask = self.tokenizer.encode([prompts])
        else:
            input_ids, attention_mask = self.tokenizer.encode(prompts)
        inputs = {
            "input_ids": input_ids.to(self.model.device),
            "attention_mask": attention_mask.to(self.model.device),
        }
        if (
            "attention_mask"
            not in inspect.signature(self.model.forward).parameters.keys()
        ):
            del inputs["attention_mask"]

        generation_kwargs = self._get_generation_kwargs(
            prompts,
            generation_parameters,
            logits_processor,
            sampling_parameters,
        )
        generated_ids = self._generate_output_seq(prompts, inputs, **generation_kwargs)

        # if single str input and single sample per input, convert to a 1D output
        if isinstance(prompts, str):
            generated_ids = generated_ids.squeeze(0)

        for i in range(generated_ids.size(-1)):
            output_group_ids = generated_ids.select(-1, i).unsqueeze(-1)
            yield self._decode_generation(output_group_ids)

    def _get_generation_kwargs(
        self,
        prompts: Union[str, List[str]],
        generation_parameters: GenerationParameters,
        logits_processor: Optional["OutlinesLogitsProcessor"],
        sampling_parameters: SamplingParameters,
    ) -> dict:
        """
        Conert outlines generation parameters into model.generate kwargs
        """
        from transformers import GenerationConfig, LogitsProcessorList, set_seed

        max_new_tokens, stop_at, seed = dataclasses.astuple(generation_parameters)
        sampler, num_samples, top_p, top_k, temperature = dataclasses.astuple(
            sampling_parameters
        )
        if max_new_tokens is None:
            max_new_tokens = int(2**30)

        # global seed, not desirable
        if seed is not None:
            set_seed(seed)

        if logits_processor is not None:
            logits_processor_list = LogitsProcessorList([logits_processor])
        else:
            logits_processor_list = None

        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            stop_strings=stop_at,
            num_return_sequences=(num_samples or 1),
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            do_sample=(sampler == "multinomial"),
            num_beams=(num_samples if sampler == "beam_search" else 1),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        return dict(
            logits_processor=logits_processor_list,
            generation_config=generation_config,
            tokenizer=self.tokenizer.tokenizer,
        )

    def _generate_output_seq(
        self, prompts, inputs, generation_config, **generation_kwargs
    ):
        input_ids = inputs["input_ids"]
        output_ids = self.model.generate(
            **inputs, generation_config=generation_config, **generation_kwargs
        )

        # encoder-decoder returns output_ids only, decoder-only returns full seq ids
        if self.model.config.is_encoder_decoder:
            generated_ids = output_ids
        else:
            generated_ids = output_ids[:, input_ids.shape[1] :]

        # if batch list inputs AND multiple samples per input, convert generated_id to 3D view
        num_samples = generation_config.num_return_sequences or 1

        if num_samples > 1 and isinstance(prompts, list):
            batch_size = input_ids.size(0)
            num_return_sequences = generation_config.num_return_sequences or 1
            generated_ids = generated_ids.view(batch_size, num_return_sequences, -1)

        return generated_ids

    def _decode_generation(self, generated_ids: "torch.Tensor"):
        if len(generated_ids.shape) == 1:
            return self.tokenizer.decode([generated_ids])[0]
        elif len(generated_ids.shape) == 2:
            return self.tokenizer.decode(generated_ids)
        elif len(generated_ids.shape) == 3:
            return [
                self.tokenizer.decode(generated_ids[i])
                for i in range(len(generated_ids))
            ]
        else:
            raise TypeError(
                f"Generated outputs aren't 1D, 2D or 3D, but instead are {generated_ids.shape}"
            )


def transformers(
    model_name: str,
    device: Optional[str] = None,
    model_kwargs: dict = {},
    tokenizer_kwargs: dict = {},
    model_class=None,
    tokenizer_class=None,
):
    """Instantiate a model from the `transformers` library and its tokenizer.

    Parameters
    ----------
    model_name
        The name of the model as listed on Hugging Face's model page.
    device
        The device(s) on which the model should be loaded. This overrides
        the `device_map` entry in `model_kwargs` when provided.
    model_kwargs
        A dictionary that contains the keyword arguments to pass to the
        `from_pretrained` method when loading the model.
    tokenizer_kwargs
        A dictionary that contains the keyword arguments to pass to the
        `from_pretrained` method when loading the tokenizer.

    Returns
    -------
    A `TransformersModel` model instance.

    """
    if model_class is None or tokenizer_class is None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "The `transformers` library needs to be installed in order to use `transformers` models."
            )
    if model_class is None:
        model_class = AutoModelForCausalLM
    if tokenizer_class is None:
        tokenizer_class = AutoTokenizer

    if device is not None:
        model_kwargs["device_map"] = device

    model = model_class.from_pretrained(model_name, **model_kwargs)

    tokenizer_kwargs.setdefault("padding_side", "left")
    tokenizer = tokenizer_class.from_pretrained(model_name, **tokenizer_kwargs)

    return Transformers(model, tokenizer)


def mamba(
    model_name: str,
    device: Optional[str] = None,
    model_kwargs: dict = {},
    tokenizer_kwargs: dict = {},
):
    try:
        from transformers import MambaForCausalLM

    except ImportError:
        raise ImportError(
            "The `mamba_ssm`, `torch` and `transformer` libraries needs to be installed in order to use Mamba."
        )

    return transformers(
        model_name=model_name,
        device=device,
        model_kwargs=model_kwargs,
        tokenizer_kwargs=tokenizer_kwargs,
        model_class=MambaForCausalLM,
    )


    from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Union

from outlines.generate.api import GenerationParameters, SamplingParameters
from outlines.models import Transformers

if TYPE_CHECKING:
    from outlines.processors import OutlinesLogitsProcessor


class TransformersVision(Transformers):
    def __init__(self, model, tokenizer, processor):
        super().__init__(model, tokenizer)
        self.processor = processor

    def generate(  # type: ignore
        self,
        prompts: Union[str, List[str]],
        media: Union[List[Any], List[List[Any]]],
        generation_parameters: GenerationParameters,
        logits_processor: Optional["OutlinesLogitsProcessor"],
        sampling_parameters: SamplingParameters,
    ) -> Union[str, List[str], List[List[str]]]:
        """Generate text using `transformers`.

        Arguments
        ---------
        prompts
            A prompt or list of prompts.
        media
            A List[PIL.Image] or List[List[PIL.Image]]
        generation_parameters
            An instance of `GenerationParameters` that contains the prompt,
            the maximum number of tokens, stop sequences and seed. All the
            arguments to `SequenceGeneratorAdapter`'s `__cal__` method.
        logits_processor
            The logits processor to use when generating text.
        sampling_parameters
            An instance of `SamplingParameters`, a dataclass that contains
            the name of the sampler to use and related parameters as available
            in Outlines.

        Returns
        -------
        The generated text
        """
        inputs = self.processor(
            text=prompts, images=media, padding=True, return_tensors="pt"
        ).to(self.model.device)

        generation_kwargs = self._get_generation_kwargs(
            prompts,
            generation_parameters,
            logits_processor,
            sampling_parameters,
        )
        generated_ids = self._generate_output_seq(prompts, inputs, **generation_kwargs)

        # if single str input and single sample per input, convert to a 1D output
        if isinstance(prompts, str):
            # Should always be true until NotImplementedError above is fixed
            generated_ids = generated_ids.squeeze(0)

        return self._decode_generation(generated_ids)

    def stream(  # type: ignore
        self,
        prompts: Union[str, List[str]],
        media: Union[Any, List[Any]],  # TODO: docstring
        generation_parameters: GenerationParameters,
        logits_processor: Optional["OutlinesLogitsProcessor"],
        sampling_parameters: SamplingParameters,
    ) -> Iterator[Union[str, List[str]]]:
        raise NotImplementedError


def transformers_vision(
    model_name: str,
    model_class,
    device: Optional[str] = None,
    model_kwargs: dict = {},
    processor_kwargs: dict = {},
    tokenizer_class=None,
    processor_class=None,
):
    """Instantiate a model from the `transformers` library and its tokenizer.

    Parameters
    ----------
    model_name
        The name of the model as listed on Hugging Face's model page.
    model_class
        The `PreTrainedModel` class from transformers to use in initializing the vision model from `model_name`.
        https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel
    device
        The device(s) on which the model should be loaded. This overrides
        the `device_map` entry in `model_kwargs` when provided.
    model_kwargs
        A dictionary that contains the keyword arguments to pass to the
        `from_pretrained` method when loading the model.
    processor_kwargs
        A dictionary that contains the keyword arguments to pass to the
        `from_pretrained` method when loading the processor.

    Returns
    -------
    A `TransformersModel` model instance.

    """
    if processor_class is None or tokenizer_class is None:
        try:
            from transformers import AutoProcessor, AutoTokenizer
        except ImportError:
            raise ImportError(
                "The `transformers` library needs to be installed in order to use `transformers` models."
            )
    if processor_class is None:
        processor_class = AutoProcessor
    if tokenizer_class is None:
        tokenizer_class = AutoTokenizer

    if device is not None:
        model_kwargs["device_map"] = device

    model = model_class.from_pretrained(model_name, **model_kwargs)

    processor_kwargs.setdefault("padding_side", "left")
    processor_kwargs.setdefault("pad_token", "[PAD]")
    processor = processor_class.from_pretrained(model_name, **processor_kwargs)

    if tokenizer_class is None:
        if getattr(processor, "tokenizer", None):
            tokenizer = processor.tokenizer
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, **processor_kwargs)
    else:
        tokenizer = tokenizer_class.from_pretrained(model_name, **processor_kwargs)

    return TransformersVision(model, tokenizer, processor)




    from typing import Dict, Hashable, List, Protocol, Set, Tuple, Union

import numpy as np
from numpy.typing import NDArray


class Tokenizer(Hashable, Protocol):
    eos_token: str
    eos_token_id: int
    pad_token_id: int
    vocabulary: Dict[str, int]
    special_tokens: Set[str]

    def encode(
        self, prompt: Union[str, List[str]]
    ) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Translate the input prompts into arrays of token ids and attention mask."""
        ...

    def decode(self, token_ids: NDArray[np.int64]) -> List[str]:
        """Translate an array of token ids to a string or list of strings."""
        ...

    def convert_token_to_string(self, token: str) -> str:
        """Convert a token to its equivalent string.

        This is for instance useful for BPE tokenizers where whitespaces are
        represented by the special characted `Ġ`. This prevents matching a raw
        token that includes `Ġ` with a string.
        """
        ...