from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    pass
from typing import List, Dict, Any, Union
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np


class WhisperEngine:
    def __init__(self, model_name: str = "openai/whisper-large-v3", **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

        # Check local cache first
        try:
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_name, cache_dir="./model_cache", **self.kwargs
            )
            self.processor = WhisperProcessor.from_pretrained(
                self.model_name, cache_dir="./model_cache", **self.kwargs
            )
            print(f"Loaded {self.model_name} from local cache.")
            return
        except Exception as e:
            print(f"Failed to load from cache: {e}")

        # If all else fails, load from Hugging Face
        print(f"Loading {self.model_name} from Hugging Face.")
        self.processor = WhisperProcessor.from_pretrained(self.model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)

    def __call__(
        self,
        raw_speech: Union[
            np.ndarray | List[float] | List[np.ndarray] | List[List[float]]
        ],
    ):
        # Convert raw_speech to numpy array if it's not already
        if isinstance(raw_speech, list):
            if isinstance(raw_speech[0], list):  # List of List[float]
                audio_sample = np.array(raw_speech)
            elif isinstance(raw_speech[0], np.ndarray):  # List of np.ndarray
                audio_sample = np.concatenate(raw_speech)
            else:  # List[float]
                audio_sample = np.array(raw_speech)
        elif isinstance(raw_speech, np.ndarray):
            audio_sample = raw_speech
        else:
            raise ValueError("Unsupported input type for raw_speech")

        # Ensure audio_sample is 1D
        audio_sample = audio_sample.flatten()

        # Create a dictionary with the array and sampling rate
        # Note: Assuming a default sampling rate of 16000 Hz, adjust if needed
        audio_sample = {"array": audio_sample, "sampling_rate": 16000}

        input_features = self.processor(
            audio_sample["array"],
            sampling_rate=audio_sample["sampling_rate"],
            return_tensors="pt",
        ).input_features

        # Generate token ids
        predicted_ids = self.model.generate(input_features)

        # Decode token ids to text
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )

        return transcription
