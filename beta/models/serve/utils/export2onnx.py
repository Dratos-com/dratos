from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    pass
import torch
from transformers import AutoModel, AutoTokenizer
import mlflow
import os
from beta.services.session_manager import SessionManager


def export_model_to_onnx(
    model_name, output_dir, task="text-classification", user_id=None
):
    # Create a session if user_id is provided
    session = None
    if user_id:
        session = SessionManager.create_session(user_id)

    # Load model and tokenizer based on task
    if task == "text-classification":
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif task == "speech-recognition":
        from transformers import AutoFeatureExtractor, AutoModelForSpeechSeq2Seq

        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    elif task == "image-generation":
        from diffusers import StableDiffusionPipeline

        model = StableDiffusionPipeline.from_pretrained(model_name)
    else:
        raise ValueError(f"Unsupported task: {task}")

    # Prepare sample input
    if task == "text-classification":
        sample_text = "This is a sample input for ONNX export"
        inputs = tokenizer(sample_text, return_tensors="pt")
        input_names = ["input_ids", "attention_mask"]
        output_names = ["last_hidden_state"]
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence"},
            "attention_mask": {0: "batch_size", 1: "sequence"},
            "last_hidden_state": {0: "batch_size", 1: "sequence"},
        }
    elif task == "speech-recognition":
        import numpy as np

        sample_audio = np.random.randn(16000)  # 1 second of audio at 16kHz
        inputs = feature_extractor(sample_audio, return_tensors="pt")
        input_names = ["input_features"]
        output_names = ["logits"]
        dynamic_axes = {
            "input_features": {0: "batch_size", 1: "channels", 2: "features"},
            "logits": {0: "batch_size", 1: "sequence"},
        }
    elif task == "image-generation":
        sample_prompt = "A beautiful landscape"
        inputs = model.tokenizer(sample_prompt, return_tensors="pt")
        input_names = ["input_ids", "attention_mask"]
        output_names = ["sample"]
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence"},
            "attention_mask": {0: "batch_size", 1: "sequence"},
            "sample": {0: "batch_size", 1: "channels", 2: "height", 3: "width"},
        }

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Export model to ONNX
    onnx_path = os.path.join(output_dir, f"{model_name.split('/')[-1]}_{task}.onnx")

    if task == "image-generation":
        # For image generation, we need to export the UNet component
        unet = model.unet
        torch.onnx.export(
            unet,
            (torch.randn(2, 4, 64, 64), torch.randn(2), torch.randn(2, 77, 768)),
            onnx_path,
            input_names=["sample", "timestep", "encoder_hidden_states"],
            output_names=["pred_sample"],
            dynamic_axes={
                "sample": {0: "batch_size", 2: "height", 3: "width"},
                "encoder_hidden_states": {0: "batch_size"},
            },
            do_constant_folding=True,
            opset_version=14,
        )
    else:
        torch.onnx.export(
            model,
            (
                (inputs.input_ids, inputs.attention_mask)
                if task == "text-classification"
                else inputs.input_features
            ),
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            opset_version=14,
        )

    # Log model to MLflow
    with mlflow.start_run():
        mlflow.onnx.log_model(onnx_path, "model")

    print(f"Model exported to ONNX and logged to MLflow: {onnx_path}")

    # Update session if it exists
    if session:
        session.data["exported_model"] = {
            "name": model_name,
            "task": task,
            "onnx_path": onnx_path,
        }
        SessionManager.update_session(session)


# Example usage
if __name__ == "__main__":
    user_id = "example_user_123"  # You would get this from your authentication system

    # Text classification example
    model_name_text = "distilbert-base-uncased-finetuned-sst-2-english"
    output_dir_text = "onnx_models/text_classification"
    export_model_to_onnx(
        model_name_text, output_dir_text, task="text-classification", user_id=user_id
    )

    # Audio transcription example
    model_name_audio = "openai/whisper-tiny"
    output_dir_audio = "onnx_models/audio_transcription"
    export_model_to_onnx(model_name_audio, output_dir_audio, task="audio-transcription")

    # Image generation example
    model_name_image = "runwayml/stable-diffusion-v1-5"
    output_dir_image = "onnx_models/image_generation"
    export_model_to_onnx(model_name_image, output_dir_image, task="image-generation")
