# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Shared model loading and generation utilities for all benchmark evaluations.
Supports Qwen2.5-VL (and compatible VLMs) with optional LoRA adapters.
"""

import logging
import re

import torch
from PIL import Image

logger = logging.getLogger(__name__)


def load_model(model_path: str, lora_path: str | None = None, device: str = "auto", torch_dtype=None):
    from transformers import AutoModelForVision2Seq, AutoProcessor

    if torch_dtype is None:
        torch_dtype = torch.bfloat16

    logger.info(f"Loading processor from {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    logger.info(f"Loading model from {model_path}")
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    )

    if lora_path:
        from peft import PeftModel

        logger.info(f"Loading LoRA adapter from {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
        logger.info("LoRA merged into base model")

    model.eval()
    return model, processor


def build_multimodal_messages(prompt_messages: list[dict], images: list[Image.Image]) -> list[dict]:
    """
    Convert text messages with <image> placeholders into multimodal message format
    expected by Qwen2.5-VL processor.apply_chat_template.
    """
    messages = []
    for msg in prompt_messages:
        content = msg["content"]
        if isinstance(content, str) and "<image>" in content and images:
            content_parts = []
            segments = re.split(r"(<image>)", content)
            img_idx = 0
            for seg in segments:
                if seg == "<image>" and img_idx < len(images):
                    content_parts.append({"type": "image", "image": images[img_idx]})
                    img_idx += 1
                elif seg:
                    content_parts.append({"type": "text", "text": seg})
            messages.append({"role": msg["role"], "content": content_parts})
        else:
            messages.append(msg)
    return messages


@torch.no_grad()
def generate_response(
    model,
    processor,
    prompt_messages: list[dict],
    images: list[Image.Image] | None = None,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    images = images or []
    messages = build_multimodal_messages(prompt_messages, images)

    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    img_list = images if images else None
    inputs = processor(text=[text], images=img_list, return_tensors="pt", padding=True)

    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0}
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    output_ids = model.generate(**inputs, **gen_kwargs)

    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, input_len:]
    return processor.decode(generated_ids, skip_special_tokens=True)
