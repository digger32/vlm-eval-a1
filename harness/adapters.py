"""Per-family model adapters.

Each adapter implements:
    load(spec)            -> handle (model + processor/tokenizer bundle)
    ask(handle, image, prompt, gen_kwargs) -> raw decoded string   (image may be None)

VLM transformers APIs differ per family and DRIFT across transformers versions. The
llava_next and qwen adapters are fully worked against known APIs. internvl and mllama
are implemented to the documented API but MUST be confirmed by scripts/smoke_test.py on
your installed transformers before the real run. If an adapter fails the gate, drop the
MODEL, not the experiment.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
from PIL import Image

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


@dataclass
class Handle:
    model: Any
    processor: Any
    spec: Any
    adapter: str


def _dtype(spec):
    return _DTYPES.get(spec.dtype, torch.float16)


# --------------------------------------------------------------- LLaVA-NeXT
def _load_llava_next(spec) -> Handle:
    from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
    proc = LlavaNextProcessor.from_pretrained(spec.hf_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        spec.hf_id, torch_dtype=_dtype(spec), device_map="cuda",
        low_cpu_mem_usage=True,
    ).eval()
    return Handle(model, proc, spec, "llava_next")


def _ask_llava_next(h: Handle, image, prompt, gen_kwargs) -> str:
    if image is None:
        conv = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = h.processor.apply_chat_template(conv, add_generation_prompt=True)
        inputs = h.processor(text=text, return_tensors="pt").to("cuda")
    else:
        conv = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        text = h.processor.apply_chat_template(conv, add_generation_prompt=True)
        inputs = h.processor(images=image, text=text, return_tensors="pt").to("cuda")
    out = h.model.generate(**inputs, **gen_kwargs)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return h.processor.decode(gen, skip_special_tokens=True).strip()


# --------------------------------------------------------------- Qwen2-VL
def _load_qwen2_vl(spec) -> Handle:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    proc = AutoProcessor.from_pretrained(spec.hf_id)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        spec.hf_id, torch_dtype=_dtype(spec), device_map="cuda",
    ).eval()
    return Handle(model, proc, spec, "qwen2_vl")


def _ask_qwen_like(h: Handle, image, prompt, gen_kwargs) -> str:
    content = ([] if image is None else [{"type": "image", "image": image}])
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = h.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    images = None if image is None else [image]
    inputs = h.processor(text=[text], images=images, return_tensors="pt").to("cuda")
    out = h.model.generate(**inputs, **gen_kwargs)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return h.processor.decode(gen, skip_special_tokens=True).strip()


# --------------------------------------------------------------- Qwen2.5-VL
def _load_qwen25_vl(spec) -> Handle:
    from transformers import AutoProcessor
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as _Cls
    except ImportError as e:
        raise ImportError(
            "Qwen2.5-VL needs transformers>=4.49. Upgrade or drop qwen25vl_7b.") from e
    proc = AutoProcessor.from_pretrained(spec.hf_id)
    model = _Cls.from_pretrained(
        spec.hf_id, torch_dtype=_dtype(spec), device_map="cuda").eval()
    return Handle(model, proc, spec, "qwen25_vl")


# --------------------------------------------------------------- Qwen3-VL
def _load_qwen3_vl(spec) -> Handle:
    from transformers import AutoProcessor
    try:
        from transformers import Qwen3VLForConditionalGeneration as _Cls
    except ImportError as e:                       # transformers too old
        raise ImportError(
            "Qwen3-VL needs a recent transformers. Upgrade or drop qwen3vl_8b.") from e
    proc = AutoProcessor.from_pretrained(spec.hf_id)
    model = _Cls.from_pretrained(
        spec.hf_id, torch_dtype=_dtype(spec), device_map="cuda").eval()
    return Handle(model, proc, spec, "qwen3_vl")


# --------------------------------------------------------------- InternVL (native -hf)
# Use the official Transformers-native checkpoints (OpenGVLab/InternVL3-8B-hf etc) with
# AutoModelForImageTextToText. The original (non-hf) OpenGVLab repos use trust_remote_code
# written for older transformers and crash on transformers 5.x (all_tied_weights_keys).
def _load_internvl(spec) -> Handle:
    from transformers import AutoProcessor, AutoModelForImageTextToText
    proc = AutoProcessor.from_pretrained(spec.hf_id)
    model = AutoModelForImageTextToText.from_pretrained(
        spec.hf_id, torch_dtype=_dtype(spec), device_map="cuda").eval()
    return Handle(model, proc, spec, "internvl")


def _ask_internvl(h: Handle, image, prompt, gen_kwargs) -> str:
    if image is None:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    else:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    inputs = h.processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to(h.model.device, dtype=_dtype(h.spec))
    input_len = inputs["input_ids"].shape[1]
    out = h.model.generate(**inputs, **gen_kwargs)
    return h.processor.decode(out[0][input_len:], skip_special_tokens=True).strip()


# --------------------------------------------------------------- Llama-3.2-Vision (mllama)
def _load_mllama(spec) -> Handle:
    from transformers import MllamaForConditionalGeneration, AutoProcessor
    proc = AutoProcessor.from_pretrained(spec.hf_id)
    model = MllamaForConditionalGeneration.from_pretrained(
        spec.hf_id, torch_dtype=_dtype(spec), device_map="cuda").eval()
    return Handle(model, proc, spec, "mllama")


def _ask_mllama(h: Handle, image, prompt, gen_kwargs) -> str:
    if image is None:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = h.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = h.processor(text=text, return_tensors="pt").to("cuda")
    else:
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        text = h.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = h.processor(image, text, return_tensors="pt").to("cuda")
    out = h.model.generate(**inputs, **gen_kwargs)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return h.processor.decode(gen, skip_special_tokens=True).strip()


# --------------------------------------------------------------- Gemma-3 (Google)
def _load_gemma3(spec) -> Handle:
    from transformers import AutoProcessor
    try:
        from transformers import Gemma3ForConditionalGeneration as _Cls
    except ImportError as e:
        raise ImportError(
            "Gemma-3 needs a recent transformers. Upgrade or pick another model.") from e
    proc = AutoProcessor.from_pretrained(spec.hf_id)
    model = _Cls.from_pretrained(
        spec.hf_id, torch_dtype=_dtype(spec), device_map="cuda").eval()
    return Handle(model, proc, spec, "gemma3")


def _ask_gemma3(h: Handle, image, prompt, gen_kwargs) -> str:
    if image is None:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    else:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    inputs = h.processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to(h.model.device, dtype=_dtype(h.spec))
    input_len = inputs["input_ids"].shape[1]
    out = h.model.generate(**inputs, **gen_kwargs)
    return h.processor.decode(out[0][input_len:], skip_special_tokens=True).strip()


# --------------------------------------------------------------- Phi-3.5-Vision (MS)
def _load_phi3v(spec) -> Handle:
    from transformers import AutoModelForCausalLM, AutoProcessor
    proc = AutoProcessor.from_pretrained(
        spec.hf_id, trust_remote_code=True, num_crops=4)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, torch_dtype=_dtype(spec), trust_remote_code=True,
        device_map="cuda", _attn_implementation="eager").eval()
    return Handle(model, proc, spec, "phi3v")


def _ask_phi3v(h: Handle, image, prompt, gen_kwargs) -> str:
    if image is None:
        full = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        inputs = h.processor(full, return_tensors="pt").to("cuda")
    else:
        full = f"<|user|>\n<|image_1|>\n{prompt}<|end|>\n<|assistant|>\n"
        inputs = h.processor(full, [image], return_tensors="pt").to("cuda")
    out = h.model.generate(**inputs, **gen_kwargs,
                           eos_token_id=h.processor.tokenizer.eos_token_id)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return h.processor.decode(gen, skip_special_tokens=True).strip()


# --------------------------------------------------------------- Molmo (AllenAI)
def _load_molmo(spec) -> Handle:
    from transformers import AutoModelForCausalLM, AutoProcessor
    proc = AutoProcessor.from_pretrained(
        spec.hf_id, trust_remote_code=True, torch_dtype=_dtype(spec), device_map="cuda")
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, trust_remote_code=True, torch_dtype=_dtype(spec),
        device_map="cuda").eval()
    return Handle(model, proc, spec, "molmo")


def _ask_molmo(h: Handle, image, prompt, gen_kwargs) -> str:
    from transformers import GenerationConfig
    if image is None:                       # text-only / blind path
        inputs = h.processor.process(text=prompt)
    else:
        inputs = h.processor.process(images=[image], text=prompt)
    inputs = {k: v.to(h.model.device).unsqueeze(0) for k, v in inputs.items()}
    gconf = GenerationConfig(
        max_new_tokens=gen_kwargs.get("max_new_tokens", 64),
        do_sample=gen_kwargs.get("do_sample", False),
        stop_strings="<|endoftext|>")
    out = h.model.generate_from_batch(inputs, gconf, tokenizer=h.processor.tokenizer)
    gen = out[0, inputs["input_ids"].size(1):]
    return h.processor.tokenizer.decode(gen, skip_special_tokens=True).strip()


# --------------------------------------------------------------- generic native proc+images
def _ask_proc_images(h: Handle, image, prompt, gen_kwargs) -> str:
    """Native path for processors that take (text=prompt, images=[img]) after a
    tokenize=False chat template. Works for Pixtral and Idefics3."""
    if image is None:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = h.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        inputs = h.processor(text=text, return_tensors="pt").to(
            h.model.device, dtype=_dtype(h.spec))
    else:
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        text = h.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        inputs = h.processor(text=text, images=[image], return_tensors="pt").to(
            h.model.device, dtype=_dtype(h.spec))
    input_len = inputs["input_ids"].shape[1]
    out = h.model.generate(**inputs, **gen_kwargs)
    return h.processor.decode(out[0][input_len:], skip_special_tokens=True).strip()


# Pixtral-12B (Mistral) — native via LlavaForConditionalGeneration
def _load_pixtral(spec) -> Handle:
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    proc = AutoProcessor.from_pretrained(spec.hf_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        spec.hf_id, torch_dtype=_dtype(spec), device_map="cuda").eval()
    return Handle(model, proc, spec, "pixtral")


# Idefics3-8B (HuggingFace) — native via AutoModelForImageTextToText
def _load_idefics3(spec) -> Handle:
    from transformers import AutoProcessor, AutoModelForImageTextToText
    proc = AutoProcessor.from_pretrained(spec.hf_id)
    model = AutoModelForImageTextToText.from_pretrained(
        spec.hf_id, torch_dtype=_dtype(spec), device_map="cuda").eval()
    return Handle(model, proc, spec, "idefics3")


# --------------------------------------------------------------- dispatch
_LOADERS = {
    "llava_next": _load_llava_next,
    "qwen2_vl": _load_qwen2_vl,
    "qwen25_vl": _load_qwen25_vl,
    "qwen3_vl": _load_qwen3_vl,
    "internvl": _load_internvl,
    "mllama": _load_mllama,
    "gemma3": _load_gemma3,
    "phi3v": _load_phi3v,
    "molmo": _load_molmo,
    "pixtral": _load_pixtral,
    "idefics3": _load_idefics3,
}
_ASKERS = {
    "llava_next": _ask_llava_next,
    "qwen2_vl": _ask_qwen_like,
    "qwen25_vl": _ask_qwen_like,
    "qwen3_vl": _ask_qwen_like,
    "internvl": _ask_internvl,
    "mllama": _ask_mllama,
    "gemma3": _ask_gemma3,
    "phi3v": _ask_phi3v,
    "molmo": _ask_molmo,
    "pixtral": _ask_proc_images,
    "idefics3": _ask_proc_images,
}


def load(spec) -> Handle:
    if spec.adapter not in _LOADERS:
        raise KeyError(f"no adapter {spec.adapter!r}")
    return _LOADERS[spec.adapter](spec)


def ask(handle: Handle, image, prompt: str, gen_kwargs: dict) -> str:
    return _ASKERS[handle.adapter](handle, image, prompt, gen_kwargs)
