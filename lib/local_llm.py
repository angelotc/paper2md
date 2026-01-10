"""Local LLM inference using LiquidAI LFM2.5-1.2B-Instruct."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

# Singleton pattern for model caching
_MODEL_CACHE: dict[str, tuple["PreTrainedModel", "PreTrainedTokenizer"]] = {}

DEFAULT_LOCAL_MODEL = "LiquidAI/LFM2.5-1.2B-Instruct"


def _get_device() -> str:
    """Determine best available device."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def load_local_model(
    model_id: str | None = None,
    device: str | None = None,
) -> tuple["PreTrainedModel", "PreTrainedTokenizer"]:
    """
    Load local LLM model and tokenizer.
    
    Uses LiquidAI/LFM2.5-1.2B-Instruct by default - a 1.2B parameter
    hybrid model optimized for edge deployment.
    
    Args:
        model_id: HuggingFace model ID (default: LiquidAI/LFM2.5-1.2B-Instruct)
        device: Device to load model on (auto-detected if None)
        
    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise RuntimeError(
            "Local LLM requires transformers and torch. "
            "Run: pip install transformers torch"
        ) from e

    model_id = model_id or os.environ.get("LOCAL_MODEL", DEFAULT_LOCAL_MODEL)
    device = device or _get_device()
    
    cache_key = f"{model_id}:{device}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    # Log device info
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[INFO] Loading local model: {model_id}")
        print(f"[INFO] Using GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    elif device == "mps":
        print(f"[INFO] Loading local model: {model_id}")
        print(f"[INFO] Using Apple Silicon (MPS)")
    else:
        print(f"[INFO] Loading local model: {model_id}")
        print(f"[INFO] Using CPU (no GPU detected - this will be slow)")
    
    # Determine dtype based on device
    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto" if device == "cuda" else None,
        dtype=dtype,
        trust_remote_code=True,
    )
    
    if device != "cuda":  # device_map="auto" handles cuda
        model = model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    _MODEL_CACHE[cache_key] = (model, tokenizer)
    print(f"[INFO] Model loaded successfully")
    
    return model, tokenizer


def generate_local(
    prompt: str,
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    max_new_tokens: int = 1024,
    temperature: float = 0.1,
    top_k: int = 50,
    top_p: float = 0.1,
    repetition_penalty: float = 1.05,
) -> str:
    """
    Generate text using local LLM.
    
    Uses LFM2.5 recommended generation parameters by default.
    
    Args:
        prompt: User prompt to send to the model
        model: Loaded model instance
        tokenizer: Loaded tokenizer instance
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.1 recommended for LFM2.5)
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        repetition_penalty: Penalty for token repetition
        
    Returns:
        Generated text response
    """
    import torch
    
    # Format as chat using tokenizer's chat template
    messages = [{"role": "user", "content": prompt}]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
    ).to(model.device)
    
    # Create attention mask (all 1s since no padding in input)
    attention_mask = torch.ones_like(input_ids)
    
    # Generate with recommended LFM2.5 parameters
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens (skip prompt)
    generated_tokens = output[0][input_ids.shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return response.strip()


class LocalLLMClient:
    """
    Client interface matching OpenAI-style API for drop-in replacement.
    
    This provides compatibility with the existing summarization code
    while using local inference.
    """
    
    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
    ):
        self.model, self.tokenizer = load_local_model(model_id, device)
        self._model_id = model_id or DEFAULT_LOCAL_MODEL
    
    @property
    def model_name(self) -> str:
        return self._model_id
    
    def generate(self, prompt: str, max_new_tokens: int = 1024) -> str:
        """Generate response for a prompt."""
        return generate_local(
            prompt=prompt,
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
        )


def is_local_available() -> bool:
    """Check if local LLM dependencies are installed."""
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM  # noqa: F401
        return True
    except ImportError:
        return False
