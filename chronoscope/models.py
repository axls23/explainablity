"""
Model loading utilities for local HuggingFace causal LMs.
Supports Qwen, DeepSeek, LLaMA, Mistral — any AutoModelForCausalLM.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.table import Table

from .config import ChronoscopeConfig

console = Console()


def detect_num_attention_heads(model) -> int | None:
    """
    Best-effort detection of the model attention head count.

    Tries known config fields first, then falls back to module attributes.
    """
    cfg = getattr(model, "config", None)
    if cfg is None and hasattr(model, "model"):
        cfg = getattr(model.model, "config", None)

    # Common transformer config keys across model families.
    config_keys = ["num_attention_heads", "n_head", "num_heads", "n_heads"]

    if cfg is not None:
        for key in config_keys:
            val = getattr(cfg, key, None)
            if isinstance(val, int) and val > 0:
                return val

        # Some architectures keep this inside nested text_config.
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None:
            for key in config_keys:
                val = getattr(text_cfg, key, None)
                if isinstance(val, int) and val > 0:
                    return val

    # Fallback: inspect attention modules for a num_heads-like field.
    target = model.model if hasattr(model, "model") else model
    for _, module in target.named_modules():
        for key in ["num_heads", "num_attention_heads", "n_head", "n_heads"]:
            val = getattr(module, key, None)
            if isinstance(val, int) and val > 0:
                return val

    return None


def detect_hidden_dim(model) -> int | None:
    """
    Best-effort detection of the model hidden dimension.

    Tries known config fields first, then falls back to checking
    the first layer's hidden dimension.
    """
    cfg = getattr(model, "config", None)
    if cfg is None and hasattr(model, "model"):
        cfg = getattr(model.model, "config", None)

    # Common transformer config keys for hidden size
    config_keys = ["hidden_size", "d_model", "hidden_dim", "dim"]

    if cfg is not None:
        for key in config_keys:
            val = getattr(cfg, key, None)
            if isinstance(val, int) and val > 0:
                return val

        # Some architectures keep this inside nested text_config
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None:
            for key in config_keys:
                val = getattr(text_cfg, key, None)
                if isinstance(val, int) and val > 0:
                    return val

    # Fallback: inspect first linear layer to infer hidden dimension
    target = model.model if hasattr(model, "model") else model
    try:
        for _, module in target.named_modules():
            if hasattr(module, "out_features"):
                # Found a linear layer; assume this is close to hidden_dim
                val = getattr(module, "out_features", None)
                if isinstance(val, int) and val > 0 and val >= 256:
                    # Heuristic: real hidden dims are typically >= 256
                    return val
    except Exception:
        pass

    return None


def load_model(config: ChronoscopeConfig):
    """
    Load a HuggingFace causal LM and tokenizer.

    Returns:
        (model, tokenizer) tuple.
    """
    console.print(
        f"[bold cyan]Loading model:[/] {config.model_name} "
        f"on {config.device} ({config.torch_dtype})"
    )

    # Keep all model/tokenizer metadata resolution offline for reproducibility
    # and to avoid intermittent hub connection resets in restricted networks.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    kwargs = {
        "torch_dtype": config.get_torch_dtype(),
        "device_map": "auto",
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "local_files_only": True, # Prevents huggingface_hub from phoning home and hitting ConnectionResetError
        "attn_implementation": "eager" if not config.use_airllm else None,
    }

    model_ref = config.model_name
    local_pin = getattr(config, "local_model_snapshot_path", None)
    if local_pin and os.path.isdir(local_pin):
        model_ref = local_pin
        console.print(f"[dim]Using pinned local snapshot:[/] {model_ref}")
    else:
        try:
            model_ref = snapshot_download(
                repo_id=config.model_name,
                local_files_only=True,
            )
            console.print(f"[dim]Using local model snapshot:[/] {model_ref}")
        except Exception:
            # Fall back to the original id. If cache is missing, the caller gets
            # a clear error from the underlying loader.
            model_ref = config.model_name

    if config.use_airllm:
        from airllm import AutoModel
        from .adaptive_airllm import AdaptiveAirLLM
        console.print("[yellow]Loading model via AirLLM...[/]")
        
        # AirLLM is extremely slow without compression on 7B+ models
        compression_arg = '4bit' if config.load_in_4bit else None
        
        base_model = AutoModel.from_pretrained(
            model_ref,
            compression=compression_arg
        )
        
        # Wrap with adaptive layer batching for auto-scaled VRAM usage
        model = AdaptiveAirLLM(
            base_model,
            vram_headroom_gb=config.airllm_vram_headroom_gb
        )
    else:
        # Optional 4-bit quantization for very tight VRAM (only for 7B+ models)
        # Disable for small models (<1B params) to avoid dependency issues
        small_models = ["gpt2", "gpt2-medium", "gpt2-large", "distilgpt2"]
        is_small_model = any(sm in config.model_name.lower() for sm in small_models)
        
        if config.load_in_4bit and not is_small_model:
            try:
                from transformers import BitsAndBytesConfig
                
                # NF4/fp4 loading with double quantization for maximum memory savings
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=config.get_torch_dtype(),
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    # Force modules onto the specified device to prevent 'accelerate' from moving them later
                    llm_int8_enable_fp32_cpu_offload=False,
                )
                console.print("[yellow]4-bit NF4 quantization enabled (with double-quant).[/]")
            except ImportError:
                console.print(
                    "[red]bitsandbytes not installed. Loading without quantization.[/]"
                )
        elif config.load_in_4bit and is_small_model:
            console.print(
                f"[dim]4-bit quantization disabled for small model '{config.model_name}' "
                "(not needed for <1B params)[/]"
            )

        model = AutoModelForCausalLM.from_pretrained(model_ref, **kwargs)
        
    tokenizer = AutoTokenizer.from_pretrained(
        model_ref, trust_remote_code=True, local_files_only=True
    )

    # Ensure pad token exists (many models lack one)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not config.use_airllm:
        model.eval()
        if hasattr(model, "config"):
            model.config.output_attentions = True
            model.config.return_dict = True

        detected_heads = detect_num_attention_heads(model)
        if detected_heads is not None:
            prev = getattr(config, "n_heads", None)
            config.n_heads = detected_heads
            if prev != detected_heads:
                console.print(
                    f"[green]Auto-detected attention heads:[/] {detected_heads} "
                    f"(was {prev})"
                )

        console.print(
            f"[bold green]Model loaded.[/] "
            f"Parameters: {sum(p.numel() for p in model.parameters()):,}"
        )
    else:
        detected_heads = detect_num_attention_heads(model)
        if detected_heads is not None:
            prev = getattr(config, "n_heads", None)
            config.n_heads = detected_heads
            if prev != detected_heads:
                console.print(
                    f"[green]Auto-detected attention heads:[/] {detected_heads} "
                    f"(was {prev})"
                )
        console.print(f"[bold green]Model loaded via AirLLM.[/]")

    return model, tokenizer


def list_hookable_layers(model, max_display: int = 60):
    """
    Walk model.named_modules() and display all hookable layers.
    Use this to identify the correct target_layers for your model architecture.
    """
    table = Table(title="Hookable Layers", show_lines=False)
    table.add_column("Index", style="dim", width=5)
    table.add_column("Layer Name", style="cyan")
    table.add_column("Type", style="green")

    layers = []
    # Handle AirLLM wrappers which keep the actual module in 'model'
    import torch.nn as nn
    target_module = model.model if hasattr(model, 'model') and isinstance(model.model, nn.Module) else model
    
    for i, (name, module) in enumerate(target_module.named_modules()):
        if name:  # Skip root
            layers.append((name, type(module).__name__))

    for i, (name, type_name) in enumerate(layers[:max_display]):
        table.add_row(str(i), name, type_name)

    if len(layers) > max_display:
        table.add_row("...", f"({len(layers) - max_display} more)", "...")

    console.print(table)
    console.print(f"\n[bold]Total hookable modules:[/] {len(layers)}")

    return layers


def get_deepest_layer(layer_names: list) -> str:
    """
    Returns the structurally deepest layer based on numerical suffix.
    Handles 'layers.0', 'layers.1', ..., 'layers.23' correctly.
    """
    if not layer_names:
        return ""
    
    def extract_index(name):
        parts = name.split('.')
        for p in reversed(parts):
            if p.isdigit():
                return int(p)
        return -1
        
    return max(layer_names, key=extract_index)
