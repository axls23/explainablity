import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from chronoscope.interceptor import ChronoscopeInterceptor
from chronoscope.config import ChronoscopeConfig

def test_live_ablation():
    config = ChronoscopeConfig()
    config.ablation_enabled = True
    config.ablation_heads = [1, 2, 12]
    config.max_new_tokens = 5
    
    print("Loading model for ablation test...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    interceptor = ChronoscopeInterceptor(model, tokenizer, config)
    
    prompt = "The quick brown fox"
    print(f"Running generation with Ablation {config.ablation_heads}...")
    
    # We use a context manager if available, but interceptor registers hooks in __init__
    # So we just run generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=config.max_new_tokens)
    
    result = tokenizer.decode(outputs[0])
    print(f"Result: {result}")
    print("Test complete.")

if __name__ == "__main__":
    test_live_ablation()
