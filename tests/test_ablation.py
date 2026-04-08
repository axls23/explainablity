import torch
import torch.nn as nn

class MockConfig:
    def __init__(self):
        self.ablation_enabled = True
        self.ablation_heads = [1, 2, 12]
        self.n_heads = 14
        self.hidden_dim = 896

def test_ablation_logic():
    config = MockConfig()
    head_dim = config.hidden_dim // config.n_heads # 64
    
    # Mock attn_output [B, S, D]
    B, S, D = 1, 5, 896
    attn_output = torch.ones(B, S, D)
    
    heads_to_kill = config.ablation_heads
    
    for head_idx in heads_to_kill:
        start = head_idx * head_dim
        end = start + head_dim
        attn_output[..., start:end] = 0.0
        
    print(f"Testing ablation for heads {heads_to_kill}...")
    
    # Verify Head 0 is NOT zeroed
    assert torch.all(attn_output[..., 0:head_dim] == 1.0)
    
    # Verify Head 1 IS zeroed
    assert torch.all(attn_output[..., head_dim:2*head_dim] == 0.0)
    
    # Verify Head 2 IS zeroed
    assert torch.all(attn_output[..., 2*head_dim:3*head_dim] == 0.0)
    
    # Verify Head 3 is NOT zeroed
    assert torch.all(attn_output[..., 3*head_dim:4*head_dim] == 1.0)
    
    # Verify Head 12 IS zeroed
    assert torch.all(attn_output[..., 12*head_dim:13*head_dim] == 0.0)
    
    print("Ablation logic verified!")

if __name__ == "__main__":
    test_ablation_logic()
