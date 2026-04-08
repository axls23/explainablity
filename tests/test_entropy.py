import torch
import torch.special

def test_entropy():
    # Simulate a small vocabulary for Qwen (actual is 151k)
    batch_size = 1
    seq_len = 5
    vocab_size = 1000
    
    # Random logits
    logits = torch.randn(batch_size, seq_len, vocab_size)
    
    # Softmax
    probs = torch.softmax(logits, dim=-1)
    
    # Entropy using torch.special.entr
    # H = -sum(p * log(p))
    # torch.special.entr(x) returns -x * log(x)
    entropy = torch.special.entr(probs).sum(dim=-1)
    
    print(f"Entropy shape: {entropy.shape}")
    print(f"Sample entropy: {entropy[0, 0].item():.4f} bits (should be > 0)")
    
    # Verify alignment
    # logits[t] predicts token t+1
    # We want to store this entropy value to align with token t+1
    
    # Manual check
    p = probs[0, 0]
    manual_h = -(p * torch.log(p)).sum()
    print(f"Manual check: {manual_h.item():.4f}")
    
    assert torch.allclose(entropy[0, 0], manual_h)
    print("Test passed!")

if __name__ == "__main__":
    test_entropy()
