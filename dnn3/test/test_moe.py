import torch
import torch.nn.functional as F
from src.moe import Router, MixtureOfExperts

@torch.no_grad()
def test_router() -> None:
    """Test Router output shapes and values."""
    torch.manual_seed(42)
    batch, seq_len, hidden_dim = 2, 4, 8
    num_experts, top_k = 4, 2
    
    router = Router(hidden_dim, num_experts, top_k)
    x = torch.randn(batch, seq_len, hidden_dim)
    
    weights, indices = router(x)
    
    assert weights.shape == (batch, seq_len, top_k)
    assert indices.shape == (batch, seq_len, top_k)
    
    # Check if weights sum to 1 (softmax property)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch, seq_len), atol=1e-5)
    
    # Check that indices are valid expert indices
    assert (indices >= 0).all() and (indices < num_experts).all()
    
    # Check that for each token, unique experts are selected
    for b in range(batch):
        for s in range(seq_len):
            idx = indices[b, s]
            assert len(torch.unique(idx)) == top_k, "Selected experts should be unique per token"

@torch.no_grad()
def test_router_deterministic() -> None:
    """Test Router selection logic with known weights."""
    hidden_dim = 4
    num_experts = 3
    top_k = 2
    router = Router(hidden_dim, num_experts, top_k)
    
    # Set weights to select specific experts
    # Expert 0 -> dim 0
    # Expert 1 -> dim 1
    # Expert 2 -> dim 2
    with torch.no_grad():
        router.lin.weight.zero_()
        router.lin.bias.zero_()
        router.lin.weight[0, 0] = 5.0
        router.lin.weight[1, 1] = 5.0
        router.lin.weight[2, 2] = 5.0
        
    x = torch.zeros(1, 1, hidden_dim)
    
    # Input has signal on dim 0 and 1, so experts 0 and 1 should be selected
    x[0, 0, 0] = 1.0
    x[0, 0, 1] = 1.0 
    
    weights, indices = router(x)
    
    # Check indices contains 0 and 1
    # (flatten to check set membership easily)
    selected = indices.flatten().tolist()
    assert 0 in selected
    assert 1 in selected
    assert 2 not in selected
    
    # Weights should be equal since inputs were equal
    # shape [1, 1, 2]
    w = weights[0, 0]
    assert torch.isclose(w[0], w[1], atol=1e-5)

@torch.no_grad()
def test_moe_forward() -> None:
    """Test MixtureOfExperts forward pass shape."""
    torch.manual_seed(42)
    batch, seq_len, hidden_dim = 2, 4, 8
    inner_dim = 16
    num_experts, top_k = 4, 2
    
    model = MixtureOfExperts(hidden_dim, inner_dim, num_experts, top_k)
    x = torch.randn(batch, seq_len, hidden_dim)
    
    output = model(x)
    
    assert output.shape == (batch, seq_len, hidden_dim)
    assert not torch.isnan(output).any()

@torch.no_grad()
def test_moe_values() -> None:
    """Test MixtureOfExperts logic with deterministic setup."""
    # We set up a simple case where we select specific experts and check if their processing is applied.
    # To do this effectively, we need to control the Router AND the Experts.
    
    hidden_dim = 2
    inner_dim = 2
    num_experts = 2
    top_k = 1
    
    model = MixtureOfExperts(hidden_dim, inner_dim, num_experts, top_k)
    
    # 1. Setup Router: Always select Expert 0
    # Router maps bias=0, weight=0 => outputs 0.
    # Set weight so that dim 0 maps to expert 0 with high value
    model.router.lin.weight.zero_()
    model.router.lin.bias.zero_()
    model.router.lin.weight[0, 0] = 10.0 # Input dim 0 -> Expert 0 (logit 10 * x0)
    # Expert 1 weight stays 0
    
    # 2. Setup Experts
    # Expert 0: Identity (using our MockSwiGLUFeedForward structure is hard to make purely identity due to silu)
    # Instead, let's just make Expert 0 output non-zero and Expert 1 output zero, or something distinguishable.
    # But Wait, we are using the MockSwiGLUFeedForward defined in this file (if we injected it).
    # If the mock uses Random weights init, we should just check the values are what we expect from the operations.
    
    # Let's simplify: Test that correct expert is CALLED.
    # Only Expert 0 should receive gradients or inputs? No, hard to check in no_grad.
    
    # Let's check output consistency.
    x = torch.ones(1, 1, hidden_dim) # [1, 1]
    
    # With Expert 0 selected (weight[0,0]=10, input[0]=1 => logit0=10, logit1=0)
    # Router output: indices=[0], weights=[1.0] (softmax of [10]) approx.
    # Actually softmax([10]) is 1.0 (if topk=1).
    
    # We'll rely on the mock implementation:
    # l3(silu(l1(x)) * l2(x))
    
    # Let's set Expert 0 weights to 1s and Expert 1 to -1s or something.
    for p in model.experts[0].parameters():
        p.data.fill_(1.0)
    for p in model.experts[1].parameters():
        p.data.fill_(-1.0)
        
    output = model(x)
    
    # Check that output roughly matches what Expert 0 would produce
    # x = [1, 1]
    # Expert 0:
    # l1(x) = [1*1+1*1, ...] = [2, 2] (inner_dim=2)
    # l2(x) = [2, 2]
    # silu(2) * 2 = 2 / (1 + e^-2) * 2 approx 1.76 * 2 = 3.52...
    # l3(...) maps back to hidden_dim.
    
    # If Expert 1 was selected (which has -1 weights):
    # l1(x) = [-2, -2]
    # l2(x) = [-2, -2]
    # silu(-2) * -2 = ...
    
    # Basically we just want to ensure the result is consistent with Expert 0 running.
    # Since we can calculate Expert 0 result directly:
    manual_exp0 = model.experts[0](x)
    assert torch.allclose(output, manual_exp0, atol=1e-5), "Output should match Expert 0"

if __name__ == "__main__":
    test_router()
    test_router_deterministic()
    test_moe_forward()
    test_moe_values()