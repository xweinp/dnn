import torch
from src import RotaryPositionalEmbedding
from src.attention import calculate_attention, GroupedQueryAttention, calculate_sliding_attention, SWAttention

@torch.no_grad()
def test_calculate_attention() -> None:
    """Test the calculate_attention function independently of module weights."""
    torch.manual_seed(42)
    batch, seq_len = 2, 4
    num_heads, num_kv_heads, head_dim = 4, 2, 4

    q = torch.randn(batch, num_heads, seq_len, head_dim)
    k = torch.randn(batch, num_kv_heads, seq_len, head_dim)
    v = torch.randn(batch, num_kv_heads, seq_len, head_dim)

    key_weights = torch.randn(num_heads)
    rope = RotaryPositionalEmbedding(head_dim)
    scale = head_dim ** -0.5

    output = calculate_attention(q, k, v, key_weights, rope, scale, q.device)

    assert output.shape == (batch, num_heads, seq_len, head_dim), \
        f"Wrong output shape: {output.shape}, expected {(batch, num_heads, seq_len, head_dim)}"
    expected = torch.tensor(
        [[[[ 1.7744, -0.9216,  0.9624, -0.3370],
          [ 0.3143, -0.2881,  0.7230,  0.4999],
          [ 0.3844,  0.4978,  0.3157,  0.0305],
          [ 0.3285,  0.6624,  0.5529,  0.1313]],

         [[ 1.7744, -0.9216,  0.9624, -0.3370],
          [-0.0153, -0.1452,  0.6690,  0.6888],
          [ 0.1277,  0.6668,  0.2440,  0.1472],
          [ 0.3011,  0.7186,  0.5328,  0.1269]],

         [[-0.8146, -1.0212, -0.4949, -0.5923],
          [-0.2359, -0.1480, -0.2879, -1.6233],
          [-0.0641,  0.5656, -0.6690, -1.0527],
          [-0.0747, -0.1315, -0.0178,  0.9354]],

         [[-0.8146, -1.0212, -0.4949, -0.5923],
          [-0.6315, -0.7449, -0.4294, -0.9186],
          [-0.6552, -0.6012, -0.6129, -0.5296],
          [ 0.1580,  0.3066, -0.0132, -1.6795]]],

        [[[-0.0045,  1.6668,  0.1539, -1.0603],
          [-0.2895,  0.8726,  0.2773,  0.4695],
          [-0.2155,  0.2792, -0.5029, -0.0560],
          [-0.1606,  0.2407, -0.3138,  0.0749]],

         [[-0.0045,  1.6668,  0.1539, -1.0603],
          [-0.2733,  0.9177,  0.2703,  0.3825],
          [-0.2363,  0.3177, -0.4039,  0.0707],
          [-0.1465,  0.1215, -0.3514,  0.1096]],

         [[-0.9291,  0.2762, -0.5389,  0.4626],
          [-0.9150,  0.2015, -0.4932,  0.7090],
          [-0.5811,  0.0459, -0.2781,  0.7316],
          [-0.2906,  0.2335, -0.0824,  0.5158]],

         [[-0.9291,  0.2762, -0.5389,  0.4626],
          [-0.8976,  0.1089, -0.4365,  1.0148],
          [-0.1522, -0.1901,  0.0176,  0.8908],
          [ 0.8761, -0.5560,  0.6310,  0.6123]]]]
    )

    assert torch.allclose(output, expected, atol=1e-4), \
        f"calculate_attention output values mismatch"


@torch.no_grad()
def test_grouped_query_attention() -> None:
    """Test GroupedQueryAttention module."""
    torch.manual_seed(42)
    hidden_dim = 16
    num_heads = 4
    head_dim = 4
    num_kv_heads = 2
    
    gqa = GroupedQueryAttention(hidden_dim, num_heads, head_dim, num_kv_heads)
    
    batch_size = 2
    seq_len = 8
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    output = gqa(x)
    
    assert output.shape == (batch_size, seq_len, hidden_dim), \
        f"Output shape mismatch: {output.shape} vs {(batch_size, seq_len, hidden_dim)}"

@torch.no_grad()
def test_calculate_sliding_attention() -> None:
    """Test the calculate_sliding_attention function independently of module weights."""
    torch.manual_seed(42)
    batch, seq_len = 2, 4
    num_heads, head_dim = 4, 4
    window_size = 2

    q = torch.randn(batch, num_heads, seq_len, head_dim)
    k = torch.randn(batch, num_heads, seq_len, head_dim)
    v = torch.randn(batch, num_heads, seq_len, head_dim)

    key_weights = torch.randn(num_heads)
    rope = RotaryPositionalEmbedding(head_dim)
    scale = head_dim ** -0.5

    output = calculate_sliding_attention(q, k, v, key_weights, rope, scale, q.device, window_size)

    assert output.shape == (batch, num_heads, seq_len, head_dim), \
        f"Wrong output shape: {output.shape}, expected {(batch, num_heads, seq_len, head_dim)}"

    expected = torch.tensor(
        [[[[-6.8548e-01,  5.6356e-01, -1.5072e+00, -1.6107e+00],
          [-7.5833e-01,  5.5151e-01, -1.3803e+00, -1.3910e+00],
          [-7.5970e-01,  5.4425e-01, -1.3694e+00, -1.4018e+00],
          [-1.0289e+00, -1.5047e-03,  1.3449e-01,  8.8395e-02]],

         [[-1.3793e+00,  6.2580e-01, -2.5850e+00, -2.4000e-02],
          [-1.0804e+00,  2.9937e-01, -1.5638e+00, -4.5193e-03],
          [-3.7572e-01,  9.0874e-01, -9.8827e-01,  3.2158e-01],
          [ 5.5610e-01,  9.3138e-01,  8.2518e-01,  3.6249e-01]],

         [[ 9.7329e-01, -1.0151e+00, -5.4192e-01, -4.4102e-01],
          [ 3.7820e-01, -6.0546e-01, -6.2194e-01, -2.5908e-01],
          [ 7.5603e-01, -4.5413e-01, -2.9462e-01, -6.9975e-02],
          [ 5.6501e-01,  6.4487e-02,  4.0517e-01,  4.1787e-01]],

         [[ 4.0380e-01, -7.1398e-01,  8.3373e-01, -9.5855e-01],
          [ 4.2490e-01,  1.1594e-01, -4.9589e-01, -1.0976e+00],
          [ 3.5349e-01, -4.0529e-01, -6.6044e-01, -1.1089e+00],
          [-2.1912e-01, -6.5963e-01,  1.6555e-01, -1.0503e+00]]],


        [[[ 4.3344e-01, -7.1719e-01,  1.0554e+00, -1.4534e+00],
          [ 4.4607e-01, -2.8344e-01,  6.3300e-01, -8.4259e-01],
          [ 4.2691e-01,  3.2082e-01, -4.8548e-01, -5.2133e-01],
          [ 3.8897e-01,  6.5369e-01, -1.5100e+00, -7.1436e-01]],

         [[ 8.8538e-01,  1.8244e-01,  7.8638e-01, -5.7920e-02],
          [ 7.1542e-01, -2.9334e-01,  1.0705e-01, -3.1831e-04],
          [ 6.7078e-01,  7.1021e-01,  4.8379e-02,  5.4688e-01],
          [ 7.3988e-01,  2.3339e-01, -3.6269e-01,  3.0450e-01]],

         [[-7.9394e-01,  3.7523e-01,  8.7910e-02, -1.2415e+00],
          [-5.1264e-01, -3.4904e-01, -2.9172e-01,  6.7694e-01],
          [ 4.9596e-01,  5.8218e-01, -1.2478e-01,  1.5970e-01],
          [ 6.7310e-02,  1.5020e-01, -2.8622e-01,  1.1465e+00]],

         [[-2.1844e-01,  1.6630e-01,  2.1442e+00,  1.7046e+00],
          [ 9.8019e-02,  4.3332e-01,  8.2743e-01,  1.1330e+00],
          [ 2.0074e-02, -5.5385e-02,  2.2436e-01,  9.4053e-01],
          [-1.2419e-01,  1.2867e-01, -6.4606e-01,  2.5874e-01]]]]
    )

    assert torch.allclose(output, expected, atol=1e-4), \
        f"calculate_sliding_attention output values mismatch"


@torch.no_grad()
def test_sw_attention() -> None:
    """Test SWAttention module."""
    torch.manual_seed(42)
    hidden_dim = 16
    num_heads = 4
    head_dim = 4
    window_size = 2
    
    swa = SWAttention(hidden_dim, num_heads, head_dim, window_size)
    
    batch_size = 2
    seq_len = 8
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    output = swa(x)
    
    assert output.shape == (batch_size, seq_len, hidden_dim), \
        f"Output shape mismatch: {output.shape} vs {(batch_size, seq_len, hidden_dim)}"


if __name__ == "__main__":
    test_calculate_attention()
    test_grouped_query_attention()
    test_calculate_sliding_attention()
    test_sw_attention()
