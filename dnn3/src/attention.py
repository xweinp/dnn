import torch
import torch.nn.functional as F
from typing import Optional
from .rope import RotaryPositionalEmbedding


def calculate_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_weights: torch.Tensor,
    rope: RotaryPositionalEmbedding,
    scale: float,
    device: torch.device,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Args:
        q: Query tensor of shape [batch, num_heads, seq_len, head_dim].
        k: Key tensor of shape [batch, num_kv_heads, seq_len, head_dim].
        v: Value tensor of shape [batch, num_kv_heads, seq_len, head_dim].
        key_weights: Per-head key weights of shape [num_heads].
        rope: Rotary positional embedding module.
        scale: Scaling factor (typically 1/sqrt(head_dim)).
        device: Device to create the causal mask on.
        mask: Optional attention mask of shape [seq_len, seq_len]. If None, uses causal mask.

    Returns:
        Output tensor of shape [batch, num_heads, seq_len, head_dim].
    """
    ### TODO: Your code starts here ###
    (batch_size, num_heads, seq_len, head_dim) = q.size()
    num_kv_heads = k.size(1)
    group_size = num_heads // num_kv_heads
    
    k = rope(k)
    q = rope(q) 

    q = q.reshape(batch_size, num_kv_heads, group_size, seq_len, head_dim)
    k = k.unsqueeze(2) # add group dim
    a = q @ k.transpose(-1, -2) * scale
    a = a * key_weights.reshape(1, num_kv_heads, group_size, 1, 1)

    if mask is None:
        infs = -torch.ones(seq_len, seq_len, device=device) * torch.inf
        mask = torch.triu(infs, diagonal=1)
    a = a + mask
    a = F.softmax(a, dim=-1)
    
    output = a @ v.unsqueeze(2)
    output = output.reshape(batch_size, num_heads, seq_len, head_dim)
    ### TODO: Your code ends here ###

    return output


class GroupedQueryAttention(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: Optional[int] = None
    ) -> None:
        """
        Args:
            hidden_dim: Input/output dimension.
            num_heads: Number of query heads.
            head_dim: Dimension of each head.
            num_kv_heads: Number of key-value heads. If None, defaults to num_heads (standard MHA).
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.scale = head_dim ** -0.5

        assert num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.num_queries_per_kv = num_heads // self.num_kv_heads

        ### TODO: Your code starts here ###
        self.rope = RotaryPositionalEmbedding(head_dim)
        self.W_Q = torch.nn.Linear(hidden_dim, head_dim*self.num_heads)
        self.W_K = torch.nn.Linear(hidden_dim, head_dim*self.num_kv_heads)
        self.W_V = torch.nn.Linear(hidden_dim, head_dim*self.num_kv_heads)
        self.W_O = torch.nn.Linear(head_dim*self.num_heads, hidden_dim)
        self.key_weights = torch.nn.Parameter(torch.randn(self.num_heads))
        ### TODO: Your code ends here ###

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch, seq_len, hidden_dim].

        Returns:
            Output tensor of shape [batch, seq_len, hidden_dim].
        """
        assert len(x.shape) == 3
        batch, seq_len, _ = x.shape

        ### TODO: Your code starts here ###
        def split_to_heads(t: torch.Tensor, n: int) -> torch.Tensor:
            t = t.reshape(batch, seq_len, n, self.head_dim)
            return t.transpose(1, 2)
        Q = split_to_heads(self.W_Q(x), self.num_heads)
        K = split_to_heads(self.W_K(x), self.num_kv_heads)
        V = split_to_heads(self.W_V(x), self.num_kv_heads)
        O = calculate_attention(
            Q, K, V, 
            self.key_weights, 
            self.rope,
            self.scale,
            x.device
        )
        O = O.transpose(1, 2).reshape(batch, seq_len, -1)
        output = self.W_O(O)
        ### TODO: Your code ends here ###

        return output


def calculate_sliding_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_weights: torch.Tensor,
    rope: RotaryPositionalEmbedding,
    scale: float,
    device: torch.device,
    window_size: int
) -> torch.Tensor:
    """
    Args:
        q: Query tensor of shape [batch, num_heads, seq_len, head_dim].
        k: Key tensor of shape [batch, num_kv_heads, seq_len, head_dim].
        v: Value tensor of shape [batch, num_kv_heads, seq_len, head_dim].
        key_weights: Per-head key weights of shape [num_heads].
        rope: Rotary positional embedding module.
        scale: Scaling factor (typically 1/sqrt(head_dim)).
        device: Device to create the causal mask on.
        window_size: Number of previous tokens each position can attend to.

    Returns:
        Output tensor of shape [batch, num_heads, seq_len, head_dim].
    """
    ### TODO: Your code starts here ###
    seq_len = q.size(2)

    infs = -torch.ones((seq_len, seq_len), device=device) * torch.inf
    mask = torch.triu(infs, diagonal=1)
    mask = mask + torch.tril(infs, diagonal=-1-window_size)
    output = calculate_attention(
        q, k, v,
        key_weights,
        rope,
        scale,
        device,
        mask
    )
    ### TODO: Your code ends here ###

    return output


class SWAttention(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        window_size: int
    ) -> None:
        """
        Args:
            hidden_dim: Input/output dimension.
            num_heads: Number of attention heads.
            head_dim: Dimension of each head.
            window_size: Size of the sliding window.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.scale = head_dim ** -0.5

        ### TODO: Your code starts here ###
        self.rope = RotaryPositionalEmbedding(head_dim)
        self.W_Q = torch.nn.Linear(hidden_dim, head_dim*num_heads)
        self.W_K = torch.nn.Linear(hidden_dim, head_dim*num_heads)
        self.W_V = torch.nn.Linear(hidden_dim, head_dim*num_heads)
        self.W_O = torch.nn.Linear(head_dim*num_heads, hidden_dim)
        self.key_weights = torch.nn.Parameter(torch.randn(num_heads))
        ### TODO: Your code ends here ###

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch, seq_len, hidden_dim].

        Returns:
            Output tensor of shape [batch, seq_len, hidden_dim].
        """
        assert len(x.shape) == 3
        batch, seq_len, _ = x.shape

        ### TODO: Your code starts here ###
        def split_to_heads(t: torch.Tensor, n: int) -> torch.Tensor:
            t = t.reshape(batch, seq_len, n, self.head_dim)
            return t.transpose(1, 2)
        Q = split_to_heads(self.W_Q(x), self.num_heads)
        K = split_to_heads(self.W_K(x), self.num_heads)
        V = split_to_heads(self.W_V(x), self.num_heads)
        O = calculate_sliding_attention(
            Q, K, V, 
            self.key_weights, 
            self.rope,
            self.scale,
            x.device,
            self.window_size
        )
        O = O.transpose(1, 2).reshape(batch, seq_len, -1)
        output = self.W_O(O)
        ### TODO: Your code ends here ###
        return output