import torch

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        """
        Args:
            head_dim: Dimension of each attention head (must be even).
            max_seq_len: Maximum sequence length to precompute embeddings for.
            base: Base for computing rotation frequencies.

        WARNING: YOUR IMPLEMENTATION MUST PRECOMPUTE THE EMBEDDINGS
        """
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"

        ### TODO: Your code starts here ###
        self.head_dim = head_dim
        self.base = base
        ### TODO: Your code ends here ###

        self._precompute_cache(max_seq_len)

    def _precompute_cache(self, seq_len: int) -> None:
        ### TODO: Your code starts here ###
        exps = -torch.arange(0., self.head_dim, 2) / self.head_dim
        thetas = self.base ** exps.unsqueeze(0)
        ms = torch.arange(0., seq_len, 1).unsqueeze(1)
        angles = ms @ thetas

        cos = torch.cos(angles).reshape(1, 1, seq_len, self.head_dim // 2)
        sin = torch.sin(angles).reshape(1, 1, seq_len, self.head_dim // 2)
        #(1 (batch size), 1 (num heads), max_seq_len, head_dim // 2)

        # I register buffer so that it model.to() moves the cache
        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)
        
        ### TODO: Your code ends here ###


    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch, num_heads, seq_len, head_dim].
            start_pos: Starting position index (for KV-cache during inference).

        Returns:
            Tensor with rotary embedding applied, same shape as input.
        """

        ### TODO: Your code starts here ###
        (_, _, seq_len, head_dim) = x.shape
        assert head_dim == self.head_dim

        x1, x2 = x.chunk(chunks=2, dim=-1)
        
        cos, sin = self.cos_cache, self.sin_cache
        sin = sin[:, :, start_pos:start_pos+seq_len]
        cos = cos[:, :, start_pos:start_pos+seq_len]

        x1_r = cos * x1 - sin * x2
        x2_r = sin * x1 + cos * x2

        return torch.cat([x1_r, x2_r], dim=-1)
        ### TODO: Your code ends here ###