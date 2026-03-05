import torch
from typing import Optional
from .attention import SWAttention, GroupedQueryAttention
from .moe import MixtureOfExperts
from .swiglu import SwiGLUFeedForward

class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ff_dim: int,
        num_heads: int,
        head_dim: int,
        use_sliding_window: bool = False,
        window_size: int = 128,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        num_kv_heads: Optional[int] = None
    ) -> None:
        """
        Args:
            hidden_dim: Hidden dimension.
            ff_dim: Feed-forward inner dimension.
            num_heads: Number of attention heads.
            head_dim: Dimension per attention head.
            use_sliding_window: Whether to use sliding window attention.
            window_size: Size of sliding window (if used).
            use_moe: Whether to use Mixture of Experts instead of single FFN.
            num_experts: Number of experts (if MoE).
            top_k: Number of experts per token (if MoE).
            num_kv_heads: Number of KV heads for GQA (None = standard MHA).
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        ### TODO: Your code starts here ###
        self.rms_norm_1 = torch.nn.RMSNorm(self.hidden_dim)
        if use_sliding_window:
            self.attention = SWAttention(
                hidden_dim, num_heads, head_dim, window_size
            )
        else:
            self.attention = GroupedQueryAttention(
                hidden_dim, num_heads, head_dim, num_kv_heads
            )
        self.rms_norm_2 = torch.nn.RMSNorm(self.hidden_dim)
        if use_moe:
            self.ffn = MixtureOfExperts(
                hidden_dim, ff_dim, num_experts, top_k
            )
        else:
            self.ffn = SwiGLUFeedForward(
                hidden_dim, ff_dim
            )
        ### TODO: Your code ends here ###

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch, seq_len, hidden_dim].

        Returns:
            Output tensor of shape [batch, seq_len, hidden_dim].
        """
        ### TODO: Your code starts here ###
        result = x + self.attention(self.rms_norm_1(x))
        result = result + self.ffn(self.rms_norm_2(result))
        ### TODO: Your code ends here ###

        assert x.shape == result.shape
        return result


class Transformer(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        hidden_dim: int,
        ff_dim: int,
        num_heads: int,
        head_dim: int,
        use_sliding_window_alternating: bool = False,
        window_size: int = 128,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        num_kv_heads: Optional[int] = None
    ) -> None:
        """
        Args:
            vocab_size: Size of the vocabulary.
            n_layers: Number of transformer layers.
            hidden_dim: Hidden dimension.
            ff_dim: Feed-forward inner dimension.
            num_heads: Number of attention heads.
            head_dim: Dimension per attention head.
            use_sliding_window_alternating: Use sliding window on every other layer
            window_size: Size of sliding window.
            use_moe: Whether to use Mixture of Experts.
            num_experts: Number of experts (if MoE).
            top_k: Number of experts per token (if MoE).
            num_kv_heads: Number of KV heads for GQA.
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        ### TODO: Your code starts here ###
        self.embedding = torch.nn.Embedding(
            vocab_size, hidden_dim
        )
        layers = []
        for i in range(n_layers):
            use_sliding_window = use_sliding_window_alternating & (i % 2 != 0)
            layers.append(TransformerBlock(
                hidden_dim,
                ff_dim,
                num_heads,
                head_dim,
                use_sliding_window,
                window_size,
                use_moe,
                num_experts,
                top_k,
                num_kv_heads
            ))
        self.layers = torch.nn.Sequential(*layers)
        self.final_norm = torch.nn.RMSNorm(hidden_dim)
        self.output_proj = torch.nn.Linear(hidden_dim, vocab_size)
        ### TODO: Your code ends here ###

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token indices of shape [batch, seq_len].

        Returns:
            Logits of shape [batch, seq_len, vocab_size].
        """
        assert len(x.shape) == 2, f"Expected 2D input, got shape {x.shape}"

        ### TODO: Your code starts here ###
        embeddings = self.embedding(x)
        trans_layers_out = self.layers(embeddings)
        logits = self.output_proj(self.final_norm(trans_layers_out))
        ### TODO: Your code ends here ###

        return logits