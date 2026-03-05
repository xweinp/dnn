import torch
import torch.nn.functional as F

class SwiGLUFeedForward(torch.nn.Module):
    def __init__(self, hidden_dim: int, inner_dim: int) -> None:
        """
        Args:
            hidden_dim: Dimension of input and output tensors.
            inner_dim: Dimension of the intermediate (inner) representation.
        """
        super().__init__()

        ### TODO: Your code starts here ###
        self.w1v = torch.nn.Linear(hidden_dim, 2 * inner_dim)
        self.w2 = torch.nn.Linear(inner_dim, hidden_dim)
        ### TODO: Your code ends here ###

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim].

        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_dim].
        """
        assert len(x.shape) == 3, f"Expected 3D tensor, got shape {x.shape}"

        ### TODO: Your code starts here ###
        x = self.w1v(x)
        h1, hv = torch.chunk(x, chunks=2, dim=-1)
        h2 = F.silu(h1) * hv
        result = self.w2(h2)
        ### TODO: Your code ends here ###

        return result