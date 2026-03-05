import torch
import torch.nn.functional as F
from .swiglu import SwiGLUFeedForward

class Router(torch.nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 2) -> None:
        """
        Args:
            hidden_dim: Input dimension.
            num_experts: Total number of experts.
            top_k: Number of experts to activate per token.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k

        ### TODO: Your code starts here ###
        self.lin = torch.nn.Linear(hidden_dim, num_experts)
        ### TODO: Your code ends here ###

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape [batch, seq_len, hidden_dim].

        Returns:
            routing_weights: Tensor of shape [batch, seq_len, top_k] with softmax weights.
            expert_indices: Tensor of shape [batch, seq_len, top_k] with selected expert indices.
        """
        assert len(x.shape) == 3
        ### TODO: Your code starts here ###
        w = self.lin(x)
        routing_weights, expert_indices = torch.topk(
            w, 
            k=self.top_k,
            dim=-1,
            sorted=False
        )
        routing_weights = F.softmax(routing_weights, dim=-1)
        ### TODO: Your code ends here ###

        return routing_weights, expert_indices


class MixtureOfExperts(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        inner_dim: int,
        num_experts: int = 8,
        top_k: int = 2
    ) -> None:
        """
        Args:
            hidden_dim: Input/output dimension.
            inner_dim: Inner dimension of each expert.
            num_experts: Total number of experts.
            top_k: Number of experts to activate per token.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k

        ### TODO: Your code starts here ###
        self.router = Router(hidden_dim, num_experts, top_k)
        self.experts = [
            SwiGLUFeedForward(hidden_dim, inner_dim)
            for _ in range(num_experts)
        ]
        ### TODO: Your code ends here ###

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch, seq_len, hidden_dim].

        Returns:
            Output tensor of shape [batch, seq_len, hidden_dim].
        """
        assert len(x.shape) == 3
        batch, seq_len, hidden_dim = x.shape

        ### TODO: Your code starts here ###
        routing_weights, expert_indices = self.router(x)
        expert_outputs = torch.zeros(batch, seq_len, self.top_k, hidden_dim)
        for i, expert in enumerate(self.experts):
            use_expert = expert_indices == i
            batch_expert_idx, seq_expert_idx, k_expert_idx = torch.nonzero(use_expert, as_tuple=True)
            relevant_embs = torch.any(use_expert, dim=-1)
            embs = x[relevant_embs]
            expert_output = expert(embs.unsqueeze(0))
            expert_outputs[batch_expert_idx, seq_expert_idx, k_expert_idx] = expert_output
        output = torch.sum(expert_outputs * routing_weights.unsqueeze(-1), dim=-2)
        ### TODO: Your code ends here ###
        return output