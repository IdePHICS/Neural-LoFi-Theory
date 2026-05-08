from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn


class DNNModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: nn.Module = nn.Tanh(),
        weight_init: Callable[[torch.Tensor], None] | None = None,
        use_bias: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=use_bias))
            if i < len(dims) - 2:
                layers.append(activation)

        self.network = nn.Sequential(*layers)

        if weight_init is not None:
            self._initialize_weights(weight_init)

    def _initialize_weights(self, init_fn: Callable[[torch.Tensor], None]) -> None:
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                init_fn(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, X: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.network(X)
