
class OpenELMFeedForwardNetwork(nn.Module):
    def __init__(self, config: OpenELMConfig, layer_idx: int) -> None:
        super().__init__()
        ffn_multiplier = config.ffn_multipliers[layer_idx]
        intermediate_dim = int(
            make_divisible(
                ffn_multiplier * config.model_dim,
                divisor=config.ffn_dim_divisor,
            )
        )
        if config.ffn_with_glu:
            # FFN with Gated linear unit, as described in https://arxiv.org/abs/2002.05202v1.
            self.proj_1 = nn.Linear(
                in_features=config.model_dim,
                out_features=2 * intermediate_dim,
                bias=False,
            )
            self.proj_2 = nn.Linear(
                in_features=intermediate_dim,
                out_features=config.model_dim,
                bias=False,
            )
            self.ffn_with_glu = True
        else:
            # Standard FFN, as described in https://arxiv.org/abs/1706.03762
            self.proj_1 = nn.Linear(
                in_features=config.model_dim,
                out_features=intermediate_dim,
                bias=False,
            )
            self.proj_2 = nn.Linear(
                in_features=intermediate_dim,
                out_features=config.model_dim,
                bias=False,
            )
            self.ffn_with_glu = False

        self.act = ACT2FN[config.activation_fn_name]

    def extra_repr(self) -> str:
        return super().extra_repr() + f"(ffn_with_glu) : {self.ffn_with_glu}"

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of FFN layer.

        Args:
            x: Input tensor of the shape [batch size, sequence length, model dimension].

        Returns:
            A tensor of the same shape as the input.
        """
        if self.ffn_with_glu:
            y_12 = self.proj_1(x)
            y_1, y_2 = y_12.chunk(2, dim=-1)
            y = self.act(y_1) * y_2
            return self.proj_2(y)
        else:
            return self.proj_2(self.act(self.proj_1(x)))