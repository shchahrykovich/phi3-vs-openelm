class OpenELMPreTrainedModel(PreTrainedModel):
    config_class = OpenELMConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OpenELMDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def __init__(self, *inputs, **kwargs) -> None:
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, OpenELMRMSNorm):
            module.weight.data.fill_(1.0)
