class OpenELMMultiHeadCausalAttention(nn.Module):
    def __init__(self, config: OpenELMConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        head_dim = config.head_dim
        q_heads = config.num_query_heads[layer_idx]
        k_heads = config.num_kv_heads[layer_idx]
        v_heads = config.num_kv_heads[layer_idx]

        self.qkv_proj = nn.Linear(
            in_features=config.model_dim,
            out_features=(q_heads + k_heads + v_heads) * head_dim,
            bias=False,
        )

        self.pos_embedding = OpenELMRotaryEmbedding(
            model_dim=config.head_dim,
            max_seq_length=config.rope_max_length,
            freq_constant=config.rope_freq_constant,
        )

        if config.normalize_qk_projections:
            self.q_norm = OpenELMRMSNorm(
                num_features=config.head_dim,
            )
            self.k_norm = OpenELMRMSNorm(
                num_features=config.head_dim,
            )
        else:
            self.q_norm = None
            self.k_norm = None

        self.out_proj = nn.Linear(
            in_features=q_heads * head_dim,
            out_features=config.model_dim,
            bias=False,
        )

        self.head_dim = config.head_dim
        self.num_q_heads = q_heads
        self.num_k_heads = k_heads
        self.num_v_heads = v_heads
        self.transformer_dim = config.model_dim
        self.num_groups = self.num_q_heads // self.num_k_heads

    def extra_repr(self) -> str:
        return (
            super().extra_repr()
            + f"query_heads={self.num_q_heads}, key_heads={self.num_k_heads}, value_heads={self.num_v_heads}"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass of multi-head self-attention.

        Args:
            hidden_states: Input tensor of the shape [batch size, sequence length, model dimension].
            past_key_value: Tensor storing the cached keys and values.
            output_attentions: output attention weights.
            use_cache: Specifies whether to use kv-cache for generation.
            cache_position: used for updating the kv-cache.

        Returns:
            The output of the same shape as the input, optionally with a tensor containing cached keys and values.
        """

        # scaled_dot_product_attention does not return attention weights, set output_attentions to False
        output_attentions = False
        batch_size, seq_length, d_model = hidden_states.size()

        # [B, S, d] --> [B, S, (q_h + k_h + v_h) * h]
        qkv = self.qkv_proj(hidden_states)
        # [B, S, (q_h + k_h + v_h) * h] --> [B, S, (q_h + k_h + v_h), h]
        qkv = qkv.reshape(
            batch_size,
            seq_length,
            self.num_q_heads + self.num_k_heads + self.num_v_heads,
            self.head_dim,
        )
        # [B, S, (q_h + k_h + v_h), h] --> [B, (q_h + k_h + v_h), S, h]
        qkv = qkv.transpose(1, 2)
        # [B, (q_h + k_h + v_h), S, h] --> [B, q_h, S h], [B, k_h, S, h], [B, v_h, S, h]
        queries, keys, values = qkv.split(
            [self.num_q_heads, self.num_k_heads, self.num_v_heads], dim=1
        )

        if self.q_norm is not None:
            queries = self.q_norm(queries)

        if self.k_norm is not None:
            keys = self.k_norm(keys)

        past_key_value = getattr(self, "past_key_value", past_key_value)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; position_ids needed for the static cache
            # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            cache_kwargs = {"cache_position": cache_position}
            keys, values = past_key_value.update(
                keys, values, self.layer_idx, cache_kwargs
            )

        # Add positional embedding
        queries, keys = self.pos_embedding(queries, keys)

        if self.num_groups != 1:
            # GQA
            # [B, k_h, S, h] --> [B, q_h, S, h]
            keys = keys.repeat_interleave(self.num_groups, dim=1)
            # [B, v_h, S, h] --> [B, q_h, S, h]
            values = values.repeat_interleave(self.num_groups, dim=1)

        causal_mask = attention_mask
        if attention_mask is not None and cache_position is not None:
            causal_mask = causal_mask[:, :, cache_position, : keys.shape[-2]]

        attn_output = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=causal_mask,
            dropout_p=0,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(
            batch_size, seq_length, self.num_q_heads * self.head_dim
        )
        attn_output = self.out_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value