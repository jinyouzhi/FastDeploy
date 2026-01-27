# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
MiniCPM4 model implementation for FastDeploy.

This module implements the MiniCPM4.1-8B model architecture from OpenBMB:
https://huggingface.co/openbmb/MiniCPM4.1-8B

Key features:
- LongRoPE position encoding for extended context length
- Scale embedding (scale_emb) and scale depth (scale_depth) for training stability
- Grouped Query Attention (GQA) with configurable num_key_value_heads
- SiLU activation in MLP
"""

from __future__ import annotations

import math
import re
from functools import partial

import paddle
from paddle import nn
from paddleformers.transformers import PretrainedModel
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig, ModelConfig
from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.graph_optimization.decorator import (
    support_graph_optimization,
)
from fastdeploy.model_executor.layers.activation import SiluAndMul
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.models.model_base import (
    ModelCategory,
    ModelForCasualLM,
    ModelRegistry,
)
from fastdeploy.model_executor.utils import (
    WeightsMapper,
    default_weight_loader,
    process_weights_after_loading,
    process_weights_before_loading,
)


class MiniCPMMLP(nn.Layer):
    """
    MiniCPM MLP module with SiLU activation.
    """

    def __init__(
        self,
        fd_config: FDConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.up_gate_proj = MergedColumnParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.up_gate_proj",
            input_size=fd_config.model_config.hidden_size,
            output_size=fd_config.model_config.intermediate_size * 2,
            with_bias=False,
            activation=fd_config.model_config.hidden_act,
        )

        self.down_proj = RowParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.down_proj",
            input_size=fd_config.model_config.intermediate_size,
            output_size=fd_config.model_config.hidden_size,
            with_bias=False,
        )

        self.act_fn = SiluAndMul(
            fd_config=fd_config,
            bias=getattr(self.up_gate_proj, "bias", None),
            act_method=fd_config.model_config.hidden_act,
        )

    def load_state_dict(self, state_dict):
        """Load model parameters from a given state dictionary."""
        self.up_gate_proj.load_state_dict(state_dict)
        self.down_proj.load_state_dict(state_dict)

    def forward(self, x, forward_meta):
        """Forward pass of the MLP."""
        gate_up_out = self.up_gate_proj(x)
        act_out = self.act_fn(gate_up_out)
        down_out = self.down_proj(act_out)
        return down_out


class MiniCPMAttention(nn.Layer):
    """
    MiniCPM Attention module with Grouped Query Attention (GQA).
    """

    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = "") -> None:
        super().__init__()

        # MiniCPM uses attention_bias=False by default
        attention_bias = getattr(fd_config.model_config, "attention_bias", False)

        self.qkv_proj = QKVParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.qkv_proj",
            with_bias=attention_bias,
        )

        self.o_proj = RowParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.o_proj",
            input_size=fd_config.model_config.hidden_size,
            output_size=fd_config.model_config.hidden_size,
            with_bias=attention_bias,
        )

        self.attn = Attention(
            fd_config=fd_config,
            layer_id=layer_id,
            prefix=prefix,
            use_neox_rotary_style=True,
        )

    def load_state_dict(self, state_dict):
        """Load model parameters from a given state dictionary."""
        self.qkv_proj.load_state_dict(state_dict)
        self.o_proj.load_state_dict(state_dict)
        self.attn.load_state_dict(state_dict)

    def forward(
        self,
        forward_meta: ForwardMeta,
        hidden_states: paddle.Tensor,
    ):
        """Forward pass of the attention module."""
        qkv_out = self.qkv_proj(hidden_states)

        atten_out = self.attn(
            qkv=qkv_out,
            forward_meta=forward_meta,
        )
        output = self.o_proj(atten_out)
        return output


class MiniCPMDecoderLayer(nn.Layer):
    """
    MiniCPM Decoder Layer with residual scaling.

    MiniCPM uses a special residual scaling factor:
        output = residual + hidden_states * (scale_depth / sqrt(num_hidden_layers))

    This helps stabilize training for deep networks.
    """

    def __init__(
        self,
        fd_config: FDConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        layer_id = int(prefix.split(sep=".")[-1])

        self.self_attn = MiniCPMAttention(
            fd_config=fd_config,
            layer_id=layer_id,
            prefix=f"{prefix}.self_attn",
        )

        self.mlp = MiniCPMMLP(
            fd_config=fd_config,
            prefix=f"{prefix}.mlp",
        )

        self.input_layernorm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{prefix}.input_layernorm",
        )

        self.post_attention_layernorm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{prefix}.post_attention_layernorm",
            layer_id=layer_id,
        )

        # MiniCPM specific scaling parameters
        self.scale_depth = getattr(fd_config.model_config, "scale_depth", 1.0)
        self.num_hidden_layers = fd_config.model_config.num_hidden_layers
        self.residual_scale = self.scale_depth / math.sqrt(self.num_hidden_layers)

    def load_state_dict(self, state_dict):
        """Load model parameters from a given state dictionary."""
        self.self_attn.load_state_dict(state_dict)
        self.mlp.load_state_dict(state_dict)
        self.input_layernorm.load_state_dict(state_dict)
        self.post_attention_layernorm.load_state_dict(state_dict)

    def forward(
        self,
        forward_meta: ForwardMeta,
        hidden_states: paddle.Tensor,
        residual: paddle.Tensor = None,
    ):
        """
        Forward pass of the decoder layer with residual scaling.

        Args:
            forward_meta: Forward metadata
            hidden_states: Input hidden states
            residual: Residual tensor from previous layer

        Returns:
            Tuple of (hidden_states, residual)
        """
        # Self Attention with residual scaling
        hidden_states, residual = self.input_layernorm(
            hidden_states, residual_input=residual, forward_meta=forward_meta
        )

        attn_output = self.self_attn(
            hidden_states=hidden_states,
            forward_meta=forward_meta,
        )

        # Apply residual with scaling
        hidden_states = attn_output * self.residual_scale

        # MLP with residual scaling
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        mlp_output = self.mlp(hidden_states, forward_meta)

        # Apply residual with scaling
        hidden_states = mlp_output * self.residual_scale

        return hidden_states, residual


@support_graph_optimization
class MiniCPMModel(nn.Layer):
    """
    MiniCPM Model backbone.

    This implements the core transformer architecture with:
    - Scaled embeddings (scale_emb)
    - LongRoPE position encoding
    - Residual scaling (scale_depth)
    """

    def __init__(
        self,
        fd_config: FDConfig = None,
    ):
        """
        Initialize the MiniCPM Model.

        Args:
            fd_config: FastDeploy configuration
        """
        super().__init__()

        self.num_layers = fd_config.model_config.num_hidden_layers
        fd_config.model_config.pretrained_config.prefix_name = "model"

        # Embedding scale factor for MiniCPM
        self.scale_emb = getattr(fd_config.model_config, "scale_emb", 1.0)

        self.embed_tokens = VocabParallelEmbedding(
            fd_config=fd_config,
            num_embeddings=fd_config.model_config.vocab_size,
            embedding_dim=fd_config.model_config.hidden_size,
            params_dtype=paddle.get_default_dtype,
            prefix=f"{fd_config.model_config.pretrained_config.prefix_name}.embed_tokens",
        )

        self.layers = nn.LayerList(
            [
                MiniCPMDecoderLayer(
                    fd_config=fd_config,
                    prefix=f"{fd_config.model_config.pretrained_config.prefix_name}.layers.{i}",
                )
                for i in range(self.num_layers)
            ]
        )

        self.norm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{fd_config.model_config.pretrained_config.prefix_name}.norm",
        )

    def load_state_dict(self, state_dict):
        """
        Load model parameters from a given state dictionary.

        Args:
            state_dict: Dictionary containing model parameters
        """
        self.embed_tokens.load_state_dict(state_dict)
        self.norm.load_state_dict(state_dict)
        for i in range(self.num_layers):
            logger.info(f"Start load layer {i}")
            self.layers[i].load_state_dict(state_dict)

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):
        """
        Forward pass of the MiniCPM model.

        Args:
            ids_remove_padding: Input token IDs with padding removed
            forward_meta: Forward metadata

        Returns:
            Output hidden states
        """
        hidden_states = self.embed_tokens(
            ids_remove_padding=ids_remove_padding, forward_meta=forward_meta
        )

        # Apply embedding scaling
        hidden_states = hidden_states * self.scale_emb

        residual = None

        for i in range(self.num_layers):
            hidden_states, residual = self.layers[i](forward_meta, hidden_states, residual)

        out = self.norm(hidden_states, residual)[0]

        return out


@ModelRegistry.register_model_class(
    architecture="MiniCPMForCausalLM",
    module_name="minicpm",
    category=[ModelCategory.TEXT_GENERATION],
    primary_use=ModelCategory.TEXT_GENERATION,
)
class MiniCPMForCausalLM(ModelForCasualLM):
    """
    MiniCPM4 For Causal Language Modeling.

    This model implements the MiniCPM4.1-8B architecture from OpenBMB.
    Key features:
    - 8B parameters with GQA (2 KV heads)
    - LongRoPE for 65536 context length
    - Scale embedding and scale depth for training stability
    """

    def __init__(self, fd_config: FDConfig):
        """
        Initialize MiniCPMForCausalLM.

        Args:
            fd_config: FastDeploy configuration
        """
        super(MiniCPMForCausalLM, self).__init__(fd_config)

        self.fd_config = fd_config
        self.model = MiniCPMModel(fd_config=fd_config)

        self.ori_vocab_size = fd_config.model_config.ori_vocab_size
        self.tie_word_embeddings = fd_config.model_config.tie_word_embeddings

        # Get dim_model_base for lm_head scaling (default to hidden_size for older models)
        self.dim_model_base = getattr(fd_config.model_config, "dim_model_base", 256)
        self.hidden_size = fd_config.model_config.hidden_size

        self.lm_head = ParallelLMHead(
            fd_config=fd_config,
            embedding_dim=fd_config.model_config.hidden_size,
            num_embeddings=fd_config.model_config.vocab_size,
            prefix="lm_head",
        )

        self.process_weights_before_loading_fn = process_weights_before_loading(
            mapper=(
                WeightsMapper(orig_to_new_prefix={"model.": "model."})
                if self.fd_config.model_config.model_format == "torch"
                else None
            ),
        )

    @paddle.no_grad()
    def load_weights(self, weights_iterator) -> None:
        """
        Load model parameters from a given weights_iterator object.

        Args:
            weights_iterator: Iterator yielding (name, weight) pairs
        """
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("up_gate_proj", "gate_proj", "gate"),
            ("up_gate_proj", "up_proj", "up"),
            ("embed_tokens.embeddings", "embed_tokens", None),
            ("lm_head.linear", "lm_head", None),
        ]

        params_dict = dict(self.named_parameters())
        process_weights_after_loading_fn = process_weights_after_loading(
            dict(self.named_sublayers()), self.fd_config
        )

        for loaded_weight_name, loaded_weight in weights_iterator:
            loaded_weight_name = (
                self.process_weights_before_loading_fn(loaded_weight_name)
                if getattr(self, "process_weights_before_loading_fn", None)
                else loaded_weight_name
            )
            if loaded_weight_name is None:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in loaded_weight_name:
                    continue
                model_param_name = loaded_weight_name.replace(weight_name, param_name)
                if model_param_name not in params_dict:
                    continue
                param = params_dict[model_param_name]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader(self.fd_config)
                )
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                model_param_name = loaded_weight_name
                if model_param_name not in params_dict:
                    continue
                param = params_dict[model_param_name]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader(self.fd_config)
                )
                weight_loader(param, loaded_weight)

            model_sublayer_name = re.sub(r"\.(weight)$", "", model_param_name)
            process_weights_after_loading_fn(model_sublayer_name, param)

        if getattr(self, "tie_word_embeddings", False):
            self.lm_head.linear.weight.set_value(
                self.model.embed_tokens.embeddings.weight.transpose([1, 0]).astype(
                    self.lm_head.linear.weight.dtype
                )
            )

    @classmethod
    def name(cls):
        """Return the model name."""
        return "MiniCPMForCausalLM"

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        """
        Load model parameters from a given state dictionary.

        Args:
            state_dict: Dictionary containing model parameters
        """
        self.model.load_state_dict(state_dict)
        self.lm_head.load_state_dict(state_dict)

    def compute_logits(self, hidden_states: paddle.Tensor):
        """
        Compute logits from hidden states.

        MiniCPM uses a special scaling for the lm_head:
            logits = lm_head(hidden_states / (hidden_size / dim_model_base))

        Args:
            hidden_states: Hidden states from the model

        Returns:
            Logits tensor
        """
        # Apply MiniCPM-specific scaling before lm_head
        scale = self.hidden_size / self.dim_model_base
        scaled_hidden_states = hidden_states / scale

        logits = self.lm_head(scaled_hidden_states)
        logits = logits.astype(paddle.float32)
        logits[:, self.ori_vocab_size:] = -float("inf")

        return logits

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):
        """
        Forward pass of the model.

        Args:
            ids_remove_padding: Input token IDs with padding removed
            forward_meta: Forward metadata

        Returns:
            Hidden states from the model
        """
        hidden_states = self.model(
            ids_remove_padding=ids_remove_padding, forward_meta=forward_meta
        )

        return hidden_states

    def clear_grpah_opt_backend(self):
        """Clear graph optimization backend, the captured cuda graph will be cleaned."""
        self.model.clear_grpah_opt_backend(fd_config=self.fd_config)


class MiniCPMPretrainedModel(PretrainedModel):
    """
    MiniCPM Pretrained Model base class for weight loading.
    """

    config_class = FDConfig

    def _init_weight(self, layer):
        """Initialize weights."""
        return None

    @classmethod
    def arch_name(cls):
        """Return architecture name."""
        return "MiniCPMForCausalLM"

    @classmethod
    def _get_tensor_parallel_mappings(cls, config: ModelConfig, is_split=True):
        """
        Get tensor parallel mappings for weight splitting/merging.

        Args:
            config: Model configuration
            is_split: Whether to split or merge

        Returns:
            Dictionary of tensor parallel mappings
        """
        from paddleformers.transformers.conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_model_parallel_size=config.tensor_model_parallel_size,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        def get_tensor_parallel_split_mappings(num_layers):
            final_actions = {}

            base_actions = {
                "lm_head.weight": partial(fn, is_column=True),
                # Row Linear
                "embed_tokens.weight": partial(fn, is_column=False),
                "layers.0.self_attn.o_proj.weight": partial(fn, is_column=False),
                "layers.0.mlp.down_proj.weight": partial(fn, is_column=False),
            }

            # Column Linear
            if config.fuse_attention_qkv:
                base_actions["layers.0.self_attn.qkv_proj.weight"] = partial(
                    fn, is_column=True
                )
            else:
                base_actions["layers.0.self_attn.q_proj.weight"] = partial(
                    fn, is_column=True
                )
                # if we have enough num_key_value_heads to split, then split it.
                if config.num_key_value_heads % config.tensor_model_parallel_size == 0:
                    base_actions["layers.0.self_attn.k_proj.weight"] = partial(
                        fn, is_column=True
                    )
                    base_actions["layers.0.self_attn.v_proj.weight"] = partial(
                        fn, is_column=True
                    )

            base_actions["layers.0.mlp.gate_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.mlp.up_proj.weight"] = partial(fn, is_column=True)

            for key, action in base_actions.items():
                if "layers.0." in key:
                    for i in range(num_layers):
                        final_actions[key.replace("layers.0.", f"layers.{i}.")] = action
                final_actions[key] = action

            return final_actions

        mappings = get_tensor_parallel_split_mappings(config.num_hidden_layers)

        return mappings
