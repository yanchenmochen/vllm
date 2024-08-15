from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from csrc.quantization.smooth_quant.w8a8_triton_kernels import matmul_kernel_dynamic_quant, per_channel_quant, per_token_quant_int8
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.utils import set_weight_attrs


class SmoothQuantConfig(QuantizationConfig):
    """Config class for SmoothQuant.

    Reference: https://arxiv.org/abs/2312.03788
    """

    def __init__(
        self,
        weight_bits: int
    ) -> None:
        self.weight_bits = weight_bits
        # self.group_size = group_size
        # self.zero_point = zero_point

        if self.weight_bits != 8:
            raise ValueError(
                "Currently, only 8-bit weight quantization is supported for "
                f"Smooth_quant, but got {self.weight_bits} bits.")
        self.pack_factor = 1

    def __repr__(self) -> str:
        return (f"SmoothQuantConfig(weight_bits={self.weight_bits}")

    def get_name(self) -> str:
        return "smooth_quant"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half]

    def get_min_capability(self) -> int:
        # The AWQ kernel only supports Turing or newer GPUs.
        return 75

    @staticmethod
    def get_config_filenames() -> List[str]:
        # quantize model need export the quantize_config.json
        return [
            "quantize_config.json",
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SmoothQuantConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        return cls(weight_bits)

    def get_quant_method(
            self, layer: torch.nn.Module) -> Optional["SmoothQuantLinearMethod"]:
        if isinstance(layer, LinearBase):
            return SmoothQuantLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class SmoothQuantLinearMethod(LinearMethodBase):
    """Linear method for SmoothQuant.

    Args:
        quant_config: The SmoothQuant quantization config.
    """

    def __init__(self, quant_config: SmoothQuantConfig):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: List[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")

        weight = Parameter(
            torch.empty(
                output_size_per_partition // self.quant_config.pack_factor,
                input_size_per_partition,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            weight, {
                "input_dim": 1,
                "output_dim": 0,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor
            })

        scale = Parameter(
            torch.empty(
                output_size_per_partition // self.quant_config.pack_factor,
                1,
                dtype=torch.float16,
            ),
            requires_grad=False,
        )
        set_weight_attrs(scale, {
            "input_dim": 1,
            "output_dim": 0,
            "packed_dim": 1,
            "pack_factor": self.quant_config.pack_factor
        })
        
        migration_scale = Parameter(torch.empty(
            1, input_size, 
            dtype=torch.float16
            ),
            requires_grad=False)
        
        set_weight_attrs(migration_scale, {
            "ignore_warning": True
        })

        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)
        
        layer.register_parameter("scale", scale)
        set_weight_attrs(scale, extra_weight_attrs)
        
        layer.register_parameter("migration_scale", migration_scale)
        set_weight_attrs(migration_scale, extra_weight_attrs)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
       
        x.div_(layer.migration_scale)
        input_quant, input_scale = per_token_quant_int8(x, 1e-7)
        out = matmul_kernel_dynamic_quant(
            input_quant,
            layer.weight,
            input_scale,
            layer.scale,
            output_dtype=torch.float16
        )
        return out