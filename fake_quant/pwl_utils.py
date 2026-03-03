"""
Piecewise-Linear (PWL) activation function approximation for fake quantization.

Simulates hardware behavior where non-linear activations (SiLU, GELU, etc.) are
approximated by piecewise-linear functions with configurable HW precision.

Static coefficients are taken from the Hailo SDK (acceleras/utils/pwl_coefficients.py).
Dynamic fitting uses the pwlf library for arbitrary segment counts.
"""

import torch
import torch.nn as nn
import numpy as np
import math
import logging
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Callable

import quant_utils
import model_utils


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PWLCoefficients:
    """Piecewise-linear approximation coefficients for an activation function.

    For N segments there are N-1 thresholds, N slopes, and N offsets.
    Segment i: y = slopes[i] * x + offsets[i]
      Segment 0:     x in (-inf,           thresholds[0])
      Segment k:     x in [thresholds[k-1], thresholds[k])   for 0 < k < N-1
      Segment N-1:   x in [thresholds[N-2], +inf)
    """
    thresholds: Tuple[float, ...]
    offsets: Tuple[float, ...]
    slopes: Tuple[float, ...]


@dataclass
class HWConfig:
    """Hardware precision parameters for PWL coefficient quantization.

    Defaults match Hailo accelerator (acceleras_definitions.py):
        APU_MANTISSA_BITS = 10, APU_EXP_BITS = 4, APU_OFFSET_BITS = 13
    """
    mantissa_bits: int = 10
    exp_bits: int = 4
    offset_bits: int = 13


# ---------------------------------------------------------------------------
# Static coefficient registry (from Hailo SDK pwl_coefficients.py)
# ---------------------------------------------------------------------------

SILU_PWL = PWLCoefficients(
    thresholds=(
        -6.0, -3.89303964, -1.3473389, -0.63382168,
        -0.02343757, 0.59187625, 1.32987503, 3.89463346,
    ),
    offsets=(
        0.0, -0.18177579, -0.40690438, -0.18371548,
        -0.01869892, -0.01169604, -0.17351164, -0.40649964, 0.0,
    ),
    slopes=(
        0.0, -0.02893179, -0.08676027, 0.07889137,
        0.33924308, 0.63803194, 0.91142625, 1.08662166, 1.0,
    ),
)

GELU_PWL = PWLCoefficients(
    thresholds=(
        -3.62189, -2.2877, -0.80275, -0.37682,
        -0.0012, 0.37014, 0.79108, 2.36739,
    ),
    offsets=(
        0.0, -0.04368, -0.26457, -0.11051,
        -0.00921, -0.00886, -0.10773, -0.26145, -0.02197,
    ),
    slopes=(
        0.0, -0.01206, -0.10862, 0.0833,
        0.35214, 0.64485, 0.91198, 1.10629, 1.00514,
    ),
)

PWL_REGISTRY: Dict[str, PWLCoefficients] = {
    'silu': SILU_PWL,
    'gelu': GELU_PWL,
}


# ---------------------------------------------------------------------------
# Dynamic fitting
# ---------------------------------------------------------------------------

def fit_pwl(func: Callable, n_segments: int,
            x_min: float = -8.0, x_max: float = 8.0,
            n_samples: int = 500) -> PWLCoefficients:
    """Fit a piecewise-linear approximation to *func* with *n_segments* pieces.

    Uses the pwlf library (same as Hailo SDK) for L2-optimal breakpoint placement.

    Args:
        func: Activation function  f(x) -> y  operating on numpy arrays.
        n_segments: Number of linear segments.
        x_min, x_max: Input range for fitting.
        n_samples: Number of sample points.

    Returns:
        PWLCoefficients with n_segments-1 thresholds, n_segments slopes/offsets.
    """
    try:
        import pwlf
    except ImportError:
        raise ImportError(
            "pwlf is required for dynamic PWL fitting. Install with: pip install pwlf"
        )

    x = np.linspace(x_min, x_max, n_samples)
    y = func(x)

    pw = pwlf.PiecewiseLinFit(x, y)
    breakpoints = pw.fit(n_segments=n_segments, seed=1)
    slopes = pw.calc_slopes()
    offsets = pw.intercepts

    # breakpoints includes x_min and x_max; internal thresholds are breakpoints[1:-1]
    thresholds = tuple(breakpoints[1:-1].tolist())
    return PWLCoefficients(
        thresholds=thresholds,
        offsets=tuple(offsets.tolist()),
        slopes=tuple(slopes.tolist()),
    )


# ---------------------------------------------------------------------------
# HW precision simulation
# ---------------------------------------------------------------------------

def _quantize_slopes_mantissa_exp(slopes: torch.Tensor,
                                  mantissa_bits: int,
                                  exp_bits: int) -> torch.Tensor:
    """Simulate Hailo's mantissa × 2^exponent decomposition of PWL slopes.

    For each slope s:
        exponent  = ceil(log2(|s|) - mantissa_bits)
        mantissa  = round(s / 2^exponent)          (mantissa_bits precision)
        s_quant   = mantissa * 2^exponent

    Mirrors acceleras/atomic_ops/activation_op.py _get_mantissa_exponent_decomposition.
    """
    result = slopes.clone()
    nonzero = slopes != 0

    if not nonzero.any():
        return result

    s_nz = slopes[nonzero].float()
    exponents = torch.ceil(torch.log2(torch.abs(s_nz)) - mantissa_bits)

    # Clamp exponent range (Hailo uses a bias-based range)
    max_exp = 2 ** exp_bits - 1
    exponents = torch.clamp(exponents, min=-max_exp, max=max_exp)

    exp_factors = (2.0 ** exponents)
    mantissas = torch.round(s_nz / exp_factors)

    # Clamp mantissa overflow
    max_mantissa = 2 ** mantissa_bits - 1
    mantissas = torch.clamp(mantissas, -max_mantissa, max_mantissa)

    result[nonzero] = (mantissas * exp_factors).to(slopes.dtype)
    return result


def _quantize_offsets_fixed(offsets: torch.Tensor,
                            offset_bits: int) -> torch.Tensor:
    """Simulate fixed-point quantization of PWL offsets.

    Offsets are quantized to *offset_bits* signed fixed-point
    relative to the maximum absolute offset value.
    """
    abs_max = offsets.abs().max()
    if abs_max == 0:
        return offsets

    # Scale so that abs_max maps to 2^(offset_bits-1) - 1
    max_int = 2 ** (offset_bits - 1) - 1
    scale = abs_max / max_int
    quantized = torch.round(offsets / scale) * scale
    return quantized


# ---------------------------------------------------------------------------
# PWLActivation module
# ---------------------------------------------------------------------------

class PWLActivation(nn.Module):
    """Piecewise-linear activation function with optional HW precision simulation.

    Drop-in replacement for torch.nn.SiLU / torch.nn.GELU etc.

    Args:
        coefficients: Static PWL coefficients.
        hw_config: If provided, slopes and offsets are precision-limited to
            match hardware (mantissa/exponent for slopes, fixed-point for offsets).
        input_bits: Bit-width for input quantization (16 = pass-through).
        output_bits: Bit-width for output quantization (16 = pass-through).
        name: Human-readable label.
    """

    def __init__(self, coefficients: PWLCoefficients, *,
                 hw_config: Optional[HWConfig] = None,
                 input_bits: int = 16,
                 output_bits: int = 16,
                 name: str = 'pwl'):
        super().__init__()
        self.name = name
        self.hw_config = hw_config

        n_segments = len(coefficients.slopes)
        assert len(coefficients.thresholds) == n_segments - 1
        assert len(coefficients.offsets) == n_segments

        # Float coefficients (buffers: move with device, saved in state_dict)
        self.register_buffer('thresholds',
                             torch.tensor(coefficients.thresholds, dtype=torch.float32))
        self.register_buffer('slopes',
                             torch.tensor(coefficients.slopes, dtype=torch.float32))
        self.register_buffer('offsets',
                             torch.tensor(coefficients.offsets, dtype=torch.float32))

        # HW-quantized versions (None until simulate_hw_precision is called)
        self.register_buffer('slopes_hw', None)
        self.register_buffer('offsets_hw', None)

        if hw_config is not None:
            self.simulate_hw_precision()

        # Input / output quantizers (reuse existing ActQuantizer)
        self.input_quantizer = quant_utils.ActQuantizer()
        self.output_quantizer = quant_utils.ActQuantizer()
        if input_bits < 16:
            self.input_quantizer.configure(bits=input_bits, sym=True)
        if output_bits < 16:
            self.output_quantizer.configure(bits=output_bits, sym=True)

    def simulate_hw_precision(self):
        """Apply HW precision constraints to slopes and offsets."""
        if self.hw_config is None:
            self.slopes_hw = None
            self.offsets_hw = None
            return

        self.slopes_hw = _quantize_slopes_mantissa_exp(
            self.slopes, self.hw_config.mantissa_bits, self.hw_config.exp_bits)
        self.offsets_hw = _quantize_offsets_fixed(
            self.offsets, self.hw_config.offset_bits)

    def make_learnable(self):
        """Convert slopes/offsets to nn.Parameters for future GPTQ optimisation."""
        self.slopes = nn.Parameter(self.slopes.clone())
        self.offsets = nn.Parameter(self.offsets.clone())
        # Thresholds stay as buffers — segment boundaries are typically fixed.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype

        # --- optional input quantization (simulate accumulator precision) ---
        if self.input_quantizer.bits < 16:
            x = self.input_quantizer(x).to(x_dtype)
            self.input_quantizer.free()

        # --- piecewise-linear computation ---
        x_f = x.float()

        # Segment index for each element: searchsorted gives the insertion
        # index into thresholds, which is exactly the segment index.
        seg_idx = torch.searchsorted(self.thresholds, x_f)

        # Choose float or HW-quantized coefficients
        slopes = self.slopes_hw if self.slopes_hw is not None else self.slopes
        offsets = self.offsets_hw if self.offsets_hw is not None else self.offsets

        s = slopes[seg_idx]
        o = offsets[seg_idx]
        y = s * x_f + o

        y = y.to(x_dtype)

        # --- optional output quantization (simulate output encoding) ---
        if self.output_quantizer.bits < 16:
            y = self.output_quantizer(y).to(x_dtype)
            self.output_quantizer.free()

        return y

    def extra_repr(self) -> str:
        parts = [f'name={self.name}', f'segments={len(self.slopes)}']
        if self.hw_config is not None:
            parts.append(f'hw=M{self.hw_config.mantissa_bits}E{self.hw_config.exp_bits}O{self.hw_config.offset_bits}')
        if self.input_quantizer.bits < 16:
            parts.append(f'in_bits={self.input_quantizer.bits}')
        if self.output_quantizer.bits < 16:
            parts.append(f'out_bits={self.output_quantizer.bits}')
        return ', '.join(parts)


# ---------------------------------------------------------------------------
# Model-level replacement
# ---------------------------------------------------------------------------

def _get_coefficients(act_name: str, n_segments: int) -> PWLCoefficients:
    """Get PWL coefficients: static registry if available, otherwise fit."""
    act_key = act_name.lower()
    if act_key in PWL_REGISTRY:
        static = PWL_REGISTRY[act_key]
        if len(static.slopes) == n_segments:
            return static
        logging.info("Requested %d segments but static registry has %d for %s; fitting dynamically",
                     n_segments, len(static.slopes), act_name)

    # Dynamic fitting
    ACTIVATION_FUNCS = {
        'silu': lambda x: x / (1 + np.exp(-x)),
        'gelu': lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))),
    }
    if act_key not in ACTIVATION_FUNCS:
        raise ValueError(f"No activation function for '{act_name}'. "
                         f"Available: {list(ACTIVATION_FUNCS.keys())}")
    return fit_pwl(ACTIVATION_FUNCS[act_key], n_segments)


def replace_activation_with_pwl(model, act_name: str = 'silu', *,
                                n_segments: int = 9,
                                hw_config: Optional[HWConfig] = None,
                                input_bits: int = 16,
                                output_bits: int = 16):
    """Replace all activation functions in the model with PWL approximations.

    Args:
        model: HuggingFace causal LM (LLaMA or OPT).
        act_name: Activation to replace ('silu', 'gelu').
        n_segments: Number of PWL segments.
        hw_config: HW precision config (None = pure float PWL).
        input_bits: Bit-width for PWL input quantizer.
        output_bits: Bit-width for PWL output quantizer.

    Returns:
        List of (layer_index, attr_path) for all replaced activations.
    """
    coefficients = _get_coefficients(act_name, n_segments)

    model_type = model_utils.get_model_type(model)
    layers = model_utils.get_transformer_layers(model, model_type)

    replaced = []
    for i, layer in enumerate(layers):
        if model_type == model_utils.LLAMA_MODEL:
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'act_fn'):
                pwl = PWLActivation(coefficients, hw_config=hw_config,
                                    input_bits=input_bits, output_bits=output_bits,
                                    name=f'pwl_{act_name}')
                layer.mlp.act_fn = pwl
                replaced.append((i, 'mlp.act_fn'))
        elif model_type == model_utils.OPT_MODEL:
            if hasattr(layer, 'activation_fn'):
                pwl = PWLActivation(coefficients, hw_config=hw_config,
                                    input_bits=input_bits, output_bits=output_bits,
                                    name=f'pwl_{act_name}')
                layer.activation_fn = pwl
                replaced.append((i, 'activation_fn'))

    return replaced


def find_pwl_activations(model):
    """Find all PWLActivation modules in the model, keyed by full name."""
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, PWLActivation):
            result[name] = module
    return result


# ---------------------------------------------------------------------------
# Tag generation (shared by calibrate_act_scales.py and dart_gptq_wxaykvz.sh)
# ---------------------------------------------------------------------------

# Defaults matching argparse in both calibrate_act_scales.py and args_config_gen.py
_PWL_DEFAULTS = dict(
    n_segments=9,
    input_bits=16,
    output_bits=16,
    no_hw_sim=False,
)


def pwl_tag(n_segments=9, input_bits=16, output_bits=16, no_hw_sim=False):
    """Build a filename-safe tag string for PWL configuration.

    Returns e.g. ``"_pwl"`` with defaults, ``"_pwl_12p"`` for 12 segments,
    ``"_pwl_12p_in8_out8_nohw"`` when everything differs.
    Only non-default values are included.
    """
    parts = ["_pwl"]
    if n_segments != _PWL_DEFAULTS['n_segments']:
        parts.append(f"{n_segments}p")
    if input_bits != _PWL_DEFAULTS['input_bits']:
        parts.append(f"in{input_bits}")
    if output_bits != _PWL_DEFAULTS['output_bits']:
        parts.append(f"out{output_bits}")
    if no_hw_sim and no_hw_sim != _PWL_DEFAULTS['no_hw_sim']:
        parts.append("nohw")
    return "_".join(parts)
