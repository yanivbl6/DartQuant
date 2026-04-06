"""R4 diagnostic stats collection for down_proj layers.

Instruments ActQuantWrapper.forward() to capture intermediate activations
at four diagnostic points (A/B/C/D) and computes distribution stats and
cross-point quality metrics (SQNR, MSE).

Usage:
    --r4_stats <path>           Save JSON report to this path
    --r4_stats_batches <N>      Limit to first N eval batches (0=all)
"""

import json
import logging
import math

import torch

import quant_utils


# ---------------------------------------------------------------------------
# Incremental accumulators
# ---------------------------------------------------------------------------

class PointStatsAccumulator:
    """Incremental distribution stats using Welford's online algorithm."""

    def __init__(self, name: str):
        self.name = name
        self.n_elements = 0
        self.running_min = float('inf')
        self.running_max = float('-inf')
        self.abs_sum = 0.0
        # Welford's accumulators
        self._mean = 0.0
        self._M2 = 0.0       # sum of (x_i - mean)^2
        self._M4 = 0.0       # sum of (x_i - mean)^4

    def update(self, x: torch.Tensor):
        """Update with a batch tensor."""
        flat = x.detach().float().reshape(-1)
        n_new = flat.numel()
        if n_new == 0:
            return

        batch_min = flat.min().item()
        batch_max = flat.max().item()
        batch_mean = flat.mean().item()
        batch_abs_sum = flat.abs().sum().item()

        self.running_min = min(self.running_min, batch_min)
        self.running_max = max(self.running_max, batch_max)
        self.abs_sum += batch_abs_sum

        # Batch variance and fourth central moment
        delta_from_batch_mean = flat - batch_mean
        batch_var = delta_from_batch_mean.pow(2).mean().item()
        batch_m4 = delta_from_batch_mean.pow(4).mean().item()

        # Merge into running stats (parallel Welford)
        n_old = self.n_elements
        n_total = n_old + n_new
        delta = batch_mean - self._mean

        new_mean = self._mean + delta * n_new / n_total

        # M2 merge
        self._M2 += batch_var * n_new + delta ** 2 * n_old * n_new / n_total

        # M4 merge (approximate — exact parallel fourth moment is complex,
        # but this is good enough for kurtosis estimation)
        self._M4 += batch_m4 * n_new + delta ** 4 * n_old * n_new * (n_old ** 2 - n_old * n_new + n_new ** 2) / (n_total ** 3)

        self._mean = new_mean
        self.n_elements = n_total

    def finalize(self) -> dict:
        """Return final statistics."""
        if self.n_elements == 0:
            return {'min': 0, 'max': 0, 'mean': 0, 'std': 0,
                    'kurtosis': 0, 'dynamic_range': 0}

        variance = self._M2 / self.n_elements
        std = math.sqrt(max(variance, 0))
        mean_abs = self.abs_sum / self.n_elements

        # Excess kurtosis: E[(x-mu)^4] / sigma^4 - 3
        if variance > 1e-30:
            kurtosis = (self._M4 / self.n_elements) / (variance ** 2) - 3.0
        else:
            kurtosis = 0.0

        # Dynamic range: max_abs / mean_abs
        max_abs = max(abs(self.running_min), abs(self.running_max))
        dynamic_range = max_abs / mean_abs if mean_abs > 1e-30 else 0.0

        return {
            'min': self.running_min,
            'max': self.running_max,
            'mean': self._mean,
            'std': std,
            'kurtosis': kurtosis,
            'dynamic_range': dynamic_range,
        }


class CrossPointMetrics:
    """Accumulates SQNR and MSE between signal and its quantized version."""

    def __init__(self, name: str):
        self.name = name
        self.signal_power_sum = 0.0
        self.noise_power_sum = 0.0
        self.n_elements = 0

    def update(self, signal: torch.Tensor, quantized: torch.Tensor):
        """Update with a batch pair (signal, quantized version)."""
        sig = signal.detach().float()
        qnt = quantized.detach().float()
        diff = sig - qnt
        self.signal_power_sum += sig.pow(2).sum().item()
        self.noise_power_sum += diff.pow(2).sum().item()
        self.n_elements += sig.numel()

    def finalize(self) -> dict:
        if self.n_elements == 0:
            return {'sqnr_db': 0.0, 'mse': 0.0}
        sqnr = 10.0 * math.log10(self.signal_power_sum / max(self.noise_power_sum, 1e-30))
        mse = self.noise_power_sum / self.n_elements
        return {'sqnr_db': sqnr, 'mse': mse}


# ---------------------------------------------------------------------------
# Per-layer collector
# ---------------------------------------------------------------------------

class R4StatsCollector:
    """Collects R4 diagnostic stats for one down_proj layer."""

    def __init__(self, layer_name: str, max_batches: int = 0):
        self.layer_name = layer_name
        self.max_batches = max_batches
        self.batch_count = 0
        self.active = True

        self.point_a = PointStatsAccumulator('A_raw_input')
        self.point_b = PointStatsAccumulator('B_post_rotation')
        self.point_c = PointStatsAccumulator('C_post_quantization')
        self.point_d = PointStatsAccumulator('D_output')
        self.cross_bc = CrossPointMetrics('SQNR_B_C')
        self.cross_ref = CrossPointMetrics('SQNR_ref_vs_quant')

    def check_budget(self):
        """Deactivate if batch budget exceeded."""
        if self.max_batches > 0 and self.batch_count >= self.max_batches:
            self.active = False

    def finalize(self) -> dict:
        return {
            'layer': self.layer_name,
            'n_batches': self.batch_count,
            'point_A': self.point_a.finalize(),
            'point_B': self.point_b.finalize(),
            'point_C': self.point_c.finalize(),
            'point_D': self.point_d.finalize(),
            'cross_B_C': self.cross_bc.finalize(),
            'cross_ref': self.cross_ref.finalize(),
        }


# ---------------------------------------------------------------------------
# Setup and save
# ---------------------------------------------------------------------------

def setup_r4_stats(model, max_batches=0):
    """Create R4StatsCollector for each down_proj ActQuantWrapper.

    Sets wrapper._r4_collector on matching layers.
    Returns dict mapping layer_name -> collector.
    """
    collectors = {}
    for name, module in model.named_modules():
        if 'down_proj' in name and isinstance(module, quant_utils.ActQuantWrapper):
            collector = R4StatsCollector(name, max_batches=max_batches)
            module._r4_collector = collector
            collectors[name] = collector
    logging.info("r4_stats: attached collectors to %d down_proj layers", len(collectors))
    return collectors


def save_r4_stats(collectors, output_path, args):
    """Finalize all collectors and write JSON report."""
    layers = []
    sqnr_bc_vals = []
    sqnr_ref_vals = []
    kurtosis_a_vals = []
    kurtosis_b_vals = []

    for name in sorted(collectors.keys()):
        result = collectors[name].finalize()
        layers.append(result)
        sqnr_bc_vals.append(result['cross_B_C']['sqnr_db'])
        sqnr_ref_vals.append(result['cross_ref']['sqnr_db'])
        kurtosis_a_vals.append(result['point_A']['kurtosis'])
        kurtosis_b_vals.append(result['point_B']['kurtosis'])

    # Summary across layers
    n = len(layers)
    mean_sqnr_bc = sum(sqnr_bc_vals) / n if n else 0
    mean_sqnr_ref = sum(sqnr_ref_vals) / n if n else 0
    mean_kurt_a = sum(kurtosis_a_vals) / n if n else 0
    mean_kurt_b = sum(kurtosis_b_vals) / n if n else 0
    kurt_reduction = (1 - mean_kurt_b / mean_kurt_a) * 100 if abs(mean_kurt_a) > 1e-6 else 0

    report = {
        'config': {
            'model': getattr(args, 'model', ''),
            'use_r4': getattr(args, 'use_r4', False),
            'late_rot4': getattr(args, 'late_rot4', False),
            'quant_out': getattr(args, 'quant_out', 'none'),
            'a_bits': getattr(args, 'a_bits', 16),
            'eq': getattr(args, 'eq', False),
            'n_layers': n,
        },
        'layers': layers,
        'summary': {
            'mean_sqnr_bc_db': mean_sqnr_bc,
            'mean_sqnr_ref_db': mean_sqnr_ref,
            'mean_kurtosis_a': mean_kurt_a,
            'mean_kurtosis_b': mean_kurt_b,
            'kurtosis_reduction_pct': kurt_reduction,
        },
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    logging.info("r4_stats: saved report to %s (%d layers, SQNR(B,C)=%.1f dB, "
                 "SQNR(ref)=%.1f dB, kurtosis A=%.1f → B=%.1f (%.0f%% reduction))",
                 output_path, n, mean_sqnr_bc, mean_sqnr_ref,
                 mean_kurt_a, mean_kurt_b, kurt_reduction)
