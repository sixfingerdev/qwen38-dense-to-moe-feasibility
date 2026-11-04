"""SixFinger: training-free SwiGLU -> expert partition research prototype.

What is exact:
    With all experts enabled, the partitioned model reconstructs the dense
    FFN, apart from floating-point accumulation order.

What is NOT guaranteed:
    Reducing active experts preserves accuracy, reduces actual GPU VRAM, or
    improves wall-clock decode latency. Those require full-model benchmarks,
    a selective expert-serving runtime, and measurements.

Use: python sixfinger_moe_prototype.py --demo

This NumPy reference implementation is intentionally transparent and runnable
without GPU/cloud credentials; it is NOT an optimized production GPU kernel.
"""

from __future__ import annotations

import argparse
import dataclasses
from typing import Literal

import numpy as np


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def dense_ffn(x: np.ndarray, gate: np.ndarray, up: np.ndarray, down: np.ndarray) -> np.ndarray:
    """SwiGLU: x [..., H], gate/up [M, H], down [H, M]."""
    return silu(x @ gate.T) * (x @ up.T) @ down.T


@dataclasses.dataclass
class ExpertPartition:
    gate: np.ndarray
    up: np.ndarray
    down: np.ndarray
    shared: np.ndarray
    routed: tuple[np.ndarray, ...]
    representative: tuple[np.ndarray, ...]
    mu: np.ndarray
    compensation: np.ndarray
    down_norm: np.ndarray

    @classmethod
    def from_dense(
        cls,
        gate: np.ndarray,
        up: np.ndarray,
        down: np.ndarray,
        calibration: np.ndarray,
        *,
        experts: int = 8,
        shared_fraction: float = 0.20,
        representatives_per_expert: int = 8,
    ) -> "ExpertPartition":
        """Partition existing neuron triplets, no training or modified weights.

        Calibration supplies only fixed activation statistics and indices;
        no gradients, optimization, or synthetic weight generation are used.
        """
        if gate.ndim != 2 or up.shape != gate.shape:
            raise ValueError("gate/up must both have shape [M,H]")
        m, h = gate.shape
        if down.shape != (h, m):
            raise ValueError("down must have shape [H,M]")
        if calibration.ndim != 2 or calibration.shape[1] != h or calibration.shape[0] == 0:
            raise ValueError("calibration must have shape [N,H], N>0")
        if not (1 <= experts <= m) or not (0 <= shared_fraction < 1):
            raise ValueError("invalid number of experts or shared fraction")
        z_cal = silu(calibration @ gate.T) * (calibration @ up.T)
        mu = z_cal.mean(axis=0)
        down_norm = np.linalg.norm(down, axis=0)
        utility = np.mean(np.abs(z_cal - mu), axis=0) * down_norm
        shared_n = min(m - experts, round(m * shared_fraction))
        priority = np.argsort(-utility, kind="stable")
        shared = np.sort(priority[:shared_n])
        rest = priority[shared_n:]
        # Round-robin distributes high-utility neurons across expert groups;
        # contiguous grouping is possible, but not inherently better.
        routed = tuple(np.sort(rest[i::experts]) for i in range(experts))
        representative = tuple(
            np.sort(group[np.argsort(-utility[group])[:min(representatives_per_expert, len(group))]])
            for group in routed
        )
        if representatives_per_expert < 1:
            raise ValueError("router needs at least one representative per expert")
        all_ids = np.concatenate((shared, *routed))
        if len(all_ids) != m or not np.array_equal(np.sort(all_ids), np.arange(m)):
            raise AssertionError("expert partition must cover each original neuron once")
        compensation = down @ mu
        return cls(gate, up, down, shared, routed, representative, mu, compensation, down_norm)

    def route(self, x: np.ndarray, k: int) -> np.ndarray:
        """Cheap input-dependent expert scores from frozen neuron probes.

        This is a heuristic. It is NOT an oracle and is not quality-certified.
        """
        if x.ndim != 1 or x.shape[0] != self.gate.shape[1]:
            raise ValueError("route expects a single hidden state [H]")
        if not 0 <= k <= len(self.routed):
            raise ValueError("k must be between 0 and the number of routed experts")
        scores = np.empty(len(self.routed), dtype=np.float64)
        for j, reps in enumerate(self.representative):
            z = silu(self.gate[reps] @ x) * (self.up[reps] @ x)
            deviation = np.abs(z - self.mu[reps]) * self.down_norm[reps]
            scores[j] = float(deviation.mean())
        return np.argsort(-scores, kind="stable")[:k]

    def forward_one(
        self,
        x: np.ndarray,
        *,
        routed_k: int,
        omitted: Literal["mean", "zero"] = "mean",
    ) -> tuple[np.ndarray, dict]:
        """Compute shared + top-k experts; optionally replace skipped with mean.

        All expert weights are NumPy-resident here. True GPU-VRAM reduction
        needs a runtime that loads only required experts into VRAM.
        """
        chosen_experts = self.route(x, routed_k)
        indices = np.concatenate([self.shared, *(self.routed[j] for j in chosen_experts)])
        if omitted == "mean":
            y = self.compensation.copy()
        elif omitted == "zero":
            y = np.zeros(self.down.shape[0], dtype=self.down.dtype)
        else:
            raise ValueError("omitted must be 'mean' or 'zero'")
        if len(indices):
            z = silu(self.gate[indices] @ x) * (self.up[indices] @ x)
            if omitted == "mean":
                z = z - self.mu[indices]
            y += self.down[:, indices] @ z
        m = self.gate.shape[0]
        info = {
            "selected_experts": chosen_experts.tolist(),
            "active_neurons": len(indices),
            "active_fraction": len(indices) / m,
            "active_ffn_weight_bytes": int(len(indices) * 3 * self.gate.shape[1] * self.gate.dtype.itemsize),
            "all_ffn_weight_bytes": int(m * 3 * self.gate.shape[1] * self.gate.dtype.itemsize),
        }
        return y, info


def demo() -> None:
    rng = np.random.default_rng(20261104)
    h, m = 64, 256
    # Scale weights to keep activations numerically reasonable.
    gate = rng.normal(0, h ** -0.5, (m, h)).astype(np.float32)
    up = rng.normal(0, h ** -0.5, (m, h)).astype(np.float32)
    down = rng.normal(0, m ** -0.5, (h, m)).astype(np.float32)
    calibration = rng.normal(0, 1, (48, h)).astype(np.float32)
    heldout = rng.normal(0, 1, (24, h)).astype(np.float32)
    model = ExpertPartition.from_dense(gate, up, down, calibration, experts=8)

    print("Frozen SwiGLU expert partition (NumPy reference, NOT Qwen benchmark)")
    for k in [8, 6, 4, 2, 0]:
        errors = []
        active = []
        for x in heldout:
            dense = dense_ffn(x, gate, up, down)
            out, info = model.forward_one(x, routed_k=k)
            errors.append(np.linalg.norm(out - dense) / max(np.linalg.norm(dense), 1e-8))
            active.append(info["active_fraction"])
        print(f"k={k}/8 mean_active={np.mean(active):.2%} mean_FFN_error={np.mean(errors):.3%} max_error={np.max(errors):.3%}")
        if k == 8 and np.max(errors) > 2e-5:
            raise AssertionError("all-expert partition must reconstruct dense FFN exactly")
    print("Exact dense reconstruction with all experts: VERIFIED")
    print("No claim of near-zero loss, actual VRAM savings, or real latency improvement at k<8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="run self-contained math verification")
    args = parser.parse_args()
    if args.demo:
        demo()
    else:
        parser.print_help()
