"""Computational cost: parameters, MACs, latency and peak memory.

Parameter count alone is an insufficient measure of deployability, which is what the
framework's resource-constrained motivation actually rests on -- so throughput is
measured on the same hardware that produced the accuracy numbers, and on CPU as well,
since a screening deployment in a low-resource setting is unlikely to have a GPU.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn


def count_macs(model: nn.Module, input_size: tuple[int, ...]) -> dict:
    """Multiply--accumulate operations for one forward pass, via a hook-based count.

    Convolutions and linear layers dominate; normalisation and activation costs are
    excluded, which is the usual convention and keeps the figure comparable with counts
    reported elsewhere.
    """
    macs = {"total": 0}
    handles = []

    def conv_hook(module, inputs, output):
        out_elems = output.numel() / output.shape[0]
        k = module.kernel_size[0] * module.kernel_size[1]
        macs["total"] += int(out_elems * (module.in_channels / module.groups) * k)

    def linear_hook(module, inputs, output):
        macs["total"] += int(module.in_features * module.out_features)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, *input_size, device=device))
    for handle in handles:
        handle.remove()
    model.train(was_training)

    # The dense graph propagation is a matmul, not a module, so it is added explicitly:
    # 3 layers x (A_hat @ H) with A_hat 100x100 and H 100x256.
    graph_macs = 0
    if hasattr(model, "graph_branch") and hasattr(model, "patch_graph"):
        n = model.patch_graph.n_nodes
        dim = model.patch_graph.proj[0].out_channels
        layers = len(getattr(model.graph_branch, "layers", []))
        graph_macs = layers * n * n * dim
    total = macs["total"] + graph_macs
    return {"macs": int(total), "gmacs": total / 1e9, "gflops": 2 * total / 1e9,
            "propagation_macs": int(graph_macs)}


@torch.no_grad()
def measure_latency(model: nn.Module, input_size: tuple[int, ...], device: str = "cuda",
                    batch_size: int = 1, warmup: int = 10, runs: int = 50,
                    tta_hflip: bool = False) -> dict:
    """Median and mean single-batch latency, after warmup, on the requested device."""
    model = model.to(device).eval()
    x = torch.randn(batch_size, *input_size, device=device)

    def forward():
        if tta_hflip and hasattr(model, "predict_proba"):
            return model.predict_proba(x, tta_hflip=True)
        return model(x)

    for _ in range(warmup):
        forward()
    if device == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        forward()
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    times_t = torch.tensor(times)
    return {"device": device, "batch_size": batch_size, "runs": runs,
            "median_ms": float(times_t.median()), "mean_ms": float(times_t.mean()),
            "std_ms": float(times_t.std()), "tta": tta_hflip,
            "throughput_img_s": float(batch_size * 1000.0 / times_t.median())}


@torch.no_grad()
def measure_peak_memory(model: nn.Module, input_size: tuple[int, ...],
                        batch_size: int = 1) -> dict:
    """Peak CUDA memory for one inference pass; returns None off GPU."""
    if not torch.cuda.is_available():
        return {"peak_mb": None, "note": "CUDA unavailable"}
    model = model.cuda().eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(batch_size, *input_size, device="cuda")
    if hasattr(model, "predict_proba"):
        model.predict_proba(x, tta_hflip=True)
    else:
        model(x)
    torch.cuda.synchronize()
    return {"peak_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
            "batch_size": batch_size}


def efficiency_report(model: nn.Module, image_size: int = 320) -> dict:
    size = (3, image_size, image_size)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report = {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "trainable_fraction": trainable / total if total else 0.0,
        "parameters_millions": total / 1e6,
        "macs": count_macs(model, size),
    }
    if hasattr(model, "component_parameters"):
        report["components"] = model.component_parameters()
    if torch.cuda.is_available():
        report["latency_gpu"] = measure_latency(model, size, "cuda", 1)
        report["latency_gpu_tta"] = measure_latency(model, size, "cuda", 1, tta_hflip=True)
        report["peak_memory"] = measure_peak_memory(model, size, 1)
        report["gpu_name"] = torch.cuda.get_device_name(0)
    report["latency_cpu"] = measure_latency(model.cpu(), size, "cpu", 1, warmup=3, runs=15)
    return report
