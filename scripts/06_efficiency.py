"""Parameters, MACs, latency and peak memory for the trained model.

    python scripts/06_efficiency.py
"""
from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
from _bootstrap import CHECKPOINTS, REPO_ROOT, RESULTS

from osteognn.config import load_config
from osteognn.efficiency import efficiency_report
from osteognn.train import load_checkpoint
from osteognn.utils import device_report, save_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--run", default="full")
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint = CHECKPOINTS / args.run / "checkpoint.pt"
    model = (load_checkpoint(cfg, checkpoint) if checkpoint.exists()
             else __import__("osteognn.models.ensemble", fromlist=["build_model"])
             .build_model(cfg))
    if not checkpoint.exists():
        print(f"note: {checkpoint} not found; measuring an untrained instance "
              f"(cost does not depend on weights)")

    report = efficiency_report(model, int(cfg.preprocess.image_size))
    report["environment"] = device_report()

    print(f"total parameters      {report['total_parameters']:,} "
          f"({report['parameters_millions']:.2f}M)")
    print(f"trainable parameters  {report['trainable_parameters']:,} "
          f"({report['trainable_fraction']*100:.1f}%)")
    print(f"MACs                  {report['macs']['gmacs']:.3f} G "
          f"({report['macs']['gflops']:.3f} GFLOPs)")
    if "latency_gpu" in report:
        print(f"GPU latency (b=1)     {report['latency_gpu']['median_ms']:.2f} ms "
              f"on {report.get('gpu_name')}")
        print(f"GPU latency + TTA     {report['latency_gpu_tta']['median_ms']:.2f} ms")
        print(f"peak inference memory {report['peak_memory']['peak_mb']:.1f} MB")
    print(f"CPU latency (b=1)     {report['latency_cpu']['median_ms']:.2f} ms")

    save_json(report, RESULTS / "efficiency.json")
    with open(RESULTS / "table_efficiency.csv", "w", encoding="utf-8") as handle:
        handle.write("quantity,value\n")
        handle.write(f"Total parameters,{report['total_parameters']}\n")
        handle.write(f"Trainable parameters,{report['trainable_parameters']}\n")
        handle.write(f"MACs (G),{report['macs']['gmacs']:.3f}\n")
        handle.write(f"GFLOPs,{report['macs']['gflops']:.3f}\n")
        if "latency_gpu" in report:
            handle.write(f"GPU latency b1 (ms),{report['latency_gpu']['median_ms']:.2f}\n")
            handle.write(f"GPU latency b1 + TTA (ms),"
                         f"{report['latency_gpu_tta']['median_ms']:.2f}\n")
            handle.write(f"Peak inference memory (MB),"
                         f"{report['peak_memory']['peak_mb']:.1f}\n")
        handle.write(f"CPU latency b1 (ms),{report['latency_cpu']['median_ms']:.2f}\n")
    print(f"\nwrote {RESULTS/'efficiency.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
