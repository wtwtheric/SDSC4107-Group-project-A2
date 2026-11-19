"""Run one E2+E3 seed-0 paper-fidelity control variant end-to-end.

Recommended parallel use: one GPU / process per variant.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

VARIANTS = {
    "simulator_only": {
        "e2": "configs/fidelity/e2e3/simulator_only_e2.yaml",
        "e3": "configs/fidelity/e2e3/simulator_only_e3.yaml",
    },
    "paper_input_only": {
        "e2": "configs/fidelity/e2e3/paper_input_only_e2.yaml",
        "e3": "configs/fidelity/e2e3/paper_input_only_e3.yaml",
    },
    "simulator_plus_paper_input": {
        "e2": "configs/fidelity/e2e3/simulator_plus_paper_input_e2.yaml",
        "e3": "configs/fidelity/e2e3/simulator_plus_paper_input_e3.yaml",
    },
}


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _compact_summary(root: Path, variant: str) -> dict[str, Any]:
    e2 = _load_json(root / "e2" / "0" / "evaluation_1000000" / "metrics.json")
    e3 = _load_json(root / "e3" / "0" / "metrics.json")
    e2m = e2["metrics"]
    e3m = e3["metrics"]
    return {
        "schema_version": "1.0",
        "variant": variant,
        "seed": 0,
        "e2": {
            "q": e2m["q"],
            "price_alpha50": e2m["price_alpha50"],
            "price_alpha99": e2m["price_alpha99"],
            "cvar50_alpha50_excess": e2m["cvar05_other"],
            "cvar99_alpha50_excess": e2m["cvar99_other"],
            "cvar50_alpha99_excess": e2m["cvar05_own"],
            "cvar99_alpha99_excess": e2m["cvar99_own"],
            "paper_targets": {
                "price_alpha50": 1.94,
                "price_alpha99": 3.49,
                "figure7_cvar50_alpha50": 0.2531,
                "figure7_cvar99_alpha50": 2.3631,
                "figure7_cvar50_alpha99": 0.5270,
                "figure7_cvar99_alpha99": 1.8034,
            },
        },
        "e3": {
            "price_recurrent": e3m["price_recurrent"],
            "price_simpler": e3m["price_simpler"],
            "price_gap_simpler_minus_recurrent": (e3m["price_simpler"] - e3m["price_recurrent"]),
            "cvar_recurrent": e3m["cvar_recurrent"],
            "cvar_simpler": e3m["cvar_simpler"],
            "mean_cost_recurrent": e3m["mean_cost_recurrent"],
            "mean_cost_simpler": e3m["mean_cost_simpler"],
            "paper_targets": {
                "price_recurrent": 5.5137,
                "price_simpler": 6.7446,
                "price_gap_simpler_minus_recurrent": 1.2309,
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, default=Path("results/e2e3_fidelity"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--audit-paths", type=int, default=200_000)
    parser.add_argument("--eval-paths", type=int, default=1_000_000)
    parser.add_argument("--eval-batch-size", type=int, default=25_000)
    args = parser.parse_args()

    variant = args.variant
    root = args.output_root / variant
    root.mkdir(parents=True, exist_ok=True)
    cfg = VARIANTS[variant]

    # Run the simulator distribution audit once with the simulator-only arm.
    # The other two arms can share this artifact because the audit itself does not
    # depend on the network feature set.
    audit_path = args.output_root / "simulator_audit.json"
    if (
        variant == "simulator_only"
        and not args.skip_audit
        and not (args.resume and audit_path.is_file())
    ):
        _run(
            [
                sys.executable,
                "-m",
                "scripts.audit_e2e3_simulators",
                "--paths",
                str(args.audit_paths),
                "--batch-size",
                str(args.eval_batch_size),
                "--device",
                args.device,
                "--output",
                str(audit_path),
            ]
        )

    e2_run = root / "e2" / "0"
    if not (args.resume and (e2_run / "metrics.json").is_file()):
        _run(
            [
                sys.executable,
                "-m",
                "src.replication.train",
                "--config",
                cfg["e2"],
                "--seed",
                "0",
                "--out",
                str(root),
            ]
        )

    e2_eval = e2_run / f"evaluation_{args.eval_paths}"
    if not (args.resume and (e2_eval / "metrics.json").is_file()):
        _run(
            [
                sys.executable,
                "-m",
                "scripts.evaluate_e2",
                "--run-dir",
                str(e2_run),
                "--output-dir",
                str(e2_eval),
                "--n-paths",
                str(args.eval_paths),
                "--batch-size",
                str(args.eval_batch_size),
                "--seed",
                "0",
                "--device",
                args.device,
            ]
        )
    _run(
        [
            sys.executable,
            "-m",
            "scripts.make_e2_figures",
            "--evaluation-dir",
            str(e2_eval),
            "--figures-dir",
            str(root / "figures"),
        ]
    )

    e3_run = root / "e3" / "0"
    if not (args.resume and (e3_run / "metrics.json").is_file()):
        cmd = [
            sys.executable,
            "-m",
            "scripts.run_e3",
            "--config",
            cfg["e3"],
            "--seed",
            "0",
            "--out",
            str(root),
            "--device",
            args.device,
        ]
        if args.resume:
            cmd.append("--resume")
        _run(cmd)

    summary = _compact_summary(root, variant)
    summary_path = root / "variant_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"saved {summary_path}")
    print(
        f"E2 p99={summary['e2']['price_alpha99']:.6f}; "
        f"E3 recurrent={summary['e3']['price_recurrent']:.6f}, "
        f"simpler={summary['e3']['price_simpler']:.6f}, "
        f"gap={summary['e3']['price_gap_simpler_minus_recurrent']:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
