"""Unified A2 plotting entry point. Plot from saved evaluation data only.

Examples:
    python -m scripts.a2.make_figures_a2 --only e4
    python -m scripts.a2.make_figures_a2 --only e4 --seeds 0 1 2 3 4
    python -m scripts.a2.make_figures_a2 --only e3 --e3-dir results/e3/0
    python -m scripts.a2.make_figures_a2 --only all --e2-dir results/e2/0

This dispatcher does not train policies or recalculate Monte Carlo prices.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=("e2", "e2cs", "e2cs-diagnostics", "e3", "e4", "all"),
        default="e4",
    )
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--e2-dir", type=Path, default=Path("results/e2/0"))
    parser.add_argument("--e2cs-dir", type=Path, default=Path("results/e2cs"))
    parser.add_argument(
        "--e2cs-checkpoint-root",
        type=Path,
        default=Path("backups/e2cs_simple_seed0/e2cs"),
    )
    parser.add_argument("--e2cs-report-dir", type=Path, default=Path("reports/e2cs"))
    parser.add_argument("--e2cs-device", default="cpu")
    parser.add_argument("--e3-dir", type=Path, default=Path("results/e3/0"))
    parser.add_argument("--e4-data-root", type=Path, default=Path("backups"))
    parser.add_argument("--e4-report-dir", type=Path, default=Path("reports/e4"))
    parser.add_argument("--seed", type=int, default=0, help="E2cs/E3 seed (E2 uses --e2-dir)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="E4 completed seeds")
    parser.add_argument("--bins", type=int, default=100, help="E3 histogram bins")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    jobs = ("e2", "e2cs", "e3", "e4") if args.only == "all" else (args.only,)
    for job in jobs:
        if job == "e2":
            from scripts.a2.make_e2_figures import make_figures

            make_figures(evaluation_dir=args.e2_dir, figures_dir=args.figures_dir)
        elif job == "e2cs":
            for alpha in ("0.95", "0.99"):
                pnl = args.e2cs_dir / alpha / str(args.seed) / "pnl.npz"
                if not pnl.is_file():
                    raise FileNotFoundError(
                        f"Missing E2cs surface data: {pnl}. "
                        "GitHub's archived E2cs results omit pnl.npz; use Kaggle results."
                    )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.a2.make_e2cs_figures",
                    "--results",
                    str(args.e2cs_dir),
                    "--figures",
                    str(args.figures_dir),
                    "--seed",
                    str(args.seed),
                ],
                check=True,
            )
        elif job == "e2cs-diagnostics":
            from scripts.a2.diagnose_e2cs import analyze
            from src.common.seed import resolve_device

            report = analyze(
                checkpoint_root=args.e2cs_checkpoint_root,
                config_path=Path("configs/exp/e2cs.yaml"),
                output_dir=args.e2cs_report_dir,
                figures_dir=args.figures_dir / "e2cs_diagnostics",
                seed=args.seed,
                step=15,
                device=resolve_device(args.e2cs_device),
            )
            print(f"saved E2cs diagnostics for seed {report['seed']}")
        elif job == "e3":
            from scripts.a2.make_e3_figures import plot_saved

            out = args.figures_dir / "fig05_e3.png"
            print(f"saved {plot_saved(run_dir=args.e3_dir, output=out, bins=args.bins)}")
        elif job == "e4":
            from scripts.a2.make_e4_figures import make_figures

            for output in make_figures(
                data_root=args.e4_data_root,
                figures_dir=args.figures_dir,
                report_dir=args.e4_report_dir,
                seeds=args.seeds,
            ):
                print(f"saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
