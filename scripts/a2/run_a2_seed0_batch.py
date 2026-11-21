"""Selectable Seed-0 runners for E2CS, E3 and E4.

Use --only to run one experiment family independently. E4 means both E4-BS
and E4-Heston. The default 'all' mode retains the sequential one-GPU workflow.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--only",
        choices=["all", "e2cs", "e3", "e4", "e4bs", "e4h", "figures"],
        default="all",
        help="Run only E2CS, E3, both E4 variants, one E4 variant, or figure generation.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    py = sys.executable
    resume = ["--resume"]
    figure_dir = str(Path(args.out) / "figures")

    jobs = {
        "e2cs": [
            py,
            "-m",
            "scripts.a2.run_e2cs",
            "--config",
            "configs/exp/e2cs.yaml",
            "--seed",
            str(args.seed),
            "--out",
            args.out,
            "--device",
            args.device,
            *resume,
        ],
        "e3": [
            py,
            "-m",
            "scripts.a2.run_e3",
            "--config",
            "configs/exp/e3.yaml",
            "--seed",
            str(args.seed),
            "--out",
            args.out,
            "--device",
            args.device,
            *resume,
        ],
        "e4bs": [
            py,
            "-m",
            "scripts.a2.run_e4_grid",
            "--config",
            "configs/exp/e4bs.yaml",
            "--seed",
            str(args.seed),
            "--out",
            args.out,
            "--device",
            args.device,
            *resume,
        ],
        "e4h": [
            py,
            "-m",
            "scripts.a2.run_e4_grid",
            "--config",
            "configs/exp/e4h.yaml",
            "--seed",
            str(args.seed),
            "--out",
            args.out,
            "--device",
            args.device,
            *resume,
        ],
        "figures": [
            py,
            "-m",
            "scripts.a2.make_e2cs_figures",
            "--results",
            str(Path(args.out) / "e2cs"),
            "--figures",
            figure_dir,
            "--seed",
            str(args.seed),
        ],
    }

    if args.only == "all":
        order = ["e2cs", "figures", "e3", "e4bs", "e4h"]
    elif args.only == "e4":
        order = ["e4bs", "e4h"]
    elif args.only == "e2cs":
        order = ["e2cs", "figures"]
    else:
        order = [args.only]

    print(
        f"A2 Seed-{args.seed}; selected={args.only}; device={args.device}; output={args.out}",
        flush=True,
    )
    for name in order:
        print("\n>>> " + " ".join(jobs[name]), flush=True)
        subprocess.run(jobs[name], cwd=root, check=True)

    if args.only == "all":
        archive_cmd = [
            py,
            "-m",
            "scripts.a2.archive_a2_seed0",
            "--results",
            args.out,
            "--figures",
            figure_dir,
            "--seed",
            str(args.seed),
        ]
        print("\n>>> " + " ".join(archive_cmd), flush=True)
        subprocess.run(archive_cmd, cwd=root, check=True)

    print(f"Selected A2 task '{args.only}' completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
