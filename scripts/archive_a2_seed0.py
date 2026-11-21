"""Bundle completed A2 Seed-0 outputs into one downloadable ZIP archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    results = Path(args.results)
    figures = Path(args.figures)
    seed = str(args.seed)
    output = Path(args.output) if args.output else results / f"A2_seed_{seed}_bundle.zip"

    files: list[Path] = []
    required_dirs = [
        results / "e2cs" / "0.95" / seed,
        results / "e2cs" / "0.99" / seed,
        results / "e3" / seed,
    ]
    for folder in required_dirs:
        if not (folder / "metrics.json").exists():
            raise FileNotFoundError(f"Missing completed result folder: {folder}")
        files.extend(p for p in folder.rglob("*") if p.is_file())

    for exp in ("e4bs", "e4h"):
        eps_folders = sorted(p for p in (results / exp).iterdir() if p.is_dir())
        if len(eps_folders) < 5:
            raise FileNotFoundError(
                f"Expected five epsilon folders for {exp}; found {len(eps_folders)}"
            )
        for eps_folder in eps_folders:
            folder = eps_folder / seed
            if not (folder / "metrics.json").exists():
                raise FileNotFoundError(f"Missing completed E4 run: {folder}")
            files.extend(p for p in folder.rglob("*") if p.is_file())
        summary = results / exp / f"seed_{seed}_epsilon_summary.json"
        if summary.exists():
            files.append(summary)

    for name in (
        f"seed_{seed}_summary.json",
        f"seed_{seed}_master.log",
    ):
        path = results / "e2cs" / name
        if path.exists():
            files.append(path)

    figure_names = (
        "fig08_e2cs_stock_position.png",
        "fig09_e2cs_varswap_position.png",
        "fig08_e2cs_stock_position_heatmap.png",
        "fig09_e2cs_varswap_position_heatmap.png",
        "fig08_e2cs_surface_bin_counts.png",
        "e2cs_surface_bin_counts.npz",
    )
    for name in figure_names:
        path = figures / name
        if not path.exists():
            raise FileNotFoundError(f"Missing E2CS figure: {path}")
        files.append(path)

    unique_files = sorted(set(files))
    manifest = {
        "seed": args.seed,
        "experiments": ["e2cs", "e3", "e4bs", "e4h"],
        "files": {
            str(path): {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in unique_files
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in unique_files:
            archive.write(path, path)
        archive.writestr("archive_manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(f"Created {output} ({output.stat().st_size:,} bytes; {len(unique_files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
