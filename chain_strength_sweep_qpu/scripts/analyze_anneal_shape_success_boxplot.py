#!/usr/bin/env python3
"""Validate and aggregate the retained six-shape AS-JJIN campaign.

The release intentionally retains processed run-level metrics rather than all
per-read sample directories.  This script therefore treats the AS-JJIN raw CSV
as its metric source, verifies it against the 120-row campaign manifest, and
rebuilds the canonical manuscript aggregate.  It never reconstructs or
overwrites the raw CSV from absent sample directories.

The manuscript PDFs are generated separately by
``JeongHun/figures/as_suite/build_as_schedule_compare.py``.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path


ROOT = Path(
    os.environ.get("QDOCK_ROOT", str(Path(__file__).resolve().parents[2]))
).expanduser().resolve()

MODE = "anneal_schedule"
LIGANDS = ("3nq9", "4jsz")
VARIANTS = (
    "linear",
    "cubic_smoothstep",
    "fourier",
    "lowpass_linear",
    "bangbang_monotone",
    "pause_quench",
)
REPEATS = tuple(range(1, 11))

DEFAULT_RESULTS = ROOT / "as_jjin" / "results"
DEFAULT_RAW = DEFAULT_RESULTS / "anneal_shape_success_quality_tts_raw.csv"
DEFAULT_AGG = (
    ROOT / "JeongHun" / "tables" / "as_suite" / "as_jjin_quality_tts_agg.csv"
)
MANIFEST_GLOB = "anneal_shape_as_jjin_fixedemb_opt1_at800_r1to10_manifest_*.csv"

AGG_FIELDS = (
    "ligand",
    "mode",
    "variant",
    "setting",
    "n_repeats",
    "quality_mean",
    "quality_min",
    "quality_max",
    "tts_0p99_mean_sec",
    "tts_0p99_min_sec",
    "tts_0p99_max_sec",
)


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [
            {(key or "").strip(): (value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def latest_manifest(results_dir: Path) -> Path:
    candidates = sorted(results_dir.glob(MANIFEST_GLOB))
    if not candidates:
        raise FileNotFoundError(f"no AS-JJIN manifest matching {MANIFEST_GLOB!r}")
    return candidates[-1]


def validate_manifest(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    expected = {
        (ligand, variant, repeat)
        for ligand in LIGANDS
        for variant in VARIANTS
        for repeat in REPEATS
    }
    observed: set[tuple[str, str, int]] = set()
    by_run_tag: dict[str, dict[str, str]] = {}

    for row in rows:
        ligand = row.get("ligand", "").lower()
        mode = row.get("mode", "")
        variant = row.get("variant", "")
        run_tag = row.get("run_tag", "")
        try:
            repeat = int(row.get("repeat", ""))
        except ValueError as error:
            raise ValueError(f"invalid repeat in manifest row: {row}") from error

        key = (ligand, variant, repeat)
        if mode != MODE or ligand not in LIGANDS or variant not in VARIANTS:
            raise ValueError(f"non-AS-JJIN setting in retained manifest: {row}")
        if row.get("status") != "ok":
            raise ValueError(f"non-ok retained AS-JJIN run: {run_tag}")
        if not parse_bool(row.get("hash_ok")) or not parse_bool(row.get("cached_ok")):
            raise ValueError(f"failed provenance flags in retained AS-JJIN run: {run_tag}")
        if key in observed:
            raise ValueError(f"duplicate ligand/variant/repeat in manifest: {key}")
        if not run_tag or run_tag in by_run_tag:
            raise ValueError(f"missing or duplicate run_tag in manifest: {run_tag!r}")

        observed.add(key)
        by_run_tag[run_tag] = row

    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ValueError(
            f"AS-JJIN manifest grid mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    return by_run_tag


def validate_raw(
    rows: list[dict[str, str]], manifest_by_run_tag: dict[str, dict[str, str]]
) -> None:
    observed_run_tags: set[str] = set()

    for row in rows:
        ligand = row.get("ligand", "").lower()
        mode = row.get("mode", "")
        variant = row.get("variant", "")
        setting = row.get("setting", "")
        run_tag = row.get("run_tag", "")

        if mode != MODE or ligand not in LIGANDS or variant not in VARIANTS:
            raise ValueError(f"non-AS-JJIN setting in retained raw CSV: {row}")
        if setting != f"{MODE}|{variant}":
            raise ValueError(f"inconsistent setting field for run: {run_tag}")
        if run_tag in observed_run_tags:
            raise ValueError(f"duplicate run_tag in raw CSV: {run_tag}")
        manifest_row = manifest_by_run_tag.get(run_tag)
        if manifest_row is None:
            raise ValueError(f"raw run absent from manifest: {run_tag}")
        if (
            manifest_row.get("ligand", "").lower() != ligand
            or manifest_row.get("mode", "") != mode
            or manifest_row.get("variant", "") != variant
        ):
            raise ValueError(f"raw/manifest setting mismatch for run: {run_tag}")
        if not parse_bool(row.get("embedding_cached")):
            raise ValueError(f"embedding_cached is false for run: {run_tag}")

        try:
            n_reads = int(row.get("n_reads", ""))
            quality = float(row.get("quality", ""))
            tts = float(row.get("tts_0p99_sec", ""))
        except ValueError as error:
            raise ValueError(f"invalid metric in raw row for run: {run_tag}") from error
        if n_reads != 1000 or not 0.0 <= quality <= 1.0 or not math.isfinite(tts):
            raise ValueError(f"out-of-range retained metric for run: {run_tag}")

        observed_run_tags.add(run_tag)

    manifest_run_tags = set(manifest_by_run_tag)
    missing = sorted(manifest_run_tags - observed_run_tags)
    extra = sorted(observed_run_tags - manifest_run_tags)
    if missing or extra:
        raise ValueError(
            f"AS-JJIN raw/manifest mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )


def build_aggregate(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], row["ligand"].lower())].append(row)

    aggregate: list[dict[str, object]] = []
    for (variant, ligand), values in sorted(grouped.items()):
        quality = [float(value["quality"]) for value in values]
        tts = [float(value["tts_0p99_sec"]) for value in values]
        aggregate.append(
            {
                "ligand": ligand,
                "mode": MODE,
                "variant": variant,
                "setting": f"{MODE}|{variant}",
                "n_repeats": len(values),
                "quality_mean": sum(quality) / len(quality),
                "quality_min": min(quality),
                "quality_max": max(quality),
                "tts_0p99_mean_sec": sum(tts) / len(tts),
                "tts_0p99_min_sec": min(tts),
                "tts_0p99_max_sec": max(tts),
            }
        )
    return aggregate


def validate_aggregate(
    expected: list[dict[str, object]], actual: list[dict[str, str]]
) -> None:
    if len(actual) != len(expected):
        raise ValueError(
            f"aggregate row count mismatch: expected={len(expected)}, actual={len(actual)}"
        )
    for expected_row, actual_row in zip(expected, actual):
        for field in AGG_FIELDS:
            expected_value = expected_row[field]
            actual_value = actual_row.get(field, "")
            if field in {"ligand", "mode", "variant", "setting"}:
                if actual_value != str(expected_value):
                    raise ValueError(
                        f"aggregate mismatch at {field}: {actual_value!r} != {expected_value!r}"
                    )
            elif not math.isclose(
                float(actual_value), float(expected_value), rel_tol=1e-12, abs_tol=1e-15
            ):
                raise ValueError(
                    f"aggregate mismatch at {field}: {actual_value!r} != {expected_value!r}"
                )


def write_aggregate(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=AGG_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and aggregate the retained six-shape AS-JJIN campaign."
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGG)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate the existing canonical aggregate without rewriting it",
    )
    args = parser.parse_args()

    manifest = args.manifest or latest_manifest(DEFAULT_RESULTS)
    manifest_rows = load_csv(manifest)
    raw_rows = load_csv(args.raw)
    manifest_by_run_tag = validate_manifest(manifest_rows)
    validate_raw(raw_rows, manifest_by_run_tag)
    aggregate = build_aggregate(raw_rows)

    if args.check_only:
        validate_aggregate(aggregate, load_csv(args.aggregate))
        action = "validated"
    else:
        write_aggregate(args.aggregate, aggregate)
        validate_aggregate(aggregate, load_csv(args.aggregate))
        action = "wrote and validated"

    print(f"manifest={manifest}")
    print(f"raw={args.raw}")
    print(f"aggregate={args.aggregate}")
    print(f"runs={len(raw_rows)} settings={len(aggregate)} action={action}")


if __name__ == "__main__":
    main()
