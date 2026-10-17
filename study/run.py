"""Prepare, run, and analyze the complete 300-task alias experiment.

Set OPENROUTER_API_KEY in the environment, then run ``python -m study.run``.
The paid request is made only when a human executes this command. The default
10 USD local planning guard is not a provider-side billing limit.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from pilot import runner
from pilot.verify import ValidationError, validate_all
from pilot.common import write_json

from .analysis import analyze_study
from .dataset import COMBINED_TASKS, MAIN_TASKS, STUDY_ROOT, combine_tasks
from .smoke import SMOKE_ESTIMATED_COST_USD, run_smoke


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true", help="Run the entire 10,800-row matrix locally without network or cost")
    parser.add_argument("--prepare-only", action="store_true", help="Combine and inspect tasks without model calls")
    parser.add_argument("--analyze-only", action="store_true", help="Analyze an existing --out run without model calls")
    parser.add_argument("--out", type=Path, help="Run directory; required with --resume or --analyze-only")
    parser.add_argument("--resume", action="store_true", help="Resume a stopped run with exactly matching frozen inputs")
    parser.add_argument("--main-tasks", type=Path, default=MAIN_TASKS)
    parser.add_argument("--max-cost-usd", type=float, default=10.0)
    parser.add_argument("--max-transport-retries", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=210.0,
                        help="Per-request client timeout for smoke and matrix (default: 210 seconds)")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args(argv)
    if (args.resume or args.analyze_only) and args.out is None:
        parser.error("--resume and --analyze-only require --out")
    if args.mock and args.analyze_only:
        parser.error("--mock and --analyze-only cannot be combined")
    if not math.isfinite(args.max_cost_usd) or args.max_cost_usd <= 0:
        parser.error("--max-cost-usd must be a positive finite number")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be a positive finite number")
    if args.max_transport_retries < 0:
        parser.error("--max-transport-retries must be nonnegative")
    if args.bootstrap_replicates < 100:
        parser.error("--bootstrap-replicates must be at least 100")
    if not args.mock and not args.prepare_only and not args.analyze_only and not args.resume:
        matrix_reserve = 3600 * sum(runner.ESTIMATED_PER_CALL_USD.values()) * (args.max_transport_retries + 1)
        smoke_headroom = max(0.10, 10 * SMOKE_ESTIMATED_COST_USD)
        if args.max_cost_usd < matrix_reserve + smoke_headroom:
            parser.error(
                f"--max-cost-usd must be at least ${matrix_reserve + smoke_headroom:.2f} "
                f"for this retry policy (matrix worst-case reservation ${matrix_reserve:.2f} plus smoke headroom). "
                "Reduce --max-transport-retries or raise the local cap before any paid call."
            )
    if args.analyze_only:
        try:
            result = analyze_study(args.out, bootstrap_replicates=args.bootstrap_replicates)
        except (ValueError, FileNotFoundError) as error:
            print(f"Analysis failed: {error}", file=sys.stderr)
            return 2
        print(f"Study status: {result['status']}. Results: {args.out / 'study_results.json'}")
        return 0
    try:
        combined = combine_tasks(args.main_tasks, COMBINED_TASKS)
    except (ValueError, KeyError, ValidationError, FileNotFoundError) as error:
        print(f"Study preparation failed: {error}", file=sys.stderr)
        return 2
    print(f"Prepared {len(combined['tasks'])} tasks: {COMBINED_TASKS}")
    if args.prepare_only:
        try:
            report = validate_all(dataset=combined)
            validation_path = STUDY_ROOT / "validation" / "verification_report.json"
            write_json(validation_path, report)
        except (ValueError, ValidationError) as error:
            print(f"Study validation failed: {error}", file=sys.stderr)
            return 2
        print(f"Validated {len(report['label_checks'])} labels and {len(report['catalogs'])} catalogs: {validation_path}")
        return 0
    if not args.mock and not os.environ.get("OPENROUTER_API_KEY"):
        print("Set OPENROUTER_API_KEY before running the paid study.", file=sys.stderr)
        return 2
    out = args.out or STUDY_ROOT / "runs" / (
        ("mock" if args.mock else "openrouter") + "-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    if args.resume and not (out / "manifest.json").exists():
        parser.error("--resume requires an existing run manifest")
    if not args.resume and out.exists() and any(out.iterdir()):
        parser.error("--out must be empty for a new run")
    matrix_cost_cap = args.max_cost_usd
    if not args.mock:
        try:
            report = validate_all(dataset=combined)
            write_json(STUDY_ROOT / "validation" / "verification_report.json", report)
        except (ValueError, ValidationError) as error:
            print(f"Mechanical preflight failed before any paid request: {error}", file=sys.stderr)
            return 2
        try:
            smoke = run_smoke(out, combined, timeout_seconds=args.timeout_seconds,
                              max_cost_usd=args.max_cost_usd, resume=args.resume,
                              max_transport_retries=args.max_transport_retries)
        except (ValueError, ValidationError) as error:
            print(f"Paid matrix was not started: {error}", file=sys.stderr)
            return 2
        matrix_cost_cap -= smoke["budget_accounted_cost_usd"]
        print(f"Compatibility smoke passed: 6 probes, {smoke['requests_issued']} physical calls, "
              f"budget-accounted cost ${smoke['budget_accounted_cost_usd']:.6f}; matrix cap ${matrix_cost_cap:.6f}.")
    runner_args = [
        "--adapter", "mock" if args.mock else "openrouter",
        "--models", "jev,luna,haiku",
        "--dataset", str(COMBINED_TASKS),
        "--out", str(out),
        "--max-cost-usd", str(matrix_cost_cap),
        "--max-transport-retries", str(args.max_transport_retries),
        "--timeout-seconds", str(args.timeout_seconds),
    ]
    if args.resume:
        runner_args.append("--resume")
    if not args.mock:
        # AI construction and mechanical verification are recorded separately
        # from independent researcher review. This user-initiated execution
        # deliberately allows the pending label state and records that override.
        runner_args.extend(["--allow-live", "--allow-unreviewed"])
        print("Labels are mechanically checked but await independent researcher review; this is recorded in the manifest.")
    code = runner.main(runner_args)
    if code:
        return code
    try:
        result = analyze_study(out, bootstrap_replicates=args.bootstrap_replicates)
    except (ValueError, FileNotFoundError) as error:
        print(f"Study analysis failed: {error}", file=sys.stderr)
        return 2
    print(f"Study status: {result['status']}. Results: {out / 'study_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
