"""Single-command entry point: python run.py"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from support_pipeline.pipeline import run_pipeline

BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the grounded support-reply pipeline")
    parser.add_argument("--policy", default=str(BASE_DIR / "policy_docs.json"), help="Path to policy_docs.json")
    parser.add_argument("--messages", default=str(BASE_DIR / "messages.json"), help="Path to messages.json")
    parser.add_argument("--config", default=str(BASE_DIR / "config.json"), help="Path to config.json")
    parser.add_argument("--out-dir", default=str(BASE_DIR), help="Directory for generated artifacts")
    parser.add_argument("--offline", action="store_true", help="Force the deterministic extractive generator")
    parser.add_argument("--no-cache", action="store_true", help="Disable the LLM response cache")
    parser.add_argument(
        "--llm-required",
        action="store_true",
        help="Fail if no LLM API key is available (overrides config)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = run_pipeline(
            policy_path=args.policy,
            messages_path=args.messages,
            config_path=args.config,
            out_dir=args.out_dir,
            offline=args.offline,
            no_cache=args.no_cache,
            llm_required=True if args.llm_required else None,
        )
    except Exception as exc:  # pragma: no cover - CLI error surface
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1

    print("Pipeline complete")
    print(f"  Run:            {summary['run_id']}")
    print(f"  Messages:       {summary['message_count']}")
    print(f"  Chunks:         {summary['chunk_count']}")
    print(f"  Final statuses: {summary['final_statuses']}")
    print(f"  Policy modes:   {summary['policy_modes']}")
    print(f"  LLM mode:       {summary['llm_mode']}")
    print(f"  Artifacts:      {summary['out_dir']}")
    print("")
    print("Next: python validate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
