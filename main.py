"""From-scratch four-agent quantitative research loop."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agents import (
    generate_universe_questions,
    resolve_universe_request,
    run_backtest_agent,
    run_implementation_agent,
    run_research_agent,
    run_verification_agent,
)
from agents.backtest import BacktestError
from utils.data import (
    MODEL_TYPES,
    UNIVERSE_CHOICES,
    MarketData,
    prepare_market_data,
    resolve_universe,
)

logger = logging.getLogger("quant_lab")

MAX_VERIFICATION_ROUNDS = 3


def run_candidate(
    strategy_idea: str,
    data: MarketData,
    dataset_context: dict[str, Any],
    cost_bps: float,
) -> dict:
    """Run one independent research -> implement <-> verify -> backtest cycle."""
    research = run_research_agent(strategy_idea, dataset_context)

    code = run_implementation_agent(research, dataset_context)
    verification = run_verification_agent(code, research, dataset_context)

    rounds = 0
    qualitative_revision_done = False
    while rounds < MAX_VERIFICATION_ROUNDS:
        objective_failure = not verification["passed"]
        qualitative_revision = (
            verification.get("needs_revision", False)
            and not objective_failure
            and not qualitative_revision_done
        )
        if not objective_failure and not qualitative_revision:
            break
        if qualitative_revision:
            qualitative_revision_done = True
        rounds += 1
        logger.info("Revision round %d/%d", rounds, MAX_VERIFICATION_ROUNDS)
        code = run_implementation_agent(
            research,
            dataset_context,
            verification_feedback=verification["issues"],
            previous_code=code,
        )
        verification = run_verification_agent(code, research, dataset_context)

    if not verification["passed"]:
        raise BacktestError(
            f"Code failed verification after {MAX_VERIFICATION_ROUNDS} revisions."
        )
    if verification.get("needs_revision", False):
        logger.warning(
            "Reviewer advisories remain after one qualitative revision; "
            "objective safety checks passed, so the candidate will be backtested."
        )
        verification["advisory_issues_remaining"] = True
        verification["needs_revision"] = False

    metrics = run_backtest_agent(code, data, cost_bps=cost_bps)
    return {
        "research": research,
        "code": code,
        "verification": verification,
        "metrics": metrics,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Four-agent quantitative research lab."
    )
    parser.add_argument(
        "--idea",
        help="Trading strategy idea to research.",
    )
    parser.add_argument("--source", choices=("yahoo", "csv"))
    parser.add_argument("--csv-path", help="Long-form CSV: date,ticker,close[,volume].")
    parser.add_argument("--universe", choices=UNIVERSE_CHOICES)
    parser.add_argument("--tickers", help="Comma-separated tickers for --universe custom.")
    parser.add_argument(
        "--universe-request",
        help='Broad request for --universe agent, e.g. "liquid crypto assets".',
    )
    parser.add_argument(
        "--universe-answer",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Clarification answer for non-interactive Universe Agent runs.",
    )
    parser.add_argument("--model-type", choices=MODEL_TYPES)
    parser.add_argument("--runs", type=int, help="Independent agent candidates (default: 10).")
    parser.add_argument("--start", help="Backtest start date (YYYY-MM-DD).")
    parser.add_argument("--end", help="Backtest end date (YYYY-MM-DD).")
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=5.0,
        help="One-way transaction cost in basis points applied to turnover (default: 5).",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Use defaults for omitted selections instead of showing the menu.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def _complete_selection(args: argparse.Namespace) -> argparse.Namespace:
    interactive = sys.stdin.isatty() and not args.non_interactive
    args.source = args.source or "yahoo"
    args.universe = args.universe or "sp500"
    args.model_type = args.model_type or "price"
    default_idea = (
        "pairs trading on the S&P 500"
        if args.universe == "sp500"
        else "pairs trading on the selected universe"
    )
    if interactive:
        print(
            f"\nMarket data ({args.universe} universe, {args.source} close prices) "
            "is downloaded and cached before the agents start. "
            "Run with --help to change the universe or dataset."
        )
        args.idea = args.idea or input(
            f"Research objective ({default_idea}): "
        ).strip() or default_idea
    else:
        args.idea = args.idea or default_idea
    if args.universe == "custom" and not args.tickers and interactive:
        args.tickers = input("Tickers (comma-separated): ").strip()
    if args.universe == "agent" and not args.universe_request and interactive:
        args.universe_request = input(
            'Describe the asset universe (for example, "liquid crypto assets"): '
        ).strip()
    if args.universe == "agent" and not args.universe_request:
        raise ValueError("--universe agent requires --universe-request.")
    if args.source == "csv" and not args.csv_path and interactive:
        args.csv_path = input("CSV path: ").strip()
    if args.runs is None:
        if interactive:
            raw_runs = input("Number of independent candidates (10): ").strip()
            args.runs = int(raw_runs or "10")
        else:
            args.runs = 10
    if interactive:
        args.start = args.start or input(
            "Backtest start date (2018-01-01): "
        ).strip() or "2018-01-01"
        args.end = args.end or input(
            "Backtest end date (2024-12-31): "
        ).strip() or "2024-12-31"
    else:
        args.start = args.start or "2018-01-01"
        args.end = args.end or "2024-12-31"
    if args.runs < 1:
        raise ValueError("--runs must be at least 1.")
    _validate_date_range(args.start, args.end)
    return args


def _validate_date_range(start: str, end: str) -> None:
    try:
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end)
    except ValueError as exc:
        raise ValueError("Backtest dates must use YYYY-MM-DD format.") from exc
    if start_date >= end_date:
        raise ValueError("Backtest start date must be earlier than the end date.")


def _resolve_agent_universe(args: argparse.Namespace) -> dict:
    supplied_answers: dict[str, str] = {}
    for item in args.universe_answer:
        if "=" not in item:
            raise ValueError("--universe-answer must use KEY=VALUE format.")
        key, value = item.split("=", 1)
        supplied_answers[key.strip()] = value.strip()

    questions = generate_universe_questions(args.universe_request)
    interactive = sys.stdin.isatty() and not args.non_interactive
    answers: dict[str, str] = {}
    if questions and interactive:
        print("\nThe Universe Agent needs a few details:")
    for question in questions:
        question_id = question["id"]
        if question_id in supplied_answers:
            answers[question_id] = supplied_answers[question_id]
        elif interactive:
            default = question["default"]
            prompt = f"{question['question']} ({default}): " if default else f"{question['question']} "
            answers[question_id] = input(prompt).strip() or default
        else:
            answers[question_id] = question["default"]
    return resolve_universe_request(args.universe_request, answers)


def _save_results(
    ranked: list[dict],
    failures: list[dict],
    config: dict[str, Any],
) -> None:
    best = ranked[0]
    Path("best_strategy.py").write_text(best["code"])
    Path("best_strategy_report.json").write_text(
        json.dumps(
            {"run": best["run"], "research": best["research"], "metrics": best["metrics"]},
            indent=2,
        )
    )
    serializable_runs = [
        {
            "rank": rank,
            "run": result["run"],
            "research": result["research"],
            "verification": result["verification"],
            "metrics": result["metrics"],
        }
        for rank, result in enumerate(ranked, start=1)
    ]
    Path("research_runs_report.json").write_text(
        json.dumps(
            {"config": config, "ranked_runs": serializable_runs, "failures": failures},
            indent=2,
        )
    )


def main() -> int:
    load_dotenv()
    args = _complete_selection(_parse_args())
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)-22s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    universe_rationale = ""
    if args.universe == "agent":
        agent_universe = _resolve_agent_universe(args)
        tickers = agent_universe["tickers"]
        universe_label = agent_universe["universe_name"]
        universe_rationale = agent_universe["rationale"]
        logger.info(
            "Universe Agent selected '%s': %d tickers (%s)",
            universe_label,
            len(tickers),
            universe_rationale,
        )
    else:
        tickers = resolve_universe(args.universe, args.tickers)
        universe_label = args.universe
        logger.info("Selected universe '%s': %d tickers", universe_label, len(tickers))
    data = prepare_market_data(
        source=args.source,
        tickers=tickers,
        start=args.start,
        end=args.end,
        model_type=args.model_type,
        csv_path=args.csv_path,
    )
    close = data["close"]
    dataset_context: dict[str, Any] = {
        "source": args.source,
        "universe": universe_label,
        "ticker_count": close.shape[1],
        "fields": sorted(data),
        "model_type": args.model_type,
        "period": f"{close.index[0].date()} to {close.index[-1].date()}",
    }
    config = {
        "idea": args.idea,
        **dataset_context,
        "runs": args.runs,
        "cost_bps": args.cost_bps,
    }
    if args.universe == "agent":
        config["universe_request"] = args.universe_request
        config["universe_rationale"] = universe_rationale

    successes: list[dict] = []
    failures: list[dict] = []
    for run_number in range(1, args.runs + 1):
        print(f"\n{'=' * 70}\nCANDIDATE {run_number}/{args.runs}\n{'=' * 70}")
        try:
            result = run_candidate(args.idea, data, dataset_context, args.cost_bps)
        except (BacktestError, ValueError, RuntimeError) as exc:
            logger.error("Candidate %d failed: %s", run_number, exc)
            failures.append({"run": run_number, "error": str(exc)})
            continue

        result["run"] = run_number
        successes.append(result)
        print(f"\n{result['research']['strategy_name']}")
        print(json.dumps(result["metrics"], indent=2))

    if not successes:
        Path("research_runs_report.json").write_text(
            json.dumps({"config": config, "ranked_runs": [], "failures": failures}, indent=2)
        )
        print("\nNo candidate produced a runnable strategy.", file=sys.stderr)
        return 1

    ranked = sorted(successes, key=lambda item: item["metrics"]["sharpe_ratio"], reverse=True)
    print(f"\n{'=' * 70}\nRANKING BY SHARPE\n{'=' * 70}")
    for rank, result in enumerate(ranked, start=1):
        print(
            f"{rank:>2}. run {result['run']:>2} | "
            f"Sharpe {result['metrics']['sharpe_ratio']:>7.3f} | "
            f"{result['research']['strategy_name']}"
        )
    _save_results(ranked, failures, config)
    print("\nSaved: best_strategy.py, best_strategy_report.json, research_runs_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
