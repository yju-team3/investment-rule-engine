from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from decision_engine.engine import DecisionEngine
from decision_engine.models import (
    DecisionReport,
    FinalDecision,
    MarketRegime,
    PortfolioConstraints,
)
from decision_engine.run import (
    build_engine,
    build_live_stock_snapshot,
    build_market_snapshot,
    sample_stock_for_ticker,
)


@dataclass(frozen=True)
class ScanResult:
    ticker: str
    decision: str
    candidate_type: str | None
    wait_reason_top: str | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decision engine scanner")
    parser.add_argument(
        "--mode",
        choices=["sample", "live"],
        default="live",
        help="Use sample data or live data",
    )
    parser.add_argument(
        "--tickers",
        required=True,
        help='Ticker list as comma-separated string or file path with tickers per line',
    )
    parser.add_argument(
        "--use-adjusted-close",
        action="store_true",
        help="Use adjusted close price when available",
    )
    return parser.parse_args(argv)


def load_tickers(value: str) -> list[str]:
    if os.path.isfile(value):
        with open(value, encoding="utf-8") as handle:
            tickers = [line.strip() for line in handle if line.strip()]
    else:
        tickers = [item.strip() for item in value.split(",") if item.strip()]
    normalized = []
    seen = set()
    for ticker in tickers:
        upper = ticker.upper()
        if upper not in seen:
            seen.add(upper)
            normalized.append(upper)
    return normalized


def extract_candidate_type(action_plan: Iterable[str]) -> str | None:
    for item in action_plan:
        if item.startswith("후보 유형:"):
            return item.replace("후보 유형:", "").strip().rstrip(".")
    return None


def summarize_wait_reason(reason_log: Iterable[str]) -> str | None:
    keywords = ("보류", "대기", "미충족", "충돌", "결정", "부족", "거절", "리스크")
    for reason in reason_log:
        if any(keyword in reason for keyword in keywords):
            return reason
    return next(iter(reason_log), None)


def evaluate_ticker(
    engine: DecisionEngine,
    ticker: str,
    mode: str,
    constraints: PortfolioConstraints,
    market_regime: MarketRegime | None,
    use_adjusted_close: bool,
) -> ScanResult:
    market = build_market_snapshot(market_regime)
    if mode == "live":
        stock, reason = build_live_stock_snapshot(ticker, use_adjusted_close=use_adjusted_close)
        if stock is None:
            report = DecisionReport(
                FinalDecision.WAIT,
                [reason or "라이브 데이터가 불완전하여 WAIT 처리."],
                ["데이터 보완 후 재평가."],
            )
        else:
            report = engine.evaluate(market, stock, constraints)
    else:
        stock = sample_stock_for_ticker(ticker)
        report = engine.evaluate(market, stock, constraints)

    candidate_type = extract_candidate_type(report.action_plan)
    wait_reason_top = None
    if report.decision == FinalDecision.WAIT:
        wait_reason_top = summarize_wait_reason(report.reason_log)

    return ScanResult(
        ticker=ticker,
        decision=report.decision.value,
        candidate_type=candidate_type,
        wait_reason_top=wait_reason_top,
    )


def write_csv(path: str, results: list[ScanResult]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ticker", "decision", "candidate_type", "wait_reason_top"],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "ticker": result.ticker,
                    "decision": result.decision,
                    "candidate_type": result.candidate_type or "",
                    "wait_reason_top": result.wait_reason_top or "",
                }
            )


def format_markdown(results: list[ScanResult]) -> str:
    decisions = Counter(result.decision for result in results)
    candidates = Counter(result.candidate_type for result in results if result.candidate_type)
    wait_reasons = Counter(
        result.wait_reason_top for result in results if result.decision == FinalDecision.WAIT.value
    )

    lines = ["# 스캔 결과", "", "## 요약 테이블", ""]
    lines.append("| Ticker | Decision | Candidate Type | WAIT Reason |")
    lines.append("| --- | --- | --- | --- |")
    for result in results:
        lines.append(
            f"| {result.ticker} | {result.decision} | {result.candidate_type or ''} | {result.wait_reason_top or ''} |"
        )

    lines.extend(["", "## 통계", ""])
    lines.append("### Decision 분포")
    for decision in [FinalDecision.APPROVE.value, FinalDecision.WAIT.value, FinalDecision.REJECT.value]:
        lines.append(f"- {decision}: {decisions.get(decision, 0)}")

    lines.extend(["", "### Candidate Type 분포"])
    if candidates:
        for candidate, count in candidates.most_common():
            lines.append(f"- {candidate}: {count}")
    else:
        lines.append("- 후보 유형 없음")

    lines.extend(["", "### WAIT 사유 Top 5"])
    if wait_reasons:
        for reason, count in wait_reasons.most_common(5):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- WAIT 사유 없음")

    if decisions.get(FinalDecision.APPROVE.value, 0) == 0:
        lines.extend(["", "⚠️ APPROVE가 0개입니다. 진입 트리거가 과도할 가능성이 있습니다."])

    return "\n".join(lines) + "\n"


def write_markdown(path: str, results: list[ScanResult]) -> None:
    content = format_markdown(results)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    tickers = load_tickers(args.tickers)
    if not tickers:
        raise SystemExit("Ticker 목록이 비어 있습니다.")

    engine = build_engine()
    constraints = PortfolioConstraints(max_position_pct=0.08, tranche_count=3, max_risk_pct=0.02)
    market_regime = None

    results: list[ScanResult] = []
    for ticker in tickers:
        try:
            result = evaluate_ticker(
                engine,
                ticker,
                args.mode,
                constraints,
                market_regime,
                args.use_adjusted_close,
            )
        except Exception as exc:  # noqa: BLE001 - continue scanning
            result = ScanResult(
                ticker=ticker,
                decision=FinalDecision.WAIT.value,
                candidate_type=None,
                wait_reason_top=f"스캔 오류로 WAIT 처리: {exc}",
            )
        results.append(result)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"scan_{timestamp}.csv")
    md_path = os.path.join(results_dir, f"scan_{timestamp}.md")
    write_csv(csv_path, results)
    write_markdown(md_path, results)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved Markdown: {md_path}")


if __name__ == "__main__":
    main()
