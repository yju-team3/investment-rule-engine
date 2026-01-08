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
    StockSnapshot,
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
    block_stage: str
    key_metrics: str
    stock: StockSnapshot | None


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


EXCLUDED_REASON_PHRASES = (
    "이벤트 리스크 없음",
    "기준 통과",
    "정합성 통과",
    "유동성 기준 통과",
    "비즈니스 구조 명확",
)


def summarize_wait_reason(reason_log: Iterable[str]) -> str:
    keywords = (
        "보류",
        "불충족",
        "필요",
        "결정할 수 없음",
        "실패",
        "데이터",
        "없음",
    )
    candidates = [
        reason
        for reason in reason_log
        if any(keyword in reason for keyword in keywords)
        and not any(phrase in reason for phrase in EXCLUDED_REASON_PHRASES)
    ]
    if candidates:
        return candidates[-1]
    return "(no blocking reason detected)"


def infer_block_stage(reason_log: Iterable[str], decision: str) -> str:
    if decision == FinalDecision.APPROVE.value:
        return "NONE"

    filtered_reasons = [
        reason for reason in reason_log if not any(phrase in reason for phrase in EXCLUDED_REASON_PHRASES)
    ]

    data_keywords = ("데이터", "라이브 데이터", "수집 실패", "지표 산출", "오류")
    candidate_keywords = ("후보 유형", "결정할 수 없음", "충돌")
    entry_keywords = (
        "지지/거래량 확인 필요",
        "반등 구조 확인 필요",
        "장기 추세 회복 확인 필요",
        "진입 조건",
        "거래량/이동평균 조건 재확인",
        "1차 진입 조건 미충족",
    )
    hard_gate_keywords = (
        "유동성",
        "변동성",
        "레짐",
        "이벤트",
        "비즈니스",
        "사업 구조",
        "규제",
        "실적",
        "거절",
    )

    if any(keyword in reason for reason in filtered_reasons for keyword in data_keywords):
        return "DATA"
    if any(keyword in reason for reason in filtered_reasons for keyword in candidate_keywords):
        return "CANDIDATE"
    if any(keyword in reason for reason in filtered_reasons for keyword in entry_keywords):
        return "ENTRY_TRIGGER"
    if any(keyword in reason for reason in filtered_reasons for keyword in hard_gate_keywords):
        return "HARD_GATE"
    return "HARD_GATE"


def format_key_metrics(stock: StockSnapshot | None) -> str:
    if stock is None:
        return ""
    price_to_ma200 = stock.price / stock.ma_200 if stock.ma_200 else None
    volume_ratio = stock.volume / stock.avg_volume if stock.avg_volume else None
    ma50_distance = abs(stock.price - stock.ma_50) / stock.ma_50 if stock.ma_50 else None
    metrics = {
        "price_to_ma200": price_to_ma200,
        "volatility_annual": stock.volatility_annual,
        "drawdown_6m": stock.drawdown_6m,
        "volume_ratio": volume_ratio,
        "ma50_distance": ma50_distance,
    }
    parts = []
    for key, value in metrics.items():
        if value is None:
            continue
        parts.append(f"{key}={value:.4f}")
    return ", ".join(parts)


def evaluate_entry_trigger_conditions(result: ScanResult) -> dict[str, bool] | None:
    if result.stock is None or not result.candidate_type:
        return None

    stock = result.stock
    candidate_type = result.candidate_type

    if candidate_type == "TREND_PULLBACK":
        volume_threshold = 1.2
        volume_pass = stock.volume >= stock.avg_volume * volume_threshold
        ma_pass = stock.price > stock.ma_50
        volatility_pass = stock.volatility_annual <= 0.45
    elif candidate_type == "MEAN_REVERSION":
        volume_threshold = 1.3
        volume_pass = stock.volume >= stock.avg_volume * volume_threshold
        ma_pass = stock.price > stock.ma_50
        volatility_pass = stock.volatility_annual <= 0.45
    elif candidate_type == "DEFENSIVE_INCOME":
        volume_threshold = 1.0
        volume_pass = stock.volume >= stock.avg_volume * volume_threshold
        ma_pass = stock.price > stock.ma_200
        volatility_pass = stock.volatility_annual <= 0.25
    else:
        return None

    return {
        "거래량 조건": volume_pass,
        "이동평균 조건": ma_pass,
        "변동성 조건": volatility_pass,
    }


def analyze_entry_trigger_waits(
    entry_trigger_waits: list[ScanResult],
) -> tuple[dict[str, Counter], list[tuple[str, int]], dict[str, int], list[str]]:
    trigger_counts: dict[str, Counter] = {
        "거래량 조건": Counter(),
        "이동평균 조건": Counter(),
        "변동성 조건": Counter(),
    }
    fail_counts = Counter()
    simulations = {
        "거래량 조건": 0,
        "이동평균 조건": 0,
        "변동성 조건": 0,
    }

    for result in entry_trigger_waits:
        evaluation = evaluate_entry_trigger_conditions(result)
        if evaluation is None:
            continue
        volume_pass = evaluation["거래량 조건"]
        ma_pass = evaluation["이동평균 조건"]
        volatility_pass = evaluation["변동성 조건"]
        for trigger_name, passed in evaluation.items():
            trigger_counts[trigger_name]["PASS" if passed else "FAIL"] += 1
            if not passed:
                fail_counts[trigger_name] += 1
        if not volume_pass and ma_pass and volatility_pass:
            simulations["거래량 조건"] += 1
        if volume_pass and not ma_pass and volatility_pass:
            simulations["이동평균 조건"] += 1
        if volume_pass and ma_pass and not volatility_pass:
            simulations["변동성 조건"] += 1

    top_fails = fail_counts.most_common(3)
    notes = [
        "TREND_PULLBACK: 거래량>=1.2배, 가격>MA50, 변동성<=0.45.",
        "MEAN_REVERSION: 거래량>=1.3배, 가격>MA50, 변동성<=0.45.",
        "DEFENSIVE_INCOME: 거래량>=1.0배, 가격>MA200, 변동성<=0.25.",
        "시뮬레이션은 해당 트리거 1개만 FAIL인 경우에 한해 APPROVE 전환 가능 수로 집계.",
    ]
    return trigger_counts, top_fails, simulations, notes


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
    block_stage = infer_block_stage(report.reason_log, report.decision.value)
    key_metrics = format_key_metrics(stock)

    return ScanResult(
        ticker=ticker,
        decision=report.decision.value,
        candidate_type=candidate_type,
        wait_reason_top=wait_reason_top,
        block_stage=block_stage,
        key_metrics=key_metrics,
        stock=stock,
    )


def write_csv(path: str, results: list[ScanResult]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "ticker",
                "decision",
                "candidate_type",
                "wait_reason_top",
                "block_stage",
                "key_metrics",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "ticker": result.ticker,
                    "decision": result.decision,
                    "candidate_type": result.candidate_type or "",
                    "wait_reason_top": result.wait_reason_top or "",
                    "block_stage": result.block_stage,
                    "key_metrics": result.key_metrics,
                }
            )


def format_markdown(results: list[ScanResult]) -> str:
    decisions = Counter(result.decision for result in results)
    candidates = Counter(result.candidate_type for result in results if result.candidate_type)
    wait_reasons = Counter(
        result.wait_reason_top for result in results if result.decision == FinalDecision.WAIT.value
    )
    block_stages = Counter(result.block_stage for result in results)
    entry_trigger_waits = [
        result
        for result in results
        if result.decision == FinalDecision.WAIT.value and result.block_stage == "ENTRY_TRIGGER"
    ]

    lines = ["# 스캔 결과", "", "## 요약 테이블", ""]
    lines.append("| Ticker | Decision | Candidate Type | WAIT Reason | Block Stage | Key Metrics |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for result in results:
        lines.append(
            "| "
            f"{result.ticker} | {result.decision} | {result.candidate_type or ''} | "
            f"{result.wait_reason_top or ''} | {result.block_stage} | {result.key_metrics} |"
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

    lines.extend(["", "### Block Stage 분포"])
    if block_stages:
        for stage, count in block_stages.most_common():
            lines.append(f"- {stage}: {count}")
    else:
        lines.append("- Block Stage 없음")

    if decisions.get(FinalDecision.APPROVE.value, 0) == 0:
        lines.extend(["", "⚠️ APPROVE가 0개입니다. 진입 트리거가 과도할 가능성이 있습니다."])

    lines.extend(["", "### ENTRY_TRIGGER WAIT 분석"])
    if entry_trigger_waits:
        trigger_counts, trigger_fails, trigger_simulations, trigger_notes = analyze_entry_trigger_waits(
            entry_trigger_waits
        )
        lines.extend(["", "#### 트리거 조건별 PASS/FAIL"])
        for trigger_name, counts in trigger_counts.items():
            lines.append(f"- {trigger_name}: PASS {counts['PASS']} / FAIL {counts['FAIL']}")

        lines.extend(["", "#### FAIL 빈도 Top 3"])
        for trigger, count in trigger_fails:
            lines.append(f"- {trigger}: {count}")

        lines.extend(["", "#### 트리거 완화 시 APPROVE 전환 시뮬레이션"])
        for trigger_name, count in trigger_simulations.items():
            lines.append(f"- {trigger_name} 완화 시 전환 예상: {count}")

        lines.extend(["", "#### 트리거 판정 기준"])
        lines.extend([f"- {note}" for note in trigger_notes])
    else:
        lines.append("- ENTRY_TRIGGER에서 WAIT된 종목이 없습니다.")

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
                block_stage="DATA",
                key_metrics="",
                stock=None,
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
