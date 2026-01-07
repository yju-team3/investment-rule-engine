from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict

from decision_engine.data_sources import yfinance_source
from decision_engine.engine import DecisionEngine
from decision_engine.indicators import build_indicators
from decision_engine.models import (
    DecisionReport,
    FinalDecision,
    MarketRegime,
    MarketSnapshot,
    PortfolioConstraints,
    StockSnapshot,
)
from decision_engine.rules import (
    BusinessClarityGate,
    Classifier,
    DefensiveIncomeEntryRule,
    DefensiveIncomeRule,
    EntryEvaluator,
    EventRiskGate,
    LiquidityGate,
    MeanReversionEntryRule,
    MeanReversionRule,
    PositionSizer,
    RegimeMismatchGate,
    RegimeRule,
    TrendPullbackEntryRule,
    TrendPullbackRule,
    VolatilityGate,
)


def build_engine() -> DecisionEngine:
    regime_rule = RegimeRule()
    gates = [
        LiquidityGate(),
        VolatilityGate(),
        RegimeMismatchGate(),
        EventRiskGate(),
        BusinessClarityGate(),
    ]
    classifier = Classifier([TrendPullbackRule(), MeanReversionRule(), DefensiveIncomeRule()])
    entry_evaluator = EntryEvaluator(
        {
            TrendPullbackRule().candidate_type(): TrendPullbackEntryRule(),
            MeanReversionRule().candidate_type(): MeanReversionEntryRule(),
            DefensiveIncomeRule().candidate_type(): DefensiveIncomeEntryRule(),
        }
    )
    position_sizer = PositionSizer()
    return DecisionEngine(regime_rule, gates, classifier, entry_evaluator, position_sizer)


def build_market_snapshot(regime: MarketRegime | None) -> MarketSnapshot:
    if regime == MarketRegime.RISK_ON:
        return MarketSnapshot(index_price=4200, index_ma_200=4000, vix=18, rate_trend_up=True)
    if regime == MarketRegime.RISK_OFF:
        return MarketSnapshot(index_price=3800, index_ma_200=4000, vix=28, rate_trend_up=False)
    if regime == MarketRegime.NEUTRAL:
        return MarketSnapshot(index_price=4050, index_ma_200=4000, vix=22, rate_trend_up=True)
    return MarketSnapshot(index_price=4200, index_ma_200=4000, vix=18, rate_trend_up=True)


def sample_stock_for_ticker(ticker: str) -> StockSnapshot:
    normalized = ticker.upper()
    samples = {
        "PG": StockSnapshot(
            ticker=normalized,
            price=150,
            avg_volume=4200000,
            volume=4800000,
            volatility_annual=0.18,
            ma_50=148,
            ma_200=140,
            drawdown_6m=-0.08,
            dividend_yield=0.035,
            earnings_risk=False,
            regulatory_risk=False,
            business_clarity=True,
            sector_defensive=True,
        ),
        "TSLA": StockSnapshot(
            ticker=normalized,
            price=220,
            avg_volume=8000000,
            volume=9000000,
            volatility_annual=0.6,
            ma_50=240,
            ma_200=260,
            drawdown_6m=-0.4,
            dividend_yield=0.0,
            earnings_risk=True,
            regulatory_risk=False,
            business_clarity=True,
            sector_defensive=False,
        ),
        "MSFT": StockSnapshot(
            ticker=normalized,
            price=410,
            avg_volume=3000000,
            volume=3200000,
            volatility_annual=0.22,
            ma_50=405,
            ma_200=390,
            drawdown_6m=-0.12,
            dividend_yield=0.008,
            earnings_risk=False,
            regulatory_risk=False,
            business_clarity=True,
            sector_defensive=False,
        ),
    }
    return samples.get(
        normalized,
        StockSnapshot(
            ticker=normalized,
            price=52,
            avg_volume=500000,
            volume=600000,
            volatility_annual=0.28,
            ma_50=50,
            ma_200=45,
            drawdown_6m=-0.12,
            dividend_yield=0.01,
            earnings_risk=False,
            regulatory_risk=False,
            business_clarity=True,
            sector_defensive=False,
        ),
    )


def build_live_stock_snapshot(
    ticker: str,
    use_adjusted_close: bool = False,
) -> tuple[StockSnapshot | None, str | None]:
    data = yfinance_source.fetch_ohlcv(ticker)
    if data is None:
        return None, "라이브 데이터 수집 실패 또는 데이터가 없어 WAIT 처리."
    indicators = build_indicators(data, use_adjusted_close=use_adjusted_close)
    if indicators is None:
        return None, "라이브 데이터 지표 산출에 필요한 데이터가 부족하여 WAIT 처리."

    volatility_annual = indicators.volatility_20d * math.sqrt(252)
    stock = StockSnapshot(
        ticker=ticker.upper(),
        price=indicators.latest_price,
        avg_volume=indicators.volume_avg_20d,
        volume=indicators.latest_volume,
        volatility_annual=volatility_annual,
        ma_50=indicators.ma_50,
        ma_200=indicators.ma_200,
        drawdown_6m=indicators.drawdown_6m,
        dividend_yield=0.0,
        earnings_risk=False,
        regulatory_risk=False,
        business_clarity=True,
        sector_defensive=False,
    )
    return stock, None


def print_report(title: str, report) -> None:
    print("=" * 60)
    print(title)
    print("(1) Decision")
    print(report.decision.value)
    print("(2) Reason Log")
    for item in report.reason_log:
        print(f"- {item}")
    print("(3) Action Plan")
    for item in report.action_plan:
        print(f"- {item}")


def report_to_json(report, ticker: str, market_regime: MarketRegime | None) -> str:
    payload = asdict(report)
    payload["decision"] = report.decision.value
    payload["ticker"] = ticker
    payload["market_regime_override"] = market_regime.value if market_regime else None
    return json.dumps(payload, ensure_ascii=False, indent=2)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decision engine CLI")
    parser.add_argument("--ticker", required=True, help="Ticker symbol to evaluate")
    parser.add_argument(
        "--market-regime",
        choices=[regime.value for regime in MarketRegime],
        help="Override market regime classification",
    )
    parser.add_argument(
        "--mode",
        choices=["sample", "live"],
        default="sample",
        help="Use sample data or live data",
    )
    parser.add_argument(
        "--use-adjusted-close",
        action="store_true",
        help="Use adjusted close price when available",
    )
    parser.add_argument("--json", action="store_true", help="Also output JSON result")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    regime_override = MarketRegime(args.market_regime) if args.market_regime else None
    engine = build_engine()
    market = build_market_snapshot(regime_override)
    constraints = PortfolioConstraints(max_position_pct=0.08, tranche_count=3, max_risk_pct=0.02)
    ticker = args.ticker.upper()
    if args.mode == "live":
        stock, reason = build_live_stock_snapshot(ticker, use_adjusted_close=args.use_adjusted_close)
        if stock is None:
            report = DecisionReport(
                FinalDecision.WAIT,
                [reason or "라이브 데이터가 불완전하여 WAIT 처리."],
                ["데이터 보완 후 재평가."],
            )
            print_report(f"라이브 종목 {ticker}", report)
        else:
            report = engine.evaluate(market, stock, constraints)
            print_report(f"라이브 종목 {stock.ticker}", report)
    else:
        stock = sample_stock_for_ticker(ticker)
        report = engine.evaluate(market, stock, constraints)
        print_report(f"샘플 종목 {stock.ticker}", report)
    if args.json:
        print("(4) JSON")
        print(report_to_json(report, ticker, regime_override))


if __name__ == "__main__":
    main()
