from __future__ import annotations

from decision_engine.engine import DecisionEngine
from decision_engine.models import MarketSnapshot, PortfolioConstraints, StockSnapshot
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


def main() -> None:
    engine = build_engine()
    market = MarketSnapshot(index_price=4200, index_ma_200=4000, vix=18, rate_trend_up=True)
    constraints = PortfolioConstraints(max_position_pct=0.08, tranche_count=3, max_risk_pct=0.02)

    sample_a = StockSnapshot(
        ticker="ABC",
        price=52,
        avg_volume=500000,
        volume=750000,
        volatility_annual=0.28,
        ma_50=50,
        ma_200=45,
        drawdown_6m=-0.12,
        dividend_yield=0.01,
        earnings_risk=False,
        regulatory_risk=False,
        business_clarity=True,
        sector_defensive=False,
    )

    sample_b = StockSnapshot(
        ticker="DEF",
        price=28,
        avg_volume=350000,
        volume=300000,
        volatility_annual=0.22,
        ma_50=30,
        ma_200=32,
        drawdown_6m=-0.35,
        dividend_yield=0.04,
        earnings_risk=False,
        regulatory_risk=False,
        business_clarity=True,
        sector_defensive=True,
    )

    report_a = engine.evaluate(market, sample_a, constraints)
    report_b = engine.evaluate(market, sample_b, constraints)

    print_report("샘플 종목 ABC", report_a)
    print_report("샘플 종목 DEF", report_b)


if __name__ == "__main__":
    main()
