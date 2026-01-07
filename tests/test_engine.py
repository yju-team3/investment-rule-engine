import unittest

from decision_engine.engine import DecisionEngine
from decision_engine.models import (
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
        LiquidityGate(min_avg_volume=1000),
        VolatilityGate(max_volatility=0.5),
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


class DecisionEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = build_engine()
        self.market = MarketSnapshot(index_price=4200, index_ma_200=4000, vix=18, rate_trend_up=True)
        self.constraints = PortfolioConstraints(max_position_pct=0.08, tranche_count=2, max_risk_pct=0.02)

    def test_regime_rule_risk_on(self) -> None:
        regime, result = RegimeRule().evaluate(self.market)
        self.assertEqual(regime, MarketRegime.RISK_ON)
        self.assertTrue(result.passed)

    def test_gate_rejects_low_liquidity(self) -> None:
        stock = StockSnapshot(
            ticker="LOW",
            price=10,
            avg_volume=500,
            volume=500,
            volatility_annual=0.2,
            ma_50=9,
            ma_200=8,
            drawdown_6m=-0.1,
            dividend_yield=0.0,
            earnings_risk=False,
            regulatory_risk=False,
            business_clarity=True,
            sector_defensive=False,
        )
        report = self.engine.evaluate(self.market, stock, self.constraints)
        self.assertEqual(report.decision, FinalDecision.REJECT)

    def test_gate_waits_on_event_risk(self) -> None:
        stock = StockSnapshot(
            ticker="EVT",
            price=30,
            avg_volume=200000,
            volume=250000,
            volatility_annual=0.2,
            ma_50=28,
            ma_200=25,
            drawdown_6m=-0.1,
            dividend_yield=0.0,
            earnings_risk=True,
            regulatory_risk=False,
            business_clarity=True,
            sector_defensive=False,
        )
        report = self.engine.evaluate(self.market, stock, self.constraints)
        self.assertEqual(report.decision, FinalDecision.WAIT)

    def test_classification_trend_pullback(self) -> None:
        stock = StockSnapshot(
            ticker="TP",
            price=52,
            avg_volume=200000,
            volume=260000,
            volatility_annual=0.2,
            ma_50=50,
            ma_200=45,
            drawdown_6m=-0.12,
            dividend_yield=0.0,
            earnings_risk=False,
            regulatory_risk=False,
            business_clarity=True,
            sector_defensive=False,
        )
        report = self.engine.evaluate(self.market, stock, self.constraints)
        self.assertIn("후보 유형", " ".join(report.action_plan))

    def test_output_format_sections(self) -> None:
        stock = StockSnapshot(
            ticker="FMT",
            price=52,
            avg_volume=200000,
            volume=260000,
            volatility_annual=0.2,
            ma_50=50,
            ma_200=45,
            drawdown_6m=-0.12,
            dividend_yield=0.0,
            earnings_risk=False,
            regulatory_risk=False,
            business_clarity=True,
            sector_defensive=False,
        )
        report = self.engine.evaluate(self.market, stock, self.constraints)
        self.assertGreater(len(report.reason_log), 0)
        self.assertGreater(len(report.action_plan), 0)


if __name__ == "__main__":
    unittest.main()
