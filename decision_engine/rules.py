from __future__ import annotations

from dataclasses import dataclass
from typing import List

from decision_engine.models import (
    CandidateType,
    ClassificationResult,
    EntryDecision,
    EntryResult,
    GateDecision,
    GateResult,
    MarketRegime,
    MarketSnapshot,
    PositionPlan,
    PortfolioConstraints,
    RuleResult,
    StockSnapshot,
)


class RegimeRule:
    name = "market_regime"

    def evaluate(self, market: MarketSnapshot) -> tuple[MarketRegime, RuleResult]:
        if market.index_price > market.index_ma_200 and market.vix < 20:
            regime = MarketRegime.RISK_ON
            message = "지수는 장기 이동평균 위이며 변동성 지표가 낮아 RISK_ON으로 분류됨."
        elif market.index_price < market.index_ma_200 and market.vix > 25:
            regime = MarketRegime.RISK_OFF
            message = "지수는 장기 이동평균 아래이고 변동성 지표가 높아 RISK_OFF으로 분류됨."
        else:
            regime = MarketRegime.NEUTRAL
            message = "지수와 변동성 지표가 혼재되어 NEUTRAL로 분류됨."
        return regime, RuleResult(self.name, True, message)


class GateRule:
    name = "gate"

    def evaluate(self, market: MarketSnapshot, stock: StockSnapshot, regime: MarketRegime) -> GateResult:
        raise NotImplementedError


@dataclass(frozen=True)
class LiquidityGate(GateRule):
    min_avg_volume: float = 200000
    name: str = "liquidity_gate"

    def evaluate(self, market: MarketSnapshot, stock: StockSnapshot, regime: MarketRegime) -> GateResult:
        if stock.avg_volume < self.min_avg_volume:
            return GateResult(
                self.name,
                GateDecision.REJECT,
                "평균 거래량이 기준치 미만이라 유동성 부족으로 즉시 거절.",
            )
        return GateResult(self.name, GateDecision.PASS, "유동성 기준 통과.")


@dataclass(frozen=True)
class VolatilityGate(GateRule):
    max_volatility: float = 0.45
    name: str = "volatility_gate"

    def evaluate(self, market: MarketSnapshot, stock: StockSnapshot, regime: MarketRegime) -> GateResult:
        if stock.volatility_annual > self.max_volatility:
            return GateResult(
                self.name,
                GateDecision.WAIT,
                "연율 변동성이 높아 과도한 변동성으로 보류.",
            )
        return GateResult(self.name, GateDecision.PASS, "변동성 기준 통과.")


@dataclass(frozen=True)
class RegimeMismatchGate(GateRule):
    name: str = "regime_mismatch_gate"

    def evaluate(self, market: MarketSnapshot, stock: StockSnapshot, regime: MarketRegime) -> GateResult:
        if regime == MarketRegime.RISK_OFF and not stock.sector_defensive:
            return GateResult(
                self.name,
                GateDecision.REJECT,
                "RISK_OFF 환경에서 방어형 섹터가 아니라 레짐 불일치로 거절.",
            )
        return GateResult(self.name, GateDecision.PASS, "레짐 정합성 통과.")


@dataclass(frozen=True)
class EventRiskGate(GateRule):
    name: str = "event_risk_gate"

    def evaluate(self, market: MarketSnapshot, stock: StockSnapshot, regime: MarketRegime) -> GateResult:
        if stock.earnings_risk or stock.regulatory_risk:
            return GateResult(
                self.name,
                GateDecision.WAIT,
                "실적/규제 이벤트 리스크가 있어 신규 진입 보류.",
            )
        return GateResult(self.name, GateDecision.PASS, "이벤트 리스크 없음.")


@dataclass(frozen=True)
class BusinessClarityGate(GateRule):
    name: str = "business_clarity_gate"

    def evaluate(self, market: MarketSnapshot, stock: StockSnapshot, regime: MarketRegime) -> GateResult:
        if not stock.business_clarity:
            return GateResult(
                self.name,
                GateDecision.REJECT,
                "사업 구조가 명확하지 않아 설명 불가능한 비즈니스로 거절.",
            )
        return GateResult(self.name, GateDecision.PASS, "비즈니스 구조 명확.")


class ClassificationRule:
    name = "classification"

    def evaluate(self, stock: StockSnapshot, regime: MarketRegime) -> RuleResult:
        raise NotImplementedError

    def candidate_type(self) -> CandidateType:
        raise NotImplementedError


@dataclass(frozen=True)
class TrendPullbackRule(ClassificationRule):
    name: str = "trend_pullback"

    def evaluate(self, stock: StockSnapshot, regime: MarketRegime) -> RuleResult:
        passed = (
            stock.price > stock.ma_200
            and stock.drawdown_6m <= -0.05
            and stock.drawdown_6m >= -0.2
        )
        message = "장기 추세 위에서 5~20% 눌림 구간." if passed else "추세 눌림 조건 불충족."
        return RuleResult(self.name, passed, message)

    def candidate_type(self) -> CandidateType:
        return CandidateType.TREND_PULLBACK


@dataclass(frozen=True)
class MeanReversionRule(ClassificationRule):
    name: str = "mean_reversion"

    def evaluate(self, stock: StockSnapshot, regime: MarketRegime) -> RuleResult:
        passed = stock.drawdown_6m <= -0.3 and stock.price < stock.ma_200
        message = "과도한 하락 구간에서 평균회귀 후보." if passed else "과도한 하락 조건 불충족."
        return RuleResult(self.name, passed, message)

    def candidate_type(self) -> CandidateType:
        return CandidateType.MEAN_REVERSION


@dataclass(frozen=True)
class DefensiveIncomeRule(ClassificationRule):
    name: str = "defensive_income"

    def evaluate(self, stock: StockSnapshot, regime: MarketRegime) -> RuleResult:
        drawdown_ok = stock.drawdown_6m >= -0.15
        volatility_ok = stock.volatility_annual <= 0.25
        price_to_ma_200 = stock.price / stock.ma_200 if stock.ma_200 else 0.0
        price_above_ma_200 = price_to_ma_200 >= 0.97
        price_not_extended = price_to_ma_200 <= 1.12
        price_distance_ma_50 = abs(stock.price - stock.ma_50) / stock.ma_50 if stock.ma_50 else 0.0
        volume_ratio = stock.volume / stock.avg_volume if stock.avg_volume else 0.0
        short_term_stable = price_distance_ma_50 <= 0.08 and volume_ratio <= 1.5

        passed = (
            drawdown_ok
            and volatility_ok
            and price_above_ma_200
            and price_not_extended
            and short_term_stable
        )
        price_band_ok = price_above_ma_200 and price_not_extended
        debug_lines = [
            f"[DEF] drawdown_6m={stock.drawdown_6m:.2f} >= -0.15 ({'PASS' if drawdown_ok else 'FAIL'})",
            f"[DEF] volatility_annual={stock.volatility_annual:.2f} <= 0.25 ({'PASS' if volatility_ok else 'FAIL'})",
            (
                f"[DEF] price_to_ma200={price_to_ma_200:.2f} within [0.97, 1.12] "
                f"({'PASS' if price_band_ok else 'FAIL'})"
            ),
            (
                "[DEF] overheat_check "
                f"(ma50_distance={price_distance_ma_50:.2f} <= 0.08, "
                f"volume_ratio={volume_ratio:.2f} <= 1.50) "
                f"({'PASS' if short_term_stable else 'FAIL'})"
            ),
        ]
        message = (
            "완만한 6개월 낙폭, 낮은 변동성, 200MA 근처 안정, 단기 과열 없음."
            if passed
            else "방어형 가격/변동성 안정 조건 불충족."
        )
        return RuleResult(self.name, passed, "\n".join(debug_lines + [message]))

    def candidate_type(self) -> CandidateType:
        return CandidateType.DEFENSIVE_INCOME


class EntryRule:
    name = "entry"

    def evaluate(self, stock: StockSnapshot) -> RuleResult:
        raise NotImplementedError

    def entry_decision(self, passed: bool) -> EntryDecision:
        raise NotImplementedError


@dataclass(frozen=True)
class TrendPullbackEntryRule(EntryRule):
    name: str = "trend_pullback_entry"

    def evaluate(self, stock: StockSnapshot) -> RuleResult:
        passed = stock.price > stock.ma_50 and stock.volume >= stock.avg_volume * 1.2
        message = "50일선 회복과 거래량 증가 확인." if passed else "지지/거래량 확인 필요."
        return RuleResult(self.name, passed, message)

    def entry_decision(self, passed: bool) -> EntryDecision:
        return EntryDecision.ENTRY_ALLOWED if passed else EntryDecision.WAIT_FOR_CONFIRMATION


@dataclass(frozen=True)
class MeanReversionEntryRule(EntryRule):
    name: str = "mean_reversion_entry"

    def evaluate(self, stock: StockSnapshot) -> RuleResult:
        passed = stock.price > stock.ma_50 and stock.volume >= stock.avg_volume * 1.3
        message = "단기 반등 신호와 거래량 급증 확인." if passed else "반등 구조 확인 필요."
        return RuleResult(self.name, passed, message)

    def entry_decision(self, passed: bool) -> EntryDecision:
        return EntryDecision.ENTRY_ALLOWED if passed else EntryDecision.WAIT_FOR_CONFIRMATION


@dataclass(frozen=True)
class DefensiveIncomeEntryRule(EntryRule):
    name: str = "defensive_income_entry"

    def evaluate(self, stock: StockSnapshot) -> RuleResult:
        passed = stock.price > stock.ma_200
        message = "장기 이동평균 위에서 방어형 유지." if passed else "장기 추세 회복 확인 필요."
        return RuleResult(self.name, passed, message)

    def entry_decision(self, passed: bool) -> EntryDecision:
        return EntryDecision.ENTRY_ALLOWED if passed else EntryDecision.WAIT_FOR_CONFIRMATION


class Classifier:
    def __init__(self, rules: List[ClassificationRule]):
        self.rules = rules

    def classify(self, stock: StockSnapshot, regime: MarketRegime) -> ClassificationResult:
        hits: List[CandidateType] = []
        messages: List[str] = []
        for rule in self.rules:
            result = rule.evaluate(stock, regime)
            message_lines = result.message.splitlines()
            messages.extend(message_lines if message_lines else [result.message])
            if result.passed:
                hits.append(rule.candidate_type())
        if len(hits) == 1:
            return ClassificationResult(hits[0], messages)
        if len(hits) > 1:
            messages.append("후보 유형이 복수로 충돌하여 보류.")
        else:
            messages.append("후보 유형을 결정할 수 없음.")
        return ClassificationResult(None, messages)


class EntryEvaluator:
    def __init__(self, rules: dict[CandidateType, EntryRule]):
        self.rules = rules

    def evaluate(self, candidate_type: CandidateType, stock: StockSnapshot) -> EntryResult:
        rule = self.rules[candidate_type]
        result = rule.evaluate(stock)
        decision = rule.entry_decision(result.passed)
        return EntryResult(decision, [result.message])


class PositionSizer:
    def size(self, stock: StockSnapshot, constraints: PortfolioConstraints) -> PositionPlan:
        volatility = max(stock.volatility_annual, 0.01)
        scaled_position = constraints.max_position_pct * (constraints.target_volatility / volatility)
        max_position_pct = min(constraints.max_position_pct, scaled_position)
        tranche_pct = max_position_pct / constraints.tranche_count
        messages = [
            "변동성에 따라 최대 비중을 축소 적용.",
            f"단일 종목 최대 비중 {max_position_pct:.2%}, 트랜치 {constraints.tranche_count}회 분할.",
        ]
        return PositionPlan(max_position_pct, tranche_pct, constraints.max_risk_pct, messages)
