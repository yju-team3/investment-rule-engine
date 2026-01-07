from __future__ import annotations

from typing import Iterable, List

from decision_engine.models import (
    CandidateType,
    DecisionReport,
    EntryDecision,
    FinalDecision,
    GateDecision,
    MarketSnapshot,
    PortfolioConstraints,
    StockSnapshot,
)
from decision_engine.rules import (
    Classifier,
    EntryEvaluator,
    PositionSizer,
    RegimeRule,
    RuleResult,
)


class DecisionEngine:
    def __init__(
        self,
        regime_rule: RegimeRule,
        gates: Iterable,
        classifier: Classifier,
        entry_evaluator: EntryEvaluator,
        position_sizer: PositionSizer,
    ) -> None:
        self.regime_rule = regime_rule
        self.gates = list(gates)
        self.classifier = classifier
        self.entry_evaluator = entry_evaluator
        self.position_sizer = position_sizer

    def evaluate(
        self,
        market: MarketSnapshot,
        stock: StockSnapshot,
        constraints: PortfolioConstraints,
    ) -> DecisionReport:
        reason_log: List[str] = []
        action_plan: List[str] = []

        regime, regime_result = self.regime_rule.evaluate(market)
        reason_log.append(regime_result.message)

        gate_decision = GateDecision.PASS
        for gate in self.gates:
            result = gate.evaluate(market, stock, regime)
            reason_log.append(result.message)
            if result.decision == GateDecision.REJECT:
                gate_decision = GateDecision.REJECT
                break
            if result.decision == GateDecision.WAIT:
                gate_decision = GateDecision.WAIT

        if gate_decision == GateDecision.REJECT:
            return DecisionReport(FinalDecision.REJECT, reason_log, ["신규 매수 금지."])
        if gate_decision == GateDecision.WAIT:
            return DecisionReport(FinalDecision.WAIT, reason_log, ["조건 개선 시 재평가."])

        classification = self.classifier.classify(stock, regime)
        reason_log.extend(classification.messages)

        if classification.candidate_type is None:
            return DecisionReport(FinalDecision.WAIT, reason_log, ["후보 유형 확정 후 재평가."])

        entry_result = self.entry_evaluator.evaluate(classification.candidate_type, stock)
        reason_log.extend(entry_result.messages)

        position_plan = self.position_sizer.size(stock, constraints)
        action_plan.extend(position_plan.messages)

        action_plan.extend(self._build_action_plan(classification.candidate_type, entry_result.decision, position_plan))

        final_decision = self._final_decision(entry_result.decision)
        return DecisionReport(final_decision, reason_log, action_plan)

    def _final_decision(self, entry_decision: EntryDecision) -> FinalDecision:
        if entry_decision == EntryDecision.ENTRY_ALLOWED:
            return FinalDecision.APPROVE
        if entry_decision == EntryDecision.WAIT_FOR_CONFIRMATION:
            return FinalDecision.WAIT
        return FinalDecision.REJECT

    def _build_action_plan(
        self,
        candidate_type: CandidateType,
        entry_decision: EntryDecision,
        position_plan,
    ) -> List[str]:
        plan: List[str] = []
        plan.append(f"후보 유형: {candidate_type.value}.")
        if entry_decision == EntryDecision.ENTRY_ALLOWED:
            plan.append("1차 진입 조건 충족 시 1트랜치 매수.")
            plan.append("추가 진입은 동일 조건 재확인 후 분할 매수.")
        elif entry_decision == EntryDecision.WAIT_FOR_CONFIRMATION:
            plan.append("1차 진입 조건 미충족으로 확인 전까지 대기.")
            plan.append("거래량/이동평균 조건 재확인 후 진입.")
        else:
            plan.append("진입 조건이 충족되지 않아 신규 매수 중단.")
        plan.append(f"비중 상한: {position_plan.max_position_pct:.2%}.")
        plan.append("무효화 조건: 변동성 급등 또는 레짐 악화 시 신규 매수 중단.")
        plan.append("금지 사항: 단일 지표 기반 매수, 감정적 판단.")
        return plan
