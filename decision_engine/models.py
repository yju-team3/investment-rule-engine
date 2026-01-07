from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List


class MarketRegime(str, Enum):
    RISK_ON = "RISK_ON"
    NEUTRAL = "NEUTRAL"
    RISK_OFF = "RISK_OFF"


class GateDecision(str, Enum):
    PASS = "PASS"
    WAIT = "WAIT"
    REJECT = "REJECT"


class EntryDecision(str, Enum):
    ENTRY_ALLOWED = "ENTRY_ALLOWED"
    WAIT_FOR_CONFIRMATION = "WAIT_FOR_CONFIRMATION"
    NO_ENTRY = "NO_ENTRY"


class CandidateType(str, Enum):
    TREND_PULLBACK = "TREND_PULLBACK"
    MEAN_REVERSION = "MEAN_REVERSION"
    DEFENSIVE_INCOME = "DEFENSIVE_INCOME"


class FinalDecision(str, Enum):
    APPROVE = "APPROVE"
    WAIT = "WAIT"
    REJECT = "REJECT"


@dataclass(frozen=True)
class MarketSnapshot:
    index_price: float
    index_ma_200: float
    vix: float
    rate_trend_up: bool


@dataclass(frozen=True)
class StockSnapshot:
    ticker: str
    price: float
    avg_volume: float
    volume: float
    volatility_annual: float
    ma_50: float
    ma_200: float
    drawdown_6m: float
    dividend_yield: float
    earnings_risk: bool
    regulatory_risk: bool
    business_clarity: bool
    sector_defensive: bool


@dataclass(frozen=True)
class PortfolioConstraints:
    max_position_pct: float
    tranche_count: int
    max_risk_pct: float
    target_volatility: float = 0.2


@dataclass(frozen=True)
class RuleResult:
    name: str
    passed: bool
    message: str


@dataclass(frozen=True)
class GateResult:
    name: str
    decision: GateDecision
    message: str


@dataclass(frozen=True)
class ClassificationResult:
    candidate_type: CandidateType | None
    messages: List[str]


@dataclass(frozen=True)
class EntryResult:
    decision: EntryDecision
    messages: List[str]


@dataclass(frozen=True)
class PositionPlan:
    max_position_pct: float
    tranche_pct: float
    risk_cap_pct: float
    messages: List[str]


@dataclass(frozen=True)
class DecisionReport:
    decision: FinalDecision
    reason_log: List[str]
    action_plan: List[str]
