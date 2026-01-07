from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class IndicatorSnapshot:
    price_column: str
    latest_price: float
    latest_volume: float
    ma_20: float
    ma_50: float
    ma_60: float
    ma_100: float
    ma_200: float
    volatility_20d: float
    volume_avg_20d: float
    volume_change_ratio: float
    drawdown_6m: float


def _select_price_column(data: Any, use_adjusted_close: bool) -> Optional[str]:
    if use_adjusted_close and "Adj Close" in data.columns:
        return "Adj Close"
    if "Close" in data.columns:
        return "Close"
    if "Adj Close" in data.columns:
        return "Adj Close"
    return None


def _calculate_drawdown(prices: Any, window: int = 126) -> Optional[float]:
    if len(prices) < window:
        return None
    window_prices = prices.iloc[-window:]
    peak = window_prices.max()
    if peak == 0:
        return None
    return prices.iloc[-1] / peak - 1


def build_indicators(
    data: Any,
    use_adjusted_close: bool = False,
) -> Optional[IndicatorSnapshot]:
    import pandas as pd
    price_column = _select_price_column(data, use_adjusted_close)
    if price_column is None or "Volume" not in data.columns:
        return None

    prices = data[price_column].dropna()
    volumes = data["Volume"].dropna()

    if len(prices) < 200 or len(volumes) < 20:
        return None

    ma_20 = prices.rolling(window=20).mean().iloc[-1]
    ma_50 = prices.rolling(window=50).mean().iloc[-1]
    ma_60 = prices.rolling(window=60).mean().iloc[-1]
    ma_100 = prices.rolling(window=100).mean().iloc[-1]
    ma_200 = prices.rolling(window=200).mean().iloc[-1]

    returns = prices.pct_change()
    volatility_20d = returns.rolling(window=20).std().iloc[-1]

    volume_avg_20d = volumes.rolling(window=20).mean().iloc[-1]
    latest_volume = volumes.iloc[-1]
    volume_change_ratio = latest_volume / volume_avg_20d if volume_avg_20d else None

    drawdown_6m = _calculate_drawdown(prices)

    values = [
        ma_20,
        ma_50,
        ma_60,
        ma_100,
        ma_200,
        volatility_20d,
        volume_avg_20d,
        latest_volume,
        drawdown_6m,
        volume_change_ratio,
    ]
    if any(pd.isna(value) for value in values):
        return None

    return IndicatorSnapshot(
        price_column=price_column,
        latest_price=prices.iloc[-1],
        latest_volume=float(latest_volume),
        ma_20=float(ma_20),
        ma_50=float(ma_50),
        ma_60=float(ma_60),
        ma_100=float(ma_100),
        ma_200=float(ma_200),
        volatility_20d=float(volatility_20d),
        volume_avg_20d=float(volume_avg_20d),
        volume_change_ratio=float(volume_change_ratio),
        drawdown_6m=float(drawdown_6m),
    )
