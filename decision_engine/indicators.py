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
    if use_adjusted_close and "Adj Close" in getattr(data, "columns", []):
        return "Adj Close"
    if "Close" in getattr(data, "columns", []):
        return "Close"
    if "Adj Close" in getattr(data, "columns", []):
        return "Adj Close"
    return None


def _calculate_drawdown(prices: Any, window: int = 126) -> Optional[float]:
    # prices: pandas Series expected
    if prices is None or len(prices) < window:
        return None
    window_prices = prices.iloc[-window:]
    peak = window_prices.max()
    if peak is None or peak == 0:
        return None
    return float(prices.iloc[-1] / peak - 1)


def _to_series(x: Any):
    """
    yfinance/pandas sometimes return a 1-col DataFrame (or weird slices).
    Convert that into a Series safely.
    """
    import pandas as pd

    if x is None:
        return None

    if isinstance(x, pd.Series):
        return x

    if isinstance(x, pd.DataFrame):
        # squeeze only works reliably for 1-col DF
        try:
            s = x.squeeze("columns")
        except Exception:
            s = x
        if isinstance(s, pd.Series):
            return s
        # still DataFrame -> take first column
        if hasattr(s, "iloc") and s.shape[1] >= 1:
            return s.iloc[:, 0]

    return None


def _is_missing_bool(v: Any) -> bool:
    """
    Convert pd.Series/pd.DataFrame any()-style results into a plain bool.
    """
    import pandas as pd

    if v is None:
        return True

    # if it's already bool-like
    if isinstance(v, (bool, int)):
        return bool(v)

    # Series -> reduce
    if isinstance(v, pd.Series):
        return bool(v.any())

    # DataFrame -> reduce twice
    if isinstance(v, pd.DataFrame):
        return bool(v.any().any())

    # fallback: try to coerce
    try:
        return bool(v)
    except Exception:
        return True


def build_indicators(
    data: Any,
    use_adjusted_close: bool = False,
) -> Optional[IndicatorSnapshot]:
    import pandas as pd

    # Basic column validation
    price_column = _select_price_column(data, use_adjusted_close)
    if price_column is None:
        return None
    if not hasattr(data, "columns") or "Volume" not in data.columns:
        return None

    # Extract and force Series
    prices = _to_series(data[price_column])
    volumes = _to_series(data["Volume"])

    if prices is None or volumes is None:
        return None

    # Clean NaNs
    prices = prices.dropna()
    volumes = volumes.dropna()

    # Need enough history
    if len(prices) < 200 or len(volumes) < 20:
        return None

    # Use recent windows for "completeness" checks
    price_window = prices.iloc[-200:]
    volume_window = volumes.iloc[-20:]

    missing_price = price_window.isna().any()
    missing_volume = volume_window.isna().any()

    if _is_missing_bool(missing_price) or _is_missing_bool(missing_volume):
        return None

    # Moving averages
    ma_20 = prices.rolling(window=20).mean().iloc[-1]
    ma_50 = prices.rolling(window=50).mean().iloc[-1]
    ma_60 = prices.rolling(window=60).mean().iloc[-1]
    ma_100 = prices.rolling(window=100).mean().iloc[-1]
    ma_200 = prices.rolling(window=200).mean().iloc[-1]

    # Volatility (20d std of pct_change)
    returns = prices.pct_change()
    volatility_20d = returns.rolling(window=20).std().iloc[-1]

    # Volume features
    volume_avg_20d = volumes.rolling(window=20).mean().iloc[-1]
    latest_volume = volumes.iloc[-1]

    # IMPORTANT: do not do `if volume_avg_20d` on pandas scalars
    if pd.isna(volume_avg_20d) or float(volume_avg_20d) == 0.0:
        volume_change_ratio = None
    else:
        volume_change_ratio = float(latest_volume) / float(volume_avg_20d)

    # Drawdown (6 months ~ 126 trading days)
    drawdown_6m = _calculate_drawdown(prices)

    # Final completeness check (must be all real numbers; allow ratio None only if avg==0)
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
    ]

    if any(pd.isna(v) for v in values):
        return None

    # If ratio still None here, treat as incomplete (keep strict)
    if volume_change_ratio is None:
        return None

    return IndicatorSnapshot(
        price_column=str(price_column),
        latest_price=float(prices.iloc[-1]),
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
