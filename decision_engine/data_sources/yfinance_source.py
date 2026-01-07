from __future__ import annotations

from typing import Any, Callable, Optional


def _default_downloader(ticker: str, period: str, interval: str) -> Any:
    import yfinance as yf

    return yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)


def fetch_ohlcv(
    ticker: str,
    years: int = 5,
    downloader: Callable[[str, str, str], Any] | None = None,
) -> Optional[Any]:
    period = f"{years}y"
    download = downloader or _default_downloader
    try:
        data = download(ticker, period, "1d")
    except Exception:
        return None
    if data is None or getattr(data, "empty", True):
        return None
    return data
