import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

POLY_API_KEY = os.getenv("POLYGON_API_KEY")
BASE_URL = "https://api.polygon.io"


# ------------------------
# RÉCUPÉRATION DES DONNÉES
# ------------------------

def get_all_snapshots():
    """
    Récupère le snapshot de tous les stocks US.
    Contient dernière transaction, volume du jour, variation, etc.
    """
    url = f"{BASE_URL}/v2/snapshot/locale/us/markets/stocks/tickers"
    params = {"apiKey": POLY_API_KEY}
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()

    tickers = data.get("tickers", [])
    rows = []
    for t in tickers:
        symbol = t.get("ticker")
        day = t.get("day") or {}
        last_trade = t.get("lastTrade") or {}

        rows.append({
            "symbol": symbol,
            "volume": day.get("v", 0),
            "open": day.get("o", None),
            "high": day.get("h", None),
            "low": day.get("l", None),
            "close": day.get("c", None),
            "change": day.get("c", None),
            "last_price": last_trade.get("p", None),
        })

    df = pd.DataFrame(rows)
    return df


def get_intraday_agg(symbol: str,
                     multiplier: int = 5,
                     timespan: str = "minute",
                     lookback_hours: int = 6) -> pd.DataFrame:
    """
    Récupère les chandelles intraday récentes pour un symbole.
    Par défaut: 5 minutes sur ~6h.
    """
    to_ = datetime.utcnow()
    from_ = to_ - timedelta(hours=lookback_hours)

    url = f"{BASE_URL}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_.date()}/{to_.date()}"
    params = {
        "apiKey": POLY_API_KEY,
        "adjusted": "true",
        "sort": "asc",
        "limit": 5000
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()

    results = data.get("results", [])
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    # Colonnes Polygon: t (timestamp ms), o,h,l,c,v
    df["datetime"] = pd.to_datetime(df["t"], unit="ms")
    df = df.set_index("datetime").sort_index()
    df.rename(columns={"o": "open", "h": "high", "l": "low",
                       "c": "close", "v": "volume"}, inplace=True)
    return df


# ------------------------
# INDICATEURS TECHNIQUES
# ------------------------

def compute_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(length).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def wilder_smooth(series: pd.Series, length: int) -> pd.Series:
    # Approximation de la moyenne de Wilder
    return series.ewm(alpha=1 / length, adjust=False).mean()


def compute_dmi_adx(high: pd.Series, low: pd.Series, close: pd.Series,
                    di_length: int = 50, adx_length: int = 5):
    up_move = high.diff()
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = wilder_smooth(tr, di_length)
    plus_di = 100 * wilder_smooth(plus_dm, di_length) / atr.replace(0, np.nan)
    minus_di = 100 * wilder_smooth(minus_dm, di_length) / atr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = wilder_smooth(dx, adx_length)

    return plus_di, minus_di, adx


def compute_cci(high: pd.Series, low: pd.Series, close: pd.Series,
                length: int = 20) -> pd.Series:
    tp = (high + low + close) / 3.0
    sma_tp = tp.rolling(length).mean()
    mean_dev = (tp - sma_tp).abs().rolling(length).mean()
    cci = (tp - sma_tp) / (0.015 * mean_dev.replace(0, np.nan))
    return cci


def compute_macd(close: pd.Series,
                 fast: int = 5,
                 slow: int = 13,
                 signal_length: int = 4):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_length, adjust=False).mean()
    return macd_line, signal_line


def compute_vwap(high: pd.Series, low: pd.Series,
                 close: pd.Series, volume: pd.Series) -> pd.Series:
    tp = (high + low + close) / 3.0
    cum_pv = (tp * volume).cumsum()
    cum_v = volume.cumsum().replace(0, np.nan)
    vwap = cum_pv / cum_v
    return vwap


# ------------------------
# LOGIQUE DU COACH SYSTEM
# ------------------------

def check_signal(df: pd.DataFrame) -> bool:
    """
    Reproduction en Python du "coach system" TradingView.
    Retourne True si au moins 6 conditions sur 7 sont respectées.
    """
    if df.empty or len(df) < 80:
        return False

    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]

    # --- Momentum ---
    momentum_length = 80
    momentum = close - close.shift(momentum_length)

    # --- DMI / ADX ---
    di_length = 50
    adx_smoothing = 5
    plus_di, minus_di, adx = compute_dmi_adx(high, low, close,
                                             di_length=di_length,
                                             adx_length=adx_smoothing)

    # --- CCI ---
    cci_length = 20
    cci = compute_cci(high, low, close, length=cci_length)

    # --- RSI ---
    rsi_length = 5
    rsi = compute_rsi(close, length=rsi_length)
    sma_rsi = rsi.rolling(5).mean()

    # --- MACD ---
    macd_line, signal_line = compute_macd(close, fast=5, slow=13, signal_length=4)

    # --- Volume ---
    volume_length = 20
    volume_avg = vol.rolling(volume_length).mean()

    # --- VWAP ---
    vwap = compute_vwap(high, low, close, vol)

    # On prend la dernière bougie
    last_idx = df.index[-1]

    mom_last = momentum.loc[last_idx]
    rsi_last = rsi.loc[last_idx]
    sma_rsi_last = sma_rsi.loc[last_idx]
    macd_last = macd_line.loc[last_idx]
    signal_last = signal_line.loc[last_idx]
    adx_last = adx.loc[last_idx]
    plus_di_last = plus_di.loc[last_idx]
    minus_di_last = minus_di.loc[last_idx]
    vol_last = vol.loc[last_idx]
    volume_avg_last = volume_avg.loc[last_idx]
    close_last = close.loc[last_idx]
    vwap_last = vwap.loc[last_idx]
    cci_last = cci.loc[last_idx]

    # --- Condition CCI spéciale (approximation de ta.valuewhen(cci > lowest(cci,5), cci, 0) ) ---
    lowest_cci_5 = cci.rolling(5).min()
    cond_series = cci > lowest_cci_5
    if cond_series.any():
        last_cci_above_low = cci[cond_series].iloc[-1]
        cci_condition = cci_last >= last_cci_above_low
    else:
        cci_condition = False

    # --- Conditions comme dans ton script ---
    momentum_condition = mom_last > 0
    rsi_condition = rsi_last > sma_rsi_last
    macd_condition = macd_last > signal_last
    dmi_condition = (adx_last >= 20) and (plus_di_last > minus_di_last)
    volume_condition = vol_last >= volume_avg_last
    price_condition = close_last > vwap_last

    conditions = [
        momentum_condition,
        rsi_condition,
        macd_condition,
        cci_condition,
        dmi_condition,
        volume_condition,
        price_condition
    ]

    condition_count = sum(bool(c) for c in conditions)

    full_signal = condition_count == 7   # 7/7
    alert_signal = condition_count == 6  # 6/7

    # Si tu veux n'avoir que le "full", remplace par: return full_signal
    return (condition_count >= 6)


# ------------------------
# SCAN DU MARCHÉ
# ------------------------

def scan_market(min_volume: int = 1_000_000,
                max_symbols: int = 300) -> pd.DataFrame:
    """
    1. Récupère tous les stocks US via snapshot.
    2. Filtre sur volume >= min_volume.
    3. Pour les 'max_symbols' premiers, calcule un signal "coach system".
    4. Retourne un DF des symboles qui déclenchent un signal.
    """
    print("📥 Récupération du snapshot global…")
    snapshot_df = get_all_snapshots()
    print(f"Total tickers: {len(snapshot_df)}")

    # Filtre sur volume
    high_vol = snapshot_df[snapshot_df["volume"] >= min_volume].copy()
    high_vol = high_vol.sort_values("volume", ascending=False)
    print(f"Tickers avec volume >= {min_volume:,}: {len(high_vol)}")

    high_vol = high_vol.head(max_symbols)

    results = []
    for i, row in high_vol.iterrows():
        symbol = row["symbol"]
        try:
            print(f"🔍 {symbol}…", end=" ", flush=True)
            df = get_intraday_agg(symbol)
            signal = check_signal(df)
            print("✔" if signal else "–")
            if signal:
                results.append({
                    "symbol": symbol,
                    "volume": int(row["volume"]),
                    "last_price": row["last_price"],
                })
            time.sleep(0.25)  # pour ménager l'API
        except Exception as e:
            print(f"Erreur {symbol}: {e}")
            continue

    if not results:
        return pd.DataFrame(columns=["symbol", "volume", "last_price"])

    result_df = pd.DataFrame(results).sort_values("volume", ascending=False)
    return result_df


if __name__ == "__main__":
    if not POLY_API_KEY:
        raise ValueError("⚠️ POLYGON_API_KEY manquant (variable d'environnement).")

    signals_df = scan_market(min_volume=1_000_000, max_symbols=300)
    print("\n📊 Résultats – Tickers avec signal coach system (≥ 6 conditions) :")
    print(signals_df.to_string(index=False))
