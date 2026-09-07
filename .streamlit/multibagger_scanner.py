#!/usr/bin/env python3
"""PROP DESK MULTIBAGGER SCANNER (Accumulation Breakout + Candle Patterns)"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

try:
    import ta
    HAS_TA = True
except ImportError:
    HAS_TA = False

MIN_BARS = 120
CONSOL_LOOKBACK = 20
VOL_MA = 20
BREAKOUT_VOL_MULT = 1.3
MAX_CONSOL_RANGE_PCT = 12.0
MIN_CLOSE_LOC = 0.65
RISK_CAP_PCT = 12.0

NIFTY50_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS",
    "SUNPHARMA.NS", "BAJFINANCE.NS", "WIPRO.NS", "ULTRACEMCO.NS", "NESTLEIND.NS",
    "POWERGRID.NS", "NTPC.NS", "HCLTECH.NS", "M&M.NS", "TECHM.NS",
    "ADANIENT.NS", "ADANIPORTS.NS", "ONGC.NS", "COALINDIA.NS", "JSWSTEEL.NS",
    "TATASTEEL.NS", "INDUSINDBK.NS", "BAJAJFINSV.NS", "HDFCLIFE.NS", "SBILIFE.NS",
    "GRASIM.NS", "DIVISLAB.NS", "CIPLA.NS", "DRREDDY.NS", "EICHERMOT.NS",
    "HEROMOTOCO.NS", "BRITANNIA.NS", "APOLLOHOSP.NS", "BPCL.NS", "IOC.NS",
    "SHREECEM.NS", "PIDILITIND.NS", "DABUR.NS", "HAVELLS.NS", "GODREJCP.NS",
]


def _body(o, h, l, c):
    return abs(c - o)


def _range(h, l):
    return h - l + 1e-9


def detect_candles(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    body = _body(o, h, l, c)
    rng = _range(h, l)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l

    df["HAMMER"] = (lower > 2 * body) & (upper < 0.3 * body) & (c > o)
    df["INV_HAMMER"] = (upper > 2 * body) & (lower < 0.3 * body) & (c > o)
    df["SHOOTING_STAR"] = (upper > 2 * body) & (lower < 0.3 * body) & (c < o)
    df["DOJI"] = body / rng < 0.1
    df["MARUBOZU_BULL"] = (body / rng > 0.85) & (c > o) & (upper < 0.05 * rng) & (lower < 0.05 * rng)
    df["MARUBOZU_BEAR"] = (body / rng > 0.85) & (c < o) & (upper < 0.05 * rng) & (lower < 0.05 * rng)

    prev_o, prev_c = o.shift(1), c.shift(1)
    prev_body = _body(prev_o, h.shift(1), l.shift(1), prev_c)
    df["ENGULF_BULL"] = (c > o) & (prev_c < prev_o) & (c >= prev_o) & (o <= prev_c) & (body > prev_body)
    df["ENGULF_BEAR"] = (c < o) & (prev_c > prev_o) & (c <= prev_o) & (o >= prev_c) & (body > prev_body)

    df["MORNING_STAR"] = (
        (c.shift(2) < o.shift(2))
        & (body.shift(1) / rng.shift(1) < 0.3)
        & (c > o) & (c > (o.shift(2) + c.shift(2)) / 2)
    )
    df["EVENING_STAR"] = (
        (c.shift(2) > o.shift(2))
        & (body.shift(1) / rng.shift(1) < 0.3)
        & (c < o) & (c < (o.shift(2) + c.shift(2)) / 2)
    )

    bull_cols = ["HAMMER", "INV_HAMMER", "MARUBOZU_BULL", "ENGULF_BULL", "MORNING_STAR"]
    bear_cols = ["SHOOTING_STAR", "MARUBOZU_BEAR", "ENGULF_BEAR", "EVENING_STAR"]
    df["BULL_PATTERN"] = df[bull_cols].any(axis=1)
    df["BEAR_PATTERN"] = df[bear_cols].any(axis=1)
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("Date").reset_index(drop=True)
    df["RET"] = df["Close"].pct_change()
    df["RANGE_PCT"] = (df["High"] - df["Low"]) / df["Close"] * 100
    df["VOL_MA20"] = df["Volume"].rolling(VOL_MA).mean()
    df["VOL_RATIO20"] = df["Volume"] / (df["VOL_MA20"] + 1e-9)
    if "Value" in df.columns:
        df["VALUE_MA20"] = df["Value"].rolling(VOL_MA).mean()
        df["VALUE_RATIO20"] = df["Value"] / (df["VALUE_MA20"] + 1e-9)
    else:
        df["VALUE_RATIO20"] = df["VOL_RATIO20"]

    df["CLOSE_LOC"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"] + 1e-9)
    win = min(252, len(df) - 1)
    df["HIGH_52"] = df["High"].rolling(win).max()
    df["LOW_52"] = df["Low"].rolling(win).min()
    df["POS52"] = (df["Close"] - df["LOW_52"]) / (df["HIGH_52"] - df["LOW_52"] + 1e-9) * 100
    df["RET20"] = df["Close"].pct_change(20)
    df["REL_STRONG20"] = df["RET20"] > 0
    df["HIGH_N"] = df["High"].rolling(CONSOL_LOOKBACK).max()
    df["LOW_N"] = df["Low"].rolling(CONSOL_LOOKBACK).min()
    df["CONSOL_RANGE_PCT"] = (df["HIGH_N"] - df["LOW_N"]) / df["Close"] * 100
    df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).cumsum()
    df["OBV_MA"] = df["OBV"].rolling(20).mean()
    df["OBV_RISING"] = df["OBV"] > df["OBV_MA"]

    if HAS_TA:
        df["ATR14"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], 14)
    else:
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift()).abs(),
            (df["Low"] - df["Close"].shift()).abs(),
        ], axis=1).max(axis=1)
        df["ATR14"] = tr.rolling(14).mean()

    return detect_candles(df)


def detect_accum_breakout(df: pd.DataFrame) -> pd.Series:
    prior_high = df["HIGH_N"].shift(1)
    tight = df["CONSOL_RANGE_PCT"].shift(1) < MAX_CONSOL_RANGE_PCT
    vol_ok = df["VOL_RATIO20"] > BREAKOUT_VOL_MULT
    accum = df["OBV_RISING"]
    breakout = df["Close"] > prior_high
    strong_close = df["CLOSE_LOC"] >= MIN_CLOSE_LOC
    bull_candle = df["BULL_PATTERN"] | (df["CLOSE_LOC"] > 0.8)
    signal = tight & vol_ok & accum & breakout & strong_close & bull_candle
    return signal.fillna(False)


def backtest_signal(df: pd.DataFrame, signal: pd.Series) -> Dict[str, float]:
    if signal.sum() < 3:
        return {
            "hit20_90": np.nan, "hit30_120": np.nan, "hit50_180": np.nan,
            "pf90": np.nan, "avg_ret_90": np.nan, "n_signals": int(signal.sum()),
        }

    closes = df["Close"].values
    idxs = np.where(signal.values)[0]
    rets_90, rets_120, rets_180 = [], [], []

    for i in idxs:
        if i + 90 < len(closes):
            rets_90.append(closes[i + 90] / closes[i] - 1)
        if i + 120 < len(closes):
            rets_120.append(closes[i + 120] / closes[i] - 1)
        if i + 180 < len(closes):
            rets_180.append(closes[i + 180] / closes[i] - 1)

    def hit_rate(rets, thresh):
        if not rets:
            return np.nan
        return 100.0 * np.mean(np.array(rets) >= thresh)

    def pf(rets):
        if not rets:
            return np.nan
        wins = [r for r in rets if r > 0]
        losses = [abs(r) for r in rets if r <= 0]
        if not losses:
            return 9.99
        return sum(wins) / (sum(losses) + 1e-9)

    return {
        "hit20_90": hit_rate(rets_90, 0.20),
        "hit30_120": hit_rate(rets_120, 0.30),
        "hit50_180": hit_rate(rets_180, 0.50),
        "pf90": pf(rets_90),
        "avg_ret_90": np.mean(rets_90) * 100 if rets_90 else np.nan,
        "n_signals": len(idxs),
  }
