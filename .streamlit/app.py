#!/usr/bin/env python3
"""
Multibagger Scanner – Mobile-Friendly Web App
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Multibagger Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .stDataFrame { font-size: 13px; }
    h1 { font-size: 1.6rem !important; }
    h2 { font-size: 1.25rem !important; }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

import sys
sys.path.insert(0, str(Path(__file__).parent))

try:
    from multibagger_scanner import (
        compute_features,
        detect_accum_breakout,
        backtest_signal,
        NIFTY50_SYMBOLS,
    )
    HAS_SCANNER = True
except Exception as e:
    HAS_SCANNER = False
    _IMPORT_ERROR = str(e)
else:
    _IMPORT_ERROR = ""

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False


@st.cache_data(ttl=3600, show_spinner=False)
def download_symbol(symbol: str, years: int = 3) -> Optional[pd.DataFrame]:
    if not HAS_YF:
        return None
    end = datetime.now()
    start = end - timedelta(days=years * 365 + 30)
    for suffix in (".NS", ".BO"):
        try:
            df = yf.download(
                f"{symbol}{suffix}",
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if df is None or df.empty or len(df) < 60:
                continue
            df = df.reset_index()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df = df.rename(columns={
                "Date": "Date", "Open": "Open", "High": "High",
                "Low": "Low", "Close": "Close", "Volume": "Volume",
            })
            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
            df = df.dropna(subset=["Close"])
            return df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        except Exception:
            continue
    return None


def scan_one(symbol: str, df: pd.DataFrame) -> Optional[dict]:
    if len(df) < 120:
        return None
    try:
        df = compute_features(df)
        signal = detect_accum_breakout(df)
        bt = backtest_signal(df, signal)
        last = df.iloc[-1]
        is_live = bool(signal.iloc[-1])

        near = (
            last["CONSOL_RANGE_PCT"] < 14.0
            and last["OBV_RISING"]
            and last["VOL_RATIO20"] > 1.0
            and last["CLOSE_LOC"] > 0.5
            and last["POS52"] > 35
        )
        if not (is_live or near):
            return None

        buy_above = float(last["High"])
        invalid_below = float(last["Low"])
        risk_pct = (buy_above - invalid_below) / buy_above * 100
        mode = "HIGH RISK" if risk_pct > 10 else "CLEAN"

        score = 0.0
        score += min(float(last["VOL_RATIO20"]), 5) * 15
        score += min(float(last["VALUE_RATIO20"]), 5) * 10
        score += float(last["CLOSE_LOC"]) * 25
        score += (float(last["POS52"]) / 100) * 20
        if last.get("BULL_PATTERN", False):
            score += 25
        if is_live:
            score += 30
        if bt.get("pf90") and not np.isnan(bt["pf90"]):
            score += min(bt["pf90"], 3) * 10

        up_prob = 50.0
        if bt.get("hit20_90") and not np.isnan(bt["hit20_90"]):
            up_prob = 0.6 * bt["hit20_90"] + 0.4 * up_prob
        if last.get("BULL_PATTERN", False):
            up_prob = min(95, up_prob + 12)
        if last.get("BEAR_PATTERN", False):
            up_prob = max(5, up_prob - 15)

        return {
            "SYMBOL": symbol.replace(".NS", "").replace(".BO", ""),
            "MODE": mode,
            "LIVE_SCORE": round(score, 1),
            "CLOSE": round(float(last["Close"]), 2),
            "BUY_ABOVE": round(buy_above, 2),
            "INVALID_BELOW": round(invalid_below, 2),
            "RISK_%": round(risk_pct, 2),
            "VOL_RATIO": round(float(last["VOL_RATIO20"]), 2),
            "CLOSE_LOC": round(float(last["CLOSE_LOC"]), 2),
            "POS52_%": round(float(last["POS52"]), 1),
            "UP_PROB_%": round(max(5, min(95, up_prob)), 1),
            "BT_HIT20_90": round(bt["hit20_90"], 1) if bt.get("hit20_90") == bt.get("hit20_90") else None,
            "BT_PF90": round(bt["pf90"], 2) if bt.get("pf90") == bt.get("pf90") else None,
            "BULL_CANDLE": bool(last.get("BULL_PATTERN", False)),
            "LIVE_BREAKOUT": is_live,
        }
    except Exception:
        return None


st.title("📈 Multibagger Scanner")
st.caption("Accumulation Breakout • NSE/BSE • Positional signals")

with st.expander("How to use (tap to open)", expanded=False):
    st.markdown("""
**Rules**
- **BUY ABOVE** = signal-day high. Enter only on a break of this level.
- **INVALID BELOW** = signal-day low. Exit if price sustains below it.
- This is a **positional** watchlist, not intraday.

**On Android**
1. Open this page in Chrome
2. Menu → **Add to Home screen**
3. Run a scan whenever you want fresh signals
""")

col1, col2, col3 = st.columns(3)
with col1:
    universe = st.selectbox("Universe", ["Nifty 50 (fast)", "Custom symbols"], index=0)
with col2:
    years = st.selectbox("History (years)", [2, 3, 4, 5], index=1)
with col3:
    max_stocks = st.number_input("Max stocks", min_value=5, max_value=50, value=15, step=5)

custom_syms = ""
if universe == "Custom symbols":
    custom_syms = st.text_input(
        "Symbols (comma separated)",
        value="RELIANCE,TCS,INFY,HDFCBANK,SBIN,ICICIBANK",
    )

run = st.button("Run Scan", type="primary", use_container_width=True)

if run:
    if not HAS_YF:
        st.error("yfinance is not installed. Cannot download data.")
        st.stop()
    if not HAS_SCANNER:
        st.error(f"Scanner module could not be loaded.\n\n`{_IMPORT_ERROR}`")
        st.info("Make sure `multibagger_scanner.py` is in the same folder as `app.py`.")
        st.stop()

    if universe.startswith("Nifty 50"):
        raw_list = [s.replace(".NS", "") for s in NIFTY50_SYMBOLS]
    else:
        raw_list = [s.strip().upper() for s in custom_syms.split(",") if s.strip()]

    symbols = raw_list[: int(max_stocks)]

    progress = st.progress(0, text="Starting…")
    status = st.empty()
    results = []

    for i, sym in enumerate(symbols):
        status.markdown(f"**Downloading & scanning** `{sym}` ({i+1}/{len(symbols)})")
        progress.progress((i) / len(symbols))

        df = download_symbol(sym, years=years)
        if df is None:
            continue
        row = scan_one(sym, df)
        if row:
            results.append(row)

        time.sleep(0.35)

    progress.progress(1.0)
    status.empty()
    progress.empty()

    if not results:
        st.warning("No accumulation-breakout setups found in the selected universe today.")
        st.info("Try increasing Max stocks or switching to Custom symbols with mid/small-cap names.")
    else:
        out = pd.DataFrame(results)
        out = out.sort_values("LIVE_SCORE", ascending=False).reset_index(drop=True)
        out.insert(0, "RANK", out.index + 1)

        st.success(f"Found **{len(out)}** setups")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Signals", len(out))
        m2.metric("Avg Score", f"{out['LIVE_SCORE'].mean():.0f}")
        m3.metric("Avg Up Prob", f"{out['UP_PROB_%'].mean():.0f}%")
        m4.metric("Live Breakouts", int(out["LIVE_BREAKOUT"].sum()))

        st.subheader("Watchlist")
        display_cols = [
            "RANK", "SYMBOL", "MODE", "LIVE_SCORE", "CLOSE",
            "BUY_ABOVE", "INVALID_BELOW", "RISK_%", "UP_PROB_%",
            "VOL_RATIO", "POS52_%", "BULL_CANDLE", "LIVE_BREAKOUT",
        ]
        st.dataframe(
            out[[c for c in display_cols if c in out.columns]],
            use_container_width=True,
            hide_index=True,
            height=min(400, 60 + 35 * len(out)),
        )

        csv = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Watchlist CSV",
            data=csv,
            file_name=f"multibagger_watchlist_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown("""
**Final Rules**
- Enter only **above BUY_ABOVE** (signal-day high)
- Exit / avoid if price sustains **below INVALID_BELOW**
- Prefer **CLEAN** mode + high LIVE_SCORE + bullish candle
- This is positional — not for intraday trading
""")

else:
    st.info("Select universe and tap **Run Scan** to generate today’s watchlist.")
    st.markdown("""
### Tips for Android
- Use **Chrome** → menu → **Add to Home screen**
- Start with **Nifty 50 + Max 15 stocks** for faster results
""")
