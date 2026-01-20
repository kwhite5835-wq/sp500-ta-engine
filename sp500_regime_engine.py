import json
from dataclasses import dataclass
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
import os
import smtplib
from email.mime.text import MIMEText



# ------------------------
# Config
# ------------------------
@dataclass
class Config:
    ticker: str = "^spx"          # Stooq ticker for S&P 500 index
    lookback_years: int = 3       # enough history to compute 200DMA reliably
    sma_mid: int = 50
    sma_long: int = 200
    slope_days: int = 20          # slope proxy window for 50DMA
    rsi_len: int = 14
    atr_len: int = 14
    atr_zone_mult: float = 0.5
    atr_stop_mult: float = 1.0


# ------------------------
# Indicators
# ------------------------
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).rolling(n).mean()

def find_swings(df: pd.DataFrame, lookback: int = 5):
    highs = df["High"]
    lows = df["Low"]
    win = 2 * lookback + 1
    swing_high = highs[(highs == highs.rolling(win, center=True).max())]
    swing_low = lows[(lows == lows.rolling(win, center=True).min())]
    return swing_high.dropna(), swing_low.dropna()


# ------------------------
# Data fetch (Stooq)
# ------------------------
def fetch_stooq_daily(ticker: str) -> pd.DataFrame:
    """
    Stooq CSV endpoint.
    """
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}&i=d"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    from io import StringIO
    df = pd.read_csv(StringIO(r.text))

    # Columns: Date, Open, High, Low, Close, Volume
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def compute_regime(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    close = df["Close"]
    df["SMA50"] = sma(close, cfg.sma_mid)
    df["SMA200"] = sma(close, cfg.sma_long)

    slope50 = df["SMA50"].diff(cfg.slope_days)
    df["REGIME"] = np.where(
        (close > df["SMA200"]) & (slope50 > 0),
        "UPTREND",
        np.where((close < df["SMA200"]) & (slope50 < 0), "DOWNTREND", "RANGE"),
    )
    df["RSI"] = rsi(close, cfg.rsi_len)
    df["ATR"] = atr(df, cfg.atr_len)
    return df


def build_levels(df: pd.DataFrame, cfg: Config) -> dict:
    """
    Support/resistance zones using latest swing points +/- 0.5*ATR.
    Stops using +/- 1.0*ATR.
    """
    swings_high, swings_low = find_swings(df, lookback=5)
    last_atr = float(df["ATR"].iloc[-1])

    last_support = float(swings_low.iloc[-1]) if len(swings_low) else float(df["Low"].tail(20).min())
    last_resist  = float(swings_high.iloc[-1]) if len(swings_high) else float(df["High"].tail(20).max())

    zone_w = cfg.atr_zone_mult * last_atr

    support_zone = (last_support - zone_w, last_support + zone_w)
    resist_zone  = (last_resist  - zone_w, last_resist  + zone_w)

    stop_long = last_support - cfg.atr_stop_mult * last_atr
    target_long = last_resist - zone_w

    stop_short = last_resist + cfg.atr_stop_mult * last_atr
    target_short = last_support + zone_w

    return {
        "support": last_support,
        "resistance": last_resist,
        "support_zone": support_zone,
        "resistance_zone": resist_zone,
        "stop_long": float(stop_long),
        "target_long": float(target_long),
        "stop_short": float(stop_short),
        "target_short": float(target_short),
        "atr": last_atr,
    }


def make_report(df: pd.DataFrame, cfg: Config) -> dict:
    last = df.iloc[-1]
    reg = str(last["REGIME"])
    action = "LONG (100%)" if reg in ("UPTREND", "RANGE") else "CASH (0%)"

    levels = build_levels(df, cfg)

    report = {
        "asof_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "ticker": cfg.ticker,
        "last_close": float(last["Close"]),
        "regime": reg,
        "action": action,
        "sma50": float(last["SMA50"]),
        "sma200": float(last["SMA200"]),
        "rsi14": float(last["RSI"]),
        "atr14": float(last["ATR"]),
        "levels": levels,
    }
    return report


def render_markdown(rep: dict) -> str:
    lv = rep["levels"]
    md = []
    md.append(f"# S&P 500 Regime Report ({rep['ticker']})")
    md.append(f"- **As of:** {rep['asof_utc']}")
    md.append(f"- **Last close:** {rep['last_close']:.2f}")
    md.append(f"- **Regime:** **{rep['regime']}**")
    md.append(f"- **Action:** **{rep['action']}**")
    md.append("")
    md.append("## Indicators")
    md.append(f"- SMA50: {rep['sma50']:.2f}")
    md.append(f"- SMA200: {rep['sma200']:.2f}")
    md.append(f"- RSI(14): {rep['rsi14']:.1f}")
    md.append(f"- ATR(14): {rep['atr14']:.2f}")
    md.append("")
    md.append("## Levels (ATR-based zones)")
    md.append(f"- Support: {lv['support']:.2f}")
    md.append(f"- Resistance: {lv['resistance']:.2f}")
    md.append(f"- Support zone: {lv['support_zone'][0]:.2f} – {lv['support_zone'][1]:.2f}")
    md.append(f"- Resistance zone: {lv['resistance_zone'][0]:.2f} – {lv['resistance_zone'][1]:.2f}")
    md.append("")
    md.append("## Risk/Targets (illustrative)")
    md.append(f"- Long stop: {lv['stop_long']:.2f} | Long target: {lv['target_long']:.2f}")
    md.append(f"- Short stop: {lv['stop_short']:.2f} | Short target: {lv['target_short']:.2f}")
    md.append("")
    md.append("> Rule: LONG in UPTREND/RANGE; CASH in DOWNTREND. Levels are for timing / risk framing.")
    md.append("")
    return "\n".join(md)
def send_email(subject: str, body_text: str):
    host = os.getenv("EMAIL_HOST")
    port = int(os.getenv("EMAIL_PORT", "587"))
    user = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PASS")
    to_addr = os.getenv("EMAIL_TO")

    # If secrets aren't set, just skip (useful for local testing)
    if not all([host, user, password, to_addr]):
        print("Email not configured; skipping email send.")
        return

    msg = MIMEText(body_text, "plain", "utf-8")
    msg["From"] = user
    msg["To"] = to_addr
    msg["Subject"] = subject

    with smtplib.SMTP(host, port, timeout=30) as server:
        server.starttls()
        server.login(user, password)
        server.send_message(msg)


def main():
    cfg = Config()

    df = fetch_stooq_daily(cfg.ticker)
    df = compute_regime(df, cfg)

    # Keep last ~3 years to keep runtime small, but we have enough for 200DMA.
    if len(df) > 800:
        df = df.tail(800).reset_index(drop=True)

    rep = make_report(df, cfg)

    # Write outputs
    with open("report.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)

    md = render_markdown(rep)
    with open("report.md", "w", encoding="utf-8") as f:
        f.write(md)

    # Print report to logs
    print(md)
    subject = f"S&P 500 Regime Report – {rep['regime']} – {rep['asof_utc'][:10]}"
    send_email(subject, md)



if __name__ == "__main__":
    main()
