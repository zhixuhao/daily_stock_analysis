# -*- coding: utf-8 -*-
"""A/H股自选股智能分析系统 - 常用技术指标

目标：
- 使用 pandas/numpy 计算常见技术指标（不新增三方依赖）
- 输出“最后值 + 关键信号解释”的小体积摘要，供 LLM 与报告层使用

说明：
- 输入 DataFrame 需至少包含列：date, open, high, low, close, volume
- 建议 df 已按日期升序排序；本模块会在内部再做一次排序兜底
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (float, int, np.floating, np.integer)):
            if np.isnan(x):
                return None
            return float(x)
        x = float(x)
        if np.isnan(x):
            return None
        return x
    except Exception:
        return None


def compute_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """计算常用技术指标并返回摘要。

    返回结构示例（字段可能为 None）：
    {
      "rsi": {"rsi14": 52.3, "signal": "中性"},
      "macd": {"dif": 0.12, "dea": 0.08, "hist": 0.08, "signal": "金叉"},
      "boll": {"mid": 10.2, "upper": 10.8, "lower": 9.6, "position": "中轨附近"},
      "kdj": {"k": 62.1, "d": 58.0, "j": 70.3, "signal": "偏强"},
      "atr": {"atr14": 0.35, "atr_pct": 2.1},
      "meta": {"rows": 60, "last_date": "2026-01-27"}
    }
    """
    if df is None or df.empty:
        return {"error": "empty_df"}

    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        return {"error": f"missing_columns: {sorted(missing)}"}

    _df = df.copy()
    _df["date"] = pd.to_datetime(_df["date"], errors="coerce")
    _df = _df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        _df[col] = pd.to_numeric(_df[col], errors="coerce")

    _df = _df.dropna(subset=["high", "low", "close"]).reset_index(drop=True)
    if _df.empty:
        return {"error": "no_valid_rows"}

    out: Dict[str, Any] = {
        "meta": {
            "rows": int(len(_df)),
            "last_date": _df["date"].iloc[-1].date().isoformat() if len(_df) else None,
        }
    }

    close = _df["close"]
    high = _df["high"]
    low = _df["low"]

    # =====================
    # RSI(14) - Wilder
    # =====================
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Wilder 平滑：alpha=1/period
    period_rsi = 14
    avg_gain = gain.ewm(alpha=1 / period_rsi, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period_rsi, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi14 = 100 - (100 / (1 + rs))
    rsi_last = _safe_float(rsi14.iloc[-1])

    rsi_signal = None
    if rsi_last is not None:
        if rsi_last >= 70:
            rsi_signal = "超买(警惕回撤)"
        elif rsi_last <= 30:
            rsi_signal = "超卖(关注反弹)"
        elif rsi_last >= 55:
            rsi_signal = "偏强"
        elif rsi_last <= 45:
            rsi_signal = "偏弱"
        else:
            rsi_signal = "中性"

    out["rsi"] = {
        "rsi14": rsi_last,
        "signal": rsi_signal,
    }

    # =====================
    # MACD(12,26,9)
    # =====================
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = 2 * (dif - dea)

    dif_last = _safe_float(dif.iloc[-1])
    dea_last = _safe_float(dea.iloc[-1])
    hist_last = _safe_float(hist.iloc[-1])

    macd_signal = None
    if len(_df) >= 2 and dif_last is not None and dea_last is not None:
        prev_dif = _safe_float(dif.iloc[-2])
        prev_dea = _safe_float(dea.iloc[-2])
        if prev_dif is not None and prev_dea is not None:
            if prev_dif <= prev_dea and dif_last > dea_last:
                macd_signal = "金叉(偏多)"
            elif prev_dif >= prev_dea and dif_last < dea_last:
                macd_signal = "死叉(偏空)"

    if macd_signal is None and dif_last is not None and dea_last is not None and hist_last is not None:
        if dif_last > dea_last and hist_last > 0:
            macd_signal = "多头(柱体为正)"
        elif dif_last < dea_last and hist_last < 0:
            macd_signal = "空头(柱体为负)"
        else:
            macd_signal = "震荡"

    out["macd"] = {
        "dif": dif_last,
        "dea": dea_last,
        "hist": hist_last,
        "signal": macd_signal,
    }

    # =====================
    # BOLL(20,2)
    # =====================
    mid = close.rolling(window=20).mean()
    std = close.rolling(window=20).std(ddof=0)
    upper = mid + 2 * std
    lower = mid - 2 * std

    mid_last = _safe_float(mid.iloc[-1])
    upper_last = _safe_float(upper.iloc[-1])
    lower_last = _safe_float(lower.iloc[-1])

    boll_pos = None
    last_close = _safe_float(close.iloc[-1])
    if last_close is not None and upper_last is not None and lower_last is not None and mid_last is not None:
        if last_close > upper_last:
            boll_pos = "突破上轨(可能过热)"
        elif last_close < lower_last:
            boll_pos = "跌破下轨(可能超跌)"
        else:
            # 计算在带内的位置 0..1
            denom = (upper_last - lower_last) if (upper_last - lower_last) else None
            if denom:
                pct_b = (last_close - lower_last) / denom
                if pct_b >= 0.75:
                    boll_pos = "上轨附近(偏强)"
                elif pct_b <= 0.25:
                    boll_pos = "下轨附近(偏弱)"
                else:
                    boll_pos = "中轨附近(震荡)"

    out["boll"] = {
        "mid": mid_last,
        "upper": upper_last,
        "lower": lower_last,
        "position": boll_pos,
    }

    # =====================
    # KDJ(9,3,3) - ewm 平滑
    # =====================
    n = 9
    low_n = low.rolling(window=n).min()
    high_n = high.rolling(window=n).max()
    rsv = (close - low_n) / (high_n - low_n).replace(0, np.nan) * 100

    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    j = 3 * k - 2 * d

    k_last = _safe_float(k.iloc[-1])
    d_last = _safe_float(d.iloc[-1])
    j_last = _safe_float(j.iloc[-1])

    kdj_signal = None
    if k_last is not None and d_last is not None:
        if k_last > d_last and (k_last >= 50 or d_last >= 50):
            kdj_signal = "偏强(K>D)"
        elif k_last < d_last and (k_last <= 50 or d_last <= 50):
            kdj_signal = "偏弱(K<D)"
        else:
            kdj_signal = "震荡"

    # 极值提示
    if k_last is not None:
        if k_last >= 80:
            kdj_signal = (kdj_signal + "，高位(警惕)" if kdj_signal else "高位(警惕)")
        elif k_last <= 20:
            kdj_signal = (kdj_signal + "，低位(关注)" if kdj_signal else "低位(关注)")

    out["kdj"] = {
        "k": k_last,
        "d": d_last,
        "j": j_last,
        "signal": kdj_signal,
    }

    # =====================
    # ATR(14)
    # =====================
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr14 = tr.rolling(window=14).mean()
    atr_last = _safe_float(atr14.iloc[-1])
    atr_pct = None
    if atr_last is not None and last_close:
        atr_pct = atr_last / last_close * 100

    out["atr"] = {
        "atr14": atr_last,
        "atr_pct": _safe_float(atr_pct),
    }

    return out
