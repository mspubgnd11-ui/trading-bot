
import os
import asyncio
import logging
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
import aiohttp
from telegram import Bot
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from telegram import Update
import ta
import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════════════════
#                         الإعدادات
# ══════════════════════════════════════════════════════════════════
TELEGRAM_TOKEN    = os.getenv("TELEGRAM_TOKEN")
CHAT_ID           = os.getenv("CHAT_ID")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

CHECK_INTERVAL    = 420      # فحص كل 7 دقايق
MAX_PRICE         = 10.0     # عملات تحت $10
MAX_SYMBOLS       = 80       # أفضل 80 عملة بالحجم
SIGNAL_COOLDOWN   = 14400    # 4 ساعات بين نفس الإشارة
MAX_PER_SCAN      = 2        # أقصى إشارتين بكل فحص
SIGNALS_FILE      = "sent_signals.pkl"  # ملف دائم — ما يصفّر لو البوت أعاد تشغيل

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════
#              حفظ وتحميل الإشارات (دائم — لا يصفّر)
# ══════════════════════════════════════════════════════════════════
def load_signals() -> dict:
    try:
        if Path(SIGNALS_FILE).exists():
            with open(SIGNALS_FILE, "rb") as f:
                return pickle.load(f)
    except Exception:
        pass
    return {}

def save_signals(data: dict):
    try:
        with open(SIGNALS_FILE, "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        logger.error(f"خطأ حفظ: {e}")

def is_on_cooldown(key: str) -> bool:
    signals = load_signals()
    last = signals.get(key)
    if not last:
        return False
    return (datetime.utcnow() - last).total_seconds() < SIGNAL_COOLDOWN

def record_signal(key: str):
    signals = load_signals()
    signals[key] = datetime.utcnow()
    # تنظيف القديم (أكثر من 24 ساعة)
    cutoff = datetime.utcnow() - timedelta(hours=24)
    signals = {k: v for k, v in signals.items() if v > cutoff}
    save_signals(signals)

# ══════════════════════════════════════════════════════════════════
#                     سحب العملات
# ══════════════════════════════════════════════════════════════════
async def get_symbols() -> list:
    url = "https://fapi.bitunix.com/api/v1/futures/market/tickers"
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(url, timeout=aiohttp.ClientTimeout(total=15)) as r:
                data = await r.json()
        tickers = data.get("data", [])
        filtered = [
            (t["symbol"], float(t.get("volume24h", 0) or 0))
            for t in tickers
            if t.get("symbol", "").endswith("USDT")
            and 0 < float(t.get("lastPrice", 0) or 0) < MAX_PRICE
        ]
        filtered.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in filtered[:MAX_SYMBOLS]]
    except Exception as e:
        logger.error(f"خطأ العملات: {e}")
        return ["XRPUSDT", "DOGEUSDT", "ADAUSDT", "TRXUSDT", "SHIBUSDT"]

# ══════════════════════════════════════════════════════════════════
#                   سحب الشمعدانات
# ══════════════════════════════════════════════════════════════════
async def get_klines(symbol: str, interval: str = "15m", limit: int = 200) -> pd.DataFrame | None:
    url = "https://fapi.bitunix.com/api/v1/futures/market/kline"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as r:
                data = await r.json()
        if data.get("code") == 0 and data.get("data"):
            df = pd.DataFrame(data["data"])
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["volume"] = pd.to_numeric(
                df.get("quoteVol", df.get("baseVol", pd.Series([1.0] * len(df)))),
                errors="coerce"
            )
            df.dropna(subset=["open", "high", "low", "close"], inplace=True)
            return df
    except Exception as e:
        logger.error(f"خطأ كلاين {symbol} {interval}: {e}")
    return None

# ══════════════════════════════════════════════════════════════════
#           التحليل الفني — متعدد الإطارات الزمنية
# ══════════════════════════════════════════════════════════════════
def analyze_timeframe(df: pd.DataFrame) -> dict | None:
    """تحليل إطار زمني واحد — يُستدعى مرتين (15m و 1h)"""
    if len(df) < 60:
        return None

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]
    price  = close.iloc[-1]

    # ── RSI ──────────────────────────────────────────────────────
    rsi     = ta.momentum.RSIIndicator(close, window=14).rsi()
    rsi_cur = rsi.iloc[-1]
    rsi_pre = rsi.iloc[-2]

    # ── MACD ─────────────────────────────────────────────────────
    macd_obj        = ta.trend.MACD(close)
    macd_line       = macd_obj.macd()
    macd_sig        = macd_obj.macd_signal()
    macd_hist       = macd_obj.macd_diff()
    cross_up        = macd_line.iloc[-1] > macd_sig.iloc[-1] and macd_line.iloc[-2] <= macd_sig.iloc[-2]
    cross_down      = macd_line.iloc[-1] < macd_sig.iloc[-1] and macd_line.iloc[-2] >= macd_sig.iloc[-2]
    hist_bull       = macd_hist.iloc[-1] > 0 and macd_hist.iloc[-1] > macd_hist.iloc[-2]
    hist_bear       = macd_hist.iloc[-1] < 0 and macd_hist.iloc[-1] < macd_hist.iloc[-2]

    # ── EMA ──────────────────────────────────────────────────────
    ema9  = ta.trend.EMAIndicator(close, window=9).ema_indicator().iloc[-1]
    ema21 = ta.trend.EMAIndicator(close, window=21).ema_indicator().iloc[-1]
    ema50 = ta.trend.EMAIndicator(close, window=50).ema_indicator().iloc[-1]
    golden = ema9 > ema21 > ema50
    death  = ema9 < ema21 < ema50

    # ── Bollinger ────────────────────────────────────────────────
    bb       = ta.volatility.BollingerBands(close, window=20)
    bb_low   = bb.bollinger_lband().iloc[-1]
    bb_high  = bb.bollinger_hband().iloc[-1]
    bb_mid   = bb.bollinger_mavg().iloc[-1]
    bb_pct   = bb.bollinger_pband().iloc[-1]

    # ── Volume ───────────────────────────────────────────────────
    vol_avg   = volume.rolling(20).mean().iloc[-1]
    vol_cur   = volume.iloc[-1]
    vol_ratio = vol_cur / vol_avg if vol_avg > 0 else 1
    vol_surge = vol_ratio >= 1.8

    # ── Stochastic ───────────────────────────────────────────────
    stoch_k = ta.momentum.StochasticOscillator(high, low, close).stoch().iloc[-1]

    # ── ATR ──────────────────────────────────────────────────────
    atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]

    # ── تغيير السعر ──────────────────────────────────────────────
    chg_1h  = round(((price - close.iloc[-4])  / close.iloc[-4])  * 100, 2) if len(close) > 4  else 0
    chg_4h  = round(((price - close.iloc[-16]) / close.iloc[-16]) * 100, 2) if len(close) > 16 else 0
    chg_24h = round(((price - close.iloc[-96]) / close.iloc[-96]) * 100, 2) if len(close) > 96 else 0

    return {
        "price": price, "rsi": round(rsi_cur, 1), "rsi_prev": round(rsi_pre, 1),
        "cross_up": cross_up, "cross_down": cross_down,
        "hist_bull": hist_bull, "hist_bear": hist_bear,
        "ema9": round(ema9, 6), "ema21": round(ema21, 6), "ema50": round(ema50, 6),
        "golden": golden, "death": death,
        "bb_pct": round(bb_pct, 2), "bb_low": round(bb_low, 8), "bb_high": round(bb_high, 8),
        "stoch_k": round(stoch_k, 1),
        "vol_surge": vol_surge, "vol_ratio": round(vol_ratio, 1),
        "atr": atr, "chg_1h": chg_1h, "chg_4h": chg_4h, "chg_24h": chg_24h,
    }

def strict_signal(tf15: dict, tf1h: dict) -> dict | None:
    """
    ══ منطق الإشارة الصارم ══
    يشترط تأكيد من إطارين زمنيين + 4 شروط من 5 على الأقل
    """

    def score_direction(tf: dict, direction: str) -> int:
        pts = 0
        if direction == "LONG":
            if tf["cross_up"]:                              pts += 4   # تقاطع MACD صاعد
            elif tf["hist_bull"]:                           pts += 2   # زخم MACD صاعد
            if tf["rsi"] < 35:                              pts += 3   # ذروة بيع
            elif tf["rsi"] < 45:                            pts += 2
            if tf["rsi"] > tf["rsi_prev"] and tf["rsi"] < 60: pts += 1  # RSI يرتد
            if tf["golden"]:                                pts += 3   # EMA golden cross
            elif tf["ema9"] > tf["ema21"]:                  pts += 1
            if tf["price"] <= tf["bb_low"] * 1.015:        pts += 3   # عند البولنجر السفلي
            if tf["vol_surge"] and tf["chg_1h"] > 0:       pts += 2   # حجم + حركة
            if tf["stoch_k"] < 25:                         pts += 2   # ذروة بيع ستوكاستيك
        else:
            if tf["cross_down"]:                            pts += 4
            elif tf["hist_bear"]:                           pts += 2
            if tf["rsi"] > 65:                              pts += 3
            elif tf["rsi"] > 55:                            pts += 2
            if tf["rsi"] < tf["rsi_prev"] and tf["rsi"] > 40: pts += 1
            if tf["death"]:                                 pts += 3
            elif tf["ema9"] < tf["ema21"]:                  pts += 1
            if tf["price"] >= tf["bb_high"] * 0.985:       pts += 3
            if tf["vol_surge"] and tf["chg_1h"] < 0:       pts += 2
            if tf["stoch_k"] > 75:                         pts += 2
        return pts

    for direction in ["LONG", "SHORT"]:
        s15 = score_direction(tf15, direction)
        s1h = score_direction(tf1h, direction)

        # ★ الشروط الصارمة: نقاط كافية على كلا الإطارين
        if s15 >= 8 and s1h >= 5:
            # ★ تأكيد المومنتوم: على الأقل تقاطع أو زخم حقيقي على 15m
            has_momentum = (
                tf15["cross_up"] if direction == "LONG" else tf15["cross_down"]
            ) or (
                tf15["hist_bull"] if direction == "LONG" else tf15["hist_bear"]
            )
            if not has_momentum:
                continue

            # ★ RSI في منطقة منطقية (ليس محايداً 45-55)
            rsi = tf15["rsi"]
            if direction == "LONG"  and 48 < rsi < 58: continue
            if direction == "SHORT" and 42 < rsi < 52: continue

            return {
                "direction": direction,
                "score_15m": s15,
                "score_1h":  s1h,
                **{k: v for k, v in tf15.items()},   # بيانات 15m
                "rsi_1h":    tf1h["rsi"],
                "golden_1h": tf1h["golden"],
                "death_1h":  tf1h["death"],
                "vol_1h":    tf1h["vol_surge"],
            }
    return None

# ══════════════════════════════════════════════════════════════════
#                    تحليل Claude AI
# ══════════════════════════════════════════════════════════════════
async def ai_signal_review(symbol: str, sig: dict) -> dict:
    """مراجعة الإشارة التلقائية — موجز"""
    if not ANTHROPIC_API_KEY:
        conf = min(95, int(((sig["score_15m"] + sig["score_1h"]) / 26) * 100))
        return {"confidence": conf, "verdict": "تحليل فني", "tip": ""}

    coin = symbol.replace("USDT", "")
    prompt = f"""أنت محلل عملات رقمية خبير. قيّم هذه الإشارة بدقة وحياد.

العملة: {coin}/USDT  |  الاتجاه: {sig["direction"]}  |  السعر: {sig["price"]}

── إطار 15 دقيقة ──
RSI: {sig["rsi"]} | Stoch: {sig["stoch_k"]} | MACD: {"تقاطع صاعد" if sig["cross_up"] else "تقاطع هابط" if sig["cross_down"] else "زخم صاعد" if sig["hist_bull"] else "زخم هابط"}
EMA: {"Golden Cross ✅" if sig["golden"] else "Death Cross ❌" if sig["death"] else "مختلطة"}
Bollinger: {int(sig["bb_pct"]*100)}%  |  حجم: {"مرتفع x"+str(sig["vol_ratio"]) if sig["vol_surge"] else "طبيعي"}
التغير: 1س={sig["chg_1h"]}%  4س={sig["chg_4h"]}%  24س={sig["chg_24h"]}%

── إطار ساعة ──
RSI: {sig["rsi_1h"]} | EMA: {"Golden ✅" if sig["golden_1h"] else "Death ❌" if sig["death_1h"] else "مختلطة"} | حجم: {"مرتفع" if sig["vol_1h"] else "طبيعي"}

── النقاط الفنية ──
15m: {sig["score_15m"]}/20 | 1h: {sig["score_1h"]}/20

قيّم الإشارة وأعط:
- confidence: نسبة ثقتك 1-99 (كن صارماً — فوق 75 فقط إذا كانت قوية حقاً)
- verdict: حكم واحد بالعربي (مثل: "قوية جداً" أو "جيدة مع مخاطرة" أو "متوسطة")
- tip: نصيحة عملية قصيرة (جملة واحدة)

أجب بـ JSON فقط:
{{"confidence": 78, "verdict": "جيدة مع مخاطرة متوسطة", "tip": "انتظر تأكيد الشمعة الحالية قبل الدخول"}}"""

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": "claude-sonnet-4-20250514", "max_tokens": 150,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=aiohttp.ClientTimeout(total=20)
            ) as r:
                resp = await r.json()
        text = resp["content"][0]["text"].strip().replace("```json", "").replace("```", "")
        result = json.loads(text)
        return {
            "confidence": max(1, min(99, int(result.get("confidence", 60)))),
            "verdict":    result.get("verdict", ""),
            "tip":        result.get("tip", "")
        }
    except Exception as e:
        logger.error(f"خطأ AI إشارة: {e}")
        conf = min(95, int(((sig["score_15m"] + sig["score_1h"]) / 26) * 100))
        return {"confidence": conf, "verdict": "تحليل فني", "tip": ""}


async def ai_deep_analysis(symbol: str, tf15: dict, tf1h: dict) -> str:
    """تحليل احترافي عميق عند الطلب اليدوي"""
    if not ANTHROPIC_API_KEY:
        return "⚠️ مفتاح AI غير مفعّل — التحليل الفني متاح فقط"

    coin = symbol.replace("USDT", "")
    prompt = f"""أنت كبير محللي التداول المتخصص في العملات الرقمية للعقود الآجلة.
قدّم تحليلاً احترافياً شاملاً لـ {coin}/USDT.

══ البيانات الفنية ══

▌ إطار 15 دقيقة
السعر: {tf15["price"]}
RSI: {tf15["rsi"]} (سابق: {tf15["rsi_prev"]})
Stochastic: {tf15["stoch_k"]}
MACD: {"تقاطع صاعد 🚀" if tf15["cross_up"] else "تقاطع هابط 🔻" if tf15["cross_down"] else "زخم صاعد" if tf15["hist_bull"] else "زخم هابط" if tf15["hist_bear"] else "محايد"}
EMA: 9={tf15["ema9"]} | 21={tf15["ema21"]} | 50={tf15["ema50"]}
وضع EMA: {"Golden Cross ✅" if tf15["golden"] else "Death Cross ❌" if tf15["death"] else "مختلط"}
Bollinger: {int(tf15["bb_pct"]*100)}% | أسفل={tf15["bb_low"]} | أعلى={tf15["bb_high"]}
الحجم: {"ارتفاع x"+str(tf15["vol_ratio"])+" 🔥" if tf15["vol_surge"] else "طبيعي"}
التغير: 1س={tf15["chg_1h"]}% | 4س={tf15["chg_4h"]}% | 24س={tf15["chg_24h"]}%

▌ إطار ساعة
RSI: {tf1h["rsi"]} | وضع EMA: {"Golden ✅" if tf1h["golden"] else "Death ❌" if tf1h["death"] else "مختلط"}
Bollinger: {int(tf1h["bb_pct"]*100)}% | حجم: {"مرتفع" if tf1h["vol_surge"] else "طبيعي"}
التغير 4س: {tf1h["chg_4h"]}% | 24س: {tf1h["chg_24h"]}%

══ التحليل المطلوب ══
قدّم تحليلاً بالعربي الفصيح يشمل:

1. **خلاصة الوضع** — جملتان تصفان وضع العملة الآن
2. **نقاط القوة** — ما يدعم الدخول (إن وُجدت)
3. **نقاط الضعف** — المخاطر والعوامل السلبية
4. **التوصية** — LONG أو SHORT أو انتظار، مع تبرير واضح
5. **نقطة الدخول المثالية** — السعر والشرط
6. **الأهداف المقترحة** — TP1 وTP2 وTP3 بناءً على ATR ({round(tf15["atr"], 8)})
7. **وقف الخسارة** — ومبرره الفني
8. **تنبيه خاص** — أي شيء يستوقفك في هذه العملة تحديداً

اكتب بأسلوب محلل محترف — دقيق، مختصر، بلا حشو."""

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": "claude-sonnet-4-20250514", "max_tokens": 800,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as r:
                resp = await r.json()
        return resp["content"][0]["text"].strip()
    except Exception as e:
        logger.error(f"خطأ AI تحليل عميق: {e}")
        return "⚠️ خطأ في الاتصال بـ AI"

# ══════════════════════════════════════════════════════════════════
#                  الأهداف ووقف الخسارة
# ══════════════════════════════════════════════════════════════════
def make_targets(price: float, direction: str, atr: float):
    a = atr if atr > 0 else price * 0.02
    if direction == "LONG":
        sl   = round(price - a * 1.5, 8)
        tps  = [round(price + a * m, 8) for m in [1.0, 2.2, 3.5, 5.5, 8.0]]
        sl_p = f"-{round((price - sl) / price * 100, 1)}%"
        pcts = [f"+{round((t - price) / price * 100, 1)}%" for t in tps]
    else:
        sl   = round(price + a * 1.5, 8)
        tps  = [round(price - a * m, 8) for m in [1.0, 2.2, 3.5, 5.5, 8.0]]
        sl_p = f"+{round((sl - price) / price * 100, 1)}%"
        pcts = [f"-{round((price - t) / price * 100, 1)}%" for t in tps]
    return sl, tps, pcts, sl_p

# ══════════════════════════════════════════════════════════════════
#              تنسيق الرسائل (محترف)
# ══════════════════════════════════════════════════════════════════
def fmt(p: float) -> str:
    if p >= 1:      return f"{p:,.4f}$"
    elif p >= 0.01: return f"{p:.6f}$"
    else:           return f"{p:.8f}$"

def signal_message(symbol: str, sig: dict, ai: dict, sl, tps, pcts, sl_p) -> str:
    coin   = symbol.replace("USDT", "")
    now    = datetime.utcnow().strftime("%d/%m/%Y  %H:%M UTC")
    conf   = ai["confidence"]
    is_long = sig["direction"] == "LONG"

    # مستوى الإشارة
    if conf >= 85:
        level = "🔥 إشارة قوية جداً"
        stars = "★★★★★"
    elif conf >= 75:
        level = "✅ إشارة جيدة"
        stars = "★★★★☆"
    else:
        level = "📊 إشارة متوسطة"
        stars = "★★★☆☆"

    bar = "█" * int(conf / 10) + "░" * (10 - int(conf / 10))

    ema_txt = "Golden Cross ✅" if sig["golden"] else "Death Cross ❌" if sig["death"] else "مختلطة"
    ema_1h  = "Golden ✅" if sig["golden_1h"] else "Death ❌" if sig["death_1h"] else "مختلطة"
    vol_txt = f"🔥 مرتفع ×{sig['vol_ratio']}" if sig["vol_surge"] else "طبيعي"

    return f"""
╔══════════════════════════════╗
║  {'📈 لونغ' if is_long else '📉 شورت'}  {coin}/USDT  {'🟢' if is_long else '🔴'}   ║
╚══════════════════════════════╝
{level}  {stars}

🕐 {now}
💰 الدخول:  {fmt(sig['price'])}

🤖 تقييم AI:  {ai['verdict']}
💡 {ai['tip']}

┌─ الثقة ──────────────────────
│  {conf}%  [{bar}]
└───────────────────────────────

┌─ الأهداف ─────────────────────
│  🎯 TP1   {fmt(tps[0])}   ({pcts[0]})
│  🎯 TP2   {fmt(tps[1])}   ({pcts[1]})
│  🎯 TP3   {fmt(tps[2])}   ({pcts[2]})
│  🎯 TP4   {fmt(tps[3])}   ({pcts[3]})
│  🎯 TP5   {fmt(tps[4])}   ({pcts[4]})
└───────────────────────────────
🛑 وقف الخسارة:  {fmt(sl)}   ({sl_p})

┌─ المؤشرات ────────────────────
│  15m │ RSI {sig['rsi']}  Stoch {sig['stoch_k']}
│      │ EMA: {ema_txt}
│      │ Bollinger: {int(sig['bb_pct']*100)}%
│      │ الحجم: {vol_txt}
│   1h │ RSI {sig['rsi_1h']}  EMA: {ema_1h}
└───────────────────────────────
📊 التغير:  1س {sig['chg_1h']:+}%  ·  4س {sig['chg_4h']:+}%  ·  24س {sig['chg_24h']:+}%
⚙️ نقاط فنية:  15m {sig['score_15m']}/20  ·  1h {sig['score_1h']}/20

⚠️  للتأكيد فقط — القرار مسؤوليتك
🏦  Bitunix Futures
""".strip()


def analysis_header(symbol: str, price: float, tf15: dict, tf1h: dict) -> str:
    """رأس رسالة التحليل اليدوي"""
    coin = symbol.replace("USDT", "")
    now  = datetime.utcnow().strftime("%d/%m/%Y  %H:%M UTC")
    return f"""
╔══════════════════════════════╗
║  🔍 تحليل احترافي            ║
║  {coin}/USDT                  
╚══════════════════════════════╝
🕐 {now}
💰 السعر الحالي:  {fmt(price)}
📊 التغير:  1س {tf15['chg_1h']:+}%  ·  4س {tf15['chg_4h']:+}%  ·  24س {tf15['chg_24h']:+}%

──────────────────────────────
""".strip()

# ══════════════════════════════════════════════════════════════════
#                   الفحص التلقائي
# ══════════════════════════════════════════════════════════════════
async def scan_market(bot: Bot) -> int:
    logger.info("🔍 بدأ الفحص...")
    symbols = await get_symbols()
    found   = 0

    for sym in symbols:
        if found >= MAX_PER_SCAN:
            break
        try:
            df15 = await get_klines(sym, "15m", 200)
            df1h = await get_klines(sym, "1h",  100)
            if df15 is None or df1h is None:
                await asyncio.sleep(0.3)
                continue

            tf15 = analyze_timeframe(df15)
            tf1h = analyze_timeframe(df1h)
            if not tf15 or not tf1h:
                await asyncio.sleep(0.3)
                continue

            sig = strict_signal(tf15, tf1h)
            if not sig:
                await asyncio.sleep(0.3)
                continue

            key = f"{sym}_{sig['direction']}"
            if is_on_cooldown(key):
                await asyncio.sleep(0.3)
                continue

            price = sig["price"]
            if price >= MAX_PRICE or price <= 0:
                await asyncio.sleep(0.3)
                continue

            ai = await ai_signal_review(sym, sig)

            # ★ فقط الإشارات فوق 70%
            if ai["confidence"] < 70:
                logger.info(f"⏭ {sym} {sig['direction']} {ai['confidence']}% — ثقة منخفضة")
                await asyncio.sleep(0.3)
                continue

            sl, tps, pcts, sl_p = make_targets(price, sig["direction"], sig["atr"])
            msg = signal_message(sym, sig, ai, sl, tps, pcts, sl_p)

            await bot.send_message(chat_id=CHAT_ID, text=msg)
            record_signal(key)
            found += 1
            logger.info(f"📤 {sym} {sig['direction']} {ai['confidence']}%")
            await asyncio.sleep(2)

        except Exception as e:
            logger.error(f"خطأ {sym}: {e}")
        await asyncio.sleep(0.5)

    logger.info(f"✅ انتهى — {found} إشارة من {len(symbols)} عملة")
    return found

# ══════════════════════════════════════════════════════════════════
#              تحليل عملة بالطلب (يدوي — عميق)
# ══════════════════════════════════════════════════════════════════
async def handle_coin_request(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip().upper()
    symbol = text.replace("/", "").replace("-", "").replace(" ", "")
    if not symbol.endswith("USDT"):
        symbol += "USDT"

    msg = await update.message.reply_text(f"⏳ جاري تحليل {symbol} ...\nهذا يأخذ ثوانٍ قليلة")

    df15 = await get_klines(symbol, "15m", 200)
    df1h = await get_klines(symbol, "1h",  100)

    if df15 is None or df1h is None:
        await msg.edit_text(
            f"❌ تعذّر جلب بيانات {symbol}\n\n"
            f"تحقق من الاسم — أمثلة:\n"
            f"XRP  ·  DOGE  ·  ADA  ·  XRPUSDT"
        )
        return

    tf15 = analyze_timeframe(df15)
    tf1h = analyze_timeframe(df1h)

    if not tf15 or not tf1h:
        await msg.edit_text(f"⚠️ {symbol}\nبيانات غير كافية للتحليل")
        return

    price = tf15["price"]

    # رأس الرسالة بالأرقام
    header = analysis_header(symbol, price, tf15, tf1h)
    await msg.edit_text(f"{header}\n\n🤖 AI يحلل...")

    # التحليل العميق
    deep = await ai_deep_analysis(symbol, tf15, tf1h)

    final = f"{header}\n\n{deep}"
    # تيليغرام حد الرسالة 4096 حرف
    if len(final) > 4090:
        await msg.edit_text(header)
        await update.message.reply_text(deep)
    else:
        await msg.edit_text(final)

# ══════════════════════════════════════════════════════════════════
#                     أوامر التلغرام
# ══════════════════════════════════════════════════════════════════
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    syms = await get_symbols()
    ai_s = "✅ Claude Sonnet" if ANTHROPIC_API_KEY else "❌ غير مفعّل"
    await update.message.reply_text(f"""
╔══════════════════════════════╗
║   🌟 سراب للإشارات v3.0     ║
╚══════════════════════════════╝

📡 يراقب {len(syms)} عملة (تحت $10)
⏱  فحص كل 7 دقايق
🤖 AI: {ai_s}
✅ تأكيد من إطارين زمنيين (15m + 1h)
💯 ثقة 70%+ فقط تعبر
🎯 أهداف ديناميكية بالـ ATR
🔒 Cooldown 4 ساعات (دائم)
📈 أقصى إشارتين بكل فحص

──────────────────────────────
الأوامر:
/scan    — فحص فوري
/status  — حالة البوت
/clear   — مسح سجل الإشارات

💡 تحليل عملة:
ابعت اسمها مباشرة مثل:
  XRP  ·  DOGE  ·  SOL  ·  BTC
وسأقدم تحليلاً احترافياً شاملاً
""".strip())

async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 جاري فحص السوق...")
    found = await scan_market(context.bot)
    await update.message.reply_text(
        f"✅ انتهى الفحص\n📤 إشارات مرسلة: {found}"
    )

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    syms     = await get_symbols()
    signals  = load_signals()
    ai_s     = "✅ Claude Sonnet" if ANTHROPIC_API_KEY else "❌ غير مفعّل"
    active   = sum(1 for t in signals.values() if (datetime.utcnow() - t).total_seconds() < SIGNAL_COOLDOWN)
    await update.message.reply_text(f"""
✅ سراب يعمل بشكل طبيعي

⏱  الفحص كل: 7 دقايق
📊 العملات: {len(syms)}
🤖 AI: {ai_s}
💯 الحد الأدنى للثقة: 70%
⚙️  الحد الأدنى للنقاط: 8/20 (15m) + 5/20 (1h)
📤 إشارات نشطة (في 4 ساعات): {active}
📁 إجمالي السجل: {len(signals)}
""".strip())

async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    signals = load_signals()
    count   = len(signals)
    save_signals({})
    await update.message.reply_text(
        f"🗑️ تم مسح سجل {count} إشارة\n✅ البوت جاهز للفحص من جديد"
    )

async def auto_scan(context: ContextTypes.DEFAULT_TYPE):
    await scan_market(context.bot)

# ══════════════════════════════════════════════════════════════════
#                      تشغيل البوت
# ══════════════════════════════════════════════════════════════════
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("scan",   cmd_scan))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("clear",  cmd_clear))

    # أي رسالة نصية = اسم عملة للتحليل
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_coin_request))

    app.job_queue.run_repeating(auto_scan, interval=CHECK_INTERVAL, first=30)

    logger.info("🚀 سراب v3.0 يعمل...")
    app.run_polling()

if __name__ == "__main__":
    main()
