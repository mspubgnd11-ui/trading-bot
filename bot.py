import os
import asyncio
import logging
import json
from datetime import datetime, date
import aiohttp
from telegram import Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram import Update
import ta
import pandas as pd
import numpy as np

# ═══════════════════════════════════════════
#           إعدادات البوت
# ═══════════════════════════════════════════
TELEGRAM_TOKEN    = os.getenv("TELEGRAM_TOKEN")
CHAT_ID           = os.getenv("CHAT_ID")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

CHECK_INTERVAL    = 180
MAX_PRICE         = 10.0
MAX_SYMBOLS       = 150
MIN_CONFIDENCE    = 70
MIN_INDICATORS    = 3
SIGNAL_COOLDOWN   = 3600
GLOBAL_COOLDOWN   = 120
DAILY_MAX         = 15
TREND_MIN_SLOPE   = 0.0003
PRICE_CHECK_INTERVAL = 60  # فحص الأسعار كل دقيقة لتحديث الأهداف

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
#     متتبع الإشارات والأهداف
# ═══════════════════════════════════════════
sent_signals:     dict  = {}   # {key: timestamp}
daily_count:      int   = 0
daily_reset_date: date  = date.today()
last_signal_time: float = 0.0

# تتبع الرسائل المرسلة لتحديث الأهداف
# {message_id: {symbol, direction, price, tps, sl, tp_hit: [F,F,F,F,F], sl_hit: F, chat_id, text}}
active_trades: dict = {}


def check_daily_limit() -> bool:
    global daily_count, daily_reset_date
    today = date.today()
    if today != daily_reset_date:
        daily_count      = 0
        daily_reset_date = today
    return daily_count < DAILY_MAX


def increment_daily():
    global daily_count
    daily_count += 1


# ═══════════════════════════════════════════
#     سحب العملات
# ═══════════════════════════════════════════
async def get_symbols():
    url = "https://fapi.bitunix.com/api/v1/futures/market/tickers"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as r:
                data = await r.json()
        tickers = data.get('data', [])
        filtered = [
            (t['symbol'], float(t.get('volume24h', 0) or 0))
            for t in tickers
            if t.get('symbol', '').endswith('USDT')
            and 0 < float(t.get('lastPrice', 0) or 0) < MAX_PRICE
        ]
        filtered.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in filtered[:MAX_SYMBOLS]]
    except Exception as e:
        logger.error(f"خطأ العملات: {e}")
        return ["XRPUSDT","DOGEUSDT","ADAUSDT","TRXUSDT","SHIBUSDT",
                "DOTUSDT","LINKUSDT","LTCUSDT","MATICUSDT","ATOMUSDT"]


# ═══════════════════════════════════════════
#     سحب السعر الحالي فقط (للتتبع)
# ═══════════════════════════════════════════
async def get_current_price(symbol: str) -> float | None:
    url = "https://fapi.bitunix.com/api/v1/futures/market/tickers"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
                data = await r.json()
        for t in data.get('data', []):
            if t.get('symbol') == symbol:
                return float(t.get('lastPrice', 0) or 0)
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════
#     سحب الشمعدانات — 5m
# ═══════════════════════════════════════════
async def get_klines(symbol: str, interval="5m", limit=200):
    url = "https://fapi.bitunix.com/api/v1/futures/market/kline"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as r:
                data = await r.json()
        if data.get('code') == 0 and data.get('data'):
            df = pd.DataFrame(data['data'])
            for col in ['open','high','low','close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            if 'quoteVol' in df.columns:
                df['volume'] = pd.to_numeric(df['quoteVol'], errors='coerce')
            elif 'baseVol' in df.columns:
                df['volume'] = pd.to_numeric(df['baseVol'], errors='coerce')
            else:
                df['volume'] = 1.0
            df.dropna(subset=['open','high','low','close'], inplace=True)
            return df
    except Exception as e:
        logger.error(f"خطأ كلاين {symbol}: {e}")
    return None


# ═══════════════════════════════════════════
#     التحليل الفني الكامل — للإشارات التلقائية
# ═══════════════════════════════════════════
def technical_analysis(df: pd.DataFrame):
    if len(df) < 50:
        return None

    close  = df['close']
    high   = df['high']
    low    = df['low']
    volume = df['volume']
    price  = close.iloc[-1]

    rsi_val = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]

    macd_obj        = ta.trend.MACD(close)
    macd_line       = macd_obj.macd()
    macd_sig        = macd_obj.macd_signal()
    macd_cross_up   = macd_line.iloc[-1] > macd_sig.iloc[-1] and macd_line.iloc[-2] <= macd_sig.iloc[-2]
    macd_cross_down = macd_line.iloc[-1] < macd_sig.iloc[-1] and macd_line.iloc[-2] >= macd_sig.iloc[-2]
    macd_bull = macd_line.iloc[-1] > macd_sig.iloc[-1]
    macd_bear = macd_line.iloc[-1] < macd_sig.iloc[-1]

    ema9  = ta.trend.EMAIndicator(close, window=9).ema_indicator().iloc[-1]
    ema21 = ta.trend.EMAIndicator(close, window=21).ema_indicator().iloc[-1]
    ema50 = ta.trend.EMAIndicator(close, window=50).ema_indicator().iloc[-1]

    bb       = ta.volatility.BollingerBands(close, window=20)
    bb_lower = bb.bollinger_lband().iloc[-1]
    bb_upper = bb.bollinger_hband().iloc[-1]
    bb_pct   = bb.bollinger_pband().iloc[-1]

    vol_avg   = volume.rolling(20).mean().iloc[-1]
    vol_cur   = volume.iloc[-1]
    vol_surge = bool(vol_cur > vol_avg * 1.6)

    atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]

    price_change_1h = round(((price - close.iloc[-12]) / (close.iloc[-12] + 1e-10)) * 100, 2)
    price_change_4h = round(((price - close.iloc[-48]) / (close.iloc[-48] + 1e-10)) * 100, 2)

    rsi_long  = rsi_val < 45
    rsi_short = rsi_val > 55
    macd_long  = macd_cross_up or macd_bull
    macd_short = macd_cross_down or macd_bear
    ema_long  = price > ema9 and ema9 > ema21
    ema_short = price < ema9 and ema9 < ema21
    bb_long   = price <= bb_lower * 1.015
    bb_short  = price >= bb_upper * 0.985
    vol_long  = vol_surge and price > close.iloc[-2]
    vol_short = vol_surge and price < close.iloc[-2]

    long_indicators  = sum([rsi_long, macd_long, ema_long, bb_long, vol_long])
    short_indicators = sum([rsi_short, macd_short, ema_short, bb_short, vol_short])

    ema50_series = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    slope = (ema50_series.iloc[-1] - ema50_series.iloc[-10]) / (ema50_series.iloc[-10] + 1e-10)
    has_uptrend   = slope > TREND_MIN_SLOPE
    has_downtrend = slope < -TREND_MIN_SLOPE

    if (long_indicators >= MIN_INDICATORS and long_indicators > short_indicators and has_uptrend):
        direction = "LONG"
        score = long_indicators
    elif (short_indicators >= MIN_INDICATORS and short_indicators > long_indicators and has_downtrend):
        direction = "SHORT"
        score = short_indicators
    else:
        return None

    if macd_cross_up:    macd_desc = "تقاطع صاعد 🚀"
    elif macd_cross_down: macd_desc = "تقاطع هابط 🔻"
    elif macd_bull:       macd_desc = "صاعد 📈"
    else:                 macd_desc = "هابط 📉"

    return {
        "direction":       direction,
        "tech_score":      score,
        "indicator_count": score,
        "price":           price,
        "rsi":             round(rsi_val, 1),
        "macd_desc":       macd_desc,
        "macd_cross_up":   macd_cross_up,
        "macd_cross_down": macd_cross_down,
        "bb_pct":          round(bb_pct, 2),
        "vol_surge":       vol_surge,
        "atr":             atr,
        "ema9":            round(ema9, 6),
        "ema21":           round(ema21, 6),
        "ema50":           round(ema50, 6),
        "price_change_1h": price_change_1h,
        "price_change_4h": price_change_4h,
        "rsi_ok":  rsi_long if direction=="LONG" else rsi_short,
        "macd_ok": macd_long if direction=="LONG" else macd_short,
        "ema_ok":  ema_long  if direction=="LONG" else ema_short,
        "bb_ok":   bb_long   if direction=="LONG" else bb_short,
        "vol_ok":  vol_long  if direction=="LONG" else vol_short,
    }


# ═══════════════════════════════════════════
#     التحليل الخفيف — للطلبات اليدوية (دايماً يرجع)
# ═══════════════════════════════════════════
def technical_analysis_light(df: pd.DataFrame) -> dict:
    close  = df['close']
    high   = df['high']
    low    = df['low']
    volume = df['volume']
    price  = close.iloc[-1]

    rsi_val = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]

    macd_obj        = ta.trend.MACD(close)
    macd_line       = macd_obj.macd()
    macd_sig        = macd_obj.macd_signal()
    macd_cross_up   = macd_line.iloc[-1] > macd_sig.iloc[-1] and macd_line.iloc[-2] <= macd_sig.iloc[-2]
    macd_cross_down = macd_line.iloc[-1] < macd_sig.iloc[-1] and macd_line.iloc[-2] >= macd_sig.iloc[-2]
    macd_bull = macd_line.iloc[-1] > macd_sig.iloc[-1]
    macd_bear = macd_line.iloc[-1] < macd_sig.iloc[-1]

    ema9  = ta.trend.EMAIndicator(close, window=9).ema_indicator().iloc[-1]
    ema21 = ta.trend.EMAIndicator(close, window=21).ema_indicator().iloc[-1]
    ema50 = ta.trend.EMAIndicator(close, window=50).ema_indicator().iloc[-1]

    bb       = ta.volatility.BollingerBands(close, window=20)
    bb_lower = bb.bollinger_lband().iloc[-1]
    bb_upper = bb.bollinger_hband().iloc[-1]
    bb_pct   = bb.bollinger_pband().iloc[-1]

    vol_avg   = volume.rolling(20).mean().iloc[-1]
    vol_cur   = volume.iloc[-1]
    vol_surge = bool(vol_cur > vol_avg * 1.5)

    atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]

    price_change_1h = round(((price - close.iloc[-12]) / (close.iloc[-12] + 1e-10)) * 100, 2)
    price_change_4h = round(((price - close.iloc[-48]) / (close.iloc[-48] + 1e-10)) * 100, 2)

    long_pts = sum([
        rsi_val < 45,
        macd_cross_up or macd_bull,
        price > ema9 and ema9 > ema21,
        price <= bb_lower * 1.02,
        vol_surge and price > close.iloc[-2],
    ])
    short_pts = sum([
        rsi_val > 55,
        macd_cross_down or macd_bear,
        price < ema9 and ema9 < ema21,
        price >= bb_upper * 0.98,
        vol_surge and price < close.iloc[-2],
    ])

    direction = "LONG" if long_pts >= short_pts else "SHORT"
    score = max(long_pts, short_pts)

    if macd_cross_up:    macd_desc = "تقاطع صاعد 🚀"
    elif macd_cross_down: macd_desc = "تقاطع هابط 🔻"
    elif macd_bull:       macd_desc = "صاعد 📈"
    else:                 macd_desc = "هابط 📉"

    rsi_ok  = rsi_val < 45 if direction=="LONG" else rsi_val > 55
    macd_ok = (macd_cross_up or macd_bull) if direction=="LONG" else (macd_cross_down or macd_bear)
    ema_ok  = (price > ema9 and ema9 > ema21) if direction=="LONG" else (price < ema9 and ema9 < ema21)
    bb_ok   = price <= bb_lower * 1.02 if direction=="LONG" else price >= bb_upper * 0.98
    vol_ok  = (vol_surge and price > close.iloc[-2]) if direction=="LONG" else (vol_surge and price < close.iloc[-2])

    return {
        "direction":       direction,
        "tech_score":      score,
        "indicator_count": score,
        "price":           price,
        "rsi":             round(rsi_val, 1),
        "macd_desc":       macd_desc,
        "macd_cross_up":   macd_cross_up,
        "macd_cross_down": macd_cross_down,
        "bb_pct":          round(bb_pct, 2),
        "vol_surge":       vol_surge,
        "atr":             atr,
        "ema9":            round(ema9, 6),
        "ema21":           round(ema21, 6),
        "ema50":           round(ema50, 6),
        "price_change_1h": price_change_1h,
        "price_change_4h": price_change_4h,
        "rsi_ok":  rsi_ok,
        "macd_ok": macd_ok,
        "ema_ok":  ema_ok,
        "bb_ok":   bb_ok,
        "vol_ok":  vol_ok,
    }


# ═══════════════════════════════════════════
#     تحليل Claude AI
# ═══════════════════════════════════════════
async def ai_analysis(symbol: str, data: dict) -> dict:
    if not ANTHROPIC_API_KEY:
        return _fallback_analysis(data)

    coin = symbol.replace("USDT","")
    prompt = f"""أنت محلل تداول خبير للعملات الرقمية. حلل هذه الصفقة بدقة وأعط نسبة ثقة.

العملة: {coin}/USDT
الاتجاه: {data['direction']}
السعر الحالي: {data['price']}
RSI: {data['rsi']}
MACD: {data['macd_desc']}
موقع السعر في Bollinger: {data['bb_pct']} (0=أسفل، 1=أعلى)
تغيير آخر ساعة: {data['price_change_1h']}%
تغيير آخر 4 ساعات: {data['price_change_4h']}%
ارتفاع الحجم: {'نعم 🔥' if data['vol_surge'] else 'لا'}
عدد المؤشرات المتوافقة: {data['indicator_count']}/5
EMA9: {data['ema9']} | EMA21: {data['ema21']} | EMA50: {data['ema50']}

أجب بـ JSON فقط بدون أي نص إضافي:
{{"confidence": 75, "comment": "تعليق قصير بالعربي جملة واحدة"}}

نسبة الثقة من 1 إلى 99. كن صارماً ودقيقاً."""

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_API_KEY,
                         "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json={"model": "claude-haiku-4-5-20251001",
                      "max_tokens": 100,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=aiohttp.ClientTimeout(total=15)
            ) as r:
                resp = await r.json()

        text = resp['content'][0]['text'].strip()
        for marker in ["```json","```"]:
            text = text.replace(marker,"")
        result = json.loads(text.strip())
        return {"confidence": max(1,min(99,int(result.get("confidence",50)))),
                "comment": result.get("comment","")}
    except Exception as e:
        logger.error(f"خطأ AI: {e}")
        return _fallback_analysis(data)


async def ai_analysis_detailed(symbol: str, data: dict) -> dict:
    """تحليل AI أكثر تفصيلاً للطلبات اليدوية."""
    if not ANTHROPIC_API_KEY:
        return _fallback_analysis(data)

    coin = symbol.replace("USDT","")
    prompt = f"""أنت محلل تداول خبير للعملات الرقمية. حلل هذه العملة بشكل احترافي.

العملة: {coin}/USDT
السعر الحالي: {data['price']}
RSI: {data['rsi']}
MACD: {data['macd_desc']}
موقع البولنجر: {data['bb_pct']:.0%} (0%=أسفل، 100%=أعلى)
تغيير آخر ساعة: {data['price_change_1h']}%
تغيير آخر 4 ساعات: {data['price_change_4h']}%
ارتفاع الحجم: {'نعم 🔥' if data['vol_surge'] else 'لا'}
EMA9: {data['ema9']} | EMA21: {data['ema21']} | EMA50: {data['ema50']}
المؤشرات المتوافقة: {data['indicator_count']}/5

مطلوب:
1. تحديد الاتجاه الفعلي: LONG أو SHORT أو SIDEWAYS
2. نسبة ثقة دقيقة 1-99
3. تعليق قصير بالعربي (جملة واحدة) يشرح السبب الرئيسي

أجب بـ JSON فقط بدون backticks:
{{"confidence": 72, "direction": "LONG", "comment": "RSI في منطقة شراء مع تقاطع MACD صاعد"}}"""

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_API_KEY,
                         "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json={"model": "claude-haiku-4-5-20251001",
                      "max_tokens": 120,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=aiohttp.ClientTimeout(total=20)
            ) as r:
                resp = await r.json()

        text = resp['content'][0]['text'].strip()
        for marker in ["```json","```"]:
            text = text.replace(marker,"")
        result = json.loads(text.strip())
        ai_dir = result.get("direction", data['direction'])
        if ai_dir in ("LONG","SHORT"):
            data['direction'] = ai_dir
        return {"confidence": max(1,min(99,int(result.get("confidence",50)))),
                "comment": result.get("comment",""),
                "ai_direction": ai_dir}
    except Exception as e:
        logger.error(f"خطأ AI تفصيلي: {e}")
        return _fallback_analysis(data)


def _fallback_analysis(data: dict) -> dict:
    score = data.get('tech_score', 2)
    base  = int((score / 5) * 100)
    bonus = 5 if (data.get('macd_cross_up') or data.get('macd_cross_down')) else 0
    bonus += 5 if data.get('vol_surge') else 0
    conf = min(92, max(45, base + bonus))
    return {"confidence": conf, "comment": "تحليل فني متقدم"}


# ═══════════════════════════════════════════
#     الأهداف الاحترافية — مع ATR
# ═══════════════════════════════════════════
def make_targets(price: float, signal: str, atr: float = None):
    """أهداف ديناميكية بناءً على ATR إذا متوفر."""
    if atr and atr > 0:
        # استخدام ATR لأهداف أكثر دقة
        atr_ratio = atr / price
        m1 = 1 + (atr_ratio * 0.8)
        m2 = 1 + (atr_ratio * 1.5)
        m3 = 1 + (atr_ratio * 2.2)
        m4 = 1 + (atr_ratio * 3.5)
        m5 = 1 + (atr_ratio * 5.5)
        sl_m = 1 - (atr_ratio * 1.5)
    else:
        m1, m2, m3, m4, m5 = 1.005, 1.010, 1.015, 1.025, 1.040
        sl_m = 0.985

    if signal == "LONG":
        sl   = round(price * sl_m, 8)
        tps  = [round(price * m, 8) for m in [m1, m2, m3, m4, m5]]
        pcts = [f"+{round((m1-1)*100,1)}%", f"+{round((m2-1)*100,1)}%",
                f"+{round((m3-1)*100,1)}%", f"+{round((m4-1)*100,1)}%",
                f"+{round((m5-1)*100,1)}%"]
        sl_p = f"-{round((1-sl_m)*100,1)}%"
    else:
        inv_sl_m = 2 - sl_m
        sl   = round(price * inv_sl_m, 8)
        tps  = [round(price * (2-m), 8) for m in [m1, m2, m3, m4, m5]]
        pcts = [f"-{round((m1-1)*100,1)}%", f"-{round((m2-1)*100,1)}%",
                f"-{round((m3-1)*100,1)}%", f"-{round((m4-1)*100,1)}%",
                f"-{round((m5-1)*100,1)}%"]
        sl_p = f"+{round((1-sl_m)*100,1)}%"

    return sl, tps, pcts, sl_p


# ═══════════════════════════════════════════
#     تنسيق الأسعار
# ═══════════════════════════════════════════
def fmt(p: float) -> str:
    if p >= 1:       return f"{p:,.4f}$"
    elif p >= 0.01:  return f"{p:.6f}$"
    else:            return f"{p:.8f}$"


# ═══════════════════════════════════════════
#     بناء رسالة الإشارة التلقائية
# ═══════════════════════════════════════════
def build_signal_message(symbol, tech, ai, sl, tps, pcts, sl_p,
                         tp_hit=None, sl_hit=False):
    """بناء رسالة الإشارة — مع دعم تحديث الأهداف."""
    if tp_hit is None:
        tp_hit = [False] * 5

    coin   = symbol.replace("USDT","")
    now    = datetime.utcnow().strftime("%d.%m.%Y %H:%M UTC+0")
    price  = tech['price']
    signal = tech['direction']
    conf   = ai['confidence']

    emoji  = "🟢🟢🟢" if signal == "LONG" else "🔴🔴🔴"
    action = "لونغ" if signal == "LONG" else "شورت"

    if conf >= 85:   stars = "⭐⭐⭐⭐⭐"
    elif conf >= 75: stars = "⭐⭐⭐⭐"
    elif conf >= 70: stars = "⭐⭐⭐"
    else:            stars = "⭐⭐"

    filled = int(conf / 10)
    bar    = "█" * filled + "░" * (10 - filled)
    ai_line = f"🤖 {ai['comment']}\n" if ai.get('comment') else ""

    # الأهداف مع علامة ✅ عند تحقيق كل هدف
    tp_lines = ""
    tp_emojis = ["🎯","🎯","🎯","🎯","🏆"]
    for i, (tp, pct) in enumerate(zip(tps, pcts)):
        check = "✅" if tp_hit[i] else "⬜"
        label = "TP5 — إغلاق الكامل" if i == 4 else f"TP{i+1}"
        tp_lines += f"{check} {tp_emojis[i]} {label}: {fmt(tp)}  ({pct})\n"

    sl_status = "🔴 وقف الخسارة فُعِّل ❌" if sl_hit else f"🛑 SL: {fmt(sl)}  ({sl_p})"

    # ملخص الأداء
    hits = sum(tp_hit)
    status_line = ""
    if hits > 0:
        status_line = f"\n📊 الأهداف المحققة: {hits}/5\n"
    if sl_hit:
        status_line = "\n❌ الصفقة أُغلقت بوقف الخسارة\n"

    extras = "📊 ارتفاع في الحجم 🔥\n" if tech.get('vol_surge') else ""

    return (
        f"🕐 {now}\n"
        f"{emoji} {coin}/USDT — {action} {emoji}\n\n"
        f"➡️ الدخول: {fmt(price)}\n"
        f"{ai_line}"
        f"💯 الثقة: {conf}%  [{bar}]  {stars}\n"
        f"📡 المؤشرات: {tech.get('indicator_count',0)}/5\n"
        f"\n{tp_lines}\n"
        f"{sl_status}\n"
        f"{status_line}"
        f"📊 RSI: {tech['rsi']}  |  MACD: {tech['macd_desc']}\n"
        f"{extras}"
        f"⚠️ القرار النهائي لك. آخر TP يغلق الباقي.\n"
        f"🏦 Bitunix Futures"
    )


# ═══════════════════════════════════════════
#     بناء رسالة التحليل اليدوي
# ═══════════════════════════════════════════
def build_analysis_message(symbol: str, tech: dict, ai: dict) -> str:
    coin  = symbol.replace("USDT","")
    conf  = ai['confidence']
    price = tech['price']
    direction = tech['direction']

    # الاتجاه من AI إذا موجود
    ai_dir = ai.get('ai_direction', direction)
    is_sideways = ai_dir == "SIDEWAYS"

    if is_sideways:
        dir_text = "↔️ تذبذب / لا اتجاه واضح"
        dir_emoji = "⚠️"
    elif direction == "LONG":
        dir_text = "📈 صاعدة"
        dir_emoji = "🟢"
    else:
        dir_text = "📉 هابطة"
        dir_emoji = "🔴"

    if conf >= 80:   stars = "⭐⭐⭐⭐⭐"
    elif conf >= 65: stars = "⭐⭐⭐⭐"
    elif conf >= 50: stars = "⭐⭐⭐"
    else:            stars = "⭐⭐"

    rsi_status  = f"{'✅' if tech['rsi_ok'] else '❌'}  {tech['rsi']} — {'شراء' if tech['rsi'] < 45 else 'بيع' if tech['rsi'] > 55 else 'محايد'}"
    macd_status = f"{'✅' if tech['macd_ok'] else '❌'}  {tech['macd_desc']}"
    ema_status  = f"{'✅' if tech['ema_ok'] else '❌'}  EMA9={tech['ema9']} | EMA21={tech['ema21']}"
    vol_status  = f"{'✅' if tech['vol_ok'] else '❌'}  {'ارتفاع مفاجئ 🔥' if tech['vol_surge'] else 'حجم عادي'}"
    bb_status   = f"{'✅' if tech['bb_ok'] else '❌'}  موقع في البولنجر: {tech['bb_pct']:.0%}"

    # توصية دخول فقط إذا ليس تذبذب
    rec = ""
    if not is_sideways and conf >= 55:
        atr = tech.get('atr')
        sl, tps, pcts, sl_p = make_targets(price, direction, atr)
        rec = (
            f"\n💡 توصية الدخول:\n"
            f"  ➡️ دخول: {fmt(price)}\n"
            f"  🎯 TP1: {fmt(tps[0])}  ({pcts[0]})\n"
            f"  🎯 TP2: {fmt(tps[1])}  ({pcts[1]})\n"
            f"  🎯 TP3: {fmt(tps[2])}  ({pcts[2]})\n"
            f"  🏆 TP4: {fmt(tps[3])}  ({pcts[3]})\n"
            f"  🛑 SL:  {fmt(sl)}  ({sl_p})\n"
        )
    elif is_sideways:
        rec = f"\n💡 السوق في تذبذب — انتظر كسر واضح للاتجاه قبل الدخول.\n"

    ai_line = f"🤖 AI: {ai['comment']}\n" if ai.get('comment') else ""

    return (
        f"🔎 تحليل عملة عند الطلب\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"{dir_emoji} العملة: {coin}/USDT\n"
        f"💰 السعر: {fmt(price)}\n"
        f"📊 الاتجاه: {dir_text}\n"
        f"💯 الثقة: {conf}%  {stars}\n"
        f"{ai_line}"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"🧪 المؤشرات:\n"
        f"  📌 RSI:   {rsi_status}\n"
        f"  📌 MACD:  {macd_status}\n"
        f"  📌 EMA:   {ema_status}\n"
        f"  📌 حجم:   {vol_status}\n"
        f"  📌 BB:    {bb_status}\n"
        f"  📡 توافق: {tech['indicator_count']}/5 مؤشرات\n"
        f"{rec}"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"⚠️ للمعلومات فقط، القرار لك.\n"
        f"🏦 Bitunix Futures"
    )


# ═══════════════════════════════════════════
#     🆕 تتبع الأهداف وتحديث الرسائل
# ═══════════════════════════════════════════
async def check_and_update_trades(bot: Bot):
    """يفحص الأسعار الحالية ويحدث رسائل الإشارات عند تحقيق هدف أو SL."""
    if not active_trades:
        return

    to_remove = []
    for msg_id, trade in list(active_trades.items()):
        try:
            current_price = await get_current_price(trade['symbol'])
            if not current_price:
                continue

            direction  = trade['direction']
            tps        = trade['tps']
            sl         = trade['sl']
            tp_hit     = trade['tp_hit']
            sl_hit     = trade['sl_hit']
            updated    = False

            if sl_hit:
                to_remove.append(msg_id)
                continue

            # فحص وقف الخسارة
            if direction == "LONG" and current_price <= sl:
                trade['sl_hit'] = True
                updated = True
            elif direction == "SHORT" and current_price >= sl:
                trade['sl_hit'] = True
                updated = True

            # فحص الأهداف
            if not trade['sl_hit']:
                for i, tp in enumerate(tps):
                    if tp_hit[i]:
                        continue
                    if direction == "LONG" and current_price >= tp:
                        trade['tp_hit'][i] = True
                        updated = True
                    elif direction == "SHORT" and current_price <= tp:
                        trade['tp_hit'][i] = True
                        updated = True

            # إذا تحقق كل الأهداف نزيل الصفقة
            if all(trade['tp_hit']):
                to_remove.append(msg_id)

            if updated:
                # بناء الرسالة المحدثة
                new_text = build_signal_message(
                    symbol  = trade['symbol'],
                    tech    = trade['tech'],
                    ai      = trade['ai'],
                    sl      = trade['sl'],
                    tps     = trade['tps'],
                    pcts    = trade['pcts'],
                    sl_p    = trade['sl_p'],
                    tp_hit  = trade['tp_hit'],
                    sl_hit  = trade['sl_hit'],
                )
                try:
                    await bot.edit_message_text(
                        chat_id    = trade['chat_id'],
                        message_id = msg_id,
                        text       = new_text
                    )
                    logger.info(f"✏️ تحديث رسالة {trade['symbol']} — TPs: {trade['tp_hit']} SL: {trade['sl_hit']}")
                except Exception as edit_err:
                    logger.warning(f"تعذر تعديل الرسالة: {edit_err}")

        except Exception as e:
            logger.error(f"خطأ تتبع {msg_id}: {e}")

    for mid in to_remove:
        active_trades.pop(mid, None)


# ═══════════════════════════════════════════
#     الفحص التلقائي
# ═══════════════════════════════════════════
async def scan_market(bot: Bot):
    global last_signal_time

    if not check_daily_limit():
        logger.info(f"⛔ الحد اليومي ({DAILY_MAX} إشارة)")
        return

    logger.info("🔍 بدأ الفحص...")
    symbols = await get_symbols()
    found   = 0

    for sym in symbols:
        if not check_daily_limit():
            break

        try:
            df = await get_klines(sym)
            if df is None or len(df) < 50:
                await asyncio.sleep(0.2)
                continue

            tech = technical_analysis(df)
            if not tech:
                await asyncio.sleep(0.2)
                continue

            key      = f"{sym}_{tech['direction']}"
            now_time = asyncio.get_event_loop().time()

            if now_time - sent_signals.get(key, 0) < SIGNAL_COOLDOWN:
                continue
            if now_time - last_signal_time < GLOBAL_COOLDOWN:
                continue

            price = tech['price']
            if price >= MAX_PRICE or price <= 0:
                continue

            ai = await ai_analysis(sym, tech)
            if ai['confidence'] < MIN_CONFIDENCE:
                await asyncio.sleep(0.2)
                continue

            atr = tech.get('atr')
            sl, tps, pcts, sl_p = make_targets(price, tech['direction'], atr)
            msg = build_signal_message(sym, tech, ai, sl, tps, pcts, sl_p)

            sent_msg = await bot.send_message(chat_id=CHAT_ID, text=msg)

            # ✅ تسجيل الصفقة للتتبع
            active_trades[sent_msg.message_id] = {
                'symbol':    sym,
                'direction': tech['direction'],
                'tech':      tech,
                'ai':        ai,
                'tps':       tps,
                'sl':        sl,
                'pcts':      pcts,
                'sl_p':      sl_p,
                'tp_hit':    [False]*5,
                'sl_hit':    False,
                'chat_id':   CHAT_ID,
            }

            sent_signals[key] = now_time
            last_signal_time  = now_time
            increment_daily()
            found += 1
            logger.info(f"📤 {sym} {tech['direction']} {ai['confidence']}% — اليوم: {daily_count}/{DAILY_MAX}")
            await asyncio.sleep(2)

        except Exception as e:
            logger.error(f"خطأ {sym}: {e}")
        await asyncio.sleep(0.3)

    logger.info(f"✅ انتهى — {found} إشارة | اليوم: {daily_count}/{DAILY_MAX}")


# ═══════════════════════════════════════════
#     أوامر التلغرام
# ═══════════════════════════════════════════
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    syms = await get_symbols()
    ai_status = "✅ مفعّل" if ANTHROPIC_API_KEY else "⚡ تحليل فني"
    await update.message.reply_text(
        f"🌟 مرحباً في سراب للإشارات! 🌟\n\n"
        f"📡 يراقب {len(syms)} عملة (تحت $10)\n"
        f"⏱ فحص تلقائي كل 3 دقائق\n"
        f"🤖 AI: {ai_status}\n"
        f"💯 حد الثقة الأدنى: 70%\n"
        f"🎯 الحد اليومي: {DAILY_MAX} إشارة\n"
        f"✏️ تحديث تلقائي للأهداف في الرسالة\n\n"
        f"الأوامر:\n"
        f"/scan — فحص فوري\n"
        f"/status — حالة البوت\n"
        f"🔎 أرسل رمز عملة (مثل XRP أو XRPUSDT) لتحليلها فوراً"
    )


async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 جاري الفحص...")
    await scan_market(context.bot)
    await update.message.reply_text(f"✅ انتهى! إشارات اليوم: {daily_count}/{DAILY_MAX}")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    syms = await get_symbols()
    ai_status = "✅ مفعّل" if ANTHROPIC_API_KEY else "⚡ تحليل فني"
    await update.message.reply_text(
        f"✅ سراب يعمل\n\n"
        f"⏱ فحص كل: 3 دقائق\n"
        f"📊 العملات: {len(syms)}\n"
        f"💰 الحد: $10\n"
        f"💯 حد الثقة: 70%+\n"
        f"📡 إشارات اليوم: {daily_count}/{DAILY_MAX}\n"
        f"📈 صفقات نشطة (تتبع): {len(active_trades)}\n"
        f"🤖 AI: {ai_status}"
    )


# ═══════════════════════════════════════════
#     تحليل عملة عند الطلب — دايماً يعطي رد
# ═══════════════════════════════════════════
async def handle_coin_request(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip().upper()

    # تجاهل الرسائل الطويلة أو التي تحتوي مسافات
    if len(text) < 2 or len(text) > 12 or ' ' in text:
        return

    symbol = text if text.endswith("USDT") else f"{text}USDT"

    await update.message.reply_text(f"🔎 جاري تحليل {symbol}...")

    try:
        df = await get_klines(symbol, interval="5m", limit=200)
        if df is None or len(df) < 50:
            await update.message.reply_text(
                f"⚠️ لم أجد بيانات لـ {symbol}\n"
                f"تأكد من الاسم — مثال: XRP أو XRPUSDT"
            )
            return

        # ✅ تحليل خفيف — يرجع دائماً نتيجة
        tech = technical_analysis_light(df)

        # ✅ AI تفصيلي يحدد الاتجاه الحقيقي
        ai = await ai_analysis_detailed(symbol, tech)

        msg = build_analysis_message(symbol, tech, ai)
        await update.message.reply_text(msg)

    except Exception as e:
        logger.error(f"خطأ تحليل {symbol}: {e}")
        await update.message.reply_text(f"❌ خطأ في تحليل {symbol}. حاول مرة أخرى.")


# ═══════════════════════════════════════════
#     Jobs التلقائية
# ═══════════════════════════════════════════
async def auto_scan(context: ContextTypes.DEFAULT_TYPE):
    await scan_market(context.bot)


async def auto_price_check(context: ContextTypes.DEFAULT_TYPE):
    """يفحص الأسعار ويحدث رسائل الأهداف."""
    await check_and_update_trades(context.bot)


# ═══════════════════════════════════════════
#     تشغيل البوت
# ═══════════════════════════════════════════
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("scan",   cmd_scan))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_coin_request))

    # فحص السوق كل 3 دقائق
    app.job_queue.run_repeating(auto_scan, interval=CHECK_INTERVAL, first=20)
    # تتبع الأهداف وتحديث الرسائل كل دقيقة
    app.job_queue.run_repeating(auto_price_check, interval=PRICE_CHECK_INTERVAL, first=30)

    logger.info("🚀 سراب للإشارات يعمل — جودة عالية + تتبع أهداف!")
    app.run_polling()


if __name__ == "__main__":
    main()
