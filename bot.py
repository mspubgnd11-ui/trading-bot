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

CHECK_INTERVAL    = 180      # فحص كل 3 دقائق
MAX_PRICE         = 10.0     # عملات تحت $10
MAX_SYMBOLS       = 150      # أكبر 150 عملة
MIN_CONFIDENCE    = 70       # ✅ رُفع من 25 إلى 70
MIN_INDICATORS    = 3        # ✅ جديد: لازم 3 مؤشرات على الأقل تتوافق
SIGNAL_COOLDOWN   = 3600     # ✅ رُفع إلى ساعة كاملة بين نفس العملة
GLOBAL_COOLDOWN   = 120      # ✅ جديد: دقيقتين بين أي إشارتين
DAILY_MAX         = 15       # ✅ جديد: حد يومي 15 إشارة
TREND_MIN_SLOPE   = 0.0003   # ✅ جديد: للتأكد من وجود اتجاه وليس تذبذب

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
#     متتبع الإشارات اليومية
# ═══════════════════════════════════════════
sent_signals:     dict = {}   # {key: timestamp}
daily_count:      int  = 0
daily_reset_date: date = date.today()
last_signal_time: float = 0.0  # آخر وقت إرسال أي إشارة


def check_daily_limit() -> bool:
    """يرجع True إذا لم نصل للحد اليومي بعد."""
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
#     سحب العملات من Bitunix
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
        return ["XRPUSDT", "DOGEUSDT", "ADAUSDT", "TRXUSDT", "SHIBUSDT",
                "DOTUSDT", "LINKUSDT", "LTCUSDT", "MATICUSDT", "ATOMUSDT"]


# ═══════════════════════════════════════════
#     سحب الشمعدانات — ✅ تغيير إلى 5m
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
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            if 'quoteVol' in df.columns:
                df['volume'] = pd.to_numeric(df['quoteVol'], errors='coerce')
            elif 'baseVol' in df.columns:
                df['volume'] = pd.to_numeric(df['baseVol'], errors='coerce')
            else:
                df['volume'] = 1.0
            df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
            return df
    except Exception as e:
        logger.error(f"خطأ كلاين {symbol}: {e}")
    return None


# ═══════════════════════════════════════════
#     التحليل الفني — ✅ مع فلتر الاتجاه والمؤشرات
# ═══════════════════════════════════════════
def technical_analysis(df: pd.DataFrame):
    if len(df) < 50:
        return None

    close  = df['close']
    high   = df['high']
    low    = df['low']
    volume = df['volume']
    price  = close.iloc[-1]

    # ─── RSI ───────────────────────────────
    rsi_val   = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    rsi_long  = rsi_val < 45
    rsi_short = rsi_val > 55

    # ─── MACD ──────────────────────────────
    macd_obj        = ta.trend.MACD(close)
    macd_line       = macd_obj.macd()
    macd_sig        = macd_obj.macd_signal()
    macd_cross_up   = macd_line.iloc[-1] > macd_sig.iloc[-1] and macd_line.iloc[-2] <= macd_sig.iloc[-2]
    macd_cross_down = macd_line.iloc[-1] < macd_sig.iloc[-1] and macd_line.iloc[-2] >= macd_sig.iloc[-2]
    macd_bull       = macd_line.iloc[-1] > macd_sig.iloc[-1]
    macd_bear       = macd_line.iloc[-1] < macd_sig.iloc[-1]
    macd_long       = macd_cross_up or macd_bull
    macd_short      = macd_cross_down or macd_bear

    # ─── EMA ───────────────────────────────
    ema9  = ta.trend.EMAIndicator(close, window=9).ema_indicator().iloc[-1]
    ema21 = ta.trend.EMAIndicator(close, window=21).ema_indicator().iloc[-1]
    ema50 = ta.trend.EMAIndicator(close, window=50).ema_indicator().iloc[-1]
    ema_long  = price > ema9 and ema9 > ema21
    ema_short = price < ema9 and ema9 < ema21

    # ─── Bollinger ─────────────────────────
    bb       = ta.volatility.BollingerBands(close, window=20)
    bb_lower = bb.bollinger_lband().iloc[-1]
    bb_upper = bb.bollinger_hband().iloc[-1]
    bb_pct   = bb.bollinger_pband().iloc[-1]
    bb_long  = price <= bb_lower * 1.015
    bb_short = price >= bb_upper * 0.985

    # ─── Volume ────────────────────────────
    vol_avg   = volume.rolling(20).mean().iloc[-1]
    vol_cur   = volume.iloc[-1]
    vol_surge = bool(vol_cur > vol_avg * 1.6)  # رُفع المعيار لـ 1.6
    vol_long  = vol_surge and price > close.iloc[-2]
    vol_short = vol_surge and price < close.iloc[-2]

    # ─── ATR ───────────────────────────────
    atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]

    # ─── ✅ فلتر الاتجاه (منع التذبذب) ─────
    # نحسب ميل EMA50 على آخر 10 شموع
    ema50_series = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    slope = (ema50_series.iloc[-1] - ema50_series.iloc[-10]) / (ema50_series.iloc[-10] + 1e-10)
    has_uptrend   = slope > TREND_MIN_SLOPE
    has_downtrend = slope < -TREND_MIN_SLOPE
    has_trend     = has_uptrend or has_downtrend

    # ─── تغيير السعر ───────────────────────
    price_change_1h = round(((price - close.iloc[-12]) / close.iloc[-12]) * 100, 2)  # 12×5m = 1h
    price_change_4h = round(((price - close.iloc[-48]) / close.iloc[-48]) * 100, 2)  # 48×5m = 4h

    # ─── ✅ حساب المؤشرات المتوافقة ──────────
    long_indicators  = sum([rsi_long, macd_long, ema_long, bb_long, vol_long])
    short_indicators = sum([rsi_short, macd_short, ema_short, bb_short, vol_short])

    # ─── تحديد الاتجاه بشروط صارمة ─────────
    if (long_indicators >= MIN_INDICATORS and long_indicators > short_indicators
            and has_trend and has_uptrend):
        direction   = "LONG"
        tech_score  = long_indicators
        indicator_n = long_indicators
    elif (short_indicators >= MIN_INDICATORS and short_indicators > long_indicators
            and has_trend and has_downtrend):
        direction   = "SHORT"
        tech_score  = short_indicators
        indicator_n = short_indicators
    else:
        return None  # لا يكفي توافق أو لا يوجد اتجاه واضح

    # ─── وصف MACD ──────────────────────────
    if macd_cross_up:    macd_desc = "تقاطع صاعد 🚀"
    elif macd_cross_down: macd_desc = "تقاطع هابط 🔻"
    elif macd_bull:       macd_desc = "صاعد 📈"
    else:                 macd_desc = "هابط 📉"

    return {
        "direction":       direction,
        "tech_score":      tech_score,
        "indicator_count": indicator_n,
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
        # تفاصيل المؤشرات لتحليل الطلب
        "rsi_ok":    rsi_long if direction == "LONG" else rsi_short,
        "macd_ok":   macd_long if direction == "LONG" else macd_short,
        "ema_ok":    ema_long if direction == "LONG" else ema_short,
        "bb_ok":     bb_long if direction == "LONG" else bb_short,
        "vol_ok":    vol_long if direction == "LONG" else vol_short,
    }


# ═══════════════════════════════════════════
#     تحليل Claude AI — ✅ مع fallback قوي
# ═══════════════════════════════════════════
async def ai_analysis(symbol: str, data: dict) -> dict:
    """يستدعي AI فقط إذا وجد ANTHROPIC_API_KEY، وإلا يستخدم التحليل الفني."""
    if not ANTHROPIC_API_KEY:
        return _fallback_analysis(data)

    coin   = symbol.replace("USDT", "")
    prompt = f"""أنت محلل تداول خبير للعملات الرقمية. حلل هذه الصفقة بدقة وأعط نسبة ثقة.

العملة: {coin}/USDT
الاتجاه: {data['direction']}
السعر الحالي: {data['price']}
RSI: {data['rsi']}
MACD: {data['macd_desc']}
موقع السعر في Bollinger: {data['bb_pct']} (0=أسفل، 1=أعلى)
تغيير السعر آخر ساعة: {data['price_change_1h']}%
تغيير السعر آخر 4 ساعات: {data['price_change_4h']}%
ارتفاع الحجم: {'نعم 🔥' if data['vol_surge'] else 'لا'}
عدد المؤشرات المتوافقة: {data['indicator_count']}/5
EMA9: {data['ema9']} | EMA21: {data['ema21']} | EMA50: {data['ema50']}

أجب بـ JSON فقط بهذا الشكل بالضبط، بدون أي نص إضافي:
{{"confidence": 75, "comment": "تعليق قصير بالعربي جملة واحدة فقط"}}

نسبة الثقة من 1 إلى 99. كن صارماً ودقيقاً."""

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=aiohttp.ClientTimeout(total=15)
            ) as r:
                resp = await r.json()

        text = resp['content'][0]['text'].strip()
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        result = json.loads(text)
        return {
            "confidence": max(1, min(99, int(result.get("confidence", 50)))),
            "comment":    result.get("comment", "")
        }
    except Exception as e:
        logger.error(f"خطأ AI (fallback): {e}")
        return _fallback_analysis(data)  # ✅ fallback تلقائي


def _fallback_analysis(data: dict) -> dict:
    """تحليل فني بديل بدون AI."""
    score = data['tech_score']
    # حساب الثقة بناءً على المؤشرات وقوة الإشارة
    base = int((score / 5) * 100)
    bonus = 0
    if data.get('macd_cross_up') or data.get('macd_cross_down'):
        bonus += 5
    if data.get('vol_surge'):
        bonus += 5
    conf = min(95, max(50, base + bonus))
    return {"confidence": conf, "comment": "تحليل فني متقدم"}


# ═══════════════════════════════════════════
#     الأهداف ووقف الخسارة
# ═══════════════════════════════════════════
def make_targets(price: float, signal: str):
    if signal == "LONG":
        sl   = round(price * 0.985, 8)
        tps  = [round(price * m, 8) for m in [1.005, 1.010, 1.015, 1.025, 1.040]]
        pcts = ["+0.5%", "+1%", "+1.5%", "+2.5%", "+4%"]
        sl_p = "-1.5%"
    else:
        sl   = round(price * 1.015, 8)
        tps  = [round(price * m, 8) for m in [0.995, 0.990, 0.985, 0.975, 0.960]]
        pcts = ["-0.5%", "-1%", "-1.5%", "-2.5%", "-4%"]
        sl_p = "+1.5%"
    return sl, tps, pcts, sl_p


# ═══════════════════════════════════════════
#     تنسيق الرسائل
# ═══════════════════════════════════════════
def fmt(p: float) -> str:
    if p >= 1:       return f"{p:,.4f}$"
    elif p >= 0.01:  return f"{p:.6f}$"
    else:            return f"{p:.8f}$"


def build_message(symbol, tech, ai, sl, tps, pcts, sl_p):
    coin   = symbol.replace("USDT", "")
    now    = datetime.utcnow().strftime("%d.%m.%Y %H:%M:%S UTC+0")
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

    ai_line = f"\n🤖 AI: {ai['comment']}\n" if ai['comment'] else ""

    extras = ""
    if tech['vol_surge']:
        extras += "📊 الحجم: ارتفاع مفاجئ 🔥\n"

    indicators_count = tech.get('indicator_count', 0)

    return (
        f"تحديث: {now}\n"
        f"{emoji} {coin}/USDT  {action}  {emoji}\n\n"
        f"➡️ نقطة الدخول: {fmt(price)}\n"
        f"{ai_line}\n"
        f"💯 نسبة الثقة: {conf}%\n"
        f"[{bar}]\n"
        f"{stars}\n"
        f"📡 المؤشرات المتوافقة: {indicators_count}/5\n\n"
        f"🎯 TP1: {fmt(tps[0])}  ({pcts[0]})\n"
        f"🎯 TP2: {fmt(tps[1])}  ({pcts[1]})\n"
        f"🎯 TP3: {fmt(tps[2])}  ({pcts[2]})\n"
        f"🎯 TP4: {fmt(tps[3])}  ({pcts[3]})\n"
        f"🎯 TP5 (إغلاق): {fmt(tps[4])}  ({pcts[4]})\n\n"
        f"🛑 SL: {fmt(sl)}  ({sl_p})\n\n"
        f"📊 RSI: {tech['rsi']}\n"
        f"📈 MACD: {tech['macd_desc']}\n"
        f"{extras}"
        f"⚠️ التثبيت حسب تقديرك، آخر TP يغلق الباقي.\n"
        f"🏦 Bitunix Futures"
    )


# ═══════════════════════════════════════════
#     ✅ جديد: رسالة تحليل العملة عند الطلب
# ═══════════════════════════════════════════
def build_analysis_message(symbol: str, tech: dict, ai: dict) -> str:
    coin  = symbol.replace("USDT", "")
    conf  = ai['confidence']
    price = tech['price']

    direction_ar = "📈 صاعدة" if tech['direction'] == "LONG" else "📉 هابطة"

    if conf >= 85:   stars = "⭐⭐⭐⭐⭐"
    elif conf >= 75: stars = "⭐⭐⭐⭐"
    elif conf >= 70: stars = "⭐⭐⭐"
    else:            stars = "⭐⭐"

    rsi_status  = f"{'✅' if tech['rsi_ok'] else '❌'}  {tech['rsi']} — {'منطقة شراء' if tech['rsi'] < 45 else 'منطقة بيع' if tech['rsi'] > 55 else 'محايد'}"
    macd_status = f"{'✅' if tech['macd_ok'] else '❌'}  {tech['macd_desc']}"
    ema_status  = f"{'✅' if tech['ema_ok'] else '❌'}  EMA9={tech['ema9']} | EMA21={tech['ema21']}"
    vol_status  = f"{'✅' if tech['vol_ok'] else '❌'}  {'ارتفاع مفاجئ 🔥' if tech['vol_surge'] else 'حجم عادي'}"
    bb_status   = f"{'✅' if tech['bb_ok'] else '❌'}  موقع السعر في البولنجر: {tech['bb_pct']:.0%}"

    rec = ""
    if conf >= 70 and tech['direction'] == "LONG":
        sl, tps, pcts, sl_p = make_targets(price, "LONG")
        rec = (
            f"\n💡 توصية الدخول:\n"
            f"  ➡️ دخول: {fmt(price)}\n"
            f"  🎯 TP1: {fmt(tps[0])} ({pcts[0]})\n"
            f"  🎯 TP2: {fmt(tps[1])} ({pcts[1]})\n"
            f"  🎯 TP3: {fmt(tps[2])} ({pcts[2]})\n"
            f"  🛑 SL:  {fmt(sl)} ({sl_p})\n"
        )
    elif conf >= 70 and tech['direction'] == "SHORT":
        sl, tps, pcts, sl_p = make_targets(price, "SHORT")
        rec = (
            f"\n💡 توصية الدخول:\n"
            f"  ➡️ دخول: {fmt(price)}\n"
            f"  🎯 TP1: {fmt(tps[0])} ({pcts[0]})\n"
            f"  🎯 TP2: {fmt(tps[1])} ({pcts[1]})\n"
            f"  🎯 TP3: {fmt(tps[2])} ({pcts[2]})\n"
            f"  🛑 SL:  {fmt(sl)} ({sl_p})\n"
        )

    ai_line = f"🤖 AI: {ai['comment']}\n" if ai.get('comment') else ""

    return (
        f"🔎 تحليل العملة عند الطلب\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"💎 العملة: {coin}/USDT\n"
        f"💰 السعر الحالي: {fmt(price)}\n"
        f"📊 الاتجاه: {direction_ar}\n"
        f"💯 نسبة الثقة: {conf}%  {stars}\n"
        f"{ai_line}"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"🧪 الأسباب:\n"
        f"  📌 RSI:    {rsi_status}\n"
        f"  📌 MACD:   {macd_status}\n"
        f"  📌 EMA:    {ema_status}\n"
        f"  📌 حجم:    {vol_status}\n"
        f"  📌 BB:     {bb_status}\n"
        f"  📡 توافق: {tech['indicator_count']}/5 مؤشرات\n"
        f"{rec}"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"⚠️ للمعلومات فقط، القرار لك.\n"
        f"🏦 Bitunix Futures"
    )


# ═══════════════════════════════════════════
#     الفحص التلقائي — ✅ مع كل الفلاتر الجديدة
# ═══════════════════════════════════════════
async def scan_market(bot: Bot):
    global last_signal_time

    if not check_daily_limit():
        logger.info(f"⛔ وصلنا للحد اليومي ({DAILY_MAX} إشارة)")
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

            # ✅ فلتر cooldown لنفس العملة
            if now_time - sent_signals.get(key, 0) < SIGNAL_COOLDOWN:
                await asyncio.sleep(0.1)
                continue

            # ✅ فلتر global cooldown (منع إرسال إشارتين متقاربتين)
            if now_time - last_signal_time < GLOBAL_COOLDOWN:
                await asyncio.sleep(0.1)
                continue

            price = tech['price']
            if price >= MAX_PRICE or price <= 0:
                await asyncio.sleep(0.2)
                continue

            # ✅ AI فقط للعملات القوية
            ai = await ai_analysis(sym, tech)

            if ai['confidence'] < MIN_CONFIDENCE:
                await asyncio.sleep(0.2)
                continue

            sl, tps, pcts, sl_p = make_targets(price, tech['direction'])
            msg = build_message(sym, tech, ai, sl, tps, pcts, sl_p)

            await bot.send_message(chat_id=CHAT_ID, text=msg)
            sent_signals[key] = now_time
            last_signal_time   = now_time
            increment_daily()
            found += 1
            logger.info(f"📤 {sym} {tech['direction']} {ai['confidence']}% (يوم: {daily_count}/{DAILY_MAX})")
            await asyncio.sleep(2)

        except Exception as e:
            logger.error(f"خطأ {sym}: {e}")
        await asyncio.sleep(0.3)

    logger.info(f"✅ انتهى — {found} إشارة جديدة | إجمالي اليوم: {daily_count}/{DAILY_MAX}")


# ═══════════════════════════════════════════
#     أوامر التلغرام
# ═══════════════════════════════════════════
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    syms = await get_symbols()
    ai_status = "✅ مفعّل" if ANTHROPIC_API_KEY else "⚡ تحليل فني (بدون AI)"
    await update.message.reply_text(
        f"🌟 مرحباً بك في سراب للإشارات! 🌟\n\n"
        f"📡 يراقب {len(syms)} عملة (تحت $10)\n"
        f"⏱ فحص تلقائي كل 3 دقائق\n"
        f"🤖 تحليل AI: {ai_status}\n"
        f"💯 حد الثقة الأدنى: 70%\n"
        f"📊 RSI + MACD + EMA + Bollinger + Volume\n"
        f"🎯 الحد اليومي: {DAILY_MAX} إشارة عالية الجودة\n"
        f"🔄 لونغ وشورت\n\n"
        f"الأوامر:\n"
        f"/scan — فحص فوري\n"
        f"/status — حالة البوت\n"
        f"🔎 أرسل اسم عملة (مثل XRPUSDT) لتحليلها فوراً"
    )


async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 جاري الفحص، انتظر قليلاً...")
    await scan_market(context.bot)
    await update.message.reply_text(f"✅ انتهى الفحص! إشارات اليوم: {daily_count}/{DAILY_MAX}")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    syms = await get_symbols()
    ai_status = "✅ مفعّل" if ANTHROPIC_API_KEY else "⚡ تحليل فني"
    await update.message.reply_text(
        f"✅ سراب يعمل بشكل طبيعي\n\n"
        f"⏱ الفحص كل: 3 دقائق\n"
        f"📊 العملات: {len(syms)}\n"
        f"💰 الحد الأقصى: $10\n"
        f"💯 حد الثقة: 70%+\n"
        f"📡 الإشارات اليوم: {daily_count}/{DAILY_MAX}\n"
        f"🤖 AI: {ai_status}"
    )


# ═══════════════════════════════════════════
#     ✅ جديد: تحليل عملة عند الطلب
# ═══════════════════════════════════════════
async def handle_coin_request(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """يستقبل اسم عملة ويحللها فوراً."""
    text   = update.message.text.strip().upper()
    symbol = text if text.endswith("USDT") else f"{text}USDT"

    await update.message.reply_text(f"🔎 جاري تحليل {symbol}، لحظة...")

    try:
        # نستخدم 5m للتحليل الفوري
        df = await get_klines(symbol, interval="5m", limit=200)
        if df is None or len(df) < 50:
            await update.message.reply_text(
                f"⚠️ لم أتمكن من جلب بيانات {symbol}.\n"
                f"تأكد من اسم العملة (مثال: XRP أو XRPUSDT)."
            )
            return

        tech = technical_analysis(df)

        if not tech:
            price = df['close'].iloc[-1]
            await update.message.reply_text(
                f"📊 {symbol}\n"
                f"💰 السعر: {fmt(price)}\n\n"
                f"⚠️ لا يوجد اتجاه واضح حالياً.\n"
                f"السوق في تذبذب أو المؤشرات متضاربة.\n"
                f"يُنصح بالانتظار لاتجاه أوضح."
            )
            return

        ai = await ai_analysis(symbol, tech)
        msg = build_analysis_message(symbol, tech, ai)
        await update.message.reply_text(msg)

    except Exception as e:
        logger.error(f"خطأ تحليل {symbol}: {e}")
        await update.message.reply_text(f"❌ حدث خطأ أثناء تحليل {symbol}. حاول مرة أخرى.")


async def auto_scan(context: ContextTypes.DEFAULT_TYPE):
    await scan_market(context.bot)


# ═══════════════════════════════════════════
#     تشغيل البوت
# ═══════════════════════════════════════════
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("scan",   cmd_scan))
    app.add_handler(CommandHandler("status", cmd_status))

    # ✅ Handler لتحليل العملات عند الطلب (أي رسالة نصية عادية)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_coin_request))

    app.job_queue.run_repeating(auto_scan, interval=CHECK_INTERVAL, first=20)

    logger.info("🚀 سراب للإشارات يعمل — جودة عالية فقط!")
    app.run_polling()


if __name__ == "__main__":
    main()
