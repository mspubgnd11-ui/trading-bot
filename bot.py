import os
import asyncio
import logging
import json
from datetime import datetime
import aiohttp
from telegram import Bot
from telegram.ext import Application, CommandHandler, ContextTypes
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

CHECK_INTERVAL  = 120    # فحص كل دقيقتين
MAX_PRICE       = 10.0   # عملات تحت $10
MAX_SYMBOLS     = 150    # أكبر 150 عملة
MIN_CONFIDENCE  = 25     # أدنى نسبة ثقة
SIGNAL_COOLDOWN = 1200   # 20 دقيقة بين نفس الإشارة

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        return ["XRPUSDT","DOGEUSDT","ADAUSDT","TRXUSDT","SHIBUSDT",
                "DOTUSDT","LINKUSDT","LTCUSDT","MATICUSDT","ATOMUSDT"]

# ═══════════════════════════════════════════
#     سحب الشمعدانات
# ═══════════════════════════════════════════
async def get_klines(symbol: str, interval="15m", limit=200):
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
#     التحليل الفني
# ═══════════════════════════════════════════
def technical_analysis(df: pd.DataFrame):
    if len(df) < 50:
        return None

    close  = df['close']
    high   = df['high']
    low    = df['low']
    volume = df['volume']
    price  = close.iloc[-1]

    # RSI
    rsi_val = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]

    # MACD
    macd_obj  = ta.trend.MACD(close)
    macd_line = macd_obj.macd()
    macd_sig  = macd_obj.macd_signal()
    macd_hist = macd_obj.macd_diff()
    macd_cross_up   = macd_line.iloc[-1] > macd_sig.iloc[-1] and macd_line.iloc[-2] <= macd_sig.iloc[-2]
    macd_cross_down = macd_line.iloc[-1] < macd_sig.iloc[-1] and macd_line.iloc[-2] >= macd_sig.iloc[-2]
    macd_bull = macd_line.iloc[-1] > macd_sig.iloc[-1]
    macd_bear = macd_line.iloc[-1] < macd_sig.iloc[-1]

    # EMA
    ema9  = ta.trend.EMAIndicator(close, window=9).ema_indicator().iloc[-1]
    ema21 = ta.trend.EMAIndicator(close, window=21).ema_indicator().iloc[-1]
    ema50 = ta.trend.EMAIndicator(close, window=50).ema_indicator().iloc[-1]

    # Bollinger
    bb       = ta.volatility.BollingerBands(close, window=20)
    bb_lower = bb.bollinger_lband().iloc[-1]
    bb_upper = bb.bollinger_hband().iloc[-1]
    bb_pct   = bb.bollinger_pband().iloc[-1]

    # Volume
    vol_avg   = volume.rolling(20).mean().iloc[-1]
    vol_cur   = volume.iloc[-1]
    vol_surge = bool(vol_cur > vol_avg * 1.5)

    # ATR
    atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]

    # تغيير السعر
    price_change_1h  = round(((price - close.iloc[-4]) / close.iloc[-4]) * 100, 2)
    price_change_4h  = round(((price - close.iloc[-16]) / close.iloc[-16]) * 100, 2)

    # إشارة أولية
    long_pts = short_pts = 0

    if rsi_val < 50: long_pts += 2
    if rsi_val < 40: long_pts += 1
    if macd_cross_up: long_pts += 3
    elif macd_bull: long_pts += 1
    if price > ema9: long_pts += 1
    if ema9 > ema21: long_pts += 1
    if price <= bb_lower * 1.02: long_pts += 2
    if vol_surge and price > close.iloc[-2]: long_pts += 1

    if rsi_val > 50: short_pts += 2
    if rsi_val > 60: short_pts += 1
    if macd_cross_down: short_pts += 3
    elif macd_bear: short_pts += 1
    if price < ema9: short_pts += 1
    if ema9 < ema21: short_pts += 1
    if price >= bb_upper * 0.98: short_pts += 2
    if vol_surge and price < close.iloc[-2]: short_pts += 1

    if long_pts >= 2 and long_pts >= short_pts:
        direction = "LONG"
        tech_score = long_pts
    elif short_pts >= 2 and short_pts > long_pts:
        direction = "SHORT"
        tech_score = short_pts
    else:
        return None

    # وصف MACD
    if macd_cross_up:    macd_desc = "تقاطع صاعد 🚀"
    elif macd_cross_down: macd_desc = "تقاطع هابط 🔻"
    elif macd_bull:       macd_desc = "صاعد 📈"
    else:                 macd_desc = "هابط 📉"

    return {
        "direction":      direction,
        "tech_score":     tech_score,
        "price":          price,
        "rsi":            round(rsi_val, 1),
        "macd_desc":      macd_desc,
        "macd_cross_up":  macd_cross_up,
        "macd_cross_down": macd_cross_down,
        "bb_pct":         round(bb_pct, 2),
        "vol_surge":      vol_surge,
        "atr":            atr,
        "ema9":           round(ema9, 6),
        "ema21":          round(ema21, 6),
        "ema50":          round(ema50, 6),
        "price_change_1h": price_change_1h,
        "price_change_4h": price_change_4h,
    }

# ═══════════════════════════════════════════
#     تحليل Claude AI
# ═══════════════════════════════════════════
async def ai_analysis(symbol: str, data: dict) -> dict:
    if not ANTHROPIC_API_KEY:
        # بدون AI نحسب نسبة الثقة رياضياً
        conf = min(99, int((data['tech_score'] / 10) * 100) + np.random.randint(0, 10))
        return {"confidence": conf, "comment": "تحليل فني"}

    coin = symbol.replace("USDT", "")
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
EMA9: {data['ema9']} | EMA21: {data['ema21']} | EMA50: {data['ema50']}

أجب بـ JSON فقط بهذا الشكل بالضبط، بدون أي نص إضافي:
{{"confidence": 75, "comment": "تعليق قصير بالعربي جملة واحدة فقط"}}

نسبة الثقة من 1 إلى 99. كن واقعياً."""

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
        # تنظيف الرد
        if "```" in text:
            text = text.split("```")[1].replace("json","").strip()
        result = json.loads(text)
        return {
            "confidence": max(1, min(99, int(result.get("confidence", 50)))),
            "comment":    result.get("comment", "")
        }
    except Exception as e:
        logger.error(f"خطأ AI: {e}")
        conf = min(99, int((data['tech_score'] / 10) * 100) + np.random.randint(0, 10))
        return {"confidence": conf, "comment": "تحليل فني"}

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
#     تنسيق الرسالة
# ═══════════════════════════════════════════
def fmt(p: float) -> str:
    if p >= 1:      return f"{p:,.4f}$"
    elif p >= 0.01: return f"{p:.6f}$"
    else:           return f"{p:.8f}$"

def build_message(symbol, tech, ai, sl, tps, pcts, sl_p):
    coin   = symbol.replace("USDT", "")
    now    = datetime.utcnow().strftime("%d.%m.%Y %H:%M:%S UTC+0")
    price  = tech['price']
    signal = tech['direction']
    conf   = ai['confidence']

    emoji  = "🟢🟢🟢" if signal == "LONG" else "🔴🔴🔴"
    action = "لونغ" if signal == "LONG" else "شورت"

    # نجوم
    if conf >= 80:   stars = "⭐⭐⭐⭐⭐"
    elif conf >= 65: stars = "⭐⭐⭐⭐"
    elif conf >= 50: stars = "⭐⭐⭐"
    elif conf >= 35: stars = "⭐⭐"
    else:            stars = "⭐"

    # شريط الثقة
    filled = int(conf / 10)
    bar    = "█" * filled + "░" * (10 - filled)

    # تعليق AI
    ai_line = ""
    if ai['comment']:
        ai_line = f"\n🤖 AI: {ai['comment']}\n"

    # إضافات
    extras = ""
    if tech['vol_surge']:
        extras += "📊 الحجم: ارتفاع مفاجئ 🔥\n"

    return (
        f"تحديث: {now}\n"
        f"{emoji} {coin}/USDT  {action}  {emoji}\n\n"
        f"➡️ نقطة الدخول: {fmt(price)}\n"
        f"{ai_line}\n"
        f"💯 نسبة الثقة: {conf}%\n"
        f"[{bar}]\n"
        f"{stars}\n\n"
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
#     الفحص التلقائي
# ═══════════════════════════════════════════
sent_signals: dict = {}

async def scan_market(bot: Bot):
    logger.info("🔍 بدأ الفحص...")
    symbols = await get_symbols()
    found   = 0

    for sym in symbols:
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
                await asyncio.sleep(0.2)
                continue

            price = tech['price']
            if price >= MAX_PRICE or price <= 0:
                await asyncio.sleep(0.2)
                continue

            # تحليل AI
            ai = await ai_analysis(sym, tech)

            if ai['confidence'] < MIN_CONFIDENCE:
                await asyncio.sleep(0.2)
                continue

            sl, tps, pcts, sl_p = make_targets(price, tech['direction'])
            msg = build_message(sym, tech, ai, sl, tps, pcts, sl_p)

            await bot.send_message(chat_id=CHAT_ID, text=msg)
            sent_signals[key] = now_time
            found += 1
            logger.info(f"📤 {sym} {tech['direction']} {ai['confidence']}%")
            await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"خطأ {sym}: {e}")
        await asyncio.sleep(0.3)

    logger.info(f"✅ انتهى — {found} إشارة من {len(symbols)} عملة")

# ═══════════════════════════════════════════
#     أوامر التلغرام
# ═══════════════════════════════════════════
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    syms = await get_symbols()
    ai_status = "✅ مفعّل" if ANTHROPIC_API_KEY else "❌ غير مفعّل"
    await update.message.reply_text(
        f"🌟 مرحباً بك في سراب للإشارات! 🌟\n\n"
        f"📡 يراقب {len(syms)} عملة (تحت $10)\n"
        f"⏱ فحص تلقائي كل دقيقتين\n"
        f"🤖 تحليل AI: {ai_status}\n"
        f"💯 نسبة ثقة لكل إشارة\n"
        f"📊 RSI + MACD + EMA + Bollinger + Volume\n"
        f"🔄 لونغ وشورت\n\n"
        f"الأوامر:\n"
        f"/scan — فحص فوري\n"
        f"/status — حالة البوت"
    )

async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 جاري الفحص، انتظر قليلاً...")
    await scan_market(context.bot)
    await update.message.reply_text("✅ انتهى الفحص!")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    syms = await get_symbols()
    ai_status = "✅ مفعّل" if ANTHROPIC_API_KEY else "❌ غير مفعّل"
    await update.message.reply_text(
        f"✅ سراب يعمل بشكل طبيعي\n\n"
        f"⏱ الفحص كل: دقيقتين\n"
        f"📊 العملات: {len(syms)}\n"
        f"💰 الحد الأقصى: $10\n"
        f"🤖 AI: {ai_status}\n"
        f"📤 إشارات مرسلة: {len(sent_signals)}"
    )

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
    app.job_queue.run_repeating(auto_scan, interval=CHECK_INTERVAL, first=20)
    logger.info("🚀 سراب للإشارات يعمل...")
    app.run_polling()

if __name__ == "__main__":
    main()
