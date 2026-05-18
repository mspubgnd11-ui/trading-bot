import os
import asyncio
import logging
import json
from datetime import datetime
import aiohttp
from telegram import Bot
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
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

CHECK_INTERVAL  = 360    # فحص كل 6 دقايق
MAX_PRICE       = 10.0   # عملات تحت $10
MAX_SYMBOLS     = 100    # أكبر 100 عملة
MIN_CONFIDENCE  = 60     # أدنى نسبة ثقة (كانت 25 — رفعناها)
MIN_TECH_SCORE  = 6      # أدنى نقاط فنية
SIGNAL_COOLDOWN = 3600   # ساعة بين نفس الإشارة

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
#     التحليل الفني المحسّن
# ═══════════════════════════════════════════
def technical_analysis(df: pd.DataFrame):
    if len(df) < 60:
        return None

    close  = df['close']
    high   = df['high']
    low    = df['low']
    volume = df['volume']
    price  = close.iloc[-1]

    # ── RSI ──────────────────────────────
    rsi_val  = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    rsi_prev = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-2]

    # ── MACD ─────────────────────────────
    macd_obj       = ta.trend.MACD(close)
    macd_line      = macd_obj.macd()
    macd_sig       = macd_obj.macd_signal()
    macd_hist      = macd_obj.macd_diff()
    macd_cross_up  = macd_line.iloc[-1] > macd_sig.iloc[-1] and macd_line.iloc[-2] <= macd_sig.iloc[-2]
    macd_cross_down= macd_line.iloc[-1] < macd_sig.iloc[-1] and macd_line.iloc[-2] >= macd_sig.iloc[-2]
    macd_bull      = macd_line.iloc[-1] > macd_sig.iloc[-1]
    macd_bear      = macd_line.iloc[-1] < macd_sig.iloc[-1]
    # هيستوغرام يتوسع؟
    hist_growing_bull = macd_hist.iloc[-1] > 0 and macd_hist.iloc[-1] > macd_hist.iloc[-2]
    hist_growing_bear = macd_hist.iloc[-1] < 0 and macd_hist.iloc[-1] < macd_hist.iloc[-2]

    # ── EMA ──────────────────────────────
    ema9  = ta.trend.EMAIndicator(close, window=9).ema_indicator()
    ema21 = ta.trend.EMAIndicator(close, window=21).ema_indicator()
    ema50 = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    ema_golden = ema9.iloc[-1] > ema21.iloc[-1] > ema50.iloc[-1]
    ema_death  = ema9.iloc[-1] < ema21.iloc[-1] < ema50.iloc[-1]

    # ── Bollinger ────────────────────────
    bb        = ta.volatility.BollingerBands(close, window=20)
    bb_lower  = bb.bollinger_lband().iloc[-1]
    bb_upper  = bb.bollinger_hband().iloc[-1]
    bb_mid    = bb.bollinger_mavg().iloc[-1]
    bb_pct    = bb.bollinger_pband().iloc[-1]
    bb_width  = (bb_upper - bb_lower) / bb_mid  # اتساع البولنجر

    # ── Volume ───────────────────────────
    vol_avg   = volume.rolling(20).mean().iloc[-1]
    vol_cur   = volume.iloc[-1]
    vol_surge = bool(vol_cur > vol_avg * 1.8)   # رفعنا الشرط

    # ── ATR ──────────────────────────────
    atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]

    # ── Stochastic RSI ───────────────────
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14)
    stoch_k = stoch.stoch().iloc[-1]

    # ── تغيير السعر ──────────────────────
    price_change_1h  = round(((price - close.iloc[-4])  / close.iloc[-4])  * 100, 2)
    price_change_4h  = round(((price - close.iloc[-16]) / close.iloc[-16]) * 100, 2)
    price_change_24h = round(((price - close.iloc[-96]) / close.iloc[-96]) * 100, 2) if len(close) >= 96 else 0

    # ── نظام النقاط المشدّد ───────────────
    long_pts = short_pts = 0

    # RSI (محسّن)
    if rsi_val < 35:              long_pts += 3
    elif rsi_val < 45:            long_pts += 2
    elif rsi_val < 50:            long_pts += 1
    if rsi_val > rsi_prev and rsi_val < 55:  long_pts += 1  # RSI يرتد

    if rsi_val > 65:              short_pts += 3
    elif rsi_val > 55:            short_pts += 2
    elif rsi_val > 50:            short_pts += 1
    if rsi_val < rsi_prev and rsi_val > 45: short_pts += 1

    # MACD
    if macd_cross_up:             long_pts  += 4
    elif hist_growing_bull:       long_pts  += 2
    elif macd_bull:               long_pts  += 1

    if macd_cross_down:           short_pts += 4
    elif hist_growing_bear:       short_pts += 2
    elif macd_bear:               short_pts += 1

    # EMA
    if ema_golden:                long_pts  += 3
    elif ema9.iloc[-1] > ema21.iloc[-1]: long_pts += 1

    if ema_death:                 short_pts += 3
    elif ema9.iloc[-1] < ema21.iloc[-1]: short_pts += 1

    # السعر فوق/تحت EMA
    if price > ema9.iloc[-1]:    long_pts  += 1
    if price < ema9.iloc[-1]:    short_pts += 1

    # Bollinger
    if price <= bb_lower * 1.01: long_pts  += 3  # عند الحد السفلي
    elif price < bb_mid:         long_pts  += 1

    if price >= bb_upper * 0.99: short_pts += 3  # عند الحد العلوي
    elif price > bb_mid:         short_pts += 1

    # Volume surge مع حركة
    if vol_surge and price > close.iloc[-2]: long_pts  += 2
    if vol_surge and price < close.iloc[-2]: short_pts += 2

    # Stochastic
    if stoch_k < 25:             long_pts  += 2
    if stoch_k > 75:             short_pts += 2

    # ── القرار النهائي (شدّدنا الشرط) ────
    if long_pts >= MIN_TECH_SCORE and long_pts >= short_pts + 2:
        direction  = "LONG"
        tech_score = long_pts
    elif short_pts >= MIN_TECH_SCORE and short_pts > long_pts + 2:
        direction  = "SHORT"
        tech_score = short_pts
    else:
        return None

    # وصف MACD
    if macd_cross_up:    macd_desc = "تقاطع صاعد 🚀"
    elif macd_cross_down: macd_desc = "تقاطع هابط 🔻"
    elif hist_growing_bull: macd_desc = "زخم صاعد 📈"
    elif hist_growing_bear: macd_desc = "زخم هابط 📉"
    elif macd_bull:       macd_desc = "صاعد 📈"
    else:                 macd_desc = "هابط 📉"

    return {
        "direction":       direction,
        "tech_score":      tech_score,
        "price":           price,
        "rsi":             round(rsi_val, 1),
        "stoch_k":         round(stoch_k, 1),
        "macd_desc":       macd_desc,
        "macd_cross_up":   macd_cross_up,
        "macd_cross_down": macd_cross_down,
        "bb_pct":          round(bb_pct, 2),
        "bb_width":        round(bb_width * 100, 2),
        "vol_surge":       vol_surge,
        "vol_ratio":       round(vol_cur / vol_avg, 1) if vol_avg else 1,
        "atr":             atr,
        "ema9":            round(ema9.iloc[-1], 6),
        "ema21":           round(ema21.iloc[-1], 6),
        "ema50":           round(ema50.iloc[-1], 6),
        "ema_golden":      ema_golden,
        "ema_death":       ema_death,
        "price_change_1h":  price_change_1h,
        "price_change_4h":  price_change_4h,
        "price_change_24h": price_change_24h,
    }

# ═══════════════════════════════════════════
#     تحليل Claude AI (محسّن)
# ═══════════════════════════════════════════
async def ai_analysis(symbol: str, data: dict, deep: bool = False) -> dict:
    if not ANTHROPIC_API_KEY:
        conf = min(99, int((data['tech_score'] / 14) * 100))
        return {"confidence": conf, "comment": "تحليل فني فقط", "reasoning": ""}

    coin = symbol.replace("USDT", "")

    # بروميت أذكى
    prompt = f"""أنت محلل تداول متخصص في العملات الرقمية على منصة Bitunix Futures.
حلل هذه الصفقة وأعط تقييماً دقيقاً.

═══ بيانات الصفقة ═══
العملة: {coin}/USDT | الاتجاه المقترح: {data['direction']}
السعر: {data['price']}

═══ المؤشرات الفنية ═══
RSI (14): {data['rsi']} {'⚠️ ذروة بيع' if data['rsi'] < 30 else '⚠️ ذروة شراء' if data['rsi'] > 70 else ''}
Stochastic: {data['stoch_k']}
MACD: {data['macd_desc']}
Bollinger %B: {data['bb_pct']} (0=أسفل، 1=أعلى) | عرض البولنجر: {data['bb_width']}%

═══ المتوسطات ═══
EMA9={data['ema9']} | EMA21={data['ema21']} | EMA50={data['ema50']}
{'✅ Golden Cross (EMA9>21>50)' if data['ema_golden'] else '❌ Death Cross (EMA9<21<50)' if data['ema_death'] else '↔️ EMA مختلطة'}

═══ الحجم والتغير ═══
الحجم: {'🔥 ارتفاع x' + str(data['vol_ratio']) if data['vol_surge'] else 'طبيعي'}
التغير: 1س={data['price_change_1h']}% | 4س={data['price_change_4h']}% | 24س={data['price_change_24h']}%
النقاط الفنية: {data['tech_score']}/14

═══ تعليمات ═══
- قيّم قوة الإشارة بموضوعية
- خذ بعين الاعتبار كل المؤشرات معاً
- نسبة الثقة: 1-99 (لا تبالغ)
- {'أعط تحليلاً مفصلاً (3-4 جمل)' if deep else 'تعليق قصير جملة واحدة'}

أجب بـ JSON فقط بدون أي نص إضافي:
{{"confidence": 72, "comment": "تعليق{'  مفصل' if deep else ' قصير'} بالعربي", "reasoning": "{'سبب مختصر في جملة' if not deep else 'تحليل مفصل'}"}}"""

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
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 200 if deep else 120,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=aiohttp.ClientTimeout(total=20)
            ) as r:
                resp = await r.json()

        text = resp['content'][0]['text'].strip()
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        result = json.loads(text)
        return {
            "confidence": max(1, min(99, int(result.get("confidence", 50)))),
            "comment":    result.get("comment", ""),
            "reasoning":  result.get("reasoning", "")
        }
    except Exception as e:
        logger.error(f"خطأ AI: {e}")
        conf = min(99, int((data['tech_score'] / 14) * 100))
        return {"confidence": conf, "comment": "تحليل فني", "reasoning": ""}

# ═══════════════════════════════════════════
#     الأهداف ووقف الخسارة (أوسع)
# ═══════════════════════════════════════════
def make_targets(price: float, signal: str, atr: float = 0):
    # نستخدم ATR لأهداف ديناميكية
    atr_pct = (atr / price) if price > 0 else 0.02

    if signal == "LONG":
        sl   = round(price * 0.975, 8)           # -2.5%
        tps  = [
            round(price * 1.010, 8),             # +1%
            round(price * 1.020, 8),             # +2%
            round(price * 1.035, 8),             # +3.5%
            round(price * 1.055, 8),             # +5.5%
            round(price * 1.080, 8),             # +8%
        ]
        pcts = ["+1%", "+2%", "+3.5%", "+5.5%", "+8%"]
        sl_p = "-2.5%"
    else:
        sl   = round(price * 1.025, 8)           # +2.5%
        tps  = [
            round(price * 0.990, 8),             # -1%
            round(price * 0.980, 8),             # -2%
            round(price * 0.965, 8),             # -3.5%
            round(price * 0.945, 8),             # -5.5%
            round(price * 0.920, 8),             # -8%
        ]
        pcts = ["-1%", "-2%", "-3.5%", "-5.5%", "-8%"]
        sl_p = "+2.5%"

    return sl, tps, pcts, sl_p

# ═══════════════════════════════════════════
#     تنسيق الرسالة
# ═══════════════════════════════════════════
def fmt(p: float) -> str:
    if p >= 1:      return f"{p:,.4f}$"
    elif p >= 0.01: return f"{p:.6f}$"
    else:           return f"{p:.8f}$"

def build_message(symbol, tech, ai, sl, tps, pcts, sl_p, is_manual=False):
    coin   = symbol.replace("USDT", "")
    now    = datetime.utcnow().strftime("%d.%m.%Y %H:%M UTC+0")
    price  = tech['price']
    signal = tech['direction']
    conf   = ai['confidence']

    emoji  = "🟢" if signal == "LONG" else "🔴"
    action = "لونغ  📈" if signal == "LONG" else "شورت  📉"

    # نجوم الثقة
    if conf >= 85:   stars = "⭐⭐⭐⭐⭐  ممتاز"
    elif conf >= 70: stars = "⭐⭐⭐⭐  جيد جداً"
    elif conf >= 60: stars = "⭐⭐⭐  جيد"
    else:            stars = "⭐⭐  متوسط"

    # شريط الثقة
    filled = int(conf / 10)
    bar    = "█" * filled + "░" * (10 - filled)

    # EMA الوضع
    if tech['ema_golden']:   ema_status = "✅ Golden Cross"
    elif tech['ema_death']:  ema_status = "❌ Death Cross"
    else:                    ema_status = "↔️ مختلطة"

    # AI reasoning
    reasoning_line = ""
    if ai.get('reasoning'):
        reasoning_line = f"📝 {ai['reasoning']}\n"

    tag = "🔍 تحليل مطلوب" if is_manual else "🤖 إشارة تلقائية"

    return (
        f"{'─'*30}\n"
        f"{tag}  •  {now}\n"
        f"{'─'*30}\n\n"
        f"{emoji} {coin}/USDT  —  {action}\n\n"
        f"💰 الدخول: {fmt(price)}\n\n"
        f"🤖 AI: {ai['comment']}\n"
        f"{reasoning_line}\n"
        f"💯 الثقة: {conf}%\n"
        f"[{bar}] {stars}\n\n"
        f"{'─'*20}\n"
        f"🎯 الأهداف:\n"
        f"   TP1: {fmt(tps[0])}  {pcts[0]}\n"
        f"   TP2: {fmt(tps[1])}  {pcts[1]}\n"
        f"   TP3: {fmt(tps[2])}  {pcts[2]}\n"
        f"   TP4: {fmt(tps[3])}  {pcts[3]}\n"
        f"   TP5: {fmt(tps[4])}  {pcts[4]}\n\n"
        f"🛑 وقف الخسارة: {fmt(sl)}  ({sl_p})\n"
        f"{'─'*20}\n\n"
        f"📊 RSI: {tech['rsi']}  |  Stoch: {tech['stoch_k']}\n"
        f"📈 MACD: {tech['macd_desc']}\n"
        f"📉 EMA: {ema_status}\n"
        f"📦 Bollinger: {int(tech['bb_pct']*100)}%\n"
        f"{'🔥 حجم مرتفع x' + str(tech['vol_ratio']) if tech['vol_surge'] else ''}\n\n"
        f"⚠️ للتأكيد فقط — القرار عليك\n"
        f"🏦 Bitunix Futures"
    )

# ═══════════════════════════════════════════
#     الفحص التلقائي
# ═══════════════════════════════════════════
sent_signals: dict = {}

async def scan_market(bot: Bot, silent: bool = False):
    logger.info("🔍 بدأ الفحص...")
    symbols = await get_symbols()
    found   = 0

    for sym in symbols:
        try:
            df = await get_klines(sym)
            if df is None or len(df) < 60:
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

            ai = await ai_analysis(sym, tech)

            if ai['confidence'] < MIN_CONFIDENCE:
                await asyncio.sleep(0.2)
                continue

            sl, tps, pcts, sl_p = make_targets(price, tech['direction'], tech['atr'])
            msg = build_message(sym, tech, ai, sl, tps, pcts, sl_p)

            await bot.send_message(chat_id=CHAT_ID, text=msg)
            sent_signals[key] = now_time
            found += 1
            logger.info(f"📤 {sym} {tech['direction']} {ai['confidence']}%")
            await asyncio.sleep(1.5)

        except Exception as e:
            logger.error(f"خطأ {sym}: {e}")
        await asyncio.sleep(0.4)

    if not silent:
        logger.info(f"✅ انتهى — {found} إشارة من {len(symbols)} عملة")
    return found

# ═══════════════════════════════════════════
#     تحليل عملة بالطلب 🆕
# ═══════════════════════════════════════════
async def analyze_coin_request(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    المستخدم يبعت اسم عملة مثل: BTC أو XRPUSDT أو xrp
    البوت يحللها فوراً بتحليل عميق
    """
    text = update.message.text.strip().upper()

    # تنظيف الاسم
    symbol = text.replace("/", "").replace("-", "").replace(" ", "")
    if not symbol.endswith("USDT"):
        symbol = symbol + "USDT"

    await update.message.reply_text(f"🔍 جاري تحليل {symbol} ...")

    # جلب البيانات
    df = await get_klines(symbol)
    if df is None or len(df) < 60:
        await update.message.reply_text(
            f"❌ تعذّر جلب بيانات {symbol}\n"
            f"تأكد من الاسم — مثال: XRP أو DOGEUSDT"
        )
        return

    tech = technical_analysis(df)
    price = df['close'].iloc[-1]

    if tech is None:
        # حتى لو ما في إشارة واضحة — نحلل ونبين الوضع
        await update.message.reply_text(
            f"📊 {symbol}\n\n"
            f"💰 السعر: {fmt(price)}\n"
            f"🔍 لا توجد إشارة واضحة حالياً\n\n"
            f"المؤشرات متضاربة — انتظر إشارة أقوى"
        )
        return

    # تحليل AI عميق للطلبات اليدوية
    ai = await ai_analysis(symbol, tech, deep=True)

    sl, tps, pcts, sl_p = make_targets(price, tech['direction'], tech['atr'])
    msg = build_message(symbol, tech, ai, sl, tps, pcts, sl_p, is_manual=True)
    await update.message.reply_text(msg)

# ═══════════════════════════════════════════
#     أوامر التلغرام
# ═══════════════════════════════════════════
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    syms = await get_symbols()
    ai_status = "✅ مفعّل (Sonnet)" if ANTHROPIC_API_KEY else "❌ غير مفعّل"
    await update.message.reply_text(
        f"🌟 سراب للإشارات — النسخة المحسّنة 🌟\n\n"
        f"📡 يراقب {len(syms)} عملة (تحت $10)\n"
        f"⏱ فحص تلقائي كل 6 دقايق\n"
        f"🤖 تحليل AI: {ai_status}\n"
        f"💯 نسبة ثقة ≥ 60% فقط\n"
        f"🎯 أهداف واسعة: حتى +8%\n"
        f"📊 RSI + MACD + EMA + Bollinger + Stochastic + Volume\n\n"
        f"─────────────────────\n"
        f"الأوامر:\n"
        f"/scan — فحص فوري للسوق\n"
        f"/status — حالة البوت\n"
        f"/clear — مسح سجل الإشارات\n\n"
        f"💡 تحليل عملة:\n"
        f"ابعت اسم أي عملة مثل:\n"
        f"XRP أو BTC أو DOGE\n"
        f"وسأحللها فوراً بالذكاء الاصطناعي! 🤖"
    )

async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 جاري فحص السوق، انتظر قليلاً...")
    found = await scan_market(context.bot)
    await update.message.reply_text(f"✅ انتهى الفحص — وجدنا {found} إشارة!")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    syms = await get_symbols()
    ai_status = "✅ Sonnet" if ANTHROPIC_API_KEY else "❌ غير مفعّل"
    active_signals = len([k for k, t in sent_signals.items()
                          if asyncio.get_event_loop().time() - t < 3600])
    await update.message.reply_text(
        f"✅ سراب يعمل بشكل طبيعي\n\n"
        f"⏱ الفحص كل: 6 دقايق\n"
        f"📊 العملات: {len(syms)}\n"
        f"💰 الحد الأقصى: $10\n"
        f"🤖 AI: {ai_status}\n"
        f"📤 إشارات نشطة (آخر ساعة): {active_signals}\n"
        f"📤 إجمالي الإشارات: {len(sent_signals)}\n"
        f"💯 الحد الأدنى للثقة: {MIN_CONFIDENCE}%"
    )

async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    count = len(sent_signals)
    sent_signals.clear()
    await update.message.reply_text(
        f"🗑️ تم مسح سجل {count} إشارة\n"
        f"البوت جاهز للفحص من جديد!"
    )

async def auto_scan(context: ContextTypes.DEFAULT_TYPE):
    await scan_market(context.bot, silent=True)

# ═══════════════════════════════════════════
#     تشغيل البوت
# ═══════════════════════════════════════════
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # أوامر
    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("scan",   cmd_scan))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("clear",  cmd_clear))

    # 🆕 تحليل عملة بالاسم — أي رسالة نصية بدون / تُعامَل كاسم عملة
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_coin_request))

    # فحص تلقائي
    app.job_queue.run_repeating(auto_scan, interval=CHECK_INTERVAL, first=30)

    logger.info("🚀 سراب للإشارات (النسخة المحسّنة) يعمل...")
    app.run_polling()

if __name__ == "__main__":
    main()
