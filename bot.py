import os
import asyncio
import logging
from datetime import datetime
import aiohttp
from telegram import Bot
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram import Update
import ta
import pandas as pd

# ═══════════════════════════════════════════
#           إعدادات البوت
# ═══════════════════════════════════════════

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

CHECK_INTERVAL = 300
TOP_SYMBOLS_LIMIT = 100

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
#     سحب العملات
# ═══════════════════════════════════════════

async def get_bitunix_symbols():
    url = "https://fapi.bitunix.com/api/v1/futures/market/tickers"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                data = await response.json()

        if data.get('code') == 0:
            tickers = data.get('data', [])

            sorted_tickers = sorted(
                [t for t in tickers if t.get('symbol', '').endswith('USDT')],
                key=lambda x: float(x.get('volume24h', 0)),
                reverse=True
            )

            symbols = [t['symbol'] for t in sorted_tickers[:TOP_SYMBOLS_LIMIT]]
            logger.info(f"تم سحب {len(symbols)} عملة")
            return symbols

    except Exception as e:
        logger.error(f"خطأ: {e}")

    return ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT"]

# ═══════════════════════════════════════════
#     بيانات السوق
# ═══════════════════════════════════════════

async def get_klines_bitunix(symbol: str, interval: str = "15m", limit: int = 150):
    url = "https://fapi.bitunix.com/api/v1/futures/market/kline"
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                data = await response.json()

        if data.get('code') == 0 and data.get('data'):
            df = pd.DataFrame(data['data'])

            for col in ['close', 'high', 'low', 'open', 'volume']:
                df[col] = df[col].astype(float)

            return df

    except Exception as e:
        logger.error(f"خطأ {symbol}: {e}")

    return None

# ═══════════════════════════════════════════
#         التحليل (مُحسّن)
# ═══════════════════════════════════════════

def analyze(df: pd.DataFrame):
    close = df['close']

    rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]

    macd_ind = ta.trend.MACD(close)
    macd = macd_ind.macd()
    macd_sig = macd_ind.macd_signal()

    ema20 = ta.trend.EMAIndicator(close, 20).ema_indicator()
    ema50 = ta.trend.EMAIndicator(close, 50).ema_indicator()

    current_price = close.iloc[-1]

    long_score = 0
    short_score = 0

    # RSI (خفيف أكثر = إشارات أكثر)
    if rsi < 50:
        long_score += 1
    if rsi > 50:
        short_score += 1

    # MACD (أقوى عامل)
    if macd.iloc[-1] > macd_sig.iloc[-1]:
        long_score += 2
    else:
        short_score += 2

    # EMA trend
    if current_price > ema20.iloc[-1]:
        long_score += 1
    else:
        short_score += 1

    if ema20.iloc[-1] > ema50.iloc[-1]:
        long_score += 1
    else:
        short_score += 1

    # ⚡ تخفيف شرط الإشارة (مهم جدًا)
    if long_score >= 2 and long_score > short_score:
        signal = "LONG"
    elif short_score >= 2 and short_score > long_score:
        signal = "SHORT"
    else:
        signal = None

    return {
        "signal": signal,
        "strength": min(max(long_score, short_score), 5),
        "rsi": round(rsi, 1),
        "price": current_price
    }

# ═══════════════════════════════════════════
#     الأهداف
# ═══════════════════════════════════════════

def calculate_targets(price: float, signal: str):
    if signal == "LONG":
        sl = price * 0.985
        targets = [price * 1.01, price * 1.02, price * 1.035, price * 1.05, price * 1.08]
    else:
        sl = price * 1.015
        targets = [price * 0.99, price * 0.98, price * 0.965, price * 0.95, price * 0.92]

    return sl, targets

# ═══════════════════════════════════════════
#     الرسالة
# ═══════════════════════════════════════════

def format_signal_message(symbol, price, signal, strength, rsi, sl, targets):
    coin = symbol.replace("USDT", "")
    time_now = datetime.utcnow().strftime("%d.%m.%Y %H:%M UTC")

    emoji = "🟢🟢🟢" if signal == "LONG" else "🔴🔴🔴"

    return (
        f"📊 تحديث: {time_now}\n\n"
        f"{emoji} {coin} {signal} {emoji}\n\n"
        f"➡️ الدخول: {price:.6f}$\n"
        f"⭐ القوة: {strength}/5\n"
        f"📊 RSI: {rsi}\n\n"
        f"🎯 TP1: {targets[0]:.6f}$\n"
        f"🎯 TP2: {targets[1]:.6f}$\n"
        f"🎯 TP3: {targets[2]:.6f}$\n"
        f"🎯 TP4: {targets[3]:.6f}$\n"
        f"🎯 TP5: {targets[4]:.6f}$\n\n"
        f"🛑 SL: {sl:.6f}$\n\n"
        f"🏦 Bitunix Futures"
    )

# ═══════════════════════════════════════════
#     فحص السوق
# ═══════════════════════════════════════════

sent_signals = {}

async def scan_market(bot: Bot):
    logger.info("🔍 فحص السوق...")

    symbols = await get_bitunix_symbols()

    for symbol in symbols:
        try:
            df = await get_klines_bitunix(symbol)

            if df is None or len(df) < 100:
                continue

            analysis = analyze(df)

            if not analysis["signal"]:
                continue

            key = symbol + analysis["signal"]
            now = asyncio.get_event_loop().time()

            if now - sent_signals.get(key, 0) < 1800:
                continue

            sl, targets = calculate_targets(analysis["price"], analysis["signal"])

            message = format_signal_message(
                symbol,
                analysis["price"],
                analysis["signal"],
                analysis["strength"],
                analysis["rsi"],
                sl,
                targets
            )

            await bot.send_message(chat_id=CHAT_ID, text=message)
            sent_signals[key] = now

            await asyncio.sleep(2)

        except Exception as e:
            logger.error(f"خطأ {symbol}: {e}")

        await asyncio.sleep(0.3)

    logger.info("✅ انتهى الفحص")

# ═══════════════════════════════════════════
#     أوامر التليجرام
# ═══════════════════════════════════════════

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔥 البوت شغال وجاهز للإشارات")

async def scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 جاري الفحص...")
    await scan_market(context.bot)
    await update.message.reply_text("✅ انتهى")

# ═══════════════════════════════════════════
#     التشغيل (Railway آمن)
# ═══════════════════════════════════════════

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("scan", scan))

    app.job_queue.run_repeating(scan_market, interval=CHECK_INTERVAL, first=10)

    logger.info("🚀 Bot running...")
    app.run_polling()

if __name__ == "__main__":
    main()
