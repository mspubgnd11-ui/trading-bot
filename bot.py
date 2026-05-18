import os
import asyncio
import aiohttp
import logging
from datetime import datetime
import pandas as pd
import ta

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ======================
# CONFIG
# ======================

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

TIMEFRAME = "15m"
MAX_PRICE = 10
SCAN_LIMIT = 100
COOLDOWN = 1800

EXCLUDED = ["BTC", "ETH"]

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bot")

bot = None
sent = {}

# ======================
# SAFE HTTP (NO SESSION PROBLEMS)
# ======================

async def fetch(url, params=None):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as r:
                return await r.json()
    except:
        return None

# ======================
# SYMBOLS
# ======================

async def get_symbols():
    url = "https://fapi.bitunix.com/api/v1/futures/market/tickers"
    data = await fetch(url)

    if not data or data.get("code") != 0:
        return []

    symbols = []

    for t in data.get("data", []):
        sym = t.get("symbol", "")
        price = float(t.get("lastPrice", 0))

        if not sym.endswith("USDT"):
            continue

        if any(x in sym for x in EXCLUDED):
            continue

        if price > MAX_PRICE:
            continue

        symbols.append(sym)

    return symbols[:SCAN_LIMIT]

# ======================
# KLINES
# ======================

async def get_klines(symbol):
    url = "https://fapi.bitunix.com/api/v1/futures/market/kline"
    params = {"symbol": symbol, "interval": TIMEFRAME, "limit": 120}

    data = await fetch(url, params)

    if not data or "data" not in data or not data["data"]:
        return None

    try:
        df = pd.DataFrame(data["data"])

        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float)

        return df
    except:
        return None

# ======================
# ANALYSIS (SMART BUT LIGHT)
# ======================

def analyze(df):
    close = df["close"]

    rsi = ta.momentum.RSIIndicator(close).rsi().iloc[-1]

    macd = ta.trend.MACD(close)
    macd_line = macd.macd().iloc[-1]
    macd_signal = macd.macd_signal().iloc[-1]

    ema20 = ta.trend.EMAIndicator(close, 20).ema_indicator().iloc[-1]
    ema50 = ta.trend.EMAIndicator(close, 50).ema_indicator().iloc[-1]

    price = close.iloc[-1]

    long_score = 0
    short_score = 0

    if rsi < 55:
        long_score += 1
    if rsi > 45:
        short_score += 1

    if macd_line > macd_signal:
        long_score += 2
    else:
        short_score += 2

    if price > ema20:
        long_score += 1
    else:
        short_score += 1

    if ema20 > ema50:
        long_score += 1
    else:
        short_score += 1

    volatility = abs(price - ema20) / price
    if volatility < 0.001:
        return None, price, rsi, 0

    max_score = 5

    if long_score >= 3:
        confidence = int((long_score / max_score) * 100)
        return "LONG", price, rsi, confidence

    if short_score >= 3:
        confidence = int((short_score / max_score) * 100)
        return "SHORT", price, rsi, confidence

    return None, price, rsi, 0

# ======================
# TARGETS
# ======================

def targets(price, side):
    if side == "LONG":
        sl = price * 0.98
        tps = [price * 1.01, price * 1.02, price * 1.04, price * 1.06, price * 1.10]
    else:
        sl = price * 1.02
        tps = [price * 0.99, price * 0.98, price * 0.96, price * 0.94, price * 0.90]

    return sl, tps

# ======================
# MESSAGE
# ======================

def format_msg(symbol, side, price, rsi, sl, tps, confidence):
    emoji = "🟢🟢🟢" if side == "LONG" else "🔴🔴🔴"

    return f"""
📊 تحديث: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

{emoji} {symbol} {side} {emoji}

📈 ثقة الإشارة: {confidence}%

➡️ الدخول: {price:.5f}$
📊 RSI: {round(rsi,1)}

🎯 TP1: {tps[0]:.5f}$
🎯 TP2: {tps[1]:.5f}$
🎯 TP3: {tps[2]:.5f}$
🎯 TP4: {tps[3]:.5f}$
🎯 TP5: {tps[4]:.5f}$

🛑 SL: {sl:.5f}$

🏦 Bitunix Futures
"""

# ======================
# PROCESS
# ======================

async def process(symbol):
    try:
        df = await get_klines(symbol)
        if df is None or len(df) < 50:
            return

        signal, price, rsi, confidence = analyze(df)

        if not signal or confidence < 55:
            return

        key = symbol + signal
        now = asyncio.get_event_loop().time()

        if key in sent and now - sent[key] < COOLDOWN:
            return

        sl, tps = targets(price, signal)

        msg = format_msg(symbol, signal, price, rsi, sl, tps, confidence)

        await bot.send_message(chat_id=CHAT_ID, text=msg)

        sent[key] = now

    except:
        pass

# ======================
# SCAN LOOP (SAFE)
# ======================

async def scan():
    symbols = await get_symbols()
    tasks = [process(s) for s in symbols]
    await asyncio.gather(*tasks)

# ======================
# LOOP (NO CRASH)
# ======================

async def auto_loop():
    while True:
        try:
            await scan()
        except:
            pass
        await asyncio.sleep(300)

# ======================
# COMMANDS
# ======================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔥 البوت شغال وجاهز")

async def scan_now(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 جاري الفحص...")
    await scan()
    await update.message.reply_text("✅ انتهى")

# ======================
# MAIN (RAILWAY SAFE)
# ======================

def main():
    global bot

    app = Application.builder().token(TOKEN).build()
    bot = app.bot

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("scan", scan_now))

    asyncio.create_task(auto_loop())

    print("Bot running...")
    app.run_polling()

if __name__ == "__main__":
    main()
