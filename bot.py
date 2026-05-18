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

CHECK_INTERVAL = 300   # فحص كل 5 دقائق
TOP_SYMBOLS_LIMIT = 100 # أفضل 100 عملة بحجم تداول

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
#     سحب العملات من Bitunix تلقائياً
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
            logger.info(f"تم سحب {len(symbols)} عملة من Bitunix")
            return symbols
    except Exception as e:
        logger.error(f"خطأ في سحب العملات: {e}")
    return [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
        "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT"
    ]

# ═══════════════════════════════════════════
#         سحب بيانات السوق من Bitunix
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
        logger.error(f"خطأ في بيانات {symbol}: {e}")
    return None

async def get_price_bitunix(symbol: str):
    url = "https://fapi.bitunix.com/api/v1/futures/market/tickers"
    params = {"symbol": symbol}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                data = await response.json()
        if data.get('code') == 0 and data.get('data'):
            return float(data['data'][0]['lastPrice'])
    except Exception as e:
        logger.error(f"خطأ في سعر {symbol}: {e}")
    return None

# ═══════════════════════════════════════════
#         تحليل المؤشرات الفنية (مخففة)
# ═══════════════════════════════════════════
def analyze(df: pd.DataFrame):
    close = df['close']
    high = df['high']
    low = df['low']

    rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
    current_rsi = rsi.iloc[-1]

    macd_ind = ta.trend.MACD(close)
    macd = macd_ind.macd()
    macd_sig = macd_ind.macd_signal()
    macd_cross_up = macd.iloc[-1] > macd_sig.iloc[-1] and macd.iloc[-2] <= macd_sig.iloc[-2]
    macd_cross_down = macd.iloc[-1] < macd_sig.iloc[-1] and macd.iloc[-2] >= macd_sig.iloc[-2]

    ema20 = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    ema50 = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    current_price = close.iloc[-1]

    bb = ta.volatility.BollingerBands(close, window=20)
    bb_lower = bb.bollinger_lband().iloc[-1]
    bb_upper = bb.bollinger_hband().iloc[-1]

    atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]

    long_score = 0
    short_score = 0

    # LONG شروط مخففة
    if current_rsi < 45: long_score += 2
    elif current_rsi < 55: long_score += 1
    if macd_cross_up: long_score += 2
    if macd.iloc[-1] > macd_sig.iloc[-1]: long_score += 1
    if current_price > ema20.iloc[-1]: long_score += 1
    if ema20.iloc[-1] > ema50.iloc[-1]: long_score += 1
    if current_price <= bb_lower * 1.02: long_score += 1

    # SHORT شروط مخففة
    if current_rsi > 55: short_score += 2
    elif current_rsi > 45: short_score += 1
    if macd_cross_down: short_score += 2
    if macd.iloc[-1] < macd_sig.iloc[-1]: short_score += 1
    if current_price < ema20.iloc[-1]: short_score += 1
    if ema20.iloc[-1] < ema50.iloc[-1]: short_score += 1
    if current_price >= bb_upper * 0.98: short_score += 1

    signal = None
    strength = 0

    if long_score >= 3 and long_score > short_score:
        signal = "LONG"
        strength = long_score
    elif short_score >= 3 and short_score > long_score:
        signal = "SHORT"
        strength = short_score

    return {
        "signal": signal,
        "strength": min(strength, 5),
        "rsi": round(current_rsi, 1),
        "atr": atr,
        "price": current_price,
    }

# ═══════════════════════════════════════════
#      حساب 5 أهداف ووقف الخسارة
# ═══════════════════════════════════════════
def calculate_targets(price: float, signal: str, atr: float):
    multipliers = [1.5, 3, 5, 8, 12]
    targets = []
    if signal == "LONG":
        sl = round(price - (atr * 2), 8)
        for m in multipliers:
            targets.append(round(price + (atr * m), 8))
    else:
        sl = round(price + (atr * 2), 8)
        for m in multipliers:
            targets.append(round(price - (atr * m), 8))
    return sl, targets

# ═══════════════════════════════════════════
#      تنسيق رسالة الإشارة الاحترافية
# ═══════════════════════════════════════════
def format_signal_message(symbol, price, signal, strength, rsi, sl, targets):
    coin = symbol.replace("USDT", "")
    time_now = datetime.utcnow().strftime("%d.%m.%Y %H:%M:%S UTC+0")
    emoji = "🟢🟢🟢" if signal == "LONG" else "🔴🔴🔴"
    stars = "⭐" * strength

    if signal == "LONG":
        sl_pct = round(((price - sl) / price) * 100, 2)
        tp1_pct = round(((targets[0] - price) / price) * 100, 2)
    else:
        sl_pct = round(((sl - price) / price) * 100, 2)
        tp1_pct = round(((price - targets[0]) / price) * 100, 2)

    def fmt(p):
        if p >= 1: return f"{p:,.4f}$"
        elif p >= 0.01: return f"{p:.6f}$"
        else: return f"{p:.8f}$"

    return (
        f"تحديث: {time_now}\n"
        f"{emoji} {coin}USDT ({coin}) {signal} {emoji}\n\n"
        f"➡️ نقطة الدخول: {fmt(price)}\n"
        f"⭐ القوة: {stars}\n"
        f"📊 RSI: {rsi}\n\n"
        f"🎯 TP1: {fmt(targets[0])}  (+{tp1_pct}%)\n"
        f"🎯 TP2: {fmt(targets[1])}\n"
        f"🎯 TP3: {fmt(targets[2])}\n"
        f"🎯 TP4: {fmt(targets[3])}\n"
        f"🎯 TP5 (إغلاق): {fmt(targets[4])}\n\n"
        f"🛑 SL (إغلاق): {fmt(sl)}  (-{sl_pct}%)\n\n"
        f"⚠️ البوت يثبت جزءاً من الصفقة عند كل TP. التثبيت حسب تقديرك، آخر TP يغلق الباقي.\n\n"
        f"🏦 المنصة: Bitunix Futures"
    )

# ═══════════════════════════════════════════
#         الفحص التلقائي للسوق
# ═══════════════════════════════════════════
sent_signals = {}

async def scan_market(bot: Bot):
    logger.info("🔍 جاري فحص السوق...")
    symbols = await get_bitunix_symbols()
    signals_found = 0

    for symbol in symbols:
        try:
            df = await get_klines_bitunix(symbol)
            if df is None or len(df) < 100:
                continue

            price = await get_price_bitunix(symbol)
            if price is None:
                price = df['close'].iloc[-1]

            analysis = analyze(df)

            if analysis['signal']:
                signal_key = f"{symbol}_{analysis['signal']}"
                last_sent = sent_signals.get(signal_key, 0)
                now = asyncio.get_event_loop().time()

                if now - last_sent < 3600:
                    continue

                sl, targets = calculate_targets(price, analysis['signal'], analysis['atr'])
                message = format_signal_message(
                    symbol=symbol, price=price,
                    signal=analysis['signal'], strength=analysis['strength'],
                    rsi=analysis['rsi'], sl=sl, targets=targets
                )

                await bot.send_message(chat_id=CHAT_ID, text=message)
                sent_signals[signal_key] = now
                signals_found += 1
                await asyncio.sleep(2)

        except Exception as e:
            logger.error(f"خطأ في {symbol}: {e}")

        await asyncio.sleep(0.3)

    logger.info(f"✅ انتهى الفحص — {signals_found} إشارة")

# ═══════════════════════════════════════════
#         أوامر التلغرام
# ═══════════════════════════════════════════
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🌟 Welcome to Sarab Signals! 🌟\n\n"
        "Your professional Bitunix Futures signal bot is ready.\n\n"
        "📡 Scanning 100 coins every 5 minutes\n"
        "📊 Analysis: RSI + MACD + EMA + Bollinger\n"
        "🔄 Auto LONG & SHORT signals\n\n"
        "Commands:\n"
        "/scan — Scan now\n"
        "/status — Bot status"
    )

async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Scanning the market, please wait...")
    await scan_market(context.bot)
    await update.message.reply_text("✅ Scan complete!")

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbols = await get_bitunix_symbols()
    await update.message.reply_text(
        f"✅ Sarab Signals is running!\n\n"
        f"⏱ Scan every: 5 minutes\n"
        f"📊 Coins monitored: {len(symbols)}\n"
        f"🔄 Signals sent: {len(sent_signals)}"
    )

async def auto_scan(context: ContextTypes.DEFAULT_TYPE):
    await scan_market(context.bot)

# ═══════════════════════════════════════════
#         تشغيل البوت
# ═══════════════════════════════════════════
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("scan", scan_command))
    app.add_handler(CommandHandler("status", status_command))
    app.job_queue.run_repeating(auto_scan, interval=CHECK_INTERVAL, first=15)
    logger.info("🚀 Sarab Signals Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
