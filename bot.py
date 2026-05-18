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
TOP_SYMBOLS_LIMIT = 50 # أفضل 50 عملة بحجم تداول

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
#     سحب العملات من Bitunix تلقائياً
# ═══════════════════════════════════════════
async def get_bitunix_symbols():
    """سحب كل عملات Bitunix الفيوتشر"""
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
        logger.error(f"خطأ في سحب العملات من Bitunix: {e}")
    
    return [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
        "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT"
    ]

# ═══════════════════════════════════════════
#         سحب بيانات السوق من Bitunix
# ═══════════════════════════════════════════
async def get_klines_bitunix(symbol: str, interval: str = "15m", limit: int = 150):
    """سحب بيانات الشمعدانات من Bitunix"""
    url = "https://fapi.bitunix.com/api/v1/futures/market/kline"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                data = await response.json()
        
        if data.get('code') == 0 and data.get('data'):
            klines = data['data']
            df = pd.DataFrame(klines)
            df['close'] = df['close'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['open'] = df['open'].astype(float)
            df['volume'] = df['volume'].astype(float)
            return df
    except Exception as e:
        logger.error(f"خطأ في سحب بيانات {symbol}: {e}")
    
    return None

async def get_price_bitunix(symbol: str):
    """سحب السعر الحالي من Bitunix"""
    url = "https://fapi.bitunix.com/api/v1/futures/market/tickers"
    params = {"symbol": symbol}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                data = await response.json()
        
        if data.get('code') == 0 and data.get('data'):
            return float(data['data'][0]['lastPrice'])
    except Exception as e:
        logger.error(f"خطأ في سحب سعر {symbol}: {e}")
    
    return None

# ═══════════════════════════════════════════
#         تحليل المؤشرات الفنية
# ═══════════════════════════════════════════
def analyze(df: pd.DataFrame):
    """تحليل RSI + MACD + EMA + Bollinger"""
    close = df['close']
    high = df['high']
    low = df['low']
    
    rsi_indicator = ta.momentum.RSIIndicator(close, window=14)
    rsi = rsi_indicator.rsi()
    current_rsi = rsi.iloc[-1]
    
    macd_indicator = ta.trend.MACD(close)
    macd = macd_indicator.macd()
    macd_signal = macd_indicator.macd_signal()
    macd_cross_up = macd.iloc[-1] > macd_signal.iloc[-1] and macd.iloc[-2] <= macd_signal.iloc[-2]
    macd_cross_down = macd.iloc[-1] < macd_signal.iloc[-1] and macd.iloc[-2] >= macd_signal.iloc[-2]
    
    ema20 = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    ema50 = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    current_price = close.iloc[-1]
    
    bb = ta.volatility.BollingerBands(close, window=20)
    bb_lower = bb.bollinger_lband().iloc[-1]
    bb_upper = bb.bollinger_hband().iloc[-1]
    
    atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
    
    long_score = 0
    short_score = 0
    
    if current_rsi < 35: long_score += 2
    elif current_rsi < 45: long_score += 1
    if macd_cross_up: long_score += 3
    if current_price > ema20.iloc[-1]: long_score += 1
    if ema20.iloc[-1] > ema50.iloc[-1]: long_score += 1
    if current_price <= bb_lower * 1.01: long_score += 2
    
    if current_rsi > 65: short_score += 2
    elif current_rsi > 55: short_score += 1
    if macd_cross_down: short_score += 3
    if current_price < ema20.iloc[-1]: short_score += 1
    if ema20.iloc[-1] < ema50.iloc[-1]: short_score += 1
    if current_price >= bb_upper * 0.99: short_score += 2
    
    signal = None
    strength = 0
    
    if long_score >= 4 and long_score > short_score:
        signal = "LONG"
        strength = long_score
    elif short_score >= 4 and short_score > long_score:
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
    sl_multiplier = 2
    targets = []
    
    if signal == "LONG":
        sl = round(price - (atr * sl_multiplier), 8)
        for m in multipliers:
            targets.append(round(price + (atr * m), 8))
    else:
        sl = round(price + (atr * sl_multiplier), 8)
        for m in multipliers:
            targets.append(round(price - (atr * m), 8))
    
    return sl, targets

# ═══════════════════════════════════════════
#      تنسيق رسالة الإشارة الاحترافية
# ═══════════════════════════════════════════
def format_signal_message(symbol: str, price: float, signal: str,
                           strength: int, rsi: float,
                           sl: float, targets: list):
    
    coin = symbol.replace("USDT", "")
    time_now = datetime.utcnow().strftime("%d.%m.%Y %H:%M:%S UTC+0")
    
    if signal == "LONG":
        emoji = "🟢🟢🟢"
        sl_pct = round(((price - sl) / price) * 100, 2)
        tp1_pct = round(((targets[0] - price) / price) * 100, 2)
    else:
        emoji = "🔴🔴🔴"
        sl_pct = round(((sl - price) / price) * 100, 2)
        tp1_pct = round(((price - targets[0]) / price) * 100, 2)
    
    stars = "⭐" * strength
    
    def fmt(p):
        if p >= 1:
            return f"{p:,.4f}$"
        elif p >= 0.01:
            return f"{p:.6f}$"
        else:
            return f"{p:.8f}$"
    
    msg = (
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
    return msg

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
                    symbol=symbol,
                    price=price,
                    signal=analysis['signal'],
                    strength=analysis['strength'],
                    rsi=analysis['rsi'],
                    sl=sl,
                    targets=targets
                )
                
                await bot.send_message(chat_id=CHAT_ID, text=message)
                sent_signals[signal_key] = now
                signals_found += 1
                await asyncio.sleep(2)
                
        except Exception as e:
            logger.error(f"خطأ في {symbol}: {e}")
        
        await asyncio.sleep(0.5)
    
    logger.info(f"✅ انتهى الفحص — {signals_found} إشارة")

# ═══════════════════════════════════════════
#         أوامر التلغرام
# ═══════════════════════════════════════════
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbols = await get_bitunix_symbols()
    await update.message.reply_text(
        f"🤖 بوت إشارات Bitunix Futures يعمل!\n\n"
        f"📡 يفحص {len(symbols)} عملة كل 5 دقائق\n"
        f"📊 يحلل: RSI + MACD + EMA + Bollinger\n"
        f"🔄 لونغ وشورت تلقائياً\n\n"
        f"الأوامر:\n"
        f"/scan — فحص فوري\n"
        f"/status — حالة البوت"
    )

async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 جاري الفحص الآن، انتظر...")
    await scan_market(context.bot)
    await update.message.reply_text("✅ انتهى الفحص!")

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbols = await get_bitunix_symbols()
    await update.message.reply_text(
        f"✅ البوت يعمل بشكل طبيعي\n\n"
        f"⏱ الفحص كل: 5 دقائق\n"
        f"📊 العملات المراقبة: {len(symbols)}\n"
        f"🔄 الإشارات المرسلة: {len(sent_signals)}"
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
    
    logger.info("🚀 البوت يعمل...")
    app.run_polling()

if __name__ == "__main__":
    main()
