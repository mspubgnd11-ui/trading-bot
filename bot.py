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
import numpy as np

# ═══════════════════════════════════════════
#           إعدادات البوت
# ═══════════════════════════════════════════
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# العملات اللي رح يراقبها البوت
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT"
]

# إعدادات التحليل
RSI_PERIOD = 14
RSI_OVERSOLD = 30      # إشارة شراء
RSI_OVERBOUGHT = 70    # إشارة بيع
CHECK_INTERVAL = 900   # كل 15 دقيقة

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
#         سحب بيانات السوق
# ═══════════════════════════════════════════
async def get_klines(symbol: str, interval: str = "15m", limit: int = 100):
    """سحب بيانات الشمعدانات من Binance"""
    url = f"https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            data = await response.json()
            
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['volume'] = df['volume'].astype(float)
    
    return df

async def get_price(symbol: str):
    """سحب السعر الحالي"""
    url = f"https://api.binance.com/api/v3/ticker/price"
    params = {"symbol": symbol}
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            data = await response.json()
    
    return float(data['price'])

# ═══════════════════════════════════════════
#         تحليل المؤشرات الفنية
# ═══════════════════════════════════════════
def analyze(df: pd.DataFrame):
    """تحليل RSI + MACD + EMA"""
    close = df['close']
    
    # RSI
    rsi = ta.momentum.RSIIndicator(close, window=RSI_PERIOD).rsi()
    current_rsi = rsi.iloc[-1]
    
    # MACD
    macd_indicator = ta.trend.MACD(close)
    macd = macd_indicator.macd().iloc[-1]
    macd_signal = macd_indicator.macd_signal().iloc[-1]
    macd_prev = macd_indicator.macd().iloc[-2]
    macd_signal_prev = macd_indicator.macd_signal().iloc[-2]
    
    # EMA
    ema20 = ta.trend.EMAIndicator(close, window=20).ema_indicator().iloc[-1]
    ema50 = ta.trend.EMAIndicator(close, window=50).ema_indicator().iloc[-1]
    current_price = close.iloc[-1]
    
    # قوة الإشارة
    signal = None
    strength = 0
    
    # إشارة شراء
    if current_rsi < RSI_OVERSOLD:
        strength += 2
    if macd > macd_signal and macd_prev <= macd_signal_prev:  # تقاطع صاعد
        strength += 2
    if current_price > ema20 > ema50:
        strength += 1
        
    if strength >= 3:
        signal = "BUY"
    
    # إشارة بيع
    sell_strength = 0
    if current_rsi > RSI_OVERBOUGHT:
        sell_strength += 2
    if macd < macd_signal and macd_prev >= macd_signal_prev:  # تقاطع هابط
        sell_strength += 2
    if current_price < ema20 < ema50:
        sell_strength += 1
        
    if sell_strength >= 3:
        signal = "SELL"
        strength = sell_strength
    
    return {
        "signal": signal,
        "strength": strength,
        "rsi": round(current_rsi, 2),
        "macd": round(macd, 4),
        "ema20": round(ema20, 4),
        "ema50": round(ema50, 4),
    }

# ═══════════════════════════════════════════
#         حساب الأهداف ووقف الخسارة
# ═══════════════════════════════════════════
def calculate_targets(price: float, signal: str, df: pd.DataFrame):
    """حساب الأهداف ووقف الخسارة بناءً على السوق"""
    # ATR لحساب التذبذب
    atr = ta.volatility.AverageTrueRange(
        df['high'], df['low'], df['close'], window=14
    ).average_true_range().iloc[-1]
    
    if signal == "BUY":
        stop_loss = round(price - (atr * 2), 6)
        target1 = round(price + (atr * 2), 6)
        target2 = round(price + (atr * 4), 6)
        target3 = round(price + (atr * 6), 6)
    else:  # SELL
        stop_loss = round(price + (atr * 2), 6)
        target1 = round(price - (atr * 2), 6)
        target2 = round(price - (atr * 4), 6)
        target3 = round(price - (atr * 6), 6)
    
    return stop_loss, target1, target2, target3

# ═══════════════════════════════════════════
#         تنسيق رسالة الإشارة
# ═══════════════════════════════════════════
def format_signal_message(symbol: str, price: float, signal: str, 
                           strength: int, rsi: float,
                           sl: float, t1: float, t2: float, t3: float):
    
    emoji = "🟢" if signal == "BUY" else "🔴"
    action = "شراء" if signal == "BUY" else "بيع"
    
    stars = "⭐" * min(strength, 5)
    
    # نسبة الربح
    if signal == "BUY":
        profit1 = round(((t1 - price) / price) * 100, 2)
        profit2 = round(((t2 - price) / price) * 100, 2)
        profit3 = round(((t3 - price) / price) * 100, 2)
        loss = round(((sl - price) / price) * 100, 2)
    else:
        profit1 = round(((price - t1) / price) * 100, 2)
        profit2 = round(((price - t2) / price) * 100, 2)
        profit3 = round(((price - t3) / price) * 100, 2)
        loss = round(((price - sl) / price) * 100, 2)
    
    coin = symbol.replace("USDT", "")
    time_now = datetime.now().strftime("%d/%m/%Y — %H:%M")
    
    message = f"""
{emoji} *إشارة {action} — {coin}/USDT*

💰 *سعر الدخول:* `${price:,.6g}`

🎯 *الهدف 1:* `${t1:,.6g}` _(+{profit1}%)_
🎯 *الهدف 2:* `${t2:,.6g}` _(+{profit2}%)_  
🎯 *الهدف 3:* `${t3:,.6g}` _(+{profit3}%)_

🛑 *وقف الخسارة:* `${sl:,.6g}` _({loss}%)_

📊 *RSI:* `{rsi}`
💪 *قوة الإشارة:* {stars}

🏦 *المنصة:* Bitunix
⏰ *{time_now}*

⚠️ _هذه إشارة تحليلية فقط، التداول على مسؤوليتك_
"""
    return message

# ═══════════════════════════════════════════
#         الفحص التلقائي للسوق
# ═══════════════════════════════════════════
async def scan_market(bot: Bot):
    """فحص كل العملات وإرسال الإشارات"""
    logger.info("🔍 جاري فحص السوق...")
    signals_found = 0
    
    for symbol in SYMBOLS:
        try:
            df = await get_klines(symbol)
            price = await get_price(symbol)
            analysis = analyze(df)
            
            if analysis['signal']:
                sl, t1, t2, t3 = calculate_targets(price, analysis['signal'], df)
                
                message = format_signal_message(
                    symbol=symbol,
                    price=price,
                    signal=analysis['signal'],
                    strength=analysis['strength'],
                    rsi=analysis['rsi'],
                    sl=sl, t1=t1, t2=t2, t3=t3
                )
                
                await bot.send_message(
                    chat_id=CHAT_ID,
                    text=message,
                    parse_mode='Markdown'
                )
                signals_found += 1
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"خطأ في {symbol}: {e}")
    
    if signals_found == 0:
        logger.info("لا توجد إشارات قوية حالياً")
    else:
        logger.info(f"تم إرسال {signals_found} إشارة")

# ═══════════════════════════════════════════
#         أوامر التلغرام
# ═══════════════════════════════════════════
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *بوت الإشارات يعمل!*\n\n"
        "📡 يفحص السوق كل 15 دقيقة\n\n"
        "الأوامر:\n"
        "/scan — فحص فوري الآن\n"
        "/status — حالة البوت\n"
        "/help — المساعدة",
        parse_mode='Markdown'
    )

async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 جاري الفحص الآن...")
    await scan_market(context.bot)

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbols_list = "\n".join([f"• {s.replace('USDT', '/USDT')}" for s in SYMBOLS])
    await update.message.reply_text(
        f"✅ *البوت يعمل بشكل طبيعي*\n\n"
        f"⏱ الفحص كل: 15 دقيقة\n"
        f"📊 العملات المراقبة:\n{symbols_list}",
        parse_mode='Markdown'
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 *المساعدة*\n\n"
        "/start — تشغيل البوت\n"
        "/scan — فحص فوري للسوق\n"
        "/status — عرض حالة البوت\n\n"
        "🔍 *كيف يعمل البوت؟*\n"
        "يحلل السوق باستخدام:\n"
        "• RSI — مؤشر القوة النسبية\n"
        "• MACD — مؤشر الزخم\n"
        "• EMA — المتوسط المتحرك",
        parse_mode='Markdown'
    )

# ═══════════════════════════════════════════
#         الفحص التلقائي الدوري
# ═══════════════════════════════════════════
async def auto_scan(context: ContextTypes.DEFAULT_TYPE):
    await scan_market(context.bot)

# ═══════════════════════════════════════════
#         تشغيل البوت
# ═══════════════════════════════════════════
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # أوامر التلغرام
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("scan", scan_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("help", help_command))
    
    # الفحص التلقائي كل 15 دقيقة
    app.job_queue.run_repeating(auto_scan, interval=CHECK_INTERVAL, first=10)
    
    logger.info("🚀 البوت يعمل...")
    app.run_polling()

if __name__ == "__main__":
    main()
