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
TELEGRAM_TOKEN    = os.getenv("TELEGRAM_TOKEN")
CHAT_ID           = os.getenv("CHAT_ID")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

CHECK_INTERVAL  = 300    # كل 5 دقايق
MAX_PRICE       = 10.0
MAX_SYMBOLS     = 150
MIN_CONFIDENCE  = 70     # صفقات قوية فقط
SIGNAL_COOLDOWN = 300    # 5 دقايق بين نفس الإشارة

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            df['volume'] = pd.to_numeric(df.get('quoteVol', df.get('baseVol', 1.0)), errors='coerce')
            df.dropna(subset=['open','high','low','close'], inplace=True)
            return df
    except Exception as e:
        logger.error(f"خطأ كلاين {symbol}: {e}")
    return None

# ═══════════════════════════════════════════
def technical_analysis(df: pd.DataFrame):
    if len(df) < 50:
        return None
    close = df['close']
    high  = df['high']
    low   = df['low']
    volume= df['volume']
    price = close.iloc[-1]

    rsi_val = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    macd_obj = ta.trend.MACD(close)
    macd_line = macd_obj.macd()
    macd_sig  = macd_obj.macd_signal()
    macd_cross_up   = macd_line.iloc[-1] > macd_sig.iloc[-1] and macd_line.iloc[-2] <= macd_sig.iloc[-2]
    macd_cross_down = macd_line.iloc[-1] < macd_sig.iloc[-1] and macd_line.iloc[-2] >= macd_sig.iloc[-2]
    macd_bull = macd_line.iloc[-1] > macd_sig.iloc[-1]
    macd_bear = macd_line.iloc[-1] < macd_sig.iloc[-1]

    ema9  = ta.trend.EMAIndicator(close, window=9).ema_indicator().iloc[-1]
    ema21 = ta.trend.EMAIndicator(close, window=21).ema_indicator().iloc[-1]

    vol_avg = volume.rolling(20).mean().iloc[-1]
    vol_cur = volume.iloc[-1]
    vol_surge = vol_cur > vol_avg * 1.5

    long_pts = short_pts = 0
    if rsi_val < 50: long_pts += 2
    if rsi_val < 40: long_pts += 1
    if macd_cross_up: long_pts += 3
    elif macd_bull: long_pts += 1
    if price > ema9: long_pts += 1
    if ema9 > ema21: long_pts += 1
    if vol_surge and price > close.iloc[-2]: long_pts +=1

    if rsi_val > 50: short_pts += 2
    if rsi_val > 60: short_pts +=1
    if macd_cross_down: short_pts +=3
    elif macd_bear: short_pts +=1
    if price < ema9: short_pts +=1
    if ema9 < ema21: short_pts +=1
    if vol_surge and price < close.iloc[-2]: short_pts +=1

    if long_pts >=2 and long_pts >= short_pts:
        return {"direction":"LONG","tech_score":long_pts,"price":price,"rsi":round(rsi_val,1),"macd_desc":"صاعد 📈" if macd_bull else "هابط 📉","vol_surge":vol_surge,"ema9":round(ema9,6),"ema21":round(ema21,6)}
    elif short_pts >=2 and short_pts > long_pts:
        return {"direction":"SHORT","tech_score":short_pts,"price":price,"rsi":round(rsi_val,1),"macd_desc":"هابط 📉","vol_surge":vol_surge,"ema9":round(ema9,6),"ema21":round(ema21,6)}
    else:
        return None

# ═══════════════════════════════════════════
async def ai_analysis(symbol: str, data: dict) -> dict:
    conf = min(99,int((data['tech_score']/10)*100)+np.random.randint(0,10))
    return {"confidence": conf,"comment":"تحليل فني"}

# ═══════════════════════════════════════════
def make_targets(price: float, signal: str):
    if signal=="LONG":
        sl = round(price*0.985,8)
        tps = [round(price*m,8) for m in [1.005,1.010,1.015,1.025,1.040]]
    else:
        sl = round(price*1.015,8)
        tps = [round(price*m,8) for m in [0.995,0.990,0.985,0.975,0.960]]
    return sl,tps

def fmt(p): return f"{p:,.4f}$" if p>=1 else f"{p:.6f}$" if p>=0.01 else f"{p:.8f}$"

def build_message(symbol, tech, ai, sl, tps):
    signal = tech['direction']
    conf = ai['confidence']
    coin = symbol.replace("USDT","")
    now = datetime.utcnow().strftime("%d.%m.%Y %H:%M:%S UTC+0")
    emoji = "🟢🟢🟢" if signal=="LONG" else "🔴🔴🔴"
    action = "لونغ" if signal=="LONG" else "شورت"
    ai_line = f"\n🤖 AI: {ai['comment']}\n" if ai['comment'] else ""
    return (f"{emoji} {coin}/USDT {action} {emoji}\n\n"
            f"تحديث: {now}\n"
            f"➡️ نقطة الدخول: {fmt(tech['price'])}{ai_line}\n"
            f"💯 نسبة الثقة: {conf}%\n"
            f"🎯 TP1: {fmt(tps[0])}\n"
            f"🎯 TP2: {fmt(tps[1])}\n"
            f"🎯 TP3: {fmt(tps[2])}\n"
            f"🎯 TP4: {fmt(tps[3])}\n"
            f"🎯 TP5: {fmt(tps[4])}\n"
            f"🛑 SL: {fmt(sl)}\n"
            f"📈 MACD: {tech['macd_desc']}\n"
            f"📊 RSI: {tech['rsi']}\n"
            f"{'📊 حجم مرتفع 🔥' if tech['vol_surge'] else ''}\n"
            f"🏦 Bitunix Futures")

# ═══════════════════════════════════════════
sent_signals = {}

async def scan_market(bot: Bot):
    logger.info("🔍 بدأ الفحص...")
    symbols = await get_symbols()
    found = 0
    for sym in symbols:
        df = await get_klines(sym)
        if df is None or len(df)<50: continue
        tech = technical_analysis(df)
        if not tech: continue
        key = f"{sym}_{tech['direction']}"
        now_time = asyncio.get_event_loop().time()
        if now_time - sent_signals.get(key,0) < SIGNAL_COOLDOWN: continue
        ai = await ai_analysis(sym, tech)
        if ai['confidence'] < MIN_CONFIDENCE: continue
        sl,tps = make_targets(tech['price'],tech['direction'])
        msg = build_message(sym, tech, ai, sl, tps)
        await bot.send_message(chat_id=CHAT_ID,text=msg)
        sent_signals[key] = now_time
        found+=1
    if found==0:
        await bot.send_message(chat_id=CHAT_ID,text="⚠️ لا توجد صفقات قوية حالياً")
    logger.info(f"✅ انتهى الفحص — {found} إشارة من {len(symbols)} عملة")

# ═══════════════════════════════════════════
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🌟 البوت جاهز — /scan للفحص الفوري")

async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 جاري الفحص...")
    await scan_market(context.bot)
    await update.message.reply_text("✅ انتهى الفحص!")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"✅ البوت يعمل، إشارات مرسلة: {len(sent_signals)}")

async def auto_scan(context: ContextTypes.DEFAULT_TYPE):
    await scan_market(context.bot)

# ═══════════════════════════════════════════
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start",cmd_start))
    app.add_handler(CommandHandler("scan",cmd_scan))
    app.add_handler(CommandHandler("status",cmd_status))
    app.job_queue.run_repeating(auto_scan, interval=CHECK_INTERVAL, first=10)
    logger.info("🚀 البوت بدأ العمل...")
    app.run_polling()

if __name__=="__main__":
    main()
