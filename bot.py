
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

CHECK_INTERVAL       = 300    # فحص كل 5 دقائق
MAX_PRICE            = 10.0   # عملات تحت $10
MAX_SYMBOLS          = 150    # أكبر 150 عملة
MIN_CONFIDENCE       = 70     # فقط الصفقات القوية
TECH_SCORE_THRESHOLD  = 5      # نقاط القوة الفنية المطلوبة
SIGNAL_COOLDOWN      = 300     # 5 دقائق بين نفس الإشارة

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
    macd_cross_up   = macd_line.iloc[-1] > macd_sig.iloc[-1] and macd_line.iloc[-2] <= macd_sig.iloc[-2]
    macd_cross_down = macd_line.iloc[-1] < macd_sig.iloc[-1] and macd_line.iloc[-2] >= macd_sig.iloc[-2]
    macd_bull = macd_line.iloc[-1] > macd_sig.iloc[-1]
    macd_bear = macd_line.iloc[-1] < macd_sig.iloc[-1]

    # EMA
    ema9  = ta.trend.EMAIndicator(close, window=9).ema_indicator().iloc[-1]
    ema21 = ta.trend.EMAIndicator(close, window=21).ema_indicator().iloc[-1]

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

    if long_pts >= TECH_SCORE_THRESHOLD and long_pts >= short_pts:
        direction = "LONG"
        tech_score = long_pts
    elif short_pts >= TECH_SCORE_THRESHOLD and short_pts > long_pts:
        direction = "SHORT"
        tech_score = short_pts
    else:
        return None

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
        "vol_surge":      vol_surge,
        "atr":            atr,
        "ema9":           round(ema9, 6),
        "ema21":          round(ema21, 6),
        "price_change_1h": round(((price - close.iloc[-4])/close.iloc[-4])*100,2),
    }

# ═══════════════════════════════════════════
#     تحليل AI
# ═══════════════════════════════════════════
async def ai_analysis(symbol: str, data: dict) -> dict:
    if not ANTHROPIC_API_KEY:
        conf = min(99, int((data['tech_score'] / 10) * 100) + np.random.randint(0, 10))
        return {"confidence": conf, "comment": "تحليل فني"}
    coin = symbol.replace("USDT","")
    prompt = f"""أنت محلل
