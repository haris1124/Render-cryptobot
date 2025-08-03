import logging
import time
import asyncio
import random
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import ccxt.async_support as ccxt
from config import Config
from Technical_analysis import TechnicalAnalysis
from Telegram_client import Telegram
from portfolio import Portfolio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SignalGenerator")

class SignalGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.binance = ccxt.binance({
            'apiKey': config.BINANCE_API_KEY,
            'secret': config.BINANCE_API_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True
            }
        })
        self.telegram = Telegram(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
        self.portfolio = Portfolio(self.binance, config)
        self.ta = TechnicalAnalysis()
        self.last_signals = {}
        self.cooldown = 900  # 15 minutes
        self.min_sl_pct = 0.5  # 0.5% minimum
        self.max_sl_pct = 1.5  # 1.5% maximum

    async def get_real_time_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch real-time market data with freshness check"""
        try:
            ohlcv = await self.binance.fetch_ohlcv(symbol, timeframe, limit=200)
            if not ohlcv or len(ohlcv) < 100:
                return None

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert to timezone-naive UTC
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize(None)
            df.set_index('timestamp', inplace=True)
            
            # Get current time as timezone-naive UTC
            current_time = pd.Timestamp.utcnow().tz_localize(None)
            last_candle_time = df.index[-1]
            
            # Check data freshness (2 minutes threshold)
            time_diff = (current_time - last_candle_time).total_seconds()
            if time_diff > 120:
                logger.warning(f"Stale data for {symbol} on {timeframe} - {time_diff:.0f} seconds old")
                return None
                
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def calculate_sl_tp(self, entry_price: float, direction: str) -> Tuple[float, List[float], float]:
        """Calculate SL and TP levels with strict 0.5-1.5% SL"""
        try:
            # Random SL between 0.5% and 1.5%
            sl_pct = random.uniform(self.min_sl_pct, self.max_sl_pct) / 100
            
            if direction == 'BUY':
                sl = entry_price * (1 - sl_pct)
                tp1 = entry_price * (1 + (sl_pct * 1.5))  # 1.5:1 RR
                tp2 = entry_price * (1 + (sl_pct * 2.0))  # 2.0:1 RR
                tp3 = entry_price * (1 + (sl_pct * 3.0))  # 3.0:1 RR
            else:  # SELL
                sl = entry_price * (1 + sl_pct)
                tp1 = entry_price * (1 - (sl_pct * 1.5))
                tp2 = entry_price * (1 - (sl_pct * 2.0))
                tp3 = entry_price * (1 - (sl_pct * 3.0))
                
            return sl, [tp1, tp2, tp3], sl_pct * 100
            
        except Exception as e:
            logger.error(f"Error in calculate_sl_tp: {str(e)}")
            # Fallback values
            if direction == 'BUY':
                return entry_price * 0.99, [entry_price * 1.02, entry_price * 1.03, entry_price * 1.05], 1.0
            else:
                return entry_price * 1.01, [entry_price * 0.98, entry_price * 0.97, entry_price * 0.95], 1.0

    async def check_indicators(self, df: pd.DataFrame) -> Tuple[str, float, Dict]:
        """Check all indicators and return direction, confidence, and indicator states"""
        try:
            indicators = {}
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # 1. EMAs
            ema20 = self.ta.ema(close, 20)
            ema50 = self.ta.ema(close, 50)
            ema200 = self.ta.ema(close, 200)
            ema_bullish = ema20[-1] > ema50[-1] > ema200[-1]
            ema_bearish = ema20[-1] < ema50[-1] < ema200[-1]
            
            # 2. MACD
            macd_line, signal_line, _ = self.ta.macd(close)
            macd_bullish = macd_line[-1] > signal_line[-1]
            macd_bearish = macd_line[-1] < signal_line[-1]
            
            # 3. RSI
            rsi = self.ta.rsi(close)
            rsi_bullish = rsi[-1] > 60
            rsi_bearish = rsi[-1] < 40
            
            # 4. Bollinger Bands
            bb_upper, _, bb_lower = self.ta.bollinger_bands(close)
            bb_bullish = close[-1] > bb_upper[-1]
            bb_bearish = close[-1] < bb_lower[-1]
            
            # 5. ADX (must be > 40)
            adx = self.ta.adx(high, low, close)
            trend_strength = adx[-1] > 40
            adx_value = adx[-1]
            
            # 6. Volume (must be above 30-period average)
            vol_ma = np.mean(volume[-30:])
            volume_ok = volume[-1] > vol_ma
            
            # 7. Stochastic RSI
            stoch_rsi = self.ta.stoch_rsi(close)
            stoch_bullish = stoch_rsi[-1] > 0.8
            stoch_bearish = stoch_rsi[-1] < 0.2
            
            # 8. SuperTrend
            supertrend = self.ta.supertrend(high, low, close)
            supertrend_bullish = supertrend[-1] < close[-1]
            supertrend_bearish = supertrend[-1] > close[-1]
            
            # Store indicator states
            indicators = {
                'ema': 'BULLISH' if ema_bullish else 'BEARISH' if ema_bearish else 'NEUTRAL',
                'macd': 'BULLISH' if macd_bullish else 'BEARISH',
                'rsi': 'BULLISH' if rsi_bullish else 'BEARISH' if rsi_bearish else 'NEUTRAL',
                'bb': 'BULLISH' if bb_bullish else 'BEARISH' if bb_bearish else 'NEUTRAL',
                'adx': adx_value,
                'volume': volume_ok,
                'stoch_rsi': 'BULLISH' if stoch_bullish else 'BEARISH' if stoch_bearish else 'NEUTRAL',
                'supertrend': 'BULLISH' if supertrend_bullish else 'BEARISH'
            }
            
            # Count confirmations
            bull_confirm = sum([
                ema_bullish,
                macd_bullish,
                rsi_bullish,
                bb_bullish,
                supertrend_bullish,
                stoch_bullish,
                trend_strength,
                volume_ok
            ])
            
            bear_confirm = sum([
                ema_bearish,
                macd_bearish,
                rsi_bearish,
                bb_bearish,
                supertrend_bearish,
                stoch_bearish,
                trend_strength,
                volume_ok
            ])
            
            # Calculate confidence (0-100%)
            confidence = max(bull_confirm, bear_confirm) / 8
            
            if bull_confirm == 8:
                return 'BUY', min(0.99, 0.7 + (confidence * 0.3)), indicators
            elif bear_confirm == 8:
                return 'SELL', min(0.99, 0.7 + (confidence * 0.3)), indicators
            else:
                return 'NEUTRAL', 0, indicators
                
        except Exception as e:
            logger.error(f"Error in check_indicators: {str(e)}")
            return 'NEUTRAL', 0, {}

    async def scan_market(self, symbols: List[str] = None) -> List[Dict]:
        """Scan market and generate trading signals"""
        if symbols is None:
            symbols = ['BTC/USDT']  # Default symbol
            
        logger.info(f"Starting market scan for {symbols}...")
        signals = []
        
        for symbol in symbols:
            try:
                # Check cooldown
                current_time = time.time()
                if symbol in self.last_signals and (current_time - self.last_signals[symbol]) < self.cooldown:
                    continue
                
                # Check all timeframes
                timeframes = ['5m', '15m', '30m', '1h']
                tf_signals = []
                
                for tf in timeframes:
                    df = await self.get_real_time_data(symbol, tf)
                    if df is None:
                        continue
                        
                    direction, confidence, indicators = await self.check_indicators(df)
                    if direction != 'NEUTRAL' and confidence >= 0.85:  # 85% confidence
                        tf_signals.append({
                            'timeframe': tf,
                            'direction': direction,
                            'price': df['close'].iloc[-1],
                            'confidence': confidence,
                            'indicators': indicators
                        })
                
                # Check if all timeframes agree
                if len(tf_signals) == 4:
                    directions = [s['direction'] for s in tf_signals]
                    if all(d == 'BUY' for d in directions):
                        final_direction = 'BUY'
                    elif all(d == 'SELL' for d in directions):
                        final_direction = 'SELL'
                    else:
                        continue
                    
                    # Use the most recent price
                    entry_price = tf_signals[-1]['price']
                    
                    # Calculate SL/TP with strict 0.5-1.5% SL
                    sl, tp_levels, sl_pct = self.calculate_sl_tp(entry_price, final_direction)
                    
                    # Calculate risk/reward
                    if final_direction == 'BUY':
                        risk = entry_price - sl
                        reward = tp_levels[0] - entry_price
                    else:
                        risk = sl - entry_price
                        reward = entry_price - tp_levels[0]
                        
                    risk_reward = reward / risk if risk > 0 else 0
                    
                    # Final validation
                    if risk_reward < 1.5:  # Minimum 1.5:1 RR
                        continue
                        
                    signal = {
                        'symbol': symbol,
                        'direction': final_direction,
                        'entry': entry_price,
                        'sl': sl,
                        'sl_pct': sl_pct,
                        'tp_levels': tp_levels,
                        'risk_reward': risk_reward,
                        'confidence': min(tf_signals[-1]['confidence'], 0.99),
                        'time': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                        'indicators': tf_signals[-1]['indicators']
                    }
                    
                    signals.append(signal)
                    self.last_signals[symbol] = current_time
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                continue
                
        return signals

    def format_signal(self, signal: Dict) -> str:
        """Format signal for display"""
        try:
            direction_emoji = "🟢" if signal['direction'] == 'BUY' else "🔴"
            tp1_pct = ((signal['tp_levels'][0] / signal['entry'] - 1) * 100) if signal['direction'] == 'BUY' else ((1 - signal['tp_levels'][0] / signal['entry']) * 100)
            
            return f"""
{direction_emoji} *{signal['symbol']} {signal['direction']} Signal* {direction_emoji}
⏰ Time: {signal['time']}
💰 Entry: {signal['entry']:.8f}
🛑 Stop Loss: {signal['sl']:.8f} ({signal['sl_pct']:.2f}%)
🎯 Take Profits:
   TP1: {signal['tp_levels'][0]:.8f} ({tp1_pct:.2f}%)
   TP2: {signal['tp_levels'][1]:.8f}
   TP3: {signal['tp_levels'][2]:.8f}
📊 Confidence: {signal['confidence']:.2%}
⚖️ Risk/Reward: 1:{signal['risk_reward']:.2f}
📈 Indicators:
   - EMA: {signal['indicators']['ema']}
   - MACD: {signal['indicators']['macd']}
   - RSI: {signal['indicators']['rsi']}
   - BB: {signal['indicators']['bb']}
   - ADX: {signal['indicators']['adx']:.2f}
   - Stoch RSI: {signal['indicators']['stoch_rsi']}
   - SuperTrend: {signal['indicators']['supertrend']}
   - Volume: {'Above Avg' if signal['indicators']['volume'] else 'Below Avg'}
"""
        except Exception as e:
            logger.error(f"Error formatting signal: {str(e)}")
            return f"Signal formatting error: {str(e)}"

    async def close(self):
        """Close all connections"""
        await self.binance.close()
