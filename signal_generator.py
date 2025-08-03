import logging
import time
import asyncio
import numpy as np
import pandas as pd
import ccxt.async_support as ccxt
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import random
from config import Config
from Technical_analysis import TechnicalAnalysis
from Telegram_client import Telegram
from portfolio import Portfolio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot_accurate_sl.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("StrictSLSignalGenerator")

class SignalGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.binance = ccxt.binance({
            'apiKey': config.BINANCE_API_KEY,
            'secret': config.BINANCE_API_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        self.telegram = Telegram(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
        self.portfolio = Portfolio(self.binance, config)
        self.ta = TechnicalAnalysis()
        self.last_signals = {}
        self.cooldown = 900  # 15 minutes
        self.required_indicators = 8
        self.min_sl_pct = 0.5  # 0.5% minimum
        self.max_sl_pct = 1.5  # 1.5% maximum

    async def get_real_time_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch real-time market data with freshness check"""
        try:
            ohlcv = await self.binance.fetch_ohlcv(symbol, timeframe, limit=200)
            if not ohlcv or len(ohlcv) < 100:
                return None

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Verify data freshness (last candle within 2 minutes)
            last_candle_time = df.index[-1]
            if (pd.Timestamp.utcnow() - last_candle_time).total_seconds() > 120:
                logger.warning(f"Stale data for {symbol} on {timeframe}")
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

    async def check_indicators(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Check all 8 indicators for confirmation"""
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
                return 'BUY', confidence
            elif bear_confirm == 8:
                return 'SELL', confidence
            else:
                return 'NEUTRAL', 0
                
        except Exception as e:
            logger.error(f"Error in check_indicators: {str(e)}")
            return 'NEUTRAL', 0

    async def generate_signal(self, symbol: str) -> Optional[Dict]:
        """Generate trading signal for a symbol"""
        try:
            current_time = time.time()
            
            # Check cooldown
            if symbol in self.last_signals and (current_time - self.last_signals[symbol]) < self.cooldown:
                return None
                
            # Check all timeframes
            timeframes = ['5m', '15m', '30m', '1h']
            signals = []
            
            for tf in timeframes:
                df = await self.get_real_time_data(symbol, tf)
                if df is None:
                    continue
                    
                direction, confidence = await self.check_indicators(df)
                if direction != 'NEUTRAL' and confidence >= 0.85:  # 85% confidence
                    signals.append({
                        'timeframe': tf,
                        'direction': direction,
                        'price': df['close'].iloc[-1],
                        'confidence': confidence
                    })
            
            # Require all timeframes to agree
            if len(signals) == 4:
                directions = [s['direction'] for s in signals]
                if all(d == 'BUY' for d in directions):
                    final_direction = 'BUY'
                elif all(d == 'SELL' for d in directions):
                    final_direction = 'SELL'
                else:
                    return None
                    
                # Use the most recent price
                entry_price = signals[-1]['price']
                
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
                    return None
                    
                signal = {
                    'symbol': symbol,
                    'direction': final_direction,
                    'entry': entry_price,
                    'sl': sl,
                    'sl_pct': sl_pct,
                    'tp_levels': tp_levels,
                    'risk_reward': risk_reward,
                    'confidence': min(signals[-1]['confidence'], 0.99),
                    'time': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                self.last_signals[symbol] = current_time
                return signal
                
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {str(e)}")
            
        return None

    async def scan_market(self, symbols: List[str] = None):
        """Scan market and generate trading signals"""
        if symbols is None:
            symbols = ['BTC/USDT']  # Default symbol if none provided
        logger.info("Starting market monitoring with strict SL (0.5-1.5%)...")
        
        while True:
            try:
                for symbol in symbols:
                    try:
                        signal = await self.generate_signal(symbol)
                        if signal:
                            # Format and send signal
                            message = self.format_signal(signal)
                            logger.info(f"New signal: {message}")
                            await self.telegram.send_message(message)
                            
                            # Execute trade
                            await self.portfolio.place_trade(
                                symbol=symbol,
                                direction=signal['direction'],
                                entry=signal['entry'],
                                sl=signal['sl'],
                                tp_levels=signal['tp_levels']
                            )
                            
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {str(e)}")
                        continue
                        
                    await asyncio.sleep(1)  # Rate limiting
                    
                # Wait for next cycle
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in monitor_market: {str(e)}")
                await asyncio.sleep(60)

    def format_signal(self, signal: Dict) -> str:
        """Format signal for display"""
        direction_emoji = "🟢" if signal['direction'] == 'BUY' else "🔴"
        return f"""
{direction_emoji} *{signal['symbol']} {signal['direction']} Signal* {direction_emoji}
⏰ Time: {signal['time']}
💰 Entry: {signal['entry']:.8f}
🛑 Stop Loss: {signal['sl']:.8f} ({signal['sl_pct']:.2f}%)
🎯 Take Profits: 
   TP1: {signal['tp_levels'][0]:.8f} (1.5R)
   TP2: {signal['tp_levels'][1]:.8f} (2.0R)
   TP3: {signal['tp_levels'][2]:.8f} (3.0R)
📊 Confidence: {signal['confidence']:.2%}
⚖️ Risk/Reward: 1:{signal['risk_reward']:.2f}
"""

async def main():
    config = Config()
    bot = StrictSLSignalGenerator(config)
    
    # Example symbols to monitor
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT']
    
    try:
        await bot.monitor_market(symbols)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
    finally:
        await bot.binance.close()
        logger.info("Bot shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())
