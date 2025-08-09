import logging
import time
import asyncio
from typing import Dict, List, Optional, Tuple
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("signal_generator")

class SignalGenerator:
    def __init__(self, config):
        """Initialize SignalGenerator with configuration"""
        self.config = config
        self.binance_client = ccxt.binance({
            'apiKey': config.BINANCE_API_KEY,
            'secret': config.BINANCE_API_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True
            }
        })
        self.telegram = getattr(config, 'TELEGRAM', None)
        self.portfolio = getattr(config, 'PORTFOLIO', None)
        self.ta = TechnicalAnalysis()
        self.last_signal = {}
        self.last_signal_time = {}
        self.failed_symbols = {}
        self.warned_symbols = set()
        self.scanned_symbols = set()
        self.used_coins = set()
        self.cooldown_period = 900  # 15 minutes
        self.major_pairs = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
            'ADA/USDT', 'AVAX/USDT', 'DOT/USDT', 'LINK/USDT', 'MATIC/USDT'
        ]
        self.timeframes = ['5m', '15m', '30m', '1h']
        self.min_volume_btc = 10  # Minimum 10 BTC volume in last 24h

    async def get_historical_data(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        """Fetch historical OHLCV data"""
        try:
            ohlcv = await self.binance_client.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv or len(ohlcv) < 100:
                logger.warning(f"Insufficient data for {symbol} on {timeframe}")
                return pd.DataFrame()
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def _calculate_sl_levels(self, df: pd.DataFrame, current_price: float, direction: str) -> Tuple[float, float]:
        """Calculate dynamic stop loss between 1% and 2%"""
        try:
            # Calculate ATR and recent volatility
            atr = self.ta.calculate_atr(df, period=14)
            atr_value = atr.iloc[-1] if not atr.empty else 0
            atr_percent = (atr_value / current_price) if current_price > 0 else 0.01
            
            # Base SL between 1% and 2%
            base_sl_pct = min(max(0.01, atr_percent * 0.8), 0.02)
            
            # Adjust based on recent volatility
            recent_high = df['high'].iloc[-10:].max()
            recent_low = df['low'].iloc[-10:].min()
            recent_range = (recent_high - recent_low) / current_price
            volatility_factor = min(max(0.8, recent_range * 4), 1.2)
            
            # Final SL calculation
            sl_pct = base_sl_pct * volatility_factor
            sl_pct = min(max(0.01, sl_pct), 0.02)  # Enforce 1-2% range
            
            # Calculate SL price
            if direction == "BULLISH":
                sl = current_price * (1 - sl_pct)
                swing_low = df['low'].iloc[-10:].min()
                sl = min(sl, swing_low)
            else:
                sl = current_price * (1 + sl_pct)
                swing_high = df['high'].iloc[-10:].max()
                sl = max(sl, swing_high)
            
            # Ensure minimum distance
            min_distance = current_price * 0.002
            if direction == "BULLISH":
                sl = min(sl, current_price - min_distance)
            else:
                sl = max(sl, current_price + min_distance)
            
            # Calculate actual SL percentage
            actual_sl_pct = abs((sl - current_price) / current_price)
            return sl, actual_sl_pct * 100
            
        except Exception as e:
            logger.error(f"Error in _calculate_sl_levels: {str(e)}")
            sl_pct = 0.015  # 1.5% default on error
            if direction == "BULLISH":
                return current_price * (1 - sl_pct), sl_pct * 100
            return current_price * (1 + sl_pct), sl_pct * 100

    async def _generate_signal(self, symbol: str, timeframe: str, df: pd.DataFrame) -> Optional[Dict]:
        """Generate trading signal for given symbol and timeframe"""
        try:
            if df.empty or len(df) < 100:
                return None

            current_price = df['close'].iloc[-1]
            if pd.isna(current_price) or current_price <= 0:
                return None

            # Calculate indicators
            indicators = self._calculate_indicators(df, current_price)
            if not indicators:
                return None

            # Determine direction based on indicators
            direction = self._determine_direction(indicators)
            if direction == "NEUTRAL":
                return None

            # Calculate SL/TP
            sl, sl_pct = self._calculate_sl_levels(df, current_price, direction)
            tp_levels, tp1_pct = self._calculate_tp_levels(current_price, direction, sl, sl_pct)
            
            # Calculate risk/reward and confidence
            risk_reward = self._calculate_risk_reward(current_price, sl, tp_levels[0], direction)
            confidence = self._calculate_confidence(indicators, direction)
            win_probability = 0.7 + (0.2 * (confidence - 0.7) / 0.3)

            signal = {
                'symbol': symbol,
                'direction': direction,
                'timeframe': timeframe,
                'confidence': min(confidence, 0.99),
                'entry': current_price,
                'tp_levels': tp_levels,
                'tp1_percent': tp1_pct * 100,
                'sl': sl,
                'sl_percent': sl_pct,
                'risk_reward': risk_reward,
                'win_probability': min(win_probability, 0.95),
                'leverage': 10,
                'indicators': indicators,
                'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            }

            return signal

        except Exception as e:
            logger.error(f"Error in _generate_signal for {symbol}: {str(e)}")
            return None

    async def analyze_pair(self, symbol: str) -> List[Dict]:
        """Analyze a trading pair across multiple timeframes"""
        try:
            logger.info(f"Analyzing {symbol}...")
            
            # Check cooldown
            current_time = time.time()
            if symbol in self.last_signal_time and (current_time - self.last_signal_time[symbol]) < self.cooldown_period:
                return []

            signals = []
            timeframe_signals = {}

            # Analyze each timeframe
            for tf in self.timeframes:
                df = await self.get_historical_data(symbol, tf)
                if df.empty:
                    continue
                    
                signal = await self._generate_signal(symbol, tf, df)
                if signal:
                    timeframe_signals[tf] = signal
                    signals.append(signal)

            # Require at least 2 timeframes to agree
            if len(signals) < 2:
                return []

            # Check direction agreement
            directions = [s['direction'] for s in signals]
            dir_counts = {
                'BULLISH': directions.count('BULLISH'),
                'BEARISH': directions.count('BEARISH')
            }
            
            if dir_counts['BULLISH'] >= 2:
                agreed_direction = 'BULLISH'
            elif dir_counts['BEARISH'] >= 2:
                agreed_direction = 'BEARISH'
            else:
                return []

            # Get the best signal
            agreeing_signals = [s for s in signals if s['direction'] == agreed_direction]
            base_signal = max(agreeing_signals, key=lambda x: x['confidence'])

            # Final validation
            if not self._validate_signal(base_signal):
                return []

            # Update state
            self.last_signal[symbol] = base_signal
            self.last_signal_time[symbol] = current_time
            self.used_coins.add(symbol)
            
            logger.info(f"✅ Valid signal for {symbol}: {agreed_direction} with {base_signal['confidence']:.1%} confidence")
            return [base_signal]

        except Exception as e:
            logger.error(f"Error in analyze_pair for {symbol}: {str(e)}")
            return []

    async def scan_market(self):
        """Scan all trading pairs for signals"""
        try:
            # Get all USDT pairs
            markets = await self.binance_client.load_markets()
            symbols = [s for s in markets if s.endswith('/USDT') and markets[s]['active']]
            
            logger.info(f"Scanning {len(symbols)} trading pairs...")
            
            for symbol in symbols:
                try:
                    signals = await self.analyze_pair(symbol)
                    for signal in signals:
                        await self._process_signal(signal)
                    await asyncio.sleep(0.5)  # Rate limiting
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in scan_market: {str(e)}")
        finally:
            await self.binance_client.close()

    async def _process_signal(self, signal: Dict):
        """Process and send valid signal"""
        try:
            message = self.format_signal(signal)
            logger.info(f"Signal found:\n{message}")
            
            if self.telegram:
                await self.telegram.send_message(message)
                
            if self.portfolio:
                await self.portfolio.open_position(
                    symbol=signal['symbol'],
                    direction=signal['direction'],
                    entry_price=signal['entry'],
                    leverage=signal['leverage'],
                    stop_loss=signal['sl'],
                    take_profit=signal['tp_levels']
                )
                
        except Exception as e:
            logger.error(f"Error processing signal: {str(e)}")

    def format_signal(self, signal: Dict) -> str:
        """Format signal as a readable message"""
        try:
            tp1, tp2, tp3 = signal['tp_levels'][:3]
            direction_emoji = "🟢" if signal['direction'] == 'BULLISH' else "🔴"
            
            return f"""
{direction_emoji} *{signal['symbol']} - {signal['direction']} Signal* {direction_emoji}
📊 *Confidence*: {signal['confidence']:.1%}
⏰ *Time*: {signal['timestamp']} UTC
📈 *Entry*: ${signal['entry']:.4f}
🎯 *TP1*: ${tp1:.4f} ({signal['tp1_percent']:.2f}%)
🎯 *TP2*: ${tp2:.4f}
🎯 *TP3*: ${tp3:.4f}
🛑 *SL*: ${signal['sl']:.4f} ({signal['sl_percent']:.2f}%)
📊 *Risk/Reward*: {signal['risk_reward']:.2f}
🎯 *Win Probability*: {signal['win_probability']:.1%}
📊 *Indicators*:
   - EMA: {signal['indicators']['ema']}
   - MACD: {signal['indicators']['macd']}
   - RSI: {signal['indicators']['rsi']}
   - ADX: {signal['indicators']['adx']:.2f}
"""
        except Exception as e:
            logger.error(f"Error formatting signal: {str(e)}")
            return f"Error formatting signal: {str(e)}"

    async def run(self):
        """Main trading loop"""
        logger.info("Starting signal generator...")
        try:
            while True:
                try:
                    await self.scan_market()
                    await asyncio.sleep(300)  # 5 minutes between scans
                except KeyboardInterrupt:
                    logger.info("Signal generator stopped by user")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {str(e)}")
                    await asyncio.sleep(60)  # Wait 1 minute on error
        finally:
            await self.binance_client.close()

# Helper classes (should be in separate files)
class TechnicalAnalysis:
    def calculate_atr(self, df, period=14):
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

class Config:
    BINANCE_API_KEY = "YOUR_API_KEY"
    BINANCE_API_SECRET = "YOUR_API_SECRET"

if __name__ == "__main__":
    config = Config()
    bot = SignalGenerator(config)
    asyncio.run(bot.run())
