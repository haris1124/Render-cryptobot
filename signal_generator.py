import logging
import time
import asyncio
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt
from typing import Dict, List, Optional, Tuple
from datetime import datetime

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

class TechnicalAnalysis:
    """Technical analysis indicators"""
    
    @staticmethod
    def calculate_ema(df, periods: List[int]) -> Dict:
        """Calculate Exponential Moving Averages"""
        return {f'ema_{period}': df['close'].ewm(span=period, adjust=False).mean() for period in periods}
    
    @staticmethod
    def calculate_macd(df, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD indicator"""
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        hist = macd - signal_line
        return {'macd': macd, 'signal': signal_line, 'hist': hist}
    
    @staticmethod
    def calculate_rsi(df, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_bollinger_bands(df, period: int = 20, std_dev: int = 2) -> Dict:
        """Calculate Bollinger Bands"""
        middle_band = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        return {'upper': upper_band, 'middle': middle_band, 'lower': lower_band}
    
    @staticmethod
    def calculate_adx(df, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        high, low, close = df['high'], df['low'], df['close']
        
        # Calculate +DM and -DM
        up = high.diff()
        down = low.diff() * -1
        plus_dm = up.where((up > down) & (up > 0), 0)
        minus_dm = down.where((down > up) & (down > 0), 0)
        
        # Calculate True Range
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Smooth the calculations
        tr_smooth = tr.rolling(window=period).sum()
        plus_dm_smooth = plus_dm.rolling(window=period).sum()
        minus_dm_smooth = minus_dm.rolling(window=period).sum()
        
        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        
        # Calculate DX and ADX
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10))
        return dx.rolling(window=period).mean()
    
    @staticmethod
    def calculate_atr(df, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high, low, close = df['high'], df['low'], df['close']
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def calculate_supertrend(df, period: int = 10, multiplier: int = 3) -> Dict:
        """Calculate SuperTrend indicator"""
        high, low, close = df['high'], df['low'], df['close']
        
        # Calculate ATR
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Calculate basic upper and lower bands
        hl2 = (high + low) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        # Initialize SuperTrend
        supertrend = pd.Series(index=df.index)
        direction = pd.Series(1, index=df.index)
        
        for i in range(1, len(df)):
            if close.iloc[i] > upper_band.iloc[i-1]:
                direction.iloc[i] = 1
            elif close.iloc[i] < lower_band.iloc[i-1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i-1]
                
                if direction.iloc[i] == 1 and lower_band.iloc[i] < lower_band.iloc[i-1]:
                    lower_band.iloc[i] = lower_band.iloc[i-1]
                if direction.iloc[i] == -1 and upper_band.iloc[i] > upper_band.iloc[i-1]:
                    upper_band.iloc[i] = upper_band.iloc[i-1]
            
            if direction.iloc[i] == 1:
                supertrend.iloc[i] = lower_band.iloc[i]
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
        
        return {'supertrend': supertrend, 'direction': direction}
    
    @staticmethod
    def calculate_vwap(df) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        tp = (df['high'] + df['low'] + df['close']) / 3
        return (tp * df['volume']).cumsum() / df['volume'].cumsum()
    
    @staticmethod
    def calculate_stoch_rsi(df, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> pd.Series:
        """Calculate Stochastic RSI"""
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate Stochastic RSI
        stoch_rsi = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min() + 1e-10)
        stoch_rsi_k = stoch_rsi.rolling(smooth_k).mean() * 100
        return stoch_rsi_k

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
        self.ta = TechnicalAnalysis()
        self.last_signal = {}
        self.last_signal_time = {}
        self.scanned_symbols = set()
        self.used_coins = set()
        self.cooldown_period = 900  # 15 minutes
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

    def _calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate all technical indicators"""
        try:
            indicators = {}
            
            # 1. Moving Averages
            emas = self.ta.calculate_ema(df, [20, 50, 200])
            indicators.update(emas)
            
            # 2. MACD
            macd = self.ta.calculate_macd(df)
            indicators['macd'] = macd['macd'].iloc[-1]
            indicators['macd_signal'] = macd['signal'].iloc[-1]
            indicators['macd_hist'] = macd['hist'].iloc[-1]
            
            # 3. RSI
            indicators['rsi'] = self.ta.calculate_rsi(df).iloc[-1]
            
            # 4. Bollinger Bands
            bb = self.ta.calculate_bollinger_bands(df)
            indicators['bb_upper'] = bb['upper'].iloc[-1]
            indicators['bb_middle'] = bb['middle'].iloc[-1]
            indicators['bb_lower'] = bb['lower'].iloc[-1]
            
            # 5. ADX
            indicators['adx'] = self.ta.calculate_adx(df).iloc[-1]
            
            # 6. ATR
            indicators['atr'] = self.ta.calculate_atr(df).iloc[-1]
            
            # 7. Volume
            indicators['volume'] = df['volume'].iloc[-1]
            indicators['volume_ma'] = df['volume'].rolling(20).mean().iloc[-1]
            
            # 8. SuperTrend
            supertrend = self.ta.calculate_supertrend(df)
            indicators['supertrend'] = supertrend['supertrend'].iloc[-1]
            indicators['supertrend_direction'] = supertrend['direction'].iloc[-1]
            
            # 9. VWAP
            indicators['vwap'] = self.ta.calculate_vwap(df).iloc[-1]
            
            # 10. Stochastic RSI
            indicators['stoch_rsi'] = self.ta.calculate_stoch_rsi(df).iloc[-1]
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error in _calculate_indicators: {str(e)}")
            return {}

    def _calculate_sl_levels(self, df: pd.DataFrame, current_price: float, direction: str) -> Tuple[float, float]:
        """Calculate stop loss levels with improved logic"""
        try:
            # Calculate ATR and recent volatility
            atr = self.ta.calculate_atr(df)
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

            # Calculate all indicators
            indicators = self._calculate_indicators(df)
            if not indicators:
                return None

            # Determine signal direction
            direction = self._determine_direction(indicators, current_price)
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
                'timeframe': timeframe,
                'direction': direction,
                'entry': current_price,
                'sl': sl,
                'sl_percent': sl_pct,
                'tp_levels': tp_levels,
                'tp1_percent': tp1_pct * 100,
                'risk_reward': risk_reward,
                'confidence': min(confidence, 0.99),
                'win_probability': min(win_probability, 0.95),
                'leverage': 10,
                'indicators': indicators,
                'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            }

            # Validate signal
            if not self._validate_signal(signal):
                return None

            return signal

        except Exception as e:
            logger.error(f"Error in _generate_signal for {symbol}: {str(e)}")
            return None

    def _determine_direction(self, indicators: Dict, current_price: float) -> str:
        """Determine trade direction based on indicators"""
        try:
            # Count bullish and bearish signals
            bullish = 0
            bearish = 0
            
            # EMA Cross
            if indicators['ema_20'] > indicators['ema_50'] > indicators['ema_200']:
                bullish += 1
            elif indicators['ema_20'] < indicators['ema_50'] < indicators['ema_200']:
                bearish += 1
                
            # MACD
            if indicators['macd'] > indicators['macd_signal']:
                bullish += 1
            else:
                bearish += 1
                
            # RSI
            if indicators['rsi'] > 50:
                bullish += 1
            else:
                bearish += 1
                
            # Price vs VWAP
            if current_price > indicators['vwap']:
                bullish += 1
            else:
                bearish += 1
                
            # SuperTrend
            if indicators['supertrend_direction'] == 1:
                bullish += 1
            else:
                bearish += 1
                
            # Determine final direction
            if bullish >= 3 and indicators['adx'] > 25:
                return "BULLISH"
            elif bearish >= 3 and indicators['adx'] > 25:
                return "BEARISH"
            return "NEUTRAL"
            
        except Exception as e:
            logger.error(f"Error in _determine_direction: {str(e)}")
            return "NEUTRAL"

    def _calculate_tp_levels(self, entry: float, direction: str, sl: float, sl_pct: float) -> Tuple[List[float], float]:
        """Calculate take profit levels"""
        try:
            # Calculate TP1 based on 1:1.5 risk:reward
            if direction == "BULLISH":
                tp1 = entry + (1.5 * (entry - sl))
                tp2 = entry + (2.0 * (entry - sl))
                tp3 = entry + (3.0 * (entry - sl))
            else:
                tp1 = entry - (1.5 * (sl - entry))
                tp2 = entry - (2.0 * (sl - entry))
                tp3 = entry - (3.0 * (sl - entry))
                
            tp1_pct = abs((tp1 - entry) / entry)
            return [tp1, tp2, tp3], tp1_pct
            
        except Exception as e:
            logger.error(f"Error in _calculate_tp_levels: {str(e)}")
            if direction == "BULLISH":
                return [entry * 1.015, entry * 1.03, entry * 1.045], 1.5
            else:
                return [entry * 0.985, entry * 0.97, entry * 0.955], 1.5

    def _calculate_risk_reward(self, entry: float, sl: float, tp: float, direction: str) -> float:
        """Calculate risk to reward ratio"""
        try:
            if direction == "BULLISH":
                risk = entry - sl
                reward = tp - entry
            else:
                risk = sl - entry
                reward = entry - tp
                
            return reward / max(risk, 1e-10)  # Avoid division by zero
        except Exception as e:
            logger.error(f"Error in _calculate_risk_reward: {str(e)}")
            return 1.0

    def _calculate_confidence(self, indicators: Dict, direction: str) -> float:
        """Calculate signal confidence score (0-1)"""
        try:
            confidence = 0.5  # Base confidence
            
            # ADX strength (0-0.2)
            adx_strength = min(indicators['adx'] / 50.0, 1.0) * 0.2
            confidence += adx_strength
            
            # Volume confirmation (0-0.15)
            volume_ratio = indicators['volume'] / max(indicators['volume_ma'], 1)
            volume_boost = min(max(0, (volume_ratio - 1.0) * 0.3), 0.15)
            confidence += volume_boost
            
            # RSI confirmation (0-0.15)
            if (direction == "BULLISH" and 30 < indicators['rsi'] < 70) or \
               (direction == "BEARISH" and 30 < indicators['rsi'] < 70):
                confidence += 0.1
                
            # MACD confirmation (0-0.1)
            if (direction == "BULLISH" and indicators['macd_hist'] > 0) or \
               (direction == "BEARISH" and indicators['macd_hist'] < 0):
                confidence += 0.1
                
            # SuperTrend confirmation (0-0.1)
            if (direction == "BULLISH" and indicators['supertrend_direction'] == 1) or \
               (direction == "BEARISH" and indicators['supertrend_direction'] == -1):
                confidence += 0.1
                
            return min(max(confidence, 0.5), 0.99)  # Keep between 0.5 and 0.99
            
        except Exception as e:
            logger.error(f"Error in _calculate_confidence: {str(e)}")
            return 0.7

    def _validate_signal(self, signal: Dict) -> bool:
        """Validate signal meets all criteria"""
        try:
            # Check minimum risk/reward
            if signal['risk_reward'] < 1.3:
                return False
                
            # Check minimum confidence
            if signal['confidence'] < 0.7:
                return False
                
            # Check ADX strength
            if signal['indicators']['adx'] < 25:
                return False
                
            # Check volume
            if signal['indicators']['volume'] < signal['indicators']['volume_ma'] * 0.8:
                return False
                
            # Check price distance from VWAP
            vwap_dist = abs(signal['entry'] - signal['indicators']['vwap']) / signal['indicators']['vwap']
            if vwap_dist > 0.02:  # More than 2% away from VWAP
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error in _validate_signal: {str(e)}")
            return False

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

    async def get_active_symbols(self) -> List[str]:
        """Get list of active trading symbols"""
        try:
            markets = await self.binance_client.load_markets()
            return [s for s in markets 
                   if s.endswith('/USDT') 
                   and markets[s]['active'] 
                   and markets[s].get('quoteVolume24h', 0) > self.min_volume_btc * 50000]  # Approx BTC value
        except Exception as e:
            logger.error(f"Error getting active symbols: {str(e)}")
            return []

    async def scan_market(self):
        """Scan all trading pairs for signals"""
        try:
            symbols = await self.get_active_symbols()
            if not symbols:
                logger.error("No active symbols found")
                return

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

    async def _process_signal(self, signal: Dict):
        """Process and send valid signal"""
        try:
            message = self.format_signal(signal)
            logger.info(f"Signal found:\n{message}")
            
            # Send to Telegram if configured
            if hasattr(self.config, 'TELEGRAM_BOT_TOKEN') and hasattr(self.config, 'TELEGRAM_CHAT_ID'):
                from telegram import Bot
                bot = Bot(token=self.config.TELEGRAM_BOT_TOKEN)
                await bot.send_message(
                    chat_id=self.config.TELEGRAM_CHAT_ID,
                    text=message,
                    parse_mode='Markdown'
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

*Indicators*:
- EMA 20/50/200: {signal['indicators']['ema_20']:.4f}/{signal['indicators']['ema_50']:.4f}/{signal['indicators']['ema_200']:.4f}
- RSI: {signal['indicators']['rsi']:.2f}
- ADX: {signal['indicators']['adx']:.2f}
- MACD: {signal['indicators']['macd']:.4f} (Signal: {signal['indicators']['macd_signal']:.4f})
- Volume: {signal['indicators']['volume']:.2f} (MA20: {signal['indicators']['volume_ma']:.2f})
- VWAP: {signal['indicators']['vwap']:.4f}
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
                    logger.info("Market scan completed. Waiting for next cycle...")
                    await asyncio.sleep(300)  # 5 minutes between scans
                except KeyboardInterrupt:
                    logger.info("Signal generator stopped by user")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {str(e)}")
                    await asyncio.sleep(60)  # Wait 1 minute on error
        finally:
            await self.binance_client.close()

class Config:
    """Configuration class"""
    BINANCE_API_KEY = "YOUR_API_KEY"
    BINANCE_API_SECRET = "YOUR_API_SECRET"
    TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
    TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"

if __name__ == "__main__":
    config = Config()
    bot = SignalGenerator(config)
    asyncio.run(bot.run())
