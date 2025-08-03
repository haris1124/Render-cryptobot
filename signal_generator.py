import logging
import time
import random
import asyncio
import json
import collections
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("signal_generator")

class SignalGenerator:
    def __init__(self, config):
        self.config = config
        self.binance_client = None
        self.telegram = None
        self.portfolio = None
        self.ta = TechnicalAnalysis()
        
        # State management
        self.last_signal = {}
        self.last_signal_time = {}
        self.scanned_symbols = set()
        self.used_coins = set()
        self.daily_signals = collections.defaultdict(int)
        
        # Constants
        self.cooldown_period = 900  # 15 minutes
        self.min_volume_usd = 100000  # $100k daily volume minimum
        self.max_spread = 0.001  # 0.1% max spread
        self.timeframes = ['5m', '15m', '30m', '1h']  # 4 timeframes for confirmation
        self.required_indicators = 8  # Number of indicators to confirm
        self.min_indicators_agree = 6  # Minimum indicators that must agree
        self.major_pairs = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'XRP/USDT',
            'SOL/USDT', 'DOT/USDT', 'DOGE/USDT', 'AVAX/USDT', 'LTC/USDT'
        ]

    async def initialize(self):
        """Initialize the signal generator"""
        try:
            self.binance_client = ccxt.binance({
                'apiKey': self.config.BINANCE_API_KEY,
                'secret': self.config.BINANCE_API_SECRET,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                    'adjustForTimeDifference': True
                }
            })
            self.telegram = Telegram(self.config.TELEGRAM_BOT_TOKEN, self.config.TELEGRAM_CHAT_ID)
            self.portfolio = Portfolio(self.binance_client, self.config)
            await self.binance_client.load_markets()
            return True
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from order book"""
        try:
            ticker = await self.binance_client.fetch_ticker(symbol)
            return float(ticker['last'])
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None

    async def get_market_quality(self, symbol: str) -> Dict[str, Any]:
        """Check if market conditions are good for trading"""
        try:
            ticker = await self.binance_client.fetch_ticker(symbol)
            orderbook = await self.binance_client.fetch_order_book(symbol, limit=5)
            
            spread = (orderbook['asks'][0][0] - orderbook['bids'][0][0]) / orderbook['bids'][0][0]
            volume_24h = float(ticker['quoteVolume']) if ticker['quoteVolume'] else 0
            
            return {
                'spread': spread,
                'volume_24h': volume_24h,
                'is_tradable': (spread < self.max_spread and 
                               volume_24h > self.min_volume_usd),
                'best_bid': orderbook['bids'][0][0],
                'best_ask': orderbook['asks'][0][0]
            }
        except Exception as e:
            logger.error(f"Error checking market quality for {symbol}: {e}")
            return {'is_tradable': False}

    async def get_historical_data(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        """Fetch OHLCV data with error handling"""
        try:
            ohlcv = await self.binance_client.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv:
                logger.warning(f"No data returned for {symbol} on {timeframe}")
                return pd.DataFrame()
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            if df.empty or len(df) < 100:  # Minimum 100 candles required
                return pd.DataFrame()
                
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def _calculate_indicators(self, df: pd.DataFrame, current_price: float) -> Dict:
        """Calculate all 8 technical indicators with error handling"""
        indicators = {
            'ema': 'NEUTRAL',
            'macd': 'NEUTRAL',
            'rsi': 'NEUTRAL',
            'bb': 'NEUTRAL',
            'adx': 0,
            'fib': 'NEUTRAL',
            'stoch_rsi': 'NEUTRAL',
            'supertrend': 'NEUTRAL',
            'vwap': 'NEUTRAL',
            'rsi_value': 50,
            'atr_value': 0
        }
        
        try:
            # 1. EMA (20, 50, 200)
            try:
                emas = self.ta.calculate_ema(df, [20, 50, 200])
                if isinstance(emas, dict) and 'ema_20' in emas and 'ema_50' in emas:
                    ema_20 = emas['ema_20'].iloc[-1] if hasattr(emas['ema_20'], 'iloc') else emas['ema_20'][-1]
                    ema_50 = emas['ema_50'].iloc[-1] if hasattr(emas['ema_50'], 'iloc') else emas['ema_50'][-1]
                    indicators['ema'] = 'BULLISH' if ema_20 > ema_50 else 'BEARISH'
            except Exception as e:
                logger.warning(f"EMA calculation warning: {e}")

            # 2. MACD
            try:
                macd = self.ta.calculate_macd(df)
                if isinstance(macd, dict) and 'macd' in macd and 'signal' in macd:
                    macd_line = macd['macd'].iloc[-1] if hasattr(macd['macd'], 'iloc') else macd['macd'][-1]
                    signal_line = macd['signal'].iloc[-1] if hasattr(macd['signal'], 'iloc') else macd['signal'][-1]
                    indicators['macd'] = 'BULLISH' if macd_line > signal_line else 'BEARISH'
            except Exception as e:
                logger.warning(f"MACD calculation warning: {e}")

            # 3. RSI
            try:
                rsi = self.ta.calculate_rsi(df)
                if not isinstance(rsi, (pd.Series, list)):
                    rsi = [rsi] if isinstance(rsi, (int, float)) else []
                
                if len(rsi) > 0:
                    rsi_value = float(rsi[-1])
                    indicators['rsi_value'] = rsi_value
                    indicators['rsi'] = 'BULLISH' if rsi_value > 60 else 'BEARISH' if rsi_value < 40 else 'NEUTRAL'
            except Exception as e:
                logger.warning(f"RSI calculation warning: {e}")

            # 4. Bollinger Bands
            try:
                bb = self.ta.calculate_bollinger_bands(df)
                if bb and 'upper' in bb and 'lower' in bb:
                    upper = bb['upper'].iloc[-1] if hasattr(bb['upper'], 'iloc') else bb['upper'][-1]
                    lower = bb['lower'].iloc[-1] if hasattr(bb['lower'], 'iloc') else bb['lower'][-1]
                    if current_price > upper:
                        indicators['bb'] = 'BULLISH'
                    elif current_price < lower:
                        indicators['bb'] = 'BEARISH'
            except Exception as e:
                logger.warning(f"Bollinger Bands calculation warning: {e}")

            # 5. ADX
            try:
                adx = self.ta.calculate_adx(df)
                if not isinstance(adx, (pd.Series, list)):
                    adx = [adx] if isinstance(adx, (int, float)) else []
                
                if len(adx) > 0:
                    indicators['adx'] = float(adx[-1])
            except Exception as e:
                logger.warning(f"ADX calculation warning: {e}")

            # 6. Fibonacci Retracement
            try:
                if len(df) >= 50:
                    high = df['high'].rolling(50).max().iloc[-1]
                    low = df['low'].rolling(50).min().iloc[-1]
                    diff = high - low
                    if diff > 0:
                        fib_618 = high - 0.618 * diff
                        indicators['fib'] = 'BULLISH' if current_price > fib_618 else 'BEARISH'
            except Exception as e:
                logger.warning(f"Fibonacci calculation warning: {e}")

            # 7. Stochastic RSI
            try:
                stoch_rsi = self.ta.calculate_stoch_rsi(df)
                if not isinstance(stoch_rsi, (pd.Series, list)):
                    stoch_rsi = [stoch_rsi] if isinstance(stoch_rsi, (int, float)) else []
                
                if len(stoch_rsi) > 0:
                    stoch_value = float(stoch_rsi[-1])
                    if stoch_value > 80:
                        indicators['stoch_rsi'] = 'BULLISH'
                    elif stoch_value < 20:
                        indicators['stoch_rsi'] = 'BEARISH'
            except Exception as e:
                logger.warning(f"Stochastic RSI calculation warning: {e}")

            # 8. SuperTrend
            try:
                supertrend = self.ta.calculate_supertrend(df)
                if supertrend is not None and 'in_uptrend' in supertrend:
                    uptrend = supertrend['in_uptrend'].iloc[-1] if hasattr(supertrend['in_uptrend'], 'iloc') else supertrend['in_uptrend'][-1]
                    indicators['supertrend'] = 'BULLISH' if uptrend else 'BEARISH'
            except Exception as e:
                logger.warning(f"SuperTrend calculation warning: {e}")

            # 9. VWAP
            try:
                vwap = self.ta.calculate_vwap(df)
                if vwap is not None:
                    vwap_value = vwap.iloc[-1] if hasattr(vwap, 'iloc') else vwap[-1]
                    indicators['vwap'] = 'BULLISH' if current_price > vwap_value else 'BEARISH'
            except Exception as e:
                logger.warning(f"VWAP calculation warning: {e}")

        except Exception as e:
            logger.error(f"Error in _calculate_indicators: {e}")
        
        return indicators

    def _get_direction(self, indicators: Dict) -> str:
        """Determine overall direction from indicators"""
        try:
            directions = [
                indicators.get('ema', 'NEUTRAL'),
                indicators.get('macd', 'NEUTRAL'),
                indicators.get('rsi', 'NEUTRAL'),
                indicators.get('bb', 'NEUTRAL'),
                indicators.get('fib', 'NEUTRAL'),
                indicators.get('stoch_rsi', 'NEUTRAL'),
                indicators.get('supertrend', 'NEUTRAL'),
                indicators.get('vwap', 'NEUTRAL')
            ]
            
            bull_count = directions.count('BULLISH')
            bear_count = directions.count('BEARISH')
            
            if bull_count >= self.min_indicators_agree:
                return 'BULLISH'
            elif bear_count >= self.min_indicators_agree:
                return 'BEARISH'
            return 'NEUTRAL'
        except Exception as e:
            logger.error(f"Error in _get_direction: {e}")
            return 'NEUTRAL'

    def _calculate_levels(self, direction: str, current_price: float, df: pd.DataFrame) -> Tuple[List[float], float, float, float]:
        """Calculate TP/SL levels with proper error handling"""
        try:
            # Calculate ATR for TP1
            atr_value = 0
            try:
                atr = self.ta.calculate_atr(df)
                if atr is not None:
                    atr_value = float(atr[-1] if hasattr(atr, '__getitem__') else atr)
            except:
                pass

            # TP1 between 0.8% and 1.5%
            atr_pct = atr_value / current_price if current_price > 0 else 0.01
            tp1_pct = min(max(0.008, atr_pct), 0.015)
            
            if direction == "BULLISH":
                tp1 = current_price * (1 + tp1_pct)
                tp2 = current_price * (1 + tp1_pct * 1.5)
                tp3 = current_price * (1 + tp1_pct * 2)
                sl_pct = random.uniform(0.011, 0.018)
                sl = min(current_price * (1 - sl_pct), df['low'].iloc[-10:].min())
            else:  # BEARISH
                tp1 = current_price * (1 - tp1_pct)
                tp2 = current_price * (1 - tp1_pct * 1.5)
                tp3 = current_price * (1 - tp1_pct * 2)
                sl_pct = random.uniform(0.011, 0.018)
                sl = max(current_price * (1 + sl_pct), df['high'].iloc[-10:].max())
                
            return [tp1, tp2, tp3], sl, tp1_pct * 100, sl_pct * 100
            
        except Exception as e:
            logger.error(f"Error in _calculate_levels: {e}")
            # Return safe defaults
            if direction == "BULLISH":
                return [current_price * 1.01, current_price * 1.015, current_price * 1.02], current_price * 0.99, 1.0, 1.0
            else:
                return [current_price * 0.99, current_price * 0.985, current_price * 0.98], current_price * 1.01, 1.0, 1.0

            def _calculate_confidence(self, indicators: Dict) -> float:
        """Calculate signal confidence based on indicator agreement"""
        try:
            directions = [
                indicators.get('ema', 'NEUTRAL'),
                indicators.get('macd', 'NEUTRAL'),
                indicators.get('rsi', 'NEUTRAL'),
                indicators.get('bb', 'NEUTRAL'),
                indicators.get('fib', 'NEUTRAL'),
                indicators.get('stoch_rsi', 'NEUTRAL'),
                indicators.get('supertrend', 'NEUTRAL'),
                indicators.get('vwap', 'NEUTRAL')
            ]
            
            # Count agreements
            main_dir = 'BULLISH' if directions.count('BULLISH') > directions.count('BEARISH') else 'BEARISH'
            agree_count = sum(1 for d in directions if d == main_dir)
            
            # Base confidence (0.7 to 0.95)
            base_confidence = 0.7 + (agree_count * 0.03)
            
            # Adjust for ADX (stronger trend = higher confidence)
            adx = indicators.get('adx', 0)
            adx_factor = min(1.0, adx / 40)  # Scale with ADX
            confidence = min(0.95, base_confidence * (1 + 0.2 * adx_factor))
            
            return round(confidence, 2)
            
        except Exception as e:
            logger.error(f"Error in _calculate_confidence: {e}")
            return 0.7  # Default confidence
    def _calculate_win_probability(self, indicators: Dict) -> float:
        """Calculate win probability based on indicators"""
        try:
            # Base probability
            prob = 0.65
            
            # Increase probability with ADX (strong trend)
            if indicators.get('adx', 0) > 40:
                prob += 0.15
                
            # Increase if RSI is not overbought/oversold
            rsi = indicators.get('rsi_value', 50)
            if 30 < rsi < 70:
                prob += 0.1
                
            return min(0.9, max(0.5, prob))
        except Exception as e:
            logger.error(f"Error in _calculate_win_probability: {e}")
            return 0.65

    async def _generate_signal(self, symbol: str, timeframe: str, df: pd.DataFrame, market_quality: Dict) -> Optional[Dict]:
        """Generate trading signal with enhanced error handling"""
        try:
            if df.empty or len(df) < 100:  # Minimum 100 candles required
                return None

            # Get current price
            current_price = await self.get_current_price(symbol)
            if not current_price:
                return None

            # Calculate indicators
            indicators = self._calculate_indicators(df, current_price)
            
            # Get direction (requires 6/8 indicators to agree)
            direction = self._get_direction(indicators)
            if direction == 'NEUTRAL':
                return None

            # Calculate TP/SL levels
            tp_levels, sl, tp1_pct, sl_pct = self._calculate_levels(direction, current_price, df)

            # Validate levels
            if not self._validate_levels(direction, current_price, tp_levels, sl):
                return None

            # Calculate confidence and win probability
            confidence = self._calculate_confidence(indicators)
            win_probability = self._calculate_win_probability(indicators)

            # Create signal
            signal = {
                'symbol': symbol,
                'direction': direction,
                'timeframe': timeframe,
                'confidence': confidence,
                'entry': current_price,
                'tp_levels': tp_levels,
                'tp1_percent': tp1_pct,
                'sl': sl,
                'sl_percent': sl_pct,
                'risk_reward': self._calculate_risk_reward(current_price, tp_levels[0], sl),
                'win_probability': win_probability,
                'leverage': 10,
                'indicators': indicators,
                'market_quality': market_quality,
                'timestamp': datetime.utcnow().isoformat()
            }

            return signal

        except Exception as e:
            logger.error(f"Error in _generate_signal for {symbol}: {e}")
            return None

    def _validate_levels(self, direction: str, entry: float, tp_levels: List[float], sl: float) -> bool:
        """Validate TP/SL levels"""
        try:
            if direction == "BULLISH":
                return (entry < tp_levels[0] < tp_levels[1] < tp_levels[2] and sl < entry)
            else:  # BEARISH
                return (entry > tp_levels[0] > tp_levels[1] > tp_levels[2] and sl > entry)
        except Exception as e:
            logger.error(f"Error in _validate_levels: {e}")
            return False

    def _calculate_risk_reward(self, entry: float, tp: float, sl: float) -> float:
        """Calculate risk/reward ratio"""
        try:
            if entry == sl:
                return 0
            return abs(tp - entry) / abs(entry - sl)
        except:
            return 0

    async def analyze_pair(self, symbol: str) -> List[Dict]:
        """Analyze a trading pair across all timeframes"""
        try:
            # Check cooldown
            current_time = time.time()
            if (symbol in self.last_signal_time and 
                (current_time - self.last_signal_time[symbol]) < self.cooldown_period):
                return []

            # Check market quality
            market_quality = await self.get_market_quality(symbol)
            if not market_quality['is_tradable']:
                return []

            signals = []
            directions = []
            valid_timeframes = 0

            # Check all timeframes
            for tf in self.timeframes:
                df = await self.get_historical_data(symbol, tf, limit=1000)
                if df.empty or len(df) < 100:
                    continue
                    
                signal = await self._generate_signal(symbol, tf, df, market_quality)
                if signal and signal['direction'] != 'NEUTRAL':
                    signals.append(signal)
                    directions.append(signal['direction'])
                    valid_timeframes += 1

            # Require all timeframes to agree
            if valid_timeframes == len(self.timeframes) and len(set(directions)) == 1:
                # Get the signal with highest confidence
                best_signal = max(signals, key=lambda s: s['confidence'])
                
                # Additional validation
                if (best_signal['confidence'] >= 0.75 and 
                    best_signal['risk_reward'] >= 1.5 and
                    best_signal['win_probability'] >= 0.7):
                    
                    self.last_signal[symbol] = best_signal
                    self.last_signal_time[symbol] = current_time
                    return [best_signal]

            return []

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return []

    def format_signal(self, signal: Dict) -> str:
        """Format signal for Telegram"""
        try:
            # Convert UTC to PKT (UTC+5)
            utc_time = datetime.fromisoformat(signal['timestamp'].replace('Z', '+00:00'))
            pkt_time = utc_time.astimezone(pytz.timezone('Asia/Karachi'))
            time_str = pkt_time.strftime('Time: %Y-%m-%d %H:%M:%S PKT')
            
            # Format TP/SL levels
            tp_levels = signal.get('tp_levels', [signal['entry']] * 3)
            if len(tp_levels) < 3:
                tp_levels.extend([tp_levels[-1]] * (3 - len(tp_levels)))
                
            # Format indicators
            indicators = signal.get('indicators', {})
            
            message = f"""⏰ {time_str}
📊 Pair: {signal['symbol']}
📈 Direction: {signal['direction']}
🕒 Timeframe: {signal['timeframe']}
🔍 Confidence: {signal['confidence']:.0%}
💰 Entry: ${signal['entry']:.8f}

🎯 Take Profits:
TP1: ${tp_levels[0]:.8f} ({signal.get('tp1_percent', 0):.2f}%)
TP2: ${tp_levels[1]:.8f}
TP3: ${tp_levels[2]:.8f}

🛑 Stop Loss: ${signal['sl']:.8f} ({signal.get('sl_percent', 0):.2f}%)
📊 Risk/Reward: {signal.get('risk_reward', 0):.2f}
🎯 Win Probability: {signal.get('win_probability', 0):.0%}

📊 Technical Indicators:
- EMA: {indicators.get('ema', 'N/A')}
- MACD: {indicators.get('macd', 'N/A')}
- RSI: {indicators.get('rsi', 'N/A')} ({indicators.get('rsi_value', 0):.2f})
- BB: {indicators.get('bb', 'N/A')}
- ADX: {indicators.get('adx', 0):.2f}
- FIB: {indicators.get('fib', 'N/A')}
- Stoch RSI: {indicators.get('stoch_rsi', 'N/A')}
- SuperTrend: {indicators.get('supertrend', 'N/A')}
- VWAP: {indicators.get('vwap', 'N/A')}

💹 Market Quality:
- Spread: {signal['market_quality'].get('spread', 0)*100:.4f}%
- 24h Volume: ${signal['market_quality'].get('volume_24h', 0):,.2f}
- Best Bid: {signal['market_quality'].get('best_bid', 0):.8f}
- Best Ask: {signal['market_quality'].get('best_ask', 0):.8f}

#Crypto #TradingSignal #{signal['direction']}"""
            
            return message
            
        except Exception as e:
            logger.error(f"Error formatting signal: {e}")
            return f"Signal formatting error: {e}"

    async def scan_market(self):
        """Main market scanning loop"""
        try:
            while True:
                try:
                    logger.info("Starting market scan...")
                    
                    # Get available symbols
                    symbols = await self.get_available_symbols()
                    if not symbols:
                        logger.warning("No symbols available for trading")
                        await asyncio.sleep(60)
                        continue
                    
                    # Process each symbol
                    for symbol in symbols:
                        try:
                            signals = await self.analyze_pair(symbol)
                            for signal in signals:
                                if await self.validate_signal(signal):
                                    message = self.format_signal(signal)
                                    await self._execute_signal(signal, message)
                                    await asyncio.sleep(1)  # Rate limiting
                        except Exception as e:
                            logger.error(f"Error processing {symbol}: {e}")
                            
                    logger.info(f"Market scan completed. Waiting {self.config.SCAN_INTERVAL} seconds...")
                    await asyncio.sleep(self.config.SCAN_INTERVAL)
                    
                except Exception as e:
                    logger.error(f"Error in scan loop: {e}")
                    await asyncio.sleep(60)  # Wait longer on error
                    
        except asyncio.CancelledError:
            logger.info("Market scan cancelled")
        except Exception as e:
            logger.error(f"Fatal error in market scan: {e}")
        finally:
            await self.cleanup()

    async def get_available_symbols(self) -> List[str]:
        """Get list of available trading symbols"""
        try:
            await self.binance_client.load_markets()
            return [s for s in self.binance_client.symbols 
                   if s.endswith('/USDT') 
                   and self.binance_client.markets[s].get('active')]
        except Exception as e:
            logger.error(f"Error getting symbols: {e}")
            return self.major_pairs

    async def _execute_signal(self, signal: Dict, message: str):
        """Execute a trading signal"""
        try:
            # Send to Telegram
            success = await self.telegram.send_message(message)
            if not success:
                logger.error("Failed to send Telegram message")
                return
                
            # Open position in portfolio
            await self.portfolio.open_position(
                symbol=signal['symbol'],
                direction=signal['direction'],
                entry_price=signal['entry'],
                leverage=signal.get('leverage', 10),
                stop_loss=signal['sl'],
                take_profit=signal['tp_levels']
            )
            
            logger.info(f"Executed signal for {signal['symbol']}")
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")

    async def validate_signal(self, signal: Dict) -> bool:
        """Final validation before sending signal"""
        try:
            # Check daily limit
            today = datetime.utcnow().date()
            daily_key = f"{signal['symbol']}_{today}"
            if self.daily_signals[daily_key] >= 2:  # Max 2 signals per day
                return False
                
            # Get current market conditions
            market_quality = await self.get_market_quality(signal['symbol'])
            if not market_quality['is_tradable']:
                return False
                
            # Check if price is still valid (within 0.2% of signal price)
            current_price = await self.get_current_price(signal['symbol'])
            if not current_price:
                return False
                
            price_diff = abs(current_price - signal['entry']) / signal['entry']
            if price_diff > 0.002:  # 0.2% threshold
                return False
                
            # Check if spread is still acceptable
            if market_quality['spread'] > self.max_spread:
                return False
                
            # All checks passed
            self.daily_signals[daily_key] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False

    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.binance_client:
                await self.binance_client.close()
        except:
            pass

    async def run(self):
        """Main entry point"""
        logger.info("Starting trading bot...")
        try:
            # Initialize
            if not await self.initialize():
                logger.error("Failed to initialize bot")
                return
                
            logger.info("Bot initialized successfully")
            
            # Start market scanner
            scanner = asyncio.create_task(self.scan_market())
            
            # Keep the bot running
            await asyncio.Event().wait()
            
        except asyncio.CancelledError:
            logger.info("Shutting down...")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
        finally:
            await self.cleanup()

# Example usage
if __name__ == "__main__":
    import asyncio
    from config import Config
    
    async def main():
        config = Config()  # Your config class
        bot = SignalGenerator(config)
        try:
            await bot.run()
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Fatal error: {e}")

    asyncio.run(main())
