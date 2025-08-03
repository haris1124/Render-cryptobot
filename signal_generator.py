import logging
import time
import random
import asyncio
import json
import collections
from typing import Dict, List, Optional, Tuple
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pytz

# Import your existing modules
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
logger = logging.getLogger("signal_generator")

class SignalGenerator:
    def __init__(self, config: Config):
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
        self.telegram = Telegram(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
        self.portfolio = Portfolio(self.binance_client, config)
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
        self.timeframes = ['5m', '15m', '30m', '1h']
        self.major_pairs = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'XRP/USDT',
            'SOL/USDT', 'DOT/USDT', 'DOGE/USDT', 'AVAX/USDT', 'LTC/USDT'
        ]

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
            df['symbol'] = symbol
            
            if df['close'].isna().any() or df.empty:
                logger.warning(f"Invalid data in DataFrame for {symbol} on {timeframe}")
                return pd.DataFrame()
                
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    async def analyze_pair(self, symbol: str) -> List[Dict]:
        """Analyze a trading pair across multiple timeframes"""
        try:
            if not symbol:
                return []

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
            timeframes_checked = 0

            for tf in self.timeframes:
                df = await self.get_historical_data(symbol, tf, limit=120)
                if df.empty or len(df) < self.config.MIN_CANDLES:
                    continue
                    
                signal = await self._generate_signal(symbol, tf, df, market_quality)
                if signal and signal['direction'] != 'NEUTRAL':
                    signals.append(signal)
                    directions.append(signal['direction'])
                    timeframes_checked += 1

            # Require all timeframes to agree
            if timeframes_checked == len(self.timeframes) and len(set(directions)) == 1:
                direction = directions[0]
                base_signal = max(signals, key=lambda s: s['confidence'])
                
                # Final validation
                if await self.validate_signal(base_signal):
                    self.last_signal[symbol] = base_signal
                    self.last_signal_time[symbol] = current_time
                    self.used_coins.add(symbol)
                    return [base_signal]

            return []

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return []

    async def _generate_signal(self, symbol: str, timeframe: str, df: pd.DataFrame, 
                             market_quality: Dict) -> Optional[Dict]:
        """Generate trading signal with real-time validation"""
        try:
            # Get current price from order book
            current_price = await self.get_current_price(symbol)
            if not current_price:
                return None

            # Calculate indicators
            indicators = self._calculate_indicators(df, current_price)
            
            # Determine direction based on indicators
            direction = self._get_direction(indicators)
            if direction == 'NEUTRAL':
                return None

            # Calculate TP/SL levels
            tp_levels, sl, tp1_pct, sl_pct = self._calculate_levels(
                direction, current_price, df
            )

            # Validate levels
            if not self._validate_levels(direction, current_price, tp_levels, sl):
                return None

            # Calculate confidence and risk metrics
            confidence = self._calculate_confidence(indicators)
            risk_reward = self._calculate_risk_reward(current_price, tp_levels[0], sl)
            win_probability = self._calculate_win_probability(indicators)

            # Create signal
            signal = {
                'symbol': symbol,
                'direction': direction,
                'timeframe': timeframe,
                'confidence': confidence,
                'entry': current_price,
                'tp_levels': tp_levels,
                'tp1_percent': tp1_pct * 100,
                'sl': sl,
                'sl_percent': sl_pct * 100,
                'risk_reward': risk_reward,
                'win_probability': win_probability,
                'leverage': 10,
                'indicators': indicators,
                'market_quality': market_quality,
                'timestamp': datetime.utcnow().isoformat()
            }

            return signal

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None

    def _calculate_indicators(self, df: pd.DataFrame, current_price: float) -> Dict:
        """Calculate all technical indicators"""
        try:
            # EMA
            emas = self.ta.calculate_ema(df, [20, 50, 200])
            ema_dir = 'BULLISH' if emas['ema_20'].iloc[-1] > emas['ema_50'].iloc[-1] else 'BEARISH'

            # MACD
            macd = self.ta.calculate_macd(df)
            macd_dir = 'BULLISH' if macd['macd'].iloc[-1] > macd['signal'].iloc[-1] else 'BEARISH'

            # RSI
            rsi = self.ta.calculate_rsi(df)
            rsi_value = rsi.iloc[-1] if not rsi.empty else 50
            rsi_dir = 'BULLISH' if rsi_value > 60 else 'BEARISH' if rsi_value < 40 else 'NEUTRAL'

            # Bollinger Bands
            bb = self.ta.calculate_bollinger_bands(df)
            bb_dir = 'NEUTRAL'
            if not bb.empty:
                bb_dir = 'BULLISH' if current_price > bb['upper'].iloc[-1] else 'BEARISH' if current_price < bb['lower'].iloc[-1] else 'NEUTRAL'

            # ADX
            adx_value = 0
            try:
                adx = self.ta.calculate_adx(df)
                adx_value = float(adx.iloc[-1]) if not adx.empty else 0
            except:
                pass

            # Fibonacci
            fib_dir = self._calculate_fib_direction(df, current_price)

            # ATR for volatility
            atr_value = 0
            try:
                atr = self.ta.calculate_atr(df)
                atr_value = float(atr.iloc[-1]) if not atr.empty else 0
            except:
                pass

            # Stochastic RSI
            stoch_rsi_dir = 'NEUTRAL'
            try:
                stoch_rsi = self.ta.calculate_stoch_rsi(df)
                stoch_value = stoch_rsi.iloc[-1] if not stoch_rsi.empty else 0.5
                stoch_rsi_dir = 'BULLISH' if stoch_value > 0.8 else 'BEARISH' if stoch_value < 0.2 else 'NEUTRAL'
            except:
                pass

            # SuperTrend
            supertrend_dir = 'NEUTRAL'
            try:
                supertrend = self.ta.calculate_supertrend(df)
                if not supertrend.empty:
                    supertrend_dir = 'BULLISH' if supertrend['in_uptrend'].iloc[-1] else 'BEARISH'
            except:
                pass

            # VWAP
            vwap_dir = 'NEUTRAL'
            try:
                vwap = self.ta.calculate_vwap(df)
                if not vwap.empty:
                    vwap_dir = 'BULLISH' if current_price > vwap.iloc[-1] else 'BEARISH'
            except:
                pass

            return {
                'ema': ema_dir,
                'macd': macd_dir,
                'rsi': rsi_dir,
                'bb': bb_dir,
                'adx': adx_value,
                'fib': fib_dir,
                'atr': atr_value,
                'stoch_rsi': stoch_rsi_dir,
                'supertrend': supertrend_dir,
                'vwap': vwap_dir,
                'rsi_value': rsi_value,
                'atr_value': atr_value
            }

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}

    def _calculate_fib_direction(self, df: pd.DataFrame, current_price: float) -> str:
        """Calculate Fibonacci retracement direction"""
        if len(df) < 50:
            return 'NEUTRAL'
            
        max_price = df['high'].iloc[-50:].max()
        min_price = df['low'].iloc[-50:].min()
        diff = max_price - min_price
        
        if diff == 0:
            return 'NEUTRAL'
            
        fib_618 = max_price - 0.618 * diff
        return 'BULLISH' if current_price > fib_618 else 'BEARISH'

    def _get_direction(self, indicators: Dict) -> str:
        """Determine overall direction from indicators"""
        directions = [
            indicators['ema'], indicators['macd'], indicators['rsi'],
            indicators['bb'], indicators['fib'], indicators['stoch_rsi'],
            indicators['supertrend'], indicators['vwap']
        ]
        
        bull_count = directions.count('BULLISH')
        bear_count = directions.count('BEARISH')
        
        if bull_count >= 6:
            return 'BULLISH'
        elif bear_count >= 6:
            return 'BEARISH'
        return 'NEUTRAL'

    def _calculate_levels(self, direction: str, current_price: float, 
                         df: pd.DataFrame) -> Tuple[List[float], float, float, float]:
        """Calculate TP/SL levels with proper risk management"""
        # Calculate ATR-based TP1 (0.8% to 1.5%)
        atr_pct = min(max(0.008, indicators.get('atr_value', 0) / current_price), 0.015)
        tp1_pct = atr_pct
        
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

    def _validate_levels(self, direction: str, entry: float, 
                        tp_levels: List[float], sl: float) -> bool:
        """Validate that TP/SL levels make sense"""
        if direction == "BULLISH":
            if not (entry < tp_levels[0] < tp_levels[1] < tp_levels[2] and sl < entry):
                return False
        else:  # BEARISH
            if not (entry > tp_levels[0] > tp_levels[1] > tp_levels[2] and sl > entry):
                return False
        return True

    def _calculate_confidence(self, indicators: Dict) -> float:
        """Calculate signal confidence based on indicator agreement"""
        directions = [
            indicators['ema'], indicators['macd'], indicators['rsi'],
            indicators['bb'], indicators['fib'], indicators['stoch_rsi'],
            indicators['supertrend'], indicators['vwap']
        ]
        
        # Count agreements
        main_dir = 'BULLISH' if directions.count('BULLISH') > directions.count('BEARISH') else 'BEARISH'
        agree_count = sum(1 for d in directions if d == main_dir)
        
        # Base confidence on agreement count
        base_confidence = 0.7 + (agree_count * 0.05)  # 0.7 to 1.1
        
        # Adjust for ADX (stronger trend = higher confidence)
        adx_factor = min(1.0, indicators['adx'] / 40)  # Scale with ADX
        confidence = min(0.95, base_confidence * (1 + 0.2 * adx_factor))
        
        return round(confidence, 2)

    def _calculate_risk_reward(self, entry: float, tp: float, sl: float) -> float:
        """Calculate risk/reward ratio"""
        if entry == sl:
            return 0
        return abs(tp - entry) / abs(entry - sl)

    def _calculate_win_probability(self, indicators: Dict) -> float:
        """Calculate win probability based on indicators"""
        # Base probability
        prob = 0.65
        
        # Increase probability with ADX (strong trend)
        if indicators['adx'] > 40:
            prob += 0.15
            
        # Increase if RSI is not overbought/oversold
        if 30 < indicators['rsi_value'] < 70:
            prob += 0.1
            
        return min(0.9, max(0.5, prob))

    async def validate_signal(self, signal: Dict) -> bool:
        """Final validation before sending signal"""
        try:
            # Check if we've already sent too many signals for this pair today
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
- ATR: {indicators.get('atr_value', 0):.8f}
- Stoch RSI: {indicators.get('stoch_rsi', 'N/A')}
- SuperTrend: {indicators.get('supertrend', 'N/A')}
- VWAP: {indicators.get('vwap', 'N/A')}

💹 Market Quality:
- Spread: {market_quality.get('spread', 0)*100:.4f}%
- 24h Volume: ${market_quality.get('volume_24h', 0):,.2f}
- Best Bid: {market_quality.get('best_bid', 0):.8f}
- Best Ask: {market_quality.get('best_ask', 0):.8f}

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
                    symbols = await self.get_available_symbols()
                    
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

    async def cleanup(self):
        """Clean up resources"""
        try:
            await self.binance_client.close()
        except:
            pass

    async def run(self):
        """Main entry point"""
        logger.info("Starting trading bot...")
        try:
            # Initialize
            await self.binance_client.load_markets()
            
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
    config = Config()  # Your config class
    bot = SignalGenerator(config)
    
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
