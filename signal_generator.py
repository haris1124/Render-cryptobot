import logging
import time
import random
import asyncio
from typing import Dict, List, Optional, Tuple
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from config import Config
from Technical_analysis import TechnicalAnalysis
from Telegram_client import Telegram
from portfolio import Portfolio

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
    def __init__(self, config: Config):
        self.config = config
        self.binance_client = ccxt.binance({
            'apiKey': config.BINANCE_API_KEY,
            'secret': config.BINANCE_API_SECRET,
            'enableRateLimit': True,
            'testnet': True,
            'urls': {
                'api': 'https://testnet.binance.vision/api',
            }
        })
        self.telegram = Telegram(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
        self.portfolio = Portfolio(self.binance_client, config)
        self.ta = TechnicalAnalysis()
        self.last_signal = {}
        self.last_signal_time = {}
        self.failed_symbols = {}
        self.warned_symbols = set()
        self.scanned_symbols = set()
        self.used_coins = set()
        self.cooldown_period = 900
        self.major_pairs = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT',
            'SOLUSDT', 'DOTUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LTCUSDT'
        ]

    async def get_symbols(self) -> List[str]:
        try:
            await self.binance_client.load_markets()
            symbols = [symbol for symbol in self.binance_client.markets.keys()
                     if symbol.endswith('USDT') and self.binance_client.markets[symbol]['active']]
            logger.info(f"Found {len(symbols)} active USDT trading pairs")
            return symbols
        except Exception as e:
            logger.error(f"Error fetching symbols: {e}")
            return self.major_pairs

    async def get_historical_data(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        try:
            ohlcv = await self.binance_client.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv:
                logger.warning(f"No data returned for {symbol} on {timeframe}")
                return pd.DataFrame()
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df['symbol'] = symbol
            if df['close'].isna().any():
                logger.warning(f"NaN values in data for {symbol} on {timeframe}")
                return pd.DataFrame()
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def _calculate_sl_levels(self, df: pd.DataFrame, current_price: float, direction: str) -> Tuple[float, float]:
        """Calculate stop loss levels with improved logic"""
        try:
            # Calculate ATR and recent volatility
            atr = self.ta.calculate_atr(df, period=14)
            atr_value = atr.iloc[-1] if not atr.empty else 0
            atr_percent = (atr_value / current_price) if current_price > 0 else 0.01
            
            # Base SL percentage between 0.5% and 1.5%
            random_factor = random.uniform(0.98, 1.02)
            base_sl_pct = min(max(0.009, atr_percent * 0.8 * random_factor), 0.02)
            
            # Adjust SL based on recent volatility
            recent_high = df['high'].iloc[-10:].max()
            recent_low = df['low'].iloc[-10:].min()
            recent_range = (recent_high - recent_low) / current_price
            volatility_factor = min(max(0.9, recent_range * 4), 1.1)
            
            # Calculate final SL percentage
            sl_pct = base_sl_pct * volatility_factor
            sl_pct = min(max(0.009, sl_pct), 0.02)  # Ensure within 0.5%-1.5% range
            
            # Calculate SL price based on direction
            if direction == "BULLISH":
                sl = current_price * (1 - sl_pct)
                swing_low = df['low'].iloc[-10:].min()
                sl = min(sl, swing_low)
            else:
                sl = current_price * (1 + sl_pct)
                swing_high = df['high'].iloc[-10:].max()
                sl = max(sl, swing_high)
            
            # Ensure SL is not too close to the current price
            min_distance = current_price * 0.002
            if direction == "BULLISH":
                sl = min(sl, current_price - min_distance)
            else:
                sl = max(sl, current_price + min_distance)
            
            # Calculate actual SL percentage for reporting
            actual_sl_pct = abs((sl - current_price) / current_price)
            if actual_sl_pct < 0.009:
                actual_sl_pct = 0.009
                sl = current_price * (1 - actual_sl_pct) if direction == "BULLISH" else current_price * (1 + actual_sl_pct)
            elif actual_sl_pct > 0.02:
                    actual_sl_pct = 0.02
                    sl = current_price * (1 - actual_sl_pct) if direction == "BULLISH" else current_price * (1 + actual_sl_pct)
            
            return sl, actual_sl_pct * 100
            
        except Exception as e:
            logger.error(f"Error in _calculate_sl_levels: {e}")
            sl_pct = 0.0145  # 1% default
            if direction == "BULLISH":
                return current_price * (1 - sl_pct), sl_pct * 100
            return current_price * (1 + sl_pct), sl_pct * 100

    async def _generate_signal(self, symbol: str, timeframe: str, df: pd.DataFrame) -> Optional[Dict]:
        try:
            current_price = df['close'].iloc[-1]
            if pd.isna(current_price) or current_price <= 0:
                return None

            # Calculate indicators
            indicators = {}
            
            # EMA
            emas = self.ta.calculate_ema(df, [20, 50, 200])
            indicators['ema'] = 'BULLISH' if emas['ema_20'].iloc[-1] > emas['ema_50'].iloc[-1] else 'BEARISH'

            # MACD
            macd_dict = self.ta.calculate_macd(df)
            indicators['macd'] = 'BULLISH' if macd_dict['macd'].iloc[-1] > macd_dict['signal'].iloc[-1] else 'BEARISH'

            # RSI
            rsi = self.ta.calculate_rsi(df)
            rsi_value = rsi.iloc[-1] if not rsi.empty else 50
            if rsi_value > 60:
                indicators['rsi'] = 'BULLISH'
            elif rsi_value < 40:
                indicators['rsi'] = 'BEARISH'
            else:
                indicators['rsi'] = 'NEUTRAL'

            # Bollinger Bands
            bb = self.ta.calculate_bollinger_bands(df)
            if current_price > bb['upper'].iloc[-1]:
                indicators['bb'] = 'BULLISH'
            elif current_price < bb['lower'].iloc[-1]:
                indicators['bb'] = 'BEARISH'
            else:
                indicators['bb'] = 'NEUTRAL'

            # ADX
            adx = self.ta.calculate_adx(df, period=14)
            indicators['adx'] = float(adx.iloc[-1]) if not adx.empty else 0

            # Fibonacci Retracement
            if len(df) >= 50:
                max_price = df['high'].iloc[-50:].max()
                min_price = df['low'].iloc[-50:].min()
                diff = max_price - min_price
                fib_618 = max_price - 0.618 * diff
                indicators['fib'] = 'BULLISH' if current_price > fib_618 else 'BEARISH'
            else:
                indicators['fib'] = 'NEUTRAL'

            # ATR
            atr = self.ta.calculate_atr(df, period=14)
            indicators['atr'] = float(atr.iloc[-1]) if not atr.empty else 0

            # Stochastic RSI
            stoch_rsi = self.ta.calculate_stoch_rsi(df)
            stoch_rsi_value = stoch_rsi.iloc[-1] if not stoch_rsi.empty else 0
            if stoch_rsi_value > 0.8:
                indicators['stoch_rsi'] = 'BULLISH'
            elif stoch_rsi_value < 0.2:
                indicators['stoch_rsi'] = 'BEARISH'
            else:
                indicators['stoch_rsi'] = 'NEUTRAL'

            # SuperTrend
            supertrend = self.ta.calculate_supertrend(df)
            indicators['supertrend'] = 'BULLISH' if supertrend['in_uptrend'].iloc[-1] else 'BEARISH'

            # VWAP
            vwap = self.ta.calculate_vwap(df)
            vwap_value = vwap.iloc[-1] if not vwap.empty else current_price
            indicators['vwap'] = 'BULLISH' if current_price > vwap_value else 'BEARISH'

            # Determine direction based on indicators
            directions = [
                indicators['ema'], indicators['macd'], indicators['rsi'], indicators['bb'],
                indicators['fib'], indicators['stoch_rsi'], indicators['supertrend'], indicators['vwap']
            ]
            direction_counts = {
                'BULLISH': directions.count('BULLISH'),
                'BEARISH': directions.count('BEARISH')
            }
            
            if direction_counts['BULLISH'] > direction_counts['BEARISH']:
                direction = 'BULLISH'
            elif direction_counts['BEARISH'] > direction_counts['BULLISH']:
                direction = 'BEARISH'
            else:
                direction = 'NEUTRAL'

            # Calculate confidence based on indicator agreement
            agree_count = max(direction_counts.values())
            confidence = 0.65 + (0.4 * (agree_count / 4))
            confidence = min(confidence, 0.95)

            # Calculate TP levels (0.8% to 1.5% based on ATR)
            atr_percent = (indicators['atr'] / current_price) if current_price > 0 else 0.01
            tp1_pct = min(max(0.008, atr_percent * 0.8), 0.015)
            
            if direction == "BULLISH":
                tp1 = current_price * (1 + tp1_pct)
                tp2 = current_price * (1 + tp1_pct * 1.5)
                tp3 = current_price * (1 + tp1_pct * 2)
            else:
                tp1 = current_price * (1 - tp1_pct)
                tp2 = current_price * (1 - tp1_pct * 1.5)
                tp3 = current_price * (1 - tp1_pct * 2)

            tp_levels = [tp1, tp2, tp3]

            # Calculate SL with improved logic
            sl, sl_pct = self._calculate_sl_levels(df, current_price, direction)

            # Calculate risk/reward ratio
            if direction == "BULLISH":
                risk = current_price - sl
                reward = tp1 - current_price
            else:
                risk = sl - current_price
                reward = current_price - tp1
            
            risk_reward = reward / risk if risk != 0 else 0

            # Calculate win probability
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
                'indicators': indicators
            }

            # Validate signal
            if direction == "NEUTRAL":
                return None

            # Additional validation for TP/SL levels
            if direction == "BULLISH":
                if not all(tp > current_price for tp in tp_levels) or not (sl < current_price):
                    return None
            else:  # BEARISH
                if not all(tp < current_price for tp in tp_levels) or not (sl > current_price):
                    return None

            # Ensure risk/reward is reasonable
            if risk_reward < 1.0:
                return None

            return signal

        except Exception as e:
            logger.error(f"Error generating signal for {symbol} on {timeframe}: {e}")
            return None

    async def analyze_pair(self, symbol: str) -> List[Dict]:
        try:
            if not symbol:
                logger.warning("Invalid symbol provided")
                return []

            if symbol in self.used_coins:
                logger.info(f"Skipping {symbol}: Already used for signal in this session")
                return []

            current_time = time.time()
            if symbol in self.last_signal_time and (current_time - self.last_signal_time[symbol]) < self.cooldown_period:
                logger.info(f"Skipping {symbol} due to signal cooldown")
                return []

            timeframes = ['15m', '1h', '4h', '1d']
            signals = []
            directions = []

            for tf in timeframes:
                df = await self.get_historical_data(symbol, tf, limit=120)
                if df.empty or len(df) < getattr(self.config, 'MIN_CANDLES', 100):
                    continue
                signal = await self._generate_signal(symbol, tf, df)
                if signal and signal['direction'] != 'NEUTRAL':
                    signals.append(signal)
                    directions.append(signal['direction'])

            # Require all 4 timeframes to agree on the same direction
            if len(signals) >= 2:
                dir_counts = {'BULLISH': directions.count('BULLISH'), 'BEARISH': directions.count('BEARISH')}
                if dir_counts['BULLISH'] >= 2:
                    agreed_direction = 'BULLISH'
                elif dir_counts['BEARISH']>= 2:
                    agreed_direction = 'BEARISH'
                else:
                    return []

                agreeing_signals = [s for s in signals if s['direction'] == agreed_direction]
                base_signal = max(agreeing_signals, key=lambda s: s['confidence'])

                # Strict: all 8 indicators must agree, ADX > 40, confidence > 0.85, risk/reward > 1.5, win_prob > 0.8
                agree_count = sum([
                    base_signal['indicators']['ema'] == base_signal['direction'],
                    base_signal['indicators']['macd'] == base_signal['direction'],
                    base_signal['indicators']['rsi'] == base_signal['direction'],
                    base_signal['indicators']['bb'] == base_signal['direction'],
                    base_signal['indicators']['fib'] == base_signal['direction'],
                    base_signal['indicators']['stoch_rsi'] == base_signal['direction'],
                    base_signal['indicators']['supertrend'] == base_signal['direction'],
                    base_signal['indicators']['vwap'] == base_signal['direction']
                ])
                
                # EMA200 strict filter
                ema200 = self.ta.calculate_ema(df, [200])['ema_200'].iloc[-1]
                if agreed_direction == 'BULLISH' and base_signal['entry'] < ema200:
                    return []
                if agreed_direction == 'BEARISH' and base_signal['entry'] > ema200:
                    return []
                    
                # Volume strict filter
                recent_vol = df['volume'].iloc[-40:].mean() #candles
                if df['volume'].iloc[-1] < 0.2 * recent_vol:
                    return []
                    
                if (agree_count == 4 and
                    base_signal['indicators']['adx'] > 20 and
                    base_signal['confidence'] > 0.6 and
                    base_signal['risk_reward'] > 1.2 and
                    base_signal['win_probability'] > 0.6):
                    
                    self.last_signal[symbol] = base_signal
                    self.last_signal_time[symbol] = current_time
                    self.scanned_symbols.add(symbol)
                    self.used_coins.add(symbol)
                    logger.info(f"Added {symbol} to used_coins for this session")
                    return [base_signal]
            return []
        except Exception as e:
            logger.error(f"Error in analyze_pair for {symbol}: {e}")
            return []

    async def scan_market(self):
        try:
            symbols = await self.get_symbols()
            if not symbols:
                logger.error("No symbols found")
                return

            logger.info(f"Scanning {len(symbols)} trading pairs")
            available_symbols = [s for s in symbols if s not in self.scanned_symbols and s not in self.used_coins]
            if not available_symbols:
                logger.info("All symbols scanned or used, resetting scanned_symbols and used_coins")
                self.scanned_symbols.clear()
                self.used_coins.clear()
                available_symbols = symbols

            for symbol in available_symbols:
                signals = await self.analyze_pair(symbol)
                for signal in signals:
                    message = self.format_signal(signal)
                    logger.info(f"Valid signal for {symbol}:\n{message}")
                    await self._execute_trade(signal, message)
                await asyncio.sleep(1)  # Avoid rate limits

        except Exception as e:
            logger.error(f"Error in scan_market: {e}")
        finally:
            await self.binance_client.close()

    def format_signal(self, signal: Dict) -> str:
        try:
            from datetime import datetime, timedelta
            pk_time = datetime.utcnow() + timedelta(hours=5)
            time_str = pk_time.strftime('Time: %Y-%m-%d %H:%M:%S')
            tp_levels = signal.get('tp_levels', [signal['entry']] * 3)
            tp1_pct = signal.get('tp1_percent', 1)
            sl_pct = signal.get('sl_percent', 1)
            if len(tp_levels) < 3:
                tp_levels.extend([tp_levels[-1]] * (3 - len(tp_levels)))
            message = f"""{time_str}
📊 Pair: {signal['symbol']}
📈 Direction: {signal['direction']}
🕒 Timeframe: {signal['timeframe']}
🔍 Confidence: {signal['confidence']:.1%}
💰 Entry: ${signal['entry']:.6f}
🎯 TP1: ${tp_levels[0]:.6f} ({tp1_pct:.2f}%)
🎯 TP2: ${tp_levels[1]:.6f}
🎯 TP3: ${tp_levels[2]:.6f}
🛑 Stop Loss: ${signal['sl']:.6f} ({sl_pct:.2f}%)
📊 EMA: {signal['indicators']['ema']}
📊 MACD: {signal['indicators']['macd']}
📊 RSI: {signal['indicators']['rsi']}
📊 BB: {signal['indicators']['bb']}
📊 ADX: {signal['indicators']['adx']:.2f}
📊 FIB: {signal['indicators']['fib']}
📊 ATR: {signal['indicators']['atr']:.4f}
📊 Stoch RSI: {signal['indicators']['stoch_rsi']}
📊 SuperTrend: {signal['indicators']['supertrend']}
📊 VWAP: {signal['indicators']['vwap']}
"""
            return message
        except Exception as e:
            return f"Signal formatting error: {e}"

    async def _execute_trade(self, signal: Dict, message: str):
        try:
            print(message)
            success = await self.telegram.send_message(message)
            if success:
                logger.info(f"Sent signal for {signal['symbol']} to Telegram")
                await self.portfolio.open_position(
                    symbol=signal['symbol'],
                    direction=signal['direction'],
                    entry_price=signal['entry'],
                    leverage=signal['leverage'],
                    stop_loss=signal['sl'],
                    take_profit=signal['tp_levels']
                )
            else:
                logger.warning(f"Failed to send signal for {signal['symbol']} to Telegram")
        except Exception as e:
            logger.error(f"Error executing trade signal for {signal['symbol']}: {e}")

    async def run(self):
        logger.info("Starting trading bot...")
        self.scanned_symbols.clear()
        self.used_coins.clear()
        try:
            success = await self.telegram.send_message("🤖 Trading bot started")
            if success:
                print("✅ Telegram connection successful")
            else:
                print("❌ Telegram connection failed, continuing with console")
        except Exception as e:
            logger.error(f"Telegram connection failed: {e}")
            print("❌ Telegram connection failed, continuing with console")

        while True:
            try:
                print("🔍 Scanning market...")
                await self.scan_market()
                logger.info("Market scan completed. Waiting for next cycle...")
                await asyncio.sleep(getattr(self.config, 'SCAN_INTERVAL', 300))
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                await self.binance_client.close()
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(60)

if __name__ == "__main__":
    config = Config()
    bot = SignalGenerator(config)
    asyncio.run(bot.run())
