#!/usr/bin/env python3
# main.py - Binance Trading Bot with Fibonacci Strategy + FastAPI healthcheck

import asyncio
import logging
import signal
import sys
import traceback
from typing import Dict, List, Optional

import pandas as pd
import ccxt.async_support as ccxt
from datetime import datetime

from config import Config
from signal_generator import SignalGenerator
from Technical_analysis import TechnicalAnalysis
from Telegram_client import Telegram
from portfolio import Portfolio

# ---- FastAPI imports ----
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "Bot is running!"}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self, config: Config):
        self.config = config
        self.running = False
        self.setup_signal_handlers()

        # Initialize components
        self.ta = TechnicalAnalysis()
        self.telegram = Telegram(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)

        # Initialize exchange
        self.exchange = self.initialize_exchange()

        # Initialize trading components
        self.portfolio = Portfolio(self.exchange, config)
        self.signal_generator = SignalGenerator(config)

        # Track state
        self.active_trades: Dict[str, Dict] = {}
        self.completed_trades: List[Dict] = []

    def initialize_exchange(self):
        """Initialize and return the exchange instance"""
        exchange = ccxt.binance({
            'apiKey': self.config.BINANCE_API_KEY,
            'secret': self.config.BINANCE_API_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',  # or 'spot' for spot trading
                'adjustForTimeDifference': True,
            }
        })
        return exchange

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)

    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Shutdown signal received...")
        self.running = False

    async def initialize(self):
        """Initialize the trading bot"""
        try:
            # Test exchange connection
            await self.exchange.load_markets()
            logger.info(f"Connected to {self.exchange.id} exchange")

            # Initialize portfolio
            await self.portfolio.initialize()

            # Test Telegram connection
            await self.telegram.connect()
            if await self.telegram.send_message("🤖 Trading bot started successfully!"):
                logger.info("Telegram connection successful")
            else:
                logger.warning("Failed to send Telegram message")

            self.running = True
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    async def run(self):
        """Main trading loop"""
        try:
            if not await self.initialize():
                logger.error("Initialization failed. Exiting...")
                return

            logger.info("Starting trading bot...")

            while self.running:
                try:
                    # 1. Scan market for trading opportunities
                    signals = await self.signal_generator.scan_market()

                    # 2. Process signals
                    for signal in signals:
                        await self.process_signal(signal)

                    # 3. Monitor open positions
                    await self.monitor_positions()

                    # 4. Check risk limits
                    await self.portfolio.check_risk_limits()

                    # 5. Wait for next scan
                    logger.info(f"Waiting {self.config.SCAN_INTERVAL} seconds until next scan...")
                    await asyncio.sleep(self.config.SCAN_INTERVAL)

                except ccxt.NetworkError as e:
                    logger.error(f"Network error: {e}. Retrying in 60 seconds...")
                    await asyncio.sleep(60)
                except ccxt.ExchangeError as e:
                    logger.error(f"Exchange error: {e}")
                    await asyncio.sleep(60)
                except Exception as e:
                    logger.error(f"Unexpected error in main loop: {e}")
                    logger.error(traceback.format_exc())
                    await asyncio.sleep(60)

        except Exception as e:
            logger.error(f"Fatal error in run(): {e}")
            logger.error(traceback.format_exc())
        finally:
            await self.cleanup()

    async def scan_market(self) -> List[Dict]:
        """Scan the market for trading opportunities"""
        try:
            logger.info("Scanning market for trading opportunities...")

            # Get all available symbols
            symbols = await self.get_tradable_symbols()
            if not symbols:
                logger.warning("No tradable symbols found")
                return []

            # Filter out symbols with open positions
            symbols = [s for s in symbols if s not in self.active_trades]

            # Limit symbols to scan
            symbols = symbols[:self.config.MAX_PAIRS_PER_CYCLE]

            # Process symbols in batches
            batch_size = min(10, len(symbols))
            signals = []

            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                tasks = [self.analyze_symbol(symbol) for symbol in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, dict) and result:
                        signals.append(result)

                # Small delay between batches
                await asyncio.sleep(0.5)

            logger.info(f"Found {len(signals)} potential trading signals")
            return signals

        except Exception as e:
            logger.error(f"Error in scan_market: {e}")
            return []

    async def get_tradable_symbols(self) -> List[str]:
        """Get list of tradable symbols"""
        try:
            # Load markets if not already loaded
            await self.exchange.load_markets()

            # Filter for USDT pairs with sufficient volume
            symbols = []
            for symbol in self.exchange.symbols:
                market = self.exchange.markets[symbol]
                if (market['quote'] == 'USDT' and 
                    market['active'] and 
                    market['type'] == 'future'):  # For futures trading
                    symbols.append(symbol)

            # Get top volume symbols
            tickers = await self.exchange.fetch_tickers()
            symbol_volumes = []

            for symbol in symbols:
                if symbol in tickers:
                    ticker = tickers[symbol]
                    if ticker['quoteVolume'] and ticker['quoteVolume'] > self.config.MIN_VOLUME_USD:
                        symbol_volumes.append((symbol, ticker['quoteVolume']))

            # Sort by volume and take top symbols
            symbol_volumes.sort(key=lambda x: x[1], reverse=True)
            top_symbols = [s[0] for s in symbol_volumes[:50]]  # Top 50 by volume

            return top_symbols

        except Exception as e:
            logger.error(f"Error getting tradable symbols: {e}")
            return []

    async def analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """Analyze a single symbol for trading opportunities"""
        try:
            # Fetch OHLCV data
            timeframe = '15m'
            limit = self.config.MIN_CANDLES
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

            if not ohlcv or len(ohlcv) < 50:  # Need at least 50 candles
                return None

            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Generate trading signal
            signal = await self.signal_generator.generate_signal(symbol, df, timeframe)

            if signal and signal['confidence'] >= self.config.MIN_CONFIDENCE:
                return signal

            return None

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None

    async def process_signal(self, signal: Dict):
        """Process a trading signal"""
        try:
            symbol = signal['symbol']

            # Check if we already have an open position
            if symbol in self.active_trades:
                logger.info(f"Ignoring signal for {symbol}: Position already open")
                return

            # Validate signal
            if not self.validate_signal(signal):
                logger.info(f"Ignoring invalid signal for {symbol}")
                return

            # Calculate position size using portfolio manager
            position_size, quantity, risk_amount = await self.portfolio.calculate_position_size(
                signal['entry'], signal['stop_loss'], self.config.RISK_PER_TRADE
            )

            if position_size <= 0:
                logger.warning(f"Invalid position size for {symbol}")
                return

            # Execute trade (simulation for now - replace with actual trading)
            logger.info(f"Executing {signal['direction']} trade for {symbol}")

            # Format signal message
            message = self.format_signal_message(signal)

            # Send to Telegram
            try:
                await self.telegram.send_message(message)
            except Exception as e:
                logger.error(f"Failed to send Telegram message: {e}")

            # Record trade (simulation)
            self.active_trades[symbol] = {
                'entry_time': datetime.utcnow(),
                'entry_price': signal['entry'],
                'direction': signal['direction'],
                'size': quantity,
                'leverage': self.config.MAX_LEVERAGE,
                'take_profit': signal['tp_levels'],
                'stop_loss': signal['stop_loss'],
                'signal': signal
            }

            logger.info(f"Trade recorded: {symbol} {signal['direction']} at {signal['entry']}")

        except Exception as e:
            logger.error(f"Error processing signal: {e}")

    def validate_signal(self, signal: Dict) -> bool:
        """Validate a trading signal"""
        if not all(key in signal for key in ['symbol', 'direction', 'entry', 'tp_levels', 'stop_loss']):
            return False

        if signal['entry'] <= 0 or signal['stop_loss'] <= 0:
            return False

        if signal['direction'] not in ['LONG', 'SHORT']:
            return False

        if signal['confidence'] < self.config.MIN_CONFIDENCE:
            return False

        return True

    async def monitor_positions(self):
        """Monitor and manage open positions"""
        if not self.active_trades:
            return

        for symbol, position in list(self.active_trades.items()):
            try:
                # Get current market price
                ticker = await self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']

                # Check take profit levels
                for i, tp in enumerate(position['take_profit']):
                    if ((position['direction'] == 'LONG' and current_price >= tp) or
                        (position['direction'] == 'SHORT' and current_price <= tp)):
                        await self.close_position(symbol, f"TP{i+1} hit at {tp}")
                        break

                # Check stop loss
                if ((position['direction'] == 'LONG' and current_price <= position['stop_loss']) or
                    (position['direction'] == 'SHORT' and current_price >= position['stop_loss'])):
                    await self.close_position(symbol, f"Stop loss hit at {position['stop_loss']}")

            except Exception as e:
                logger.error(f"Error monitoring position {symbol}: {e}")

    async def close_position(self, symbol: str, reason: str):
        """Close an open position"""
        try:
            if symbol not in self.active_trades:
                return

            position = self.active_trades.pop(symbol)

            # Get current price
            ticker = await self.exchange.fetch_ticker(symbol)
            exit_price = ticker['last']

            # Calculate P&L
            entry_price = position['entry_price']
            if position['direction'] == 'LONG':
                pnl_pct = (exit_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - exit_price) / entry_price * 100

            # Record trade
            trade = {
                'symbol': symbol,
                'direction': position['direction'],
                'entry_time': position['entry_time'],
                'exit_time': datetime.utcnow(),
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'leverage': position['leverage'],
                'size': position['size'],
                'reason': reason
            }
            self.completed_trades.append(trade)

            # Send notification
            message = (
                f"📊 <b>CLOSED POSITION</b> 📊\n\n"
                f"🏷 <b>Pair:</b> {symbol}\n"
                f"📈 <b>Direction:</b> {position['direction']}\n"
                f"💰 <b>Entry:</b> ${entry_price:.4f}\n"
                f"🏁 <b>Exit:</b> ${exit_price:.4f}\n"
                f"📊 <b>P&L:</b> {pnl_pct:+.2f}%\n"
                f"📝 <b>Reason:</b> {reason}\n"
                f"⏱ <b>Duration:</b> {datetime.utcnow() - position['entry_time']}"
            )
            await self.telegram.send_message(message)

            logger.info(f"Closed {symbol} position: {reason}")

        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")

    def format_signal_message(self, signal: Dict) -> str:
        """Format a trading signal as a message"""
        try:
            tp_levels = signal.get('tp_levels', [])
            if len(tp_levels) < 3:
                tp_levels.extend([0] * (3 - len(tp_levels)))

            # Get Fibonacci levels
            fib = signal['indicators'].get('fibonacci', {}).get('levels', {})
            fib_levels = "\n".join([f"• {level}: ${price:,.2f}" for level, price in fib.items() if price > 0])

            return f"""
📊 <b>TRADING SIGNAL - FIBONACCI STRATEGY</b> 📊

🏷 <b>Pair:</b> {signal.get('symbol', 'N/A')}
📈 <b>Direction:</b> {signal.get('direction', 'N/A')}
⏰ <b>Timeframe:</b> {signal.get('timeframe', '15m')}
📊 <b>Confidence:</b> {signal.get('confidence', 0)*100:.1f}%

💰 <b>Entry:</b> ${signal.get('entry', 0):,.4f}
🎯 <b>TP1:</b> ${tp_levels[0]:,.4f} (0.618 Fib)
🎯 <b>TP2:</b> ${tp_levels[1]:,.4f} (0.5 Fib)
🎯 <b>TP3:</b> ${tp_levels[2]:,.4f} (0.382 Fib)
🛑 <b>Stop Loss:</b> ${signal.get('stop_loss', 0):,.4f}

📊 <b>Fibonacci Levels:</b>
{fib_levels}

⚡ <b>Leverage:</b> {self.config.MAX_LEVERAGE}x
📊 <b>Risk/Reward:</b> {signal.get('risk_reward_ratio', 0):.2f}
"""
        except Exception as e:
            logger.error(f"Error formatting signal message: {e}")
            return f"Signal for {signal.get('symbol', 'Unknown')}: {signal.get('direction', 'Unknown')}"

    async def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up...")
        try:
            # Close any remaining positions
            for symbol in list(self.active_trades.keys()):
                await self.close_position(symbol, "Bot shutting down")

            # Close connections
            await self.exchange.close()
            await self.telegram.close()
            await self.portfolio.close()

            logger.info("Cleanup complete")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

async def start_bot():
    try:
        config = Config()
        bot = TradingBot(config)
        await bot.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
    finally:
        if 'bot' in locals():
            await bot.cleanup()
        logger.info("Trading bot stopped")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    # Start the trading bot in the background
    loop.create_task(start_bot())
    # Start the FastAPI web server (port 10000 is Render default, but check your Render settings)
    uvicorn.run(app, host="0.0.0.0", port=10000)
