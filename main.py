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

# Configure logging
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
    # ... [Your TradingBot class exactly as before] ...
    # (Paste your TradingBot class here unchanged)

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
