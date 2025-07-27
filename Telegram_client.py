import logging
import aiohttp
from typing import Dict

logger = logging.getLogger(__name__)

class Telegram:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.session = None

    async def connect(self):
        """Initialize aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
            logger.info("Telegram aiohttp session started.")

    async def close(self):
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("Telegram aiohttp session closed.")

    async def send_message(self, message: str) -> bool:
        try:
            if not self.session or self.session.closed:
                await self.connect()
            url = f"{self.base_url}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            # Remove any bools from payload (Telegram only accepts str/int/float)
            for k, v in payload.items():
                if isinstance(v, bool):
                    payload[k] = str(v)
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    logger.info("Message sent successfully to Telegram")
                    return True
                else:
                    logger.error(f"Failed to send message to Telegram: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Error sending message to Telegram: {e}")
            return False

    async def send_signal(self, signal_data: dict) -> bool:
        try:
            message = self.format_signal_message(signal_data)
            return await self.send_message(message)
        except Exception as e:
            logger.error(f"Error sending signal to Telegram: {e}")
            return False

    def format_signal_message(self, signal: Dict) -> str:
        try:
            tp_levels = signal.get('tp_levels', [])
            if len(tp_levels) < 3:
                tp_levels.extend([signal.get('entry', 0)] * (3 - len(tp_levels)))

            # Convert all bool values to 'Yes'/'No' for display
            def bool_to_str(val):
                if isinstance(val, bool):
                    return 'Yes' if val else 'No'
                return val

            message = f"""
🚀 <b>TRADING SIGNAL</b> 🚀

📊 <b>Pair:</b> {signal.get('symbol', 'Unknown')}
📈 <b>Direction:</b> {signal.get('direction', 'NEUTRAL')}
🕒 <b>Timeframe:</b> {signal.get('timeframe', '15m')}
🔍 <b>Confidence:</b> {signal.get('confidence', 0):.1%}

💰 <b>Entry:</b> ${signal.get('entry', 0):.4f}
🎯 <b>TP1:</b> ${tp_levels[0] if len(tp_levels) > 0 else 0:.4f}
🎯 <b>TP2:</b> ${tp_levels[1] if len(tp_levels) > 1 else 0:.4f}
🎯 <b>TP3:</b> ${tp_levels[2] if len(tp_levels) > 2 else 0:.4f}
🛑 <b>Stop Loss:</b> ${signal.get('sl', 0):.4f}

⚖️ <b>Risk/Reward:</b> {signal.get('risk_reward', 0):.2f}
🎯 <b>Win Probability:</b> {signal.get('win_probability', 0):.1%}
📈 <b>Leverage:</b> {signal.get('leverage', 1)}x
"""
            # Replace any bools in the formatted message
            for k, v in signal.items():
                if isinstance(v, bool):
                    message = message.replace(str(v), bool_to_str(v))
            return message
        except Exception as e:
            logger.error(f"Error formatting signal message: {e}")
            return f"Signal for {signal.get('symbol', 'Unknown')}: {signal.get('direction', 'Unknown')}"