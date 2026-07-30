import logging
import os
from collections import defaultdict

from dotenv import load_dotenv

load_dotenv()

from telegram import Update
from telegram.ext import Application, MessageHandler, ContextTypes, filters
from telegram.request import HTTPXRequest

from agent import run_agent

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
LOG_FILE_PATH = os.environ.get("LOG_FILE_PATH", "run.jsonl")
PUBLIC_LOG_URL = os.environ["PUBLIC_LOG_URL"]

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# in-memory per-chat conversation history: chat_id -> list of {"role","content"}
CHAT_HISTORY = defaultdict(list)
MAX_HISTORY_MESSAGES = 2  # keep last N messages of context per chat


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = update.message.text or ""
    if not text.strip():
        return

    logger.info("Message from chat %s: %s", chat_id, text[:200])

    history = CHAT_HISTORY[chat_id]
    history.append({"role": "user", "content": text})
    history = history[-MAX_HISTORY_MESSAGES:]
    CHAT_HISTORY[chat_id] = history

    try:
        reply_json = run_agent(
            conversation_history=history,
            log_path=LOG_FILE_PATH,
            public_log_url=PUBLIC_LOG_URL,
        )
    except Exception as e:
        logger.exception("Agent failed")
        reply_json = (
            '{"answer": null, "log_url": "%s", "error": "%s"}'
            % (PUBLIC_LOG_URL, str(e).replace('"', "'"))
        )

    await update.message.reply_text(reply_json)
    CHAT_HISTORY[chat_id] = []


def main():
    request = HTTPXRequest(
        connect_timeout=30.0,
        read_timeout=30.0,
        write_timeout=30.0,
        pool_timeout=30.0,
    )
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).request(request).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("Bot starting (polling)...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
