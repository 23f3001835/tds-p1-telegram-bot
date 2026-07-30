from flask import Flask
from threading import Thread
import asyncio

from bot import main

app = Flask(__name__)

@app.route("/")
def health():
    return "Bot is running"

def run_bot():
    asyncio.set_event_loop(asyncio.new_event_loop())
    main()

Thread(target=run_bot, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
