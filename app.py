import os
import threading
from flask import Flask, send_file
from bot import main

app = Flask(__name__)

@app.get("/")
def home():
    return "Bot is running"

@app.get("/run.jsonl")
def run_log():
    if os.path.exists("run.jsonl"):
        return send_file("run.jsonl", mimetype="application/json")
    return "run.jsonl not found", 404

# Start the Telegram bot in a background thread
threading.Thread(target=main, daemon=True).start()

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),
    )
