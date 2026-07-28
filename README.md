# TDS Project 1 — Data Analyst Telegram Bot

An LLM agent (Claude, with `run_python` and `fetch_url` tools) that answers
data-analysis questions sent over Telegram and replies with a single JSON object.

## Files
- `bot.py` — Telegram polling loop, per-chat history, sends replies
- `agent.py` — Claude tool-use loop, extracts final JSON, writes `run.jsonl`
- `tools.py` — `fetch_url` and `run_python` tool implementations
- `requirements.txt`, `.env.example`

## 1. Local setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env: TELEGRAM_BOT_TOKEN, ANTHROPIC_API_KEY, PUBLIC_LOG_URL
```

`PUBLIC_LOG_URL` should point at where `run.jsonl` will be publicly reachable —
easiest option is the raw GitHub URL once you push it:
`https://raw.githubusercontent.com/<user>/<repo>/main/run.jsonl`

### Using Google Cloud Vertex AI instead of an Anthropic API key
If you'd rather spend your $300 GCP trial credit than pay for an Anthropic key:

1. In Google Cloud Console, enable the **Vertex AI API** for your project.
2. Go to **Vertex AI → Model Garden → search "Claude"** and request/enable access
   to the Claude model you want (e.g. Sonnet). This can take a few minutes to
   a few hours to approve.
3. Authenticate locally with:
   ```bash
   gcloud auth application-default login
   ```
   (or use a service account key + `GOOGLE_APPLICATION_CREDENTIALS` when deployed).
4. In `.env`, set:
   ```
   USE_VERTEX=1
   GOOGLE_CLOUD_PROJECT=your-gcp-project-id
   GOOGLE_CLOUD_REGION=global
   ```
   and leave `ANTHROPIC_API_KEY` blank — no Anthropic key is needed at all.
5. If a region-specific 404 shows up, the model isn't enabled in `global`/that
   region for your project yet — check the exact model ID shown on its Model
   Garden page and set `GOOGLE_CLOUD_REGION` to a supported region (e.g. `us-east5`).

Note the $300 credit covers *all* GCP usage, not just Vertex AI — keep an eye on
the billing dashboard if you're running other GCP services too.

## 2. Run locally

```bash
python3 bot.py
```

Message your bot on Telegram to test it end-to-end before deploying.

## 3. Keep the log public
After each test run, commit and push the updated `run.jsonl` so the
`PUBLIC_LOG_URL` stays current:

```bash
git add run.jsonl && git commit -m "update run log" && git push
```

(For continuous grading you may want a small cron/loop that auto-commits
`run.jsonl` every few minutes instead of doing it manually — optional.)

## 4. Deploy (Railway example)
1. Push this repo to GitHub (public).
2. Create a new Railway project → "Deploy from GitHub repo" → select this repo.
3. Set it as a **worker** (not a web service) with start command:
   ```
   python3 bot.py
   ```
4. Add environment variables in Railway's dashboard: `TELEGRAM_BOT_TOKEN`,
   `ANTHROPIC_API_KEY`, `PUBLIC_LOG_URL`, `LOG_FILE_PATH`.
5. Deploy, check logs to confirm "Bot starting (polling)..." appears with no errors.
6. Message the bot from Telegram again to confirm it replies from the deployed instance.

(Render works the same way — "Background Worker" service type instead of "Web Service".)

## 5. Test against the official grading pipeline

```bash
git clone https://github.com/Jivraj-18/tds-p1-t2-2026-telegram-bot
cd tds-p1-t2-2026-telegram-bot
# add sample questions to evals/questions.json
# point it at your bot's username and run per its own instructions
```

## Notes / known limitations to fix if time allows
- `run_python` executes in a plain subprocess with a timeout — fine for this
  assignment, not hardened sandboxing.
- Binary files (`.xlsx`, `.zip`) should be downloaded/parsed via `run_python`
  (`requests` + `pandas.read_excel`), not `fetch_url`.
- Chat history is in-memory only — if the bot restarts, per-chat context resets
  (fine for grading since each question is mostly self-contained).
