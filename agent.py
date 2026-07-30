import json
import os
import re
from datetime import datetime, timezone

from tools import TOOL_DEFINITIONS, TOOL_FUNCTIONS

MODEL = "claude-sonnet-4-6"
MAX_TOOL_ITERATIONS = 30

# Which backend to use — set exactly one of these in .env:
# PROVIDER=vertex    -> Google Cloud Vertex AI (needs GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_REGION)
# PROVIDER=anthropic -> direct Anthropic API (needs ANTHROPIC_API_KEY)
# PROVIDER=aipipe    -> proxy for IIT Madras students (needs AIPIPE_TOKEN)
# PROVIDER=gemini    -> free Google AI Studio key (needs GEMINI_API_KEY)
# PROVIDER=openai    -> direct OpenAI API (needs OPENAI_API_KEY)
PROVIDER = os.environ.get(
    "PROVIDER", "vertex" if os.environ.get("USE_VERTEX") == "1" else "anthropic"
)

SYSTEM_PROMPT = """You are a data-analysis agent answering questions sent over Telegram.

Rules:
- The user's message tells you EXACTLY what JSON shape to reply with. Read it carefully.
- NEVER guess or make up a URL. If you don't already know the exact URL of a
  dataset, use search_web first to find it, then fetch_url or run_python on a
  URL that actually appeared in the search results.
- When a PDF/page has multiple tables or sections, NEVER pick one just by
  position/index (e.g. "the 5th table"). ALWAYS verify the table's actual
  heading, caption, or surrounding text explicitly names the metric the
  question is asking about (e.g. confirm it says "Maternal Mortality Ratio",
  not a different metric like "Under-5 Mortality Rate" or "Infant Mortality
  Rate" that happens to have similar-looking state/number rows). If a table's
  header text is garbled by OCR (e.g. jumbled random-looking letters), print
  and read the surrounding page text/section title to confirm what the table
  actually represents before trusting its numbers as the answer.
- Prefer locating data via extract_text() and searching for the exact
  section heading (e.g. "Maternal Mortality Ratio") over blindly indexing
  extract_tables() results by position — text search lets you confirm you
  found the right section; a table index number does not.
- Pay close attention to which COUNTRY/REGION the question is actually about.
  If the question names "MOSPI", that is the Ministry of Statistics and
  Programme Implementation of the GOVERNMENT OF INDIA — your answer MUST be
  an Indian state or union territory, NEVER a US state, UK region, or any
  other country. This is an absolute rule, not a judgment call.
- When searching for MOSPI-related questions, always include "India" in your
  search_web query (e.g. "MOSPI India maternal mortality by state"), and
  COMPLETELY IGNORE/DISCARD any search result about the United States, US
  states, CDC, US Census Bureau, or any other country — even if it appears
  as the first or most detailed search result. Only use results that
  reference India, an Indian ministry/government source, or Indian state
  names (e.g. Uttar Pradesh, Bihar, Assam, Tamil Nadu, Kerala, etc.).
- Use the run_python and fetch_url tools to actually fetch and compute answers from
  real data (e.g. MOSPI or other public datasets) — never guess or make up numbers.
- run_python is STATELESS — each call is a brand-new Python process. NOTHING
  carries over from a previous run_python call: not variables, not
  downloaded PDFs, not parsed tables, not imports beyond the ones already
  preloaded. Every single run_python call must download/fetch/parse
  everything it needs from scratch, in that same code block, ending with
  print(). If you find yourself referencing a variable you defined in an
  earlier tool call, stop — rewrite the whole thing as one self-contained
  snippet instead.
- When extracting data from a PDF, table, or messy text: FIRST print() a small
  sample of the raw extracted content (or use pdfplumber's extract_tables())
  to see its real structure, THEN write targeted parsing logic based on what
  you actually see. Do not write a generic regex/parsing attempt blind and
  assume it's right. Prefer extract_tables() over regex on raw extract_text()
  output whenever a real table is present — regex on raw text is fragile and
  tends to match unrelated numbers (dates, page numbers, footnotes).
 - WATCH FOR CHARACTER-DOUBLING PDF ARTIFACTS: pdfplumber sometimes extracts
  bold or emphasized text as every character duplicated (e.g. "Tamil Nadu"
  becomes "TTaammiill NNaadduu", and critically the NUMBERS in that same row
  get doubled too — "79" becomes "7799", "134" becomes "113344"). If you fix
  a garbled state/column NAME by de-duplicating repeated characters, you
  MUST apply the exact same de-duplication to the NUMERIC values in that
  row — never assume only the name was corrupted. A simple fix: apply
  re.sub(r'(.)\\1', r'\\1', s) to the whole raw row string (not just the
  name) before splitting into columns.
- SANITY-CHECK EVERY EXTRACTED NUMBER against its neighbors in the same
  column/table before trusting it. If one row's value is wildly outside the
  range of every other row in that column (e.g. 10x-100x larger or smaller),
  that is a strong signal of a PDF extraction artifact, not a real outlier —
  re-extract that specific row (e.g. via extract_text and manual parsing,
  or de-duplication) rather than accepting it as the answer.
- If a run_python attempt produces a clearly wrong or nonsensical result
  (e.g. a date, a page number, or a word instead of a real state/number),
  do NOT repeat the same or a near-identical approach — change strategy
  entirely (e.g. print raw data first, try extract_tables instead of
  extract_text+regex, or look at a different page/source). Never retry an
  approach that already failed more than once in a row; if you notice
  yourself about to write nearly the same code again, stop and pick a
  genuinely different method instead.
- NEVER fabricate, mock, or invent data to stand in for a real dataset you
  couldn't fetch. If after real effort (search_web + fetch_url + run_python)
  you genuinely cannot access the real data, say so honestly in the "answer"
  field (e.g. {"answer": null, "error": "could not access source data"})
  rather than making up numbers — a fabricated answer is worse than admitting
  you don't know.
- If the message contains data inline (in the message text itself), use that directly.
- If it references a public dataset, search_web for the exact source page first,
  then fetch_url / run_python (with requests+pandas) to download and analyze it.
- Once you have the final answer, your LAST message must contain ONLY the single
  JSON object requested — no markdown fences, no explanation, no extra text before
  or after it. This exact text will be sent back to the user verbatim, so it must
  be valid, parseable JSON and nothing else.
- For the "log_url" field, just write the literal text "PLACEHOLDER" — it will be
  replaced automatically with the real log URL. Do not invent or guess a URL.
- If a multi-turn conversation is shown, answer only the most recent question, using
  earlier messages as context if relevant.
"""

FORCE_FINAL_PROMPT = (
    "You are out of tool-call turns. Based on everything found so far in "
    "this conversation, reply now with ONLY the final JSON object exactly "
    "as originally requested — your best answer from the data already "
    'gathered, or {"answer": null, "error": "could not determine answer"} '
    "only if truly nothing usable was found. No more tool calls, no "
    "markdown, no explanation — just the JSON object."
)


def _find_all_json_objects(text: str):
    """Find every balanced top-level {...} object in text."""
    objects = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "{":
            depth = 0
            start = i
            for j in range(i, n):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        objects.append(text[start : j + 1])
                        i = j
                        break
            else:
                break
        i += 1
    return objects


def _extract_json(text: str) -> str:
    """
    Pull the best candidate JSON object out of a text blob. Prefers an object
    that actually has an "answer" key (in case the model printed intermediate
    JSON-looking snippets before its real final answer); otherwise falls back
    to the last balanced object found, since models usually put the final
    answer last.
    """
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    text = text.strip()

    candidates = _find_all_json_objects(text)
    if not candidates:
        return text

    for c in candidates:
        try:
            parsed = json.loads(c)
            if isinstance(parsed, dict) and "answer" in parsed:
                return c
        except Exception:
            continue

    return candidates[-1]


def _finalize(final_text, conversation_history, tool_call_log, log_path, public_log_url):
    final_json = _extract_json(final_text or "{}")

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "conversation": conversation_history,
        "tool_calls": tool_call_log,
        "final_answer_raw": final_json,
    }
    try:
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass

    try:
        parsed = json.loads(final_json)
        if not isinstance(parsed, dict):
            parsed = {"answer": parsed}
        if "answer" not in parsed:
            parsed["answer"] = final_text
        # Always use the real configured log URL — never trust a placeholder
        # the model may have hallucinated into its JSON output.
        parsed["log_url"] = public_log_url
        return json.dumps(parsed)
    except Exception:
        return json.dumps({"answer": final_json, "log_url": public_log_url})


# ---------------------------------------------------------------------------
# Anthropic-format providers (direct Anthropic API or Vertex AI)
# ---------------------------------------------------------------------------

def _run_anthropic_style(conversation_history, log_path, public_log_url):
    if PROVIDER == "vertex":
        from anthropic import AnthropicVertex

        client = AnthropicVertex(
            project_id=os.environ["GOOGLE_CLOUD_PROJECT"],
            region=os.environ.get("GOOGLE_CLOUD_REGION", "us-east5"),
        )
    else:
        import anthropic

        client = anthropic.Anthropic()

    messages = [
        {"role": "user", "content": msg["content"]} for msg in conversation_history
    ]

    tool_call_log = []
    final_text = None

    for _ in range(MAX_TOOL_ITERATIONS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        assistant_content = []
        tool_results = []
        text_parts = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                assistant_content.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
                fn = TOOL_FUNCTIONS.get(block.name)
                result = fn(**block.input) if fn else f"[error] unknown tool {block.name}"
                tool_call_log.append(
                    {"tool": block.name, "input": block.input, "output": result[:2000]}
                )
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result}
                )

        messages.append({"role": "assistant", "content": assistant_content})

        if response.stop_reason == "tool_use":
            messages.append({"role": "user", "content": tool_results})
            continue
        else:
            final_text = "\n".join(text_parts)
            break

    if final_text is None:
        # Ran out of tool-call iterations without a final answer — force one
        # last no-tools call so we get a real (possibly null-but-honest) JSON
        # reply instead of silently falling back to "{}" -> answer: null.
        messages.append({"role": "user", "content": FORCE_FINAL_PROMPT})
        try:
            final_response = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
            final_text = "".join(
                b.text for b in final_response.content if b.type == "text"
            )
        except Exception as e:
            final_text = json.dumps(
                {"answer": None, "error": f"ran out of iterations: {e}"}
            )

    return _finalize(final_text, conversation_history, tool_call_log, log_path, public_log_url)


# ---------------------------------------------------------------------------
# AI Pipe / OpenRouter (OpenAI-compatible chat.completions + function calling)
# ---------------------------------------------------------------------------

OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
AIPIPE_BASE_URL = os.environ.get("AIPIPE_BASE_URL", "https://aipipe.org/openrouter/v1")

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_BASE_URL = "https://api.openai.com/v1"


def _to_openai_tools():
    tools = []
    for t in TOOL_DEFINITIONS:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
        )
    return tools


OPENAI_TOOLS = _to_openai_tools()


def _run_openai_compatible(conversation_history, log_path, public_log_url, base_url, api_key, model):
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=120,)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
        {"role": "user", "content": msg["content"]} for msg in conversation_history
    ]

    tool_call_log = []
    final_text = None

    for _ in range(MAX_TOOL_ITERATIONS):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=OPENAI_TOOLS,
        )
        choice = response.choices[0]
        msg = choice.message

        if msg.tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )
            for tc in msg.tool_calls:
                fn = TOOL_FUNCTIONS.get(tc.function.name)
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {}
                result = fn(**args) if fn else f"[error] unknown tool {tc.function.name}"
                tool_call_log.append(
                    {"tool": tc.function.name, "input": args, "output": result[:2000]}
                )
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )
            continue
        else:
            final_text = msg.content or ""
            break

    if final_text is None:
        # Ran out of tool-call iterations without a final answer — force one
        # last no-tools call so we get a real (possibly null-but-honest) JSON
        # reply instead of silently falling back to "{}" -> answer: null.
        messages.append({"role": "user", "content": FORCE_FINAL_PROMPT})
        try:
            final_response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            final_text = final_response.choices[0].message.content or ""
        except Exception as e:
            final_text = json.dumps(
                {"answer": None, "error": f"ran out of iterations: {e}"}
            )

    return _finalize(final_text, conversation_history, tool_call_log, log_path, public_log_url)


# ---------------------------------------------------------------------------

def run_agent(conversation_history, log_path, public_log_url):
    """
    conversation_history: list of {"role": "user", "content": str} for the chat,
      in order, ending with the message to answer.
    Returns the final JSON string to send back to Telegram.
    """
    if PROVIDER == "aipipe":
        return _run_openai_compatible(
            conversation_history,
            log_path,
            public_log_url,
            base_url=AIPIPE_BASE_URL,
            api_key=os.environ["AIPIPE_TOKEN"],
            model=OPENROUTER_MODEL,
        )
    elif PROVIDER == "gemini":
        return _run_openai_compatible(
            conversation_history,
            log_path,
            public_log_url,
            base_url=GEMINI_BASE_URL,
            api_key=os.environ["GEMINI_API_KEY"],
            model=GEMINI_MODEL,
        )
    elif PROVIDER == "openai":
        return _run_openai_compatible(
            conversation_history,
            log_path,
            public_log_url,
            base_url=OPENAI_BASE_URL,
            api_key=os.environ["OPENAI_API_KEY"],
            model=OPENAI_MODEL,
        )
    else:
        return _run_anthropic_style(conversation_history, log_path, public_log_url)
