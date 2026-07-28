"""
Tools the LLM agent can call:
- search_web: search the web and return top result titles/URLs/snippets
- fetch_url: download a web page / CSV / file and return text (truncated)
- run_python: execute a python snippet (with pandas/requests/bs4 preloaded)
  and return stdout/stderr
"""

import os
import subprocess
import tempfile
import textwrap
import requests
from bs4 import BeautifulSoup

MAX_FETCH_CHARS = 20000
RUN_TIMEOUT_SECONDS = 30

PY_PRELUDE = """
import pandas as pd
import numpy as np
import requests
import json
import re
import warnings
from bs4 import BeautifulSoup
import pdfplumber
from io import BytesIO
warnings.filterwarnings("ignore")
# Tip: if a .gov.in URL raises an SSLError, retry the same request with
# verify=False (e.g. requests.get(url, verify=False)).
# Tip: to read a PDF from a URL:
#   resp = requests.get(pdf_url, verify=False)
#   with pdfplumber.open(BytesIO(resp.content)) as pdf:
#       for page in pdf.pages:
#           text = page.extract_text()
#           tables = page.extract_tables()
# IMPORTANT: prefer page.extract_tables() over regex on raw text — PDF text
# extraction often interleaves numbers from unrelated rows (dates, page
# numbers, footnotes), so a generic regex like state-name-then-number will
# match garbage. ALWAYS print() a sample of extract_text()/extract_tables()
# output first to see the actual structure before writing any parsing logic
# — never write a final extraction regex/loop on your first attempt without
# looking at the real data shape first.
"""


def search_web(query: str) -> str:
    """
    Search the web. Uses Tavily (a proper search API for LLM agents) if
    TAVILY_API_KEY is set — reliable and not rate-limited like scraping.
    Falls back to scraping DuckDuckGo's HTML if no Tavily key is configured
    or the Tavily call fails.
    """
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if tavily_key:
        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={"api_key": tavily_key, "query": query, "max_results": 8},
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if results:
                lines = []
                for r in results:
                    lines.append(
                        f"{r.get('title', '')}\n  URL: {r.get('url', '')}\n  {r.get('content', '')[:300]}"
                    )
                return "\n\n".join(lines)
        except Exception as e:
            # fall through to DuckDuckGo fallback below
            pass

    return _search_web_duckduckgo(query)


def _search_web_duckduckgo(query: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        resp = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            timeout=20,
            headers=headers,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for result in soup.select(".result")[:8]:
            link_tag = result.select_one(".result__a")
            snippet_tag = result.select_one(".result__snippet")
            if not link_tag:
                continue
            title = link_tag.get_text(strip=True)
            url = link_tag.get("href", "")
            snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
            results.append(f"{title}\n  URL: {url}\n  {snippet}")
        if results:
            return "\n\n".join(results)

        # Fallback: DuckDuckGo's lite endpoint (simpler markup, sometimes
        # works when the main HTML endpoint blocks the request)
        resp2 = requests.get(
            "https://lite.duckduckgo.com/lite/",
            params={"q": query},
            timeout=20,
            headers=headers,
        )
        resp2.raise_for_status()
        soup2 = BeautifulSoup(resp2.text, "html.parser")
        links = []
        for a in soup2.select("a.result-link")[:8]:
            links.append(f"{a.get_text(strip=True)}\n  URL: {a.get('href', '')}")
        if links:
            return "\n\n".join(links)

        return "[search_web] No results found."
    except Exception as e:
        return f"[search_web error] {type(e).__name__}: {e}"


def fetch_url(url: str) -> str:
    """Fetch a URL and return its text content (truncated to keep tokens sane)."""
    headers = {"User-Agent": "Mozilla/5.0 (data-analyst-agent)"}
    try:
        try:
            resp = requests.get(url, timeout=20, headers=headers)
        except requests.exceptions.SSLError:
            # Some government sites (.gov.in etc) have misconfigured cert
            # chains; retry without verification as a fallback.
            resp = requests.get(url, timeout=20, headers=headers, verify=False)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "text" in content_type or "json" in content_type or "csv" in content_type:
            text = resp.text
        else:
            # Binary (xlsx, etc.) - report size, agent should use run_python + requests instead
            return (
                f"[binary content, content-type={content_type}, "
                f"{len(resp.content)} bytes] Use run_python with requests/pandas "
                f"to download and parse this URL directly instead of fetch_url."
            )
        if len(text) > MAX_FETCH_CHARS:
            text = text[:MAX_FETCH_CHARS] + f"\n...[truncated, {len(resp.text)} total chars]"
        return text
    except Exception as e:
        return f"[fetch_url error] {type(e).__name__}: {e}"


def run_python(code: str) -> str:
    """
    Execute a python snippet in a subprocess with pandas/requests/bs4 available.
    The snippet should print() whatever it wants returned.
    """
    full_code = PY_PRELUDE + "\n" + textwrap.dedent(code)
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(full_code)
            path = f.name

        result = subprocess.run(
            ["python3", path],
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT_SECONDS,
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        if err:
            out += f"\n[stderr]\n{err}"
        if not out:
            out = "[no output — remember to print() the result]"
        # keep it bounded
        if len(out) > 8000:
            out = out[:8000] + "\n...[truncated]"
        return out
    except subprocess.TimeoutExpired:
        return f"[run_python error] Timed out after {RUN_TIMEOUT_SECONDS}s"
    except Exception as e:
        return f"[run_python error] {type(e).__name__}: {e}"


# Tool schema passed to the Anthropic API
TOOL_DEFINITIONS = [
    {
        "name": "search_web",
        "description": (
            "Search the web for a query and get back a list of result titles, "
            "URLs, and snippets. ALWAYS use this first when you don't already "
            "know the exact URL of a dataset or page — do not guess URLs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_url",
        "description": (
            "Fetch a public web page or small text/CSV/JSON file and return its "
            "text content. For binary files (xlsx, zip) use run_python with "
            "requests/pandas instead. Only use this on a URL you already know "
            "(e.g. from search_web results) — never guess a URL."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch"}
            },
            "required": ["url"],
        },
    },
    {
        "name": "run_python",
        "description": (
            "Execute a Python snippet to download and analyze data. "
            "pandas (pd), numpy (np), requests, json, re, BeautifulSoup, "
            "pdfplumber, and BytesIO are already imported — use pdfplumber "
            "to extract text/tables from PDF reports (a common format for "
            "government statistical publications like MOSPI/census reports). "
            "Always print() the value(s) you need back. "
            "Never use a made-up/placeholder URL in this code — only URLs "
            "confirmed real via search_web or fetch_url."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to run"}
            },
            "required": ["code"],
        },
    },
]

TOOL_FUNCTIONS = {
    "search_web": search_web,
    "fetch_url": fetch_url,
    "run_python": run_python,
}
