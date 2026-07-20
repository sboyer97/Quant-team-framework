"""Universe Agent: clarify a broad asset universe before data preparation."""

from __future__ import annotations

import json
import logging

from utils.llm import chat_json

logger = logging.getLogger(__name__)

QUESTION_PROMPT = """\
You are an asset-universe specialist helping configure a quantitative research
experiment. Given a broad universe request, ask only the questions whose
answers materially change the list of tradable instruments.

Examples of useful dimensions are asset count, geography, market-cap segment,
liquidity, exchange, quote currency, stablecoin inclusion, instrument type and
minimum history. Do not ask for information already present in the request.
Ask at most four short questions, all in English.

Respond as JSON:
{
  "questions": [
    {
      "id": "short_snake_case_id",
      "question": "Question shown in the terminal?",
      "default": "sensible default answer"
    }
  ]
}
"""

RESOLUTION_PROMPT = """\
You are an asset-universe specialist. Resolve the user's request and answers
into a concrete list of instruments available through Yahoo Finance.

Rules:
- Return between 2 and 50 liquid instruments.
- Use exact Yahoo Finance symbols: e.g. BTC-USD, MC.PA, AIR.DE, 7203.T.
- Respect exclusions and requested asset count.
- Do not include indices when the user requested tradable constituents.
- For crypto, default to USD spot pairs and exclude stablecoins unless asked.
- Never invent symbols. Prefer established, liquid instruments.

Respond as JSON:
{
  "universe_name": "short descriptive name",
  "rationale": "one sentence describing the applied filters",
  "tickers": ["SYMBOL1", "SYMBOL2"]
}
"""


def generate_universe_questions(request: str) -> list[dict]:
    """Return focused English clarification questions for a broad request."""
    logger.info("Universe Agent: clarifying '%s'", request)
    result = chat_json(QUESTION_PROMPT, f"Universe request: {request}", temperature=0.1)
    questions = result.get("questions", [])
    if not isinstance(questions, list):
        raise ValueError("Universe Agent returned invalid questions.")
    return [
        {
            "id": str(question.get("id", f"question_{index}")),
            "question": str(question.get("question", "Please clarify the universe.")),
            "default": str(question.get("default", "")),
        }
        for index, question in enumerate(questions[:4], start=1)
    ]


def resolve_universe_request(request: str, answers: dict[str, str]) -> dict:
    """Resolve a request and clarification answers to Yahoo tickers."""
    user_prompt = (
        f"Universe request: {request}\n\n"
        f"Clarification answers:\n{json.dumps(answers, indent=2)}"
    )
    result = chat_json(RESOLUTION_PROMPT, user_prompt, temperature=0.1)
    tickers = [
        str(ticker).strip().upper()
        for ticker in result.get("tickers", [])
        if str(ticker).strip()
    ]
    tickers = list(dict.fromkeys(tickers))
    if len(tickers) < 2:
        raise ValueError("Universe Agent must return at least two Yahoo Finance tickers.")
    if len(tickers) > 50:
        tickers = tickers[:50]
    return {
        "universe_name": str(result.get("universe_name", request)),
        "rationale": str(result.get("rationale", "")),
        "tickers": tickers,
    }
