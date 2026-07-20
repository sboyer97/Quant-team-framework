"""Research Agent.

Given a trading strategy idea, searches the web for relevant academic work,
then asks the LLM to distill a structured research summary that the
Implementation Agent can act on.
"""

from __future__ import annotations

import json
import logging
from urllib.parse import urlparse

import requests

from utils.llm import chat_json

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a quantitative research analyst at a systematic hedge fund.

Given a trading strategy idea, its selected universe, available dataset fields
and web search results, produce one rigorous, implementable research summary.
Be specific and quantitative: name concrete signals, lookback windows,
entry/exit thresholds and rebalancing frequency. Prefer parameter choices
supported by the literature; state defaults when the literature is silent.

Implementation constraints:
- The strategy receives an already filtered dataset. Do not ask the
  implementation to identify or validate universe membership.
- The implementation may use only pandas and numpy. Every requested signal
  must be computable from the explicitly available fields.
- Every value used for a date must be available on or before that date.
- Current-close signals are valid because execution is lagged by one day in
  the backtester. Do not require an additional signal shift.
- Do not require fields that are absent from the selected model type.
- Prefer stateless daily target signals that can be expressed with vectorized
  rolling calculations. Avoid requiring a persistent position state machine.
- With pandas/numpy-only pairs strategies, never require Engle-Granger, ADF,
  Johansen or any formal cointegration test. Describe the strategy as rolling
  spread mean reversion using rolling correlation, covariance/variance hedge
  ratios and z-scores.
- Prefer daily targets. If calendar resampling is essential, use pandas 3.x
  aliases such as ME/QE/YE, never the removed M/Q/Y aliases.
- The universe may contain hundreds of tickers; the strategy must stay
  computable in under two minutes on a laptop. Never require scanning every
  ticker pair with rolling statistics. For pairs strategies, require
  selecting a limited candidate set (at most ~20 pairs) from a single
  correlation screen over exactly the first 252 rows of data (a fixed row
  count), with zero positions during that window and pairs traded only
  afterwards.

Respond with a JSON object with exactly these keys:
{
  "strategy_name": "short descriptive name",
  "strategy_logic": "2-4 sentence description of the economic rationale and mechanics",
  "key_parameters": {"parameter_name": "value and justification", ...},
  "signals": ["precise description of each signal used", ...],
  "references": ["papers or sources informing the design", ...]
}
"""


def _search_literature(query_context: str, max_results: int = 5) -> list[dict]:
    """Prefer Semantic Scholar, then fall back to an academic-domain search."""
    try:
        response = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": query_context,
                "limit": max_results,
                "fields": "title,url,abstract,year,authors",
            },
            headers={"User-Agent": "quant-team-framework/1.0"},
            timeout=15,
        )
        response.raise_for_status()
        papers = response.json().get("data", [])
        results = [
            {
                "title": paper.get("title", ""),
                "url": paper.get("url", ""),
                "year": paper.get("year"),
                "authors": [
                    author.get("name", "") for author in paper.get("authors", [])[:5]
                ],
                "snippet": (paper.get("abstract") or "")[:1200],
                "source": "Semantic Scholar",
            }
            for paper in papers
            if paper.get("title") and paper.get("url")
        ]
        if results:
            return results
    except Exception as exc:
        logger.warning("Semantic Scholar unavailable (%s); trying Crossref.", exc)

    try:
        response = requests.get(
            "https://api.crossref.org/works",
            params={
                "query.bibliographic": query_context,
                "rows": max_results,
                "select": "title,URL,abstract,published,author,DOI",
            },
            headers={"User-Agent": "quant-team-framework/1.0"},
            timeout=15,
        )
        response.raise_for_status()
        works = response.json().get("message", {}).get("items", [])
        results = []
        for work in works:
            titles = work.get("title") or []
            url = work.get("URL") or (
                f"https://doi.org/{work['DOI']}" if work.get("DOI") else ""
            )
            if not titles or not url:
                continue
            date_parts = work.get("published", {}).get("date-parts", [[]])
            results.append(
                {
                    "title": titles[0],
                    "url": url,
                    "year": date_parts[0][0] if date_parts and date_parts[0] else None,
                    "authors": [
                        " ".join(
                            part
                            for part in (author.get("given", ""), author.get("family", ""))
                            if part
                        )
                        for author in work.get("author", [])[:5]
                    ],
                    "snippet": (work.get("abstract") or "")[:1200],
                    "source": "Crossref",
                }
            )
        if results:
            return results
    except Exception as exc:
        logger.warning("Crossref unavailable (%s); trying academic web search.", exc)

    try:
        from ddgs import DDGS

        query = (
            f"{query_context} quantitative trading paper "
            "(site:arxiv.org OR site:ssrn.com OR site:doi.org)"
        )
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        allowed_domains = ("arxiv.org", "ssrn.com", "doi.org")
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
                "source": "Academic web search",
            }
            for r in results
            if r.get("title") and r.get("href")
            and any(
                urlparse(r["href"]).netloc.lower().endswith(domain)
                for domain in allowed_domains
            )
        ]
    except Exception as exc:  # network errors, rate limits, missing package
        logger.warning("Academic search unavailable (%s); using model knowledge only.", exc)
        return []


def run_research_agent(
    strategy_idea: str,
    dataset_context: dict,
) -> dict:
    """Return one independent research proposal for the selected data."""
    logger.info("Research Agent: analyzing '%s'", strategy_idea)
    search_context = (
        f"{strategy_idea} universe {dataset_context['universe']} "
        f"fields {' '.join(dataset_context['fields'])}"
    )
    search_results = _search_literature(search_context)

    user_prompt_parts = [
        f"Strategy idea: {strategy_idea}",
        "Selected dataset context:\n" + json.dumps(dataset_context, indent=2),
    ]
    if search_results:
        user_prompt_parts.append(
            "Web search results:\n" + json.dumps(search_results, indent=2)
        )
    else:
        user_prompt_parts.append(
            "No web search results available; rely on your knowledge of the literature."
        )
    summary = chat_json(SYSTEM_PROMPT, "\n\n".join(user_prompt_parts))

    required_keys = {"strategy_name", "strategy_logic", "key_parameters", "signals"}
    missing = required_keys - summary.keys()
    if missing:
        raise ValueError(f"Research summary missing required keys: {missing}")
    # Never publish model-invented citations. References come only from
    # records actually returned by the academic search.
    summary["references"] = [
        f"{result['title']} — {result['url']}" for result in search_results
    ]

    logger.info("Research Agent: produced summary for '%s'", summary["strategy_name"])
    return summary
