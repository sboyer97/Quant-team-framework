# Quant Team Framework

Personal project I built to practice agent frameworks from scratch: no
LangChain or LangGraph, just the OpenAI Python package and a loop.

You give a research question in plain English ("pairs trading on the S&P 500")
and a small team of 4 agents turns it into backtested strategy candidates:

1. a research agent that searches the web for relevant academic papers
2. an implementation agent that writes the strategy code
3. a verification agent that reviews the code and sends it back for fixes
4. a backtest agent, which is deliberately not an LLM: metrics are computed
   with plain pandas so the numbers can't be hallucinated

The loop runs several independent times (10 by default) and ranks the valid
candidates by Sharpe ratio.

Market data is downloaded and cached before the agents start. I didn't build
a data agent on purpose: the agents never fetch or generate market data.

The verifier is the part I spent the most time on. On top of an LLM review it
runs hard checks on the generated code: only pandas/numpy imports, the
function actually runs and returns the right shape, and past signals must not
change when future rows are removed (my cheap test against lookahead bias).
Code that fails these checks goes back to the implementation agent, 3 rounds
max, then the candidate is dropped.

## Setup

Needs Python 3.10+ and an OpenAI API key.

```bash
git clone https://github.com/sboyer97/Quant-team-framework.git
cd Quant-team-framework
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then put your key in .env
```

## Run it

```bash
python main.py
```

It asks for the research question, the number of candidates and the backtest
dates. Default universe is the S&P 500 (current constituents, downloaded from
Yahoo Finance and cached locally). At the end you get `best_strategy.py` (the
generated code with the best Sharpe) plus two JSON reports with the research
summaries, the ranking and the failed candidates.

Other universes are available through CLI flags (`python main.py --help`):
curated sectors (banking, technology, energy, healthcare, consumer), a few
indexes (nasdaq100, cac40, dax40, ftse100, nikkei225), custom tickers, your
own CSV (long format: `date,ticker,close[,volume]`, see
`examples/market_data.csv`), or `--universe agent` to let an agent pick the
tickers from a free-text description. Example:

```bash
python main.py --non-interactive \
  --universe custom --tickers JPM,BAC,WFC,C \
  --idea "pairs trading on S&P 500 banks" \
  --runs 10 --start 2018-01-01 --end 2024-12-31
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Tests run fully offline: LLM calls are mocked and no market data is
downloaded, so you don't need an API key to run them.

## Fair warning

This is a training project, not an investment system. Candidates are
generated, selected and ranked on the same historical sample — no
walk-forward, no out-of-sample — so the reported Sharpe is heavily exposed to
selection bias and overfitting. Using current index constituents on past
periods also adds survivorship bias. And the generated code runs in-process
after verification; a real system would sandbox it. Don't trade this.
