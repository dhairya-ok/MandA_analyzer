"""Ingestion utilities for loading a universe of companies into the DB.

This module provides a simple, resilient ingestion flow using yfinance and
pandas as free data sources. It's intentionally conservative: it creates
company records and stores snapshot financial statements as JSON payloads.

Note: network access is required to fetch data from Wikipedia and yfinance.
If network calls fail, a small fallback ticker list will be used.
"""
from typing import List, Dict, Any
import time
import logging

import pandas as pd
import yfinance as yf

from .db import SessionLocal, init_db
from .models.models import Company, Financial

logger = logging.getLogger("backend.ingest")
logging.basicConfig(level=logging.INFO)


def _get_sp500_tickers(limit: int) -> List[str]:
    """Try to fetch S&P 500 tickers from Wikipedia as a default universe.

    Falls back to a small hard-coded list if the fetch fails.
    """
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        tickers = df["Symbol"].astype(str).tolist()
        return tickers[:limit]
    except Exception as e:
        logger.warning("Could not fetch S&P500 list from Wikipedia: %s", e)
        return ["AAPL", "MSFT", "AMZN", "GOOGL", "META"][:limit]


def _parse_period(col) -> Dict[str, Any]:
    """Parse a column label (period) into year and quarter where possible."""
    try:
        ts = pd.to_datetime(col)
        return {"year": int(ts.year), "quarter": int(ts.quarter)}
    except Exception:
        return {"year": None, "quarter": None}


def _create_mock_financial(company_id: int, stmt_type: str, year: int) -> Financial:
    """Create mock financial statement data."""
    mock_data = {
        "income": {
            "Revenue": 1e10 * (1.1 ** (year - 2020)),
            "Net Income": 2e9 * (1.15 ** (year - 2020)),
            "EBITDA": 3e9 * (1.12 ** (year - 2020))
        },
        "balance": {
            "Total Assets": 5e10,
            "Total Liabilities": 3e10,
            "Shareholders Equity": 2e10
        },
        "cashflow": {
            "Operating Cash Flow": 2.5e9 * (1.1 ** (year - 2020)),
            "Free Cash Flow": 2e9 * (1.1 ** (year - 2020)),
            "Capital Expenditure": -5e8
        }
    }
    
    return Financial(
        company_id=company_id,
        statement_type=stmt_type,
        period="annual",
        year=year,
        quarter=None,
        data={"values": mock_data[stmt_type]}
    )

def ingest_universe(limit: int = 200, pause: float = 2.0, use_mock: bool = True) -> Dict[str, Any]:
    """Ingest a universe of tickers into the backend DB with rate limiting.

    Returns a summary dict containing counts and any errors encountered.
    """
    init_db()
    session = SessionLocal()
    tickers = _get_sp500_tickers(limit)
    summary = {"requested": limit, "processed": 0, "errors": []}

    mock_data = {
        'AAPL': ('Apple Inc.', 'Technology', 'Consumer Electronics'),
        'MSFT': ('Microsoft Corporation', 'Technology', 'Software'),
        'AMZN': ('Amazon.com Inc.', 'Consumer Cyclical', 'Internet Retail'),
        'GOOGL': ('Alphabet Inc.', 'Technology', 'Internet Services'),
        'META': ('Meta Platforms Inc.', 'Technology', 'Social Media')
    }

    for i, ticker in enumerate(tickers, start=1):
        if i > limit:
            break
        try:
            logger.info("Processing %s (%d/%d)", ticker, i, limit)
            
            if use_mock:
                name, sector, industry = mock_data.get(ticker, (f"{ticker} Inc.", "Technology", "Software"))
                company = Company(
                    ticker=str(ticker),
                    name=name,
                    sector=sector,
                    industry=industry,
                    market_cap=1e11,
                    revenue=1e10,
                    net_income=2e9,
                    employees=10000,
                    ebitda=3e9,
                    net_debt=1e9
                )
            else:
                t = yf.Ticker(ticker)
                info = t.info or {}

                company = Company(
                    ticker=str(ticker),
                    name=info.get("longName") or info.get("shortName") or str(ticker),
                    sector=info.get("sector"),
                    market_cap=info.get("marketCap"),
                )
            
            session.add(company)
            session.commit()
            session.refresh(company)

            if use_mock:
                # Create mock financial statements for the last 3 years
                current_year = 2025  # You can adjust this as needed
                for year in range(current_year - 2, current_year + 1):
                    for stmt_type in ["income", "balance", "cashflow"]:
                        fin = _create_mock_financial(company.id, stmt_type, year)
                        session.add(fin)
                session.commit()
            else:
                # Real data ingestion from yfinance
                statements = {
                    "income": getattr(t, "financials", None),
                    "balance": getattr(t, "balance_sheet", None),
                    "cashflow": getattr(t, "cashflow", None),
                }

                for stmt_type, df in statements.items():
                    try:
                        if df is None or df.empty:
                            continue
                        for col in df.columns:
                            per = _parse_period(col)
                            data = df[col].fillna(0).to_dict()
                            fin = Financial(
                                company_id=company.id,
                                statement_type=stmt_type,
                                period="annual",
                                year=per.get("year") or 0,
                                quarter=per.get("quarter"),
                                data={"values": data},
                            )
                            session.add(fin)
                        session.commit()
                except Exception as e:
                    # Non-fatal for single statement
                    logger.warning("Failed to ingest statement %s for %s: %s", stmt_type, ticker, e)

            summary["processed"] += 1
            # polite pause to reduce rate pressure on free services
            time.sleep(pause)
        except Exception as e:
            logger.exception("Error ingesting %s", ticker)
            summary["errors"].append({"ticker": ticker, "error": str(e)})

    session.close()
    return summary


def seed_sample_universe() -> Dict[str, Any]:
    """Seed the DB with a small, static universe for demos/tests when external
    data sources are unavailable or rate-limited.
    """
    init_db()
    session = SessionLocal()
    sample = [
        {"ticker": "ACQ1", "name": "Acquirer One", "sector": "Technology", "market_cap": 200e9},
        {"ticker": "TGT1", "name": "Target Alpha", "sector": "Technology", "market_cap": 10e9},
        {"ticker": "TGT2", "name": "Target Beta", "sector": "Healthcare", "market_cap": 8e9},
        {"ticker": "TGT3", "name": "Target Gamma", "sector": "Technology", "market_cap": 50e8},
        {"ticker": "TGT4", "name": "Target Delta", "sector": "Financials", "market_cap": 6e9},
    ]
    created = 0
    for s in sample:
        try:
            c = Company(ticker=s["ticker"], name=s["name"], sector=s.get("sector"), market_cap=s.get("market_cap"))
            session.add(c)
            session.commit()
            session.refresh(c)
            # add a dummy income financial with revenue history
            fin = Financial(company_id=c.id, statement_type="income", period="annual", year=2022, quarter=None, data={"values": {"Total Revenue": 1000.0 if s["ticker"].startswith("ACQ") else 100.0}})
            session.add(fin)
            session.commit()
            created += 1
        except Exception as e:
            logger.warning("Failed to create sample company %s: %s", s.get("ticker"), e)
    session.close()
    return {"created": created}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("ingest_universe")
    parser.add_argument("--limit", type=int, default=50, help="Number of tickers to ingest")
    parser.add_argument("--seed", action="store_true", help="Seed a small sample universe instead of fetching from internet")
    args = parser.parse_args()
    if args.seed:
        result = seed_sample_universe()
    else:
        result = ingest_universe(limit=args.limit)
    print(result)
