import os
import json
import asyncio
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from starlette.applications import Starlette
from starlette.responses import JSONResponse, FileResponse
from starlette.routing import Route
from starlette.requests import Request
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from backend.models.models import Company, Financial, DealPair, Valuation
from backend.pdf_generator import generate_deal_brief
from backend.valuation import (
    calculate_base_fcf,
    project_cash_flows,
    calculate_dcf_confidence,
    generate_dcf_sensitivity_grid,
    assess_data_completeness
)
from backend.metrics import fetch_market_data
from backend.logger import setup_logger
from backend.db import get_db, init_db, SessionLocal
from backend import ingest as ingest_module
from ..pairing import generate_top_pairs

logger = setup_logger(__name__)


# ───────────────────────────────
# STARTUP & BASIC ROUTES
# ───────────────────────────────

async def on_startup() -> None:
    init_db()


async def root(request: Request) -> JSONResponse:
    """Root endpoint for Render check"""
    return JSONResponse({
        "message": "Backend is running successfully 🚀",
        "timestamp": datetime.utcnow().isoformat()
    })


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


# ───────────────────────────────
# MOCK + INGESTION
# ───────────────────────────────

def create_mock_company(ticker: str) -> Company:
    mock_data = {
        'AAPL': ('Apple Inc.', 'Technology', 'Consumer Electronics'),
        'MSFT': ('Microsoft Corporation', 'Technology', 'Software'),
        'AMZN': ('Amazon.com Inc.', 'Consumer Cyclical', 'Internet Retail'),
        'GOOGL': ('Alphabet Inc.', 'Technology', 'Internet Services'),
        'META': ('Meta Platforms Inc.', 'Technology', 'Social Media')
    }
    name, sector, industry = mock_data.get(ticker, (f"{ticker} Inc.", "Technology", "Software"))
    return Company(
        ticker=ticker,
        name=name,
        sector=sector,
        industry=industry,
        country="USA",
        market_cap=1e11,
        revenue=1e10,
        net_income=2e9,
        employees=10000,
        ebitda=3e9,
        net_debt=1e9
    )


async def ingest_endpoint(request: Request) -> JSONResponse:
    """Trigger ingestion (query: ?limit=50&mock=true)."""
    params = request.query_params
    try:
        limit = int(params.get("limit", 50))
        use_mock = params.get("mock", "true").lower() == "true"
    except Exception:
        limit, use_mock = 50, True

    result = ingest_module.ingest_universe(limit=limit, use_mock=use_mock)
    return JSONResponse(result)


# ───────────────────────────────
# DEAL PAIRS, DCF, COMPS
# ───────────────────────────────

async def pairs_endpoint(request: Request) -> JSONResponse:
    params = request.query_params
    acquirer = params.get("acquirer")
    try:
        top = int(params.get("top", 20))
    except Exception:
        top = 20

    if not acquirer:
        return JSONResponse({"error": "missing acquirer parameter"}, status_code=400)

    try:
        results = generate_top_pairs(acquirer.upper(), top=top)
        return JSONResponse({"acquirer": acquirer.upper(), "top": top, "results": results})
    except Exception as e:
        logger.exception("Pair generation failed")
        return JSONResponse({"error": str(e)}, status_code=500)


async def dcf(request: Request) -> JSONResponse:
    try:
        pair_id = request.path_params.get("pair_id", "")
        body = await request.json()

        session = SessionLocal()
        pair = session.query(DealPair).filter(DealPair.id == pair_id).first()
        if not pair:
            return JSONResponse({"error": "Pair not found"}, status_code=404)

        target = pair.target
        financials = session.query(Financial).filter(
            Financial.company_id == target.id,
            Financial.statement_type.ilike("%income%")
        ).order_by(Financial.year.desc()).all()

        growth_rate = body.get("growth_rate", 0.03)
        wacc = body.get("wacc", 0.10)
        projection_years = body.get("projection_years", 5)
        terminal_growth = body.get("terminal_growth", 0.02)

        base_fcf = calculate_base_fcf(financials)
        projected_fcfs = project_cash_flows(base_fcf, growth_rate, projection_years)
        terminal_value = projected_fcfs[-1] * (1 + terminal_growth) / (wacc - terminal_growth)

        discount_factors = [(1 + wacc) ** -i for i in range(1, projection_years + 1)]
        pv_fcfs = sum(fcf * df for fcf, df in zip(projected_fcfs, discount_factors))
        pv_terminal = terminal_value * discount_factors[-1]
        enterprise_value = pv_fcfs + pv_terminal

        confidence = calculate_dcf_confidence(financials, growth_rate, wacc)

        return JSONResponse({
            "meta": {"model": "DCF", "timestamp": datetime.now().timestamp()},
            "data": {
                "pair_id": pair_id,
                "enterprise_value": enterprise_value,
                "confidence": confidence,
                "assumptions": {
                    "growth_rate": growth_rate,
                    "wacc": wacc,
                    "projection_years": projection_years,
                    "terminal_growth": terminal_growth,
                    "base_fcf": base_fcf
                },
                "projections": {
                    "fcfs": projected_fcfs,
                    "terminal_value": terminal_value
                },
                "sensitivity": generate_dcf_sensitivity_grid(
                    base_fcf, growth_rate, wacc, terminal_growth
                ),
                "provenance": {
                    "source": "historical_financials",
                    "data_completeness": assess_data_completeness(financials),
                    "last_actual_year": max(f.year for f in financials) if financials else None
                }
            }
        })
    except Exception as e:
        logger.exception("DCF calculation failed")
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        session.close()


# ───────────────────────────────
# ROUTES
# ───────────────────────────────

routes = [
    Route("/", endpoint=root),
    Route("/health", endpoint=health),
    Route("/ingest", endpoint=ingest_endpoint, methods=["POST", "GET"]),
    Route("/pairs", endpoint=pairs_endpoint, methods=["GET"]),
    Route("/api/valuations/{pair_id}/dcf", endpoint=dcf),
]


# ───────────────────────────────
# MIDDLEWARE
# ───────────────────────────────

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests=100, window_seconds=60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)

    async def dispatch(self, request, call_next):
        now = datetime.now()
        client_ip = request.client.host
        self.requests[client_ip] = [
            ts for ts in self.requests[client_ip]
            if now - ts < timedelta(seconds=self.window_seconds)
        ]

        if len(self.requests[client_ip]) >= self.max_requests:
            return JSONResponse(
                {"error": "Rate limit exceeded. Please try again later."}, status_code=429
            )

        self.requests[client_ip].append(now)
        response = await call_next(request)
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers.update({
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains"
        })
        return response


middleware = [
    Middleware(GZipMiddleware, minimum_size=500),
    Middleware(RateLimitMiddleware),
    Middleware(SecurityHeadersMiddleware),
    Middleware(TrustedHostMiddleware, allowed_hosts=["*"]),
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    ),
]

app = Starlette(
    debug=False,
    routes=routes,
    on_startup=[on_startup],
    middleware=middleware
)
