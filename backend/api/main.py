import os
from starlette.applications import Starlette
from starlette.responses import JSONResponse, FileResponse
from starlette.routing import Route
from starlette.requests import Request
from datetime import datetime
import numpy as np
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
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from sqlalchemy.orm import Session
from backend.logger import setup_logger

logger = setup_logger(__name__)
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict
from backend.db import get_db, init_db, SessionLocal
from backend import ingest as ingest_module
import json
from ..pairing import generate_top_pairs


async def on_startup() -> None:
	init_db()


async def health(request: Request) -> JSONResponse:
	return JSONResponse({"status": "ok"})


def create_mock_company(ticker: str) -> Company:
    """Create a mock company for testing purposes."""
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
    """Trigger ingestion. Query params: ?limit=50&mock=true"""
    params = request.query_params
    try:
        limit = int(params.get("limit", 50))
        use_mock = params.get("mock", "true").lower() == "true"
    except Exception:
        limit = 50
        use_mock = True

    # Run ingestion synchronously here; for large runs you should use background tasks or a worker.
    result = ingest_module.ingest_universe(limit=limit, use_mock=use_mock)
    return JSONResponse(result)


async def pairs_endpoint(request: Request) -> JSONResponse:
    """Return top target pairs for an acquirer ticker. Query params: acquirer, top"""
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
        return JSONResponse({"error": str(e)}, status_code=500)


async def dcf(request: Request) -> JSONResponse:
    """Calculate DCF valuation for a deal pair."""
    try:
        pair_id = request.path_params.get("pair_id", "")
        body = await request.json()
        
        session = SessionLocal()
        pair = session.query(DealPair).filter(DealPair.id == pair_id).first()
        if not pair:
            return JSONResponse({"error": "Pair not found"}, status_code=404)
            
        # Get historical financials for target
        target = pair.target
        financials = session.query(Financial).filter(
            Financial.company_id == target.id,
            Financial.statement_type.ilike("%income%")
        ).order_by(Financial.year.desc()).all()
        
        # Extract and process assumptions
        growth_rate = body.get("growth_rate", 0.03)
        wacc = body.get("wacc", 0.10)
        projection_years = body.get("projection_years", 5)
        terminal_growth = body.get("terminal_growth", 0.02)
        
        # Calculate projected cash flows
        base_fcf = calculate_base_fcf(financials)
        projected_fcfs = project_cash_flows(
            base_fcf, 
            growth_rate, 
            projection_years
        )
        
        # Calculate terminal value
        terminal_value = projected_fcfs[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
        
        # Discount all cash flows
        discount_factors = [(1 + wacc) ** -i for i in range(1, projection_years + 1)]
        pv_fcfs = sum(fcf * df for fcf, df in zip(projected_fcfs, discount_factors))
        pv_terminal = terminal_value * discount_factors[-1]
        
        enterprise_value = pv_fcfs + pv_terminal
        
        # Calculate confidence score based on data quality
        confidence = calculate_dcf_confidence(financials, growth_rate, wacc)
        
        return JSONResponse({
            "meta": {
                "model": "DCF",
                "timestamp": datetime.now().timestamp()
            },
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
                    base_fcf, 
                    growth_rate, 
                    wacc, 
                    terminal_growth
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


async def comps(request: Request) -> JSONResponse:
    """Calculate valuation using comparable companies analysis."""
    try:
        pair_id = request.path_params.get("pair_id", "")
        session = SessionLocal()
        
        # Get the deal pair and target company
        pair = session.query(DealPair).filter(DealPair.id == pair_id).first()
        if not pair:
            return JSONResponse({"error": "Pair not found"}, status_code=404)
        
        target = pair.target
        
        # Find comparable companies (same sector, similar size)
        comparable_companies = session.query(Company).filter(
            Company.sector == target.sector,
            Company.id != target.id,
            Company.market_cap.between(target.market_cap * 0.3, target.market_cap * 3.0)
        ).limit(10).all()
        
        if not comparable_companies:
            return JSONResponse({"error": "No comparable companies found"}, status_code=404)
        
        # Calculate multiples for each comparable
        multiples = []
        for comp in comparable_companies:
            market_data = fetch_market_data(comp.ticker)
            if not market_data:
                continue
                
            financials = session.query(Financial).filter(
                Financial.company_id == comp.id,
                Financial.statement_type.ilike("%income%")
            ).order_by(Financial.year.desc()).first()
            
            if not financials or not financials.data:
                continue
                
            try:
                revenue = float(financials.data.get("revenue", 0))
                ebitda = float(financials.data.get("ebitda", 0))
                
                if revenue > 0 and ebitda > 0:
                    ev = market_data.get("enterprise_value", comp.market_cap)
                    multiples.append({
                        "company": comp.ticker,
                        "ev_revenue": ev / revenue,
                        "ev_ebitda": ev / ebitda,
                        "market_cap": comp.market_cap
                    })
            except (ValueError, TypeError):
                continue
        
        if not multiples:
            return JSONResponse({"error": "Could not calculate multiples"}, status_code=404)
        
        # Calculate median multiples
        ev_revenue_median = np.median([m["ev_revenue"] for m in multiples])
        ev_ebitda_median = np.median([m["ev_ebitda"] for m in multiples])
        
        # Get target financials
        target_financials = session.query(Financial).filter(
            Financial.company_id == target.id,
            Financial.statement_type.ilike("%income%")
        ).order_by(Financial.year.desc()).first()
        
        if not target_financials or not target_financials.data:
            return JSONResponse({"error": "Target financials not found"}, status_code=404)
        
        # Calculate implied values
        target_revenue = float(target_financials.data.get("revenue", 0))
        target_ebitda = float(target_financials.data.get("ebitda", 0))
        
        implied_ev_revenue = target_revenue * ev_revenue_median
        implied_ev_ebitda = target_ebitda * ev_ebitda_median
        
        # Final enterprise value (weighted average)
        enterprise_value = (implied_ev_revenue * 0.4) + (implied_ev_ebitda * 0.6)
        
        # Calculate confidence score based on data quality
        confidence = min(1.0, len(multiples) / 5.0)  # More comps = higher confidence
        
        return JSONResponse({
            "meta": {
                "model": "Comps",
                "timestamp": datetime.now().timestamp()
            },
            "data": {
                "pair_id": pair_id,
                "enterprise_value": enterprise_value,
                "confidence": confidence,
                "assumptions": {
                    "ev_revenue_multiple": ev_revenue_median,
                    "ev_ebitda_multiple": ev_ebitda_median,
                    "comps_used": len(multiples)
                },
                "comparables": multiples,
                "implied_values": {
                    "ev_revenue": implied_ev_revenue,
                    "ev_ebitda": implied_ev_ebitda
                },
                "provenance": {
                    "source": "market_data_and_financials",
                    "comps_tickers": [m["company"] for m in multiples]
                }
            }
        })
    except Exception as e:
        logger.exception("Comps calculation failed")
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        session.close()


async def generate_deal_brief_endpoint(request: Request) -> JSONResponse:
    """Generate a Deal Brief PDF for a pair."""
    try:
        pair_id = request.path_params.get("pair_id", "")
        session = SessionLocal()
        
        # Get the deal pair and related data
        pair = session.query(DealPair).filter(DealPair.id == pair_id).first()
        if not pair:
            return JSONResponse({"error": "Pair not found"}, status_code=404)
        
        # Get valuation data
        valuation = session.query(Valuation).filter(Valuation.pair_id == pair_id).first()
        if not valuation:
            return JSONResponse({"error": "Valuation not found"}, status_code=404)
        
        # Compile deal data
        deal_data = {
            "acquirer": pair.acquirer.ticker,
            "target": pair.target.ticker,
            "executive_summary": f"Proposed acquisition of {pair.target.name} by {pair.acquirer.name}",
            "strategic_rationale": {
                "key_points": [
                    f"Strong strategic fit with compatibility score of {pair.compatibility_score:.1f}/100",
                    f"Complementary {pair.target.sector} sector capabilities",
                    "Revenue and cost synergy opportunities",
                    "Market expansion potential"
                ]
            },
            "valuation": {
                "dcf_value": valuation.dcf_value,
                "comps_value": valuation.comps_value,
                "precedent_value": valuation.ensemble_value,  # Using ensemble as precedent for now
                "dcf_confidence": valuation.confidence_scores.get("dcf", 0.7),
                "comps_confidence": valuation.confidence_scores.get("comps", 0.8),
                "precedent_confidence": valuation.confidence_scores.get("precedent", 0.6)
            },
            "projections": {
                "years": [str(year) for year in range(2024, 2029)],
                "values": [100, 120, 150, 185, 230]  # Example growth projection
            },
            "risks": [
                {
                    "description": "Integration complexity due to different technology stacks",
                    "mitigation": "Detailed technical assessment and phased integration plan"
                },
                {
                    "description": "Potential customer overlap in key markets",
                    "mitigation": "Customer retention program and clear product roadmap"
                }
            ]
        }
        
        # Generate PDF
        pdf_path = generate_deal_brief(deal_data)
        
        # Read PDF file
        with open(pdf_path, 'rb') as f:
            pdf_content = f.read()
        
        # Clean up temp file
        os.remove(pdf_path)
        
        # Return PDF as download
        return FileResponse(
            pdf_content,
            media_type='application/pdf',
            headers={
                'Content-Disposition': f'attachment; filename="deal_brief_{pair.acquirer.ticker}_{pair.target.ticker}.pdf"'
            }
        )
        
    except Exception as e:
        logger.exception("Deal brief generation failed")
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        session.close()

routes = [
    Route("/health", endpoint=health),
    Route("/ingest", endpoint=ingest_endpoint, methods=["POST", "GET"]),
    Route("/pairs", endpoint=pairs_endpoint, methods=["GET"]),
    Route("/api/valuations/{pair_id}/dcf", endpoint=dcf),
    Route("/api/valuations/{pair_id}/comps", endpoint=comps),
    Route("/api/deal-brief/{pair_id}", endpoint=generate_deal_brief_endpoint),
]

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests=100, window_seconds=60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)

    async def dispatch(self, request, call_next):
        now = datetime.now()
        client_ip = request.client.host

        # Clean old requests
        self.requests[client_ip] = [ts for ts in self.requests[client_ip] 
                                  if now - ts < timedelta(seconds=self.window_seconds)]
        
        # Check rate limit
        if len(self.requests[client_ip]) >= self.max_requests:
            return JSONResponse(
                {"error": "Rate limit exceeded. Please try again later."},
                status_code=429
            )

        # Add current request
        self.requests[client_ip].append(now)
        
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            return JSONResponse(
                {"error": str(e), "type": type(e).__name__},
                status_code=500
            )

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        return response

class InputValidationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        path = request.url.path
        params = request.query_params
        
        # Validate acquirer ticker format
        if path == "/pairs" and "acquirer" in params:
            ticker = params["acquirer"]
            if not ticker.isalnum() or len(ticker) > 10:
                return JSONResponse(
                    {"error": "Invalid ticker format"},
                    status_code=400
                )
        
        response = await call_next(request)
        return response

middleware = [
    Middleware(GZipMiddleware, minimum_size=500),
    Middleware(RateLimitMiddleware, max_requests=100, window_seconds=60),
    Middleware(SecurityHeadersMiddleware),
    Middleware(InputValidationMiddleware),
    Middleware(TrustedHostMiddleware, allowed_hosts=["*"]),
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
		allow_headers=["*"],
	),
]

app = Starlette(debug=False, routes=routes, on_startup=[on_startup], middleware=middleware)
