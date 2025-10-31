"""Pairwise compatibility scoring engine.

This module provides a simple, interpretable scoring function to rank target
companies for a given acquirer. The scoring is intentionally transparent and
easy to extend: weights and sub-scores are visible and based on available data.
"""
from typing import List, Dict, Any, Tuple, Optional
from sqlalchemy.orm import Session
from .db import SessionLocal
from .models.models import Company, DealPair, Financial
import math
import logging
from functools import lru_cache
from datetime import datetime, timedelta

logger = logging.getLogger("backend.pairing")


def _size_score(acq_cap: float, tgt_cap: float) -> float:
    """Score how size-compatible two companies are. Returns 0..1.
    
    Considers both absolute size difference and relative scale for strategic fit.
    - Ideal target size: 10-50% of acquirer
    - Penalties for too small (<5%) or too large (>70%) targets
    """
    if not acq_cap or not tgt_cap or acq_cap <= 0 or tgt_cap <= 0:
        return 0.0
    
    ratio = tgt_cap / acq_cap
    
    # Define ideal ranges
    IDEAL_MIN = 0.10  # 10% of acquirer size
    IDEAL_MAX = 0.50  # 50% of acquirer size
    ABSOLUTE_MIN = 0.05  # Too small below 5%
    ABSOLUTE_MAX = 0.70  # Too large above 70%
    
    if ratio < ABSOLUTE_MIN or ratio > ABSOLUTE_MAX:
        return 0.0
    elif IDEAL_MIN <= ratio <= IDEAL_MAX:
        return 1.0
    else:
        # Linear falloff outside ideal range
        if ratio < IDEAL_MIN:
            return (ratio - ABSOLUTE_MIN) / (IDEAL_MIN - ABSOLUTE_MIN)
        else:
            return 1.0 - (ratio - IDEAL_MAX) / (ABSOLUTE_MAX - IDEAL_MAX)


def _sector_score(acq_sector: str, tgt_sector: str, acq_industry: str = None, tgt_industry: str = None) -> float:
    """Calculate sector and industry compatibility score.
    
    Considers both sector-level matches and specific industry synergies:
    - Perfect match (same sector & industry): 1.0
    - Same sector, complementary industry: 0.8
    - Same sector, different industry: 0.6
    - Adjacent sectors with synergies: 0.3
    - Others: 0.0
    """
    if not acq_sector or not tgt_sector:
        return 0.0
        
    acq_sector = acq_sector.strip().lower()
    tgt_sector = tgt_sector.strip().lower()
    
    # Define sector adjacency and synergies
    ADJACENT_SECTORS = {
        'technology': {'communications', 'consumer cyclical'},
        'healthcare': {'technology', 'consumer defensive'},
        'financial': {'technology', 'real estate'},
        'consumer cyclical': {'technology', 'consumer defensive'},
        'industrial': {'technology', 'materials'}
    }
    
    if acq_sector == tgt_sector:
        if acq_industry and tgt_industry:
            acq_industry = acq_industry.strip().lower()
            tgt_industry = tgt_industry.strip().lower()
            if acq_industry == tgt_industry:
                return 1.0  # Perfect match
            
            # Define complementary industries within sectors
            COMPLEMENTARY = {
                'technology': {
                    'software': {'semiconductors', 'hardware'},
                    'hardware': {'software', 'semiconductors'},
                    'semiconductors': {'hardware', 'software'}
                }
            }
            
            if acq_sector in COMPLEMENTARY:
                if acq_industry in COMPLEMENTARY[acq_sector]:
                    if tgt_industry in COMPLEMENTARY[acq_sector][acq_industry]:
                        return 0.8  # Complementary industries
            return 0.6  # Same sector, different industries
            
        return 0.7  # Same sector, industries unknown
    
    # Check for adjacent sector synergies
    if acq_sector in ADJACENT_SECTORS and tgt_sector in ADJACENT_SECTORS[acq_sector]:
        return 0.3
        
    return 0.0  # No clear synergies


@lru_cache(maxsize=1000)
def _get_cached_financials(company_id: str) -> List[Dict[str, Any]]:
    """Cache financial data for companies to improve performance."""
    session = SessionLocal()
    try:
        financials = (
            session.query(Financial)
            .filter(Financial.company_id == company_id)
            .filter(Financial.statement_type.ilike("%income%"))
            .order_by(Financial.year.desc())
            .all()
        )
        return [{"year": f.year, "data": f.data} for f in financials]
    finally:
        session.close()

def _calculate_growth_metrics(financials: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate comprehensive growth and profitability metrics.
    
    Metrics included:
    - Revenue growth (YoY and CAGR)
    - EBITDA margins and growth
    - Net income margins and growth
    - Operating cash flow trends
    """
    if len(financials) < 2:
        return {
            "revenue_growth": 0.0,
            "revenue_cagr": 0.0,
            "ebitda_margin": 0.0,
            "ebitda_growth": 0.0,
            "net_margin": 0.0,
            "margin_trend": 0.0
        }
    
    # Extract time series for each metric
    metrics = {
        "revenue": [],
        "ebitda": [],
        "net_income": [],
        "operating_cash_flow": []
    }
    
    for f in financials:
        data = f.get("data", {}).get("values", {})
        year = f["year"]
        
        if isinstance(data, dict):
            # Extract revenue
            for k, v in data.items():
                if isinstance(k, str):
                    k_lower = k.lower()
                    try:
                        val = float(v)
                        if "revenue" in k_lower or "sales" in k_lower:
                            metrics["revenue"].append((year, val))
                        elif "ebitda" in k_lower:
                            metrics["ebitda"].append((year, val))
                        elif "net income" in k_lower:
                            metrics["net_income"].append((year, val))
                        elif "operating cash flow" in k_lower:
                            metrics["operating_cash_flow"].append((year, val))
                    except (ValueError, TypeError):
                        continue
    
    # Sort all metrics by year
    for metric in metrics.values():
        metric.sort(key=lambda x: x[0])
    
    results = {}
    
    # Calculate revenue metrics
    if len(metrics["revenue"]) >= 2:
        latest_rev = metrics["revenue"][-1][1]
        prev_rev = metrics["revenue"][-2][1]
        first_rev = metrics["revenue"][0][1]
        years = metrics["revenue"][-1][0] - metrics["revenue"][0][0]
        
        results["revenue_growth"] = ((latest_rev / prev_rev) - 1) if prev_rev > 0 else 0
        results["revenue_cagr"] = ((latest_rev / first_rev) ** (1/years) - 1) if years > 0 and first_rev > 0 else 0
    else:
        results["revenue_growth"] = results["revenue_cagr"] = 0.0
    
    # Calculate profitability metrics
    if metrics["revenue"] and metrics["ebitda"]:
        latest_rev = metrics["revenue"][-1][1]
        latest_ebitda = next((v[1] for v in metrics["ebitda"] if v[0] == metrics["revenue"][-1][0]), 0)
        results["ebitda_margin"] = latest_ebitda / latest_rev if latest_rev > 0 else 0
        
        # Calculate EBITDA growth
        if len(metrics["ebitda"]) >= 2:
            results["ebitda_growth"] = (metrics["ebitda"][-1][1] / metrics["ebitda"][-2][1] - 1) if metrics["ebitda"][-2][1] > 0 else 0
        else:
            results["ebitda_growth"] = 0.0
    else:
        results["ebitda_margin"] = results["ebitda_growth"] = 0.0
    
    # Calculate margin trend
    if metrics["revenue"] and metrics["net_income"]:
        margins = []
        for year, rev in metrics["revenue"]:
            ni = next((v[1] for v in metrics["net_income"] if v[0] == year), None)
            if ni is not None and rev > 0:
                margins.append(ni / rev)
        
        if len(margins) >= 2:
            results["margin_trend"] = (margins[-1] - margins[0]) / len(margins)  # Average annual margin change
            results["net_margin"] = margins[-1]
        else:
            results["margin_trend"] = results["net_margin"] = 0.0
    else:
        results["margin_trend"] = results["net_margin"] = 0.0
    
    # Normalize and cap growth rates
    for key in results:
        if "growth" in key or "cagr" in key:
            results[key] = max(-0.5, min(1.0, results[key]))
            
    return results
        
    return {
        "revenue_growth": max(-0.5, min(1.0, yoy_growth)),
        "cagr": max(-0.5, min(1.0, cagr))
    }

def _growth_score_from_financials(session: Session, company: Company) -> float:
    """Calculate a comprehensive growth score using multiple metrics."""
    try:
        rows = session.query(Company).filter(Company.id == company.id).first()
        # we didn't eager-load financials in many contexts; try to use relationship
        fin_rows = company.financials
        if not fin_rows:
            return 0.0
        # look for any 'income' statement rows and compute simple growth
        growths = []
        for f in fin_rows:
            if f.statement_type and f.statement_type.lower().startswith("income"):
                values = f.data.get("values") if isinstance(f.data, dict) else {}
                # try to find a revenue-like field
                for k, v in values.items():
                    key = str(k).lower()
                    if "total" in key and "revenue" in key or "revenue" in key:
                        try:
                            growths.append(float(v))
                        except Exception:
                            pass
        if len(growths) < 2:
            return 0.0
        # naive growth: last / first - 1
        g = (growths[-1] / max(1.0, growths[0])) - 1.0
        # clamp and normalize (assume -0.5 .. +1.0 maps to 0..1)
        g = max(-0.5, min(1.0, g))
        return float((g + 0.5) / 1.5)
    except Exception as e:
        logger.debug("growth score lookup failed for %s: %s", getattr(company, 'ticker', None), e)
        return 0.0


def score_pair(acquirer: Company, target: Company, session: Session) -> Tuple[float, Dict[str, float]]:
    """Compute total compatibility score (0..100) and return sub-scores."""
    acq_cap = acquirer.market_cap or 0.0
    tgt_cap = target.market_cap or 0.0

    # Calculate basic scores
    size = _size_score(acq_cap, tgt_cap)
    sector = _sector_score(acquirer.sector or "", target.sector or "")
    
    # Get growth metrics for both companies
    acq_financials = _get_cached_financials(str(acquirer.id))
    tgt_financials = _get_cached_financials(str(target.id))
    
    acq_growth_metrics = _calculate_growth_metrics(acq_financials)
    tgt_growth_metrics = _calculate_growth_metrics(tgt_financials)
    
    # Calculate growth synergy score
    growth_synergy = max(0, min(1, 
        0.7 * tgt_growth_metrics["cagr"] +  # Higher weight on long-term growth
        0.3 * tgt_growth_metrics["revenue_growth"]  # Some weight on recent growth
    ))
    
    # Calculate market position score
    relative_size = tgt_cap / acq_cap if acq_cap > 0 else 0
    market_position = max(0, min(1, 
        1 - abs(0.3 - relative_size)  # Optimal target size is 30% of acquirer
    ))
    
    # Weights (tweakable)
    w_size = 0.4
    w_sector = 0.25
    w_growth = 0.2
    w_market = 0.15
    
    total = (w_size * size + 
             w_sector * sector + 
             w_growth * growth_synergy +
             w_market * market_position)
             
    return float(total * 100.0), {
        "size": size,
        "sector": sector,
        "growth_synergy": growth_synergy,
        "market_position": market_position,
        "target_cagr": tgt_growth_metrics["cagr"],
        "target_recent_growth": tgt_growth_metrics["revenue_growth"]
    }

    total = w_size * size + w_sector * sector + w_growth * growth
    return float(total * 100.0), {"size": size, "sector": sector, "growth": growth}


def generate_top_pairs(acquirer_ticker: str, top: int = 20) -> List[Dict[str, Any]]:
    """Generate and persist top target pairs for an acquirer ticker.

    Returns a list of dicts describing ranked targets.
    """
    session = SessionLocal()
    try:
        acquirer = session.query(Company).filter(Company.ticker == acquirer_ticker).first()
        if not acquirer:
            raise ValueError(f"Acquirer ticker {acquirer_ticker} not found")

        candidates = session.query(Company).filter(Company.id != acquirer.id).all()
        scored = []
        for tgt in candidates:
            score, subs = score_pair(acquirer, tgt, session)
            scored.append((tgt, score, subs))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_items = scored[:top]

        results = []
        for tgt, score, subs in top_items:
            # upsert DealPair record
            pair = session.query(DealPair).filter(DealPair.acquirer_id == acquirer.id, DealPair.target_id == tgt.id).first()
            if not pair:
                pair = DealPair(acquirer_id=acquirer.id, target_id=tgt.id, compatibility_score=score, metadata_json=subs)
                session.add(pair)
            else:
                pair.compatibility_score = score
                pair.metadata_json = subs
            session.commit()

            results.append({
                "acquirer": acquirer.ticker,
                "target": tgt.ticker,
                "target_name": tgt.name,
                "score": score,
                "subscores": subs,
                "target_market_cap": tgt.market_cap,
            })

        return results
    finally:
        session.close()
