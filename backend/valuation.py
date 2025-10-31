"""
Valuation engines for M&A analysis.
Includes DCF, Comps, Precedents, and ensemble methods.
"""
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime
import logging
from .models.models import Financial

logger = logging.getLogger(__name__)

def calculate_base_fcf(financials: List[Financial]) -> Dict[str, float]:
    """Calculate normalized base free cash flow and supporting metrics.
    
    Returns:
        Dict containing:
        - base_fcf: Normalized free cash flow
        - ebit_margin: Operating margin
        - capex_ratio: CapEx as % of revenue
        - fcf_margin: FCF as % of revenue
        - stability_score: Measure of FCF stability
    """
    if not financials:
        return {
            "base_fcf": 0.0,
            "ebit_margin": 0.0,
            "capex_ratio": 0.0,
            "fcf_margin": 0.0,
            "stability_score": 0.0
        }
    
    try:
        # Get last 3 years of data if available
        years_data = []
        for fin in financials[:3]:  # Most recent 3 years
            data = fin.data.get("values", {})
            
            revenue = float(data.get("Revenue", 0))
            ebit = float(data.get("Operating Income", 0))
            depreciation = float(data.get("Depreciation & Amortization", 0))
            capex = abs(float(data.get("Capital Expenditure", 0)))  # Make positive
            nwc_change = float(data.get("Change in Working Capital", 0))
            
            if revenue > 0:  # Only include years with valid revenue
                year_fcf = ebit * (1 - 0.25)  # Apply standard tax rate
                year_fcf += depreciation  # Add back non-cash charges
                year_fcf -= capex  # Subtract CapEx
                year_fcf -= nwc_change  # Adjust for working capital
                
                years_data.append({
                    "revenue": revenue,
                    "ebit": ebit,
                    "fcf": year_fcf,
                    "capex": capex
                })
        
        if not years_data:
            return {
                "base_fcf": 0.0,
                "ebit_margin": 0.0,
                "capex_ratio": 0.0,
                "fcf_margin": 0.0,
                "stability_score": 0.0
            }
            
        # Calculate normalized metrics
        avg_revenue = sum(y["revenue"] for y in years_data) / len(years_data)
        avg_ebit = sum(y["ebit"] for y in years_data) / len(years_data)
        avg_fcf = sum(y["fcf"] for y in years_data) / len(years_data)
        avg_capex = sum(y["capex"] for y in years_data) / len(years_data)
        
        # Calculate margins and ratios
        ebit_margin = avg_ebit / avg_revenue if avg_revenue > 0 else 0
        capex_ratio = avg_capex / avg_revenue if avg_revenue > 0 else 0
        fcf_margin = avg_fcf / avg_revenue if avg_revenue > 0 else 0
        
        # Calculate FCF stability score (0-1)
        fcf_values = [y["fcf"] for y in years_data]
        if len(fcf_values) > 1 and avg_fcf > 0:
            fcf_std = np.std(fcf_values)
            stability_score = 1.0 - min(1.0, fcf_std / abs(avg_fcf))
        else:
            stability_score = 0.0
            
        return {
            "base_fcf": max(0.0, avg_fcf),  # Floor at zero
            "ebit_margin": ebit_margin,
            "capex_ratio": capex_ratio,
            "fcf_margin": fcf_margin,
            "stability_score": stability_score
        }
        
    except Exception as e:
        logger.error(f"FCF calculation failed: {e}")
        return {
            "base_fcf": 0.0,
            "ebit_margin": 0.0,
            "capex_ratio": 0.0,
            "fcf_margin": 0.0,
            "stability_score": 0.0
        }

def project_cash_flows(
    base_metrics: Dict[str, float],
    growth_rate: float,
    years: int = 5
) -> Dict[str, Any]:
    """Project future cash flows using growth assumptions and base metrics.
    
    Incorporates:
    - Base FCF trends and stability
    - Operating leverage assumptions
    - Margin expansion/contraction
    - Normalized capex cycles
    
    Returns dict with:
    - projected_fcfs: List of projected cash flows
    - margin_trends: Expected margin evolution
    - capex_forecast: Capital expenditure forecast
    - growth_assumptions: Detailed growth assumptions
    """
    base_fcf = base_metrics["base_fcf"]
    stability = base_metrics["stability_score"]
    ebit_margin = base_metrics["ebit_margin"]
    capex_ratio = base_metrics["capex_ratio"]
    
    # Adjust growth rate based on stability
    effective_growth = growth_rate * (0.7 + 0.3 * stability)  # More conservative for unstable FCF
    
    # Model operating leverage
    if ebit_margin > 0:
        margin_expansion = min(0.02, growth_rate * 0.15)  # Margin expands with growth, capped
    else:
        margin_expansion = 0
        
    # Project FCFs with margin evolution
    fcfs = []
    margins = []
    capex = []
    
    for year in range(1, years + 1):
        # Calculate year's FCF with compounding growth
        year_growth = effective_growth * (1 - 0.02 * year)  # Slight decay in growth rate
        
        # Base FCF grows with effective growth rate
        fcf_before_margin = base_fcf * (1 + year_growth) ** year
        
        # Apply margin evolution
        year_margin = ebit_margin + margin_expansion * min(year, 3)  # Margin expansion phases in
        margins.append(year_margin)
        
        # Model capex cycles (higher in years 1-2, normalized later)
        if year <= 2:
            year_capex = capex_ratio * 1.2  # 20% higher capex in early years
        else:
            year_capex = capex_ratio
        capex.append(year_capex)
        
        # Final FCF incorporates margin and capex effects
        final_fcf = fcf_before_margin * (1 + (year_margin - ebit_margin))
        final_fcf = final_fcf * (1 - (year_capex - capex_ratio))  # Adjust for capex cycle
        
        fcfs.append(max(0.0, final_fcf))  # Floor at zero
    
    return {
        "projected_fcfs": fcfs,
        "margin_trends": margins,
        "capex_forecast": capex,
        "growth_assumptions": {
            "initial_growth": effective_growth,
            "terminal_growth": effective_growth * 0.5,  # More conservative terminal growth
            "margin_expansion": margin_expansion
        }
    }

def _calculate_margin_stability(financials: List[Financial]) -> float:
    """Calculate stability of profit margins over time."""
    try:
        margins = []
        for fin in financials:
            data = fin.data.get("values", {})
            revenue = float(data.get("Revenue", 0))
            operating_income = float(data.get("Operating Income", 0))
            
            if revenue > 0:
                margins.append(operating_income / revenue)
        
        if len(margins) >= 2:
            margin_std = np.std(margins)
            mean_margin = np.mean(margins)
            if mean_margin != 0:
                return 1.0 - min(1.0, margin_std / abs(mean_margin))
    except Exception as e:
        logger.error(f"Margin stability calculation failed: {e}")
    
    return 0.0

def _calculate_historical_growth(financials: List[Financial]) -> float:
    """Calculate historical revenue growth rate."""
    try:
        revenues = []
        for fin in financials:
            data = fin.data.get("values", {})
            revenue = float(data.get("Revenue", 0))
            if revenue > 0:
                revenues.append((fin.year, revenue))
        
        if len(revenues) >= 2:
            revenues.sort(key=lambda x: x[0])
            years = revenues[-1][0] - revenues[0][0]
            if years > 0:
                cagr = (revenues[-1][1] / revenues[0][1]) ** (1/years) - 1
                return max(-0.5, min(1.0, cagr))  # Cap between -50% and 100%
    except Exception as e:
        logger.error(f"Historical growth calculation failed: {e}")
    
    return 0.0

def calculate_dcf_confidence(
    financials: List[Financial],
    growth_rate: float,
    wacc: float
) -> Dict[str, float]:
    """Calculate detailed confidence metrics for DCF valuation.
    
    Evaluates multiple factors to determine reliability:
    1. Data Quality (30% weight)
       - Completeness of financial statements
       - Consistency of reporting
       - Historical depth
    
    2. FCF Stability (25% weight)
       - Historical FCF volatility
       - Margin consistency
       - Working capital stability
    
    3. Growth Credibility (25% weight)
       - Growth rate vs historical
       - Industry context
       - Market conditions
    
    4. Risk Assessment (20% weight)
       - WACC appropriateness
       - Business model stability
       - Market position
    """
    # 1. Data Quality Score
    data_quality = assess_data_completeness(financials)
    years_of_data = len(set(f.year for f in financials))
    historical_depth = min(1.0, years_of_data / 5)  # Prefer 5+ years
    
    # Calculate reporting consistency
    required_metrics = {"Revenue", "Operating Income", "Net Income", "Operating Cash Flow"}
    consistency_scores = []
    
    for fin in financials:
        available_metrics = set(fin.data.get("values", {}).keys())
        metric_coverage = len(required_metrics.intersection(available_metrics)) / len(required_metrics)
        consistency_scores.append(metric_coverage)
    
    reporting_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0
    
    data_quality_score = 0.4 * data_quality + 0.3 * historical_depth + 0.3 * reporting_consistency
    
    # 2. FCF Stability Score
    base_metrics = calculate_base_fcf(financials)
    fcf_stability = base_metrics["stability_score"]
    margin_stability = _calculate_margin_stability(financials)
    
    stability_score = 0.5 * fcf_stability + 0.5 * margin_stability
    
    # 3. Growth Credibility
    historical_growth = _calculate_historical_growth(financials)
    growth_deviation = 1.0 - min(1.0, abs(growth_rate - historical_growth) / max(abs(historical_growth), 0.05))
    
    # Assess if growth rate is reasonable given WACC
    growth_wacc_ratio = growth_rate / wacc if wacc > 0 else 0
    growth_reasonableness = 1.0 - min(1.0, max(0.0, growth_wacc_ratio - 0.8))  # Penalize if growth > 80% of WACC
    
    growth_score = 0.5 * growth_deviation + 0.5 * growth_reasonableness
    
    # 4. Risk Assessment
    # Check if WACC is reasonable (typically 8-15% for mature companies)
    wacc_reasonableness = 1.0 - min(1.0, abs(wacc - 0.115) / 0.115)  # Center around 11.5%
    
    # Business stability from margins
    business_stability = base_metrics["ebit_margin"] * (1 + base_metrics["stability_score"])
    
    risk_score = 0.5 * wacc_reasonableness + 0.5 * min(1.0, business_stability)
    
    # Calculate weighted final confidence score
    final_confidence = (
        0.30 * data_quality_score +
        0.25 * stability_score +
        0.25 * growth_score +
        0.20 * risk_score
    )
    
    return {
        "overall_confidence": final_confidence,
        "data_quality": data_quality_score,
        "stability": stability_score,
        "growth_credibility": growth_score,
        "risk_assessment": risk_score,
        "sub_metrics": {
            "historical_depth": historical_depth,
            "reporting_consistency": reporting_consistency,
            "fcf_stability": fcf_stability,
            "growth_deviation": growth_deviation,
            "wacc_reasonableness": wacc_reasonableness
        }
    }
    if not financials:
        return 0.0
    
    # Data completeness score
    completeness = len(financials) / 5.0  # Ideally want 5 years
    
    # Growth stability score
    if len(financials) >= 2:
        growths = []
        for i in range(len(financials)-1):
            try:
                current = float(financials[i].data.get("operating_income", 0))
                prev = float(financials[i+1].data.get("operating_income", 0))
                if prev > 0:
                    growths.append(abs((current/prev) - 1))
            except:
                continue
        growth_stability = 1.0 - min(1.0, np.std(growths) if growths else 1.0)
    else:
        growth_stability = 0.0
    
    # Parameter reasonableness
    growth_reasonable = 1.0 - min(1.0, abs(growth_rate) / 0.2)  # Penalize high growth
    wacc_reasonable = 1.0 - min(1.0, abs(wacc - 0.1) / 0.1)  # Center around 10%
    
    # Weighted average
    weights = [0.4, 0.3, 0.15, 0.15]
    components = [completeness, growth_stability, growth_reasonable, wacc_reasonable]
    
    return sum(w * c for w, c in zip(weights, components))

def generate_dcf_sensitivity_grid(
    base_fcf: float,
    growth_rate: float,
    wacc: float,
    terminal_growth: float,
    growth_delta: float = 0.01,
    wacc_delta: float = 0.01
) -> Dict[str, Any]:
    """Generate sensitivity analysis grid varying growth and WACC."""
    growth_rates = [
        growth_rate + i * growth_delta
        for i in range(-2, 3)
    ]
    
    wacc_rates = [
        wacc + i * wacc_delta
        for i in range(-2, 3)
    ]
    
    grid = []
    for g in growth_rates:
        row = []
        for w in wacc_rates:
            # Simple 5-year DCF for sensitivity
            fcfs = project_cash_flows(base_fcf, g, 5)
            terminal = fcfs[-1] * (1 + terminal_growth) / (w - terminal_growth)
            dfs = [(1 + w) ** -i for i in range(1, 6)]
            value = sum(fcf * df for fcf, df in zip(fcfs, dfs)) + terminal * dfs[-1]
            row.append(round(value, 1))
        grid.append(row)
    
    return {
        "growth_rates": growth_rates,
        "wacc_rates": wacc_rates,
        "values": grid
    }

def assess_data_completeness(financials: List[Financial]) -> Dict[str, Any]:
    """Assess completeness and quality of financial data."""
    required_fields = [
        "operating_income",
        "revenue",
        "depreciation",
        "capital_expenditure",
        "change_in_working_capital"
    ]
    
    completeness = {}
    if not financials:
        return {"score": 0.0, "missing_fields": required_fields}
    
    missing = set()
    for field in required_fields:
        found = False
        for f in financials:
            if field in f.data:
                found = True
                break
        if not found:
            missing.add(field)
    
    years_available = len(financials)
    field_completeness = (len(required_fields) - len(missing)) / len(required_fields)
    
    return {
        "score": min(years_available / 5.0, 1.0) * field_completeness,
        "years_available": years_available,
        "missing_fields": list(missing)
    }