"""Deal metrics and data handlers."""
from typing import Dict, Any, List, Optional
from datetime import datetime
import yfinance as yf
import numpy as np
from scipy.stats import linregress
from .models.models import Company, Financial

def fetch_market_data(ticker: str, period: str = "5y") -> Dict[str, Any]:
    """Fetch market data for a company using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        # Calculate key metrics
        returns = hist['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        beta = calculate_beta(returns)
        
        market_data = {
            "current_price": float(hist['Close'][-1]),
            "volume": float(hist['Volume'][-1]),
            "volatility": float(volatility),
            "beta": float(beta),
            "price_history": hist['Close'].tolist(),
            "volume_history": hist['Volume'].tolist(),
            "dates": [str(d) for d in hist.index],
        }
        
        # Get additional info
        info = stock.info
        market_data.update({
            "market_cap": info.get("marketCap"),
            "enterprise_value": info.get("enterpriseValue"),
            "pe_ratio": info.get("forwardPE"),
            "dividend_yield": info.get("dividendYield"),
        })
        
        return market_data
    except Exception as e:
        print(f"Error fetching market data for {ticker}: {e}")
        return {}

def calculate_beta(returns: np.ndarray, market_returns: Optional[np.ndarray] = None) -> float:
    """Calculate beta relative to market (S&P 500 by default)."""
    if market_returns is None:
        spy = yf.Ticker("SPY")
        market_hist = spy.history(period="5y")
        market_returns = market_hist['Close'].pct_change().dropna()
    
    # Ensure same length
    min_len = min(len(returns), len(market_returns))
    returns = returns[-min_len:]
    market_returns = market_returns[-min_len:]
    
    # Calculate beta using linear regression
    slope, _, r_value, _, _ = linregress(market_returns, returns)
    return slope * (r_value ** 2)  # Adjusted for R-squared

def calculate_wacc(company: Company, risk_free_rate: float = 0.04) -> float:
    """Calculate Weighted Average Cost of Capital."""
    try:
        # Get market data
        market_data = fetch_market_data(company.ticker)
        beta = market_data.get("beta", 1.0)
        
        # Assume market risk premium of 6%
        market_risk_premium = 0.06
        
        # Cost of equity using CAPM
        cost_of_equity = risk_free_rate + beta * market_risk_premium
        
        # Simple cost of debt (could be enhanced with bond yields)
        cost_of_debt = risk_free_rate + 0.02  # Assume 2% spread
        
        # Assume 30% debt ratio for now
        debt_ratio = 0.3
        tax_rate = 0.25
        
        # Calculate WACC
        wacc = (cost_of_equity * (1 - debt_ratio) + 
                cost_of_debt * (1 - tax_rate) * debt_ratio)
        
        return float(wacc)
    except Exception:
        return 0.10  # Default to 10% if calculation fails

def calculate_growth_rates(financials: List[Financial]) -> Dict[str, float]:
    """Calculate historical growth rates from financial statements."""
    try:
        revenues = []
        ebitda = []
        years = []
        
        for f in sorted(financials, key=lambda x: x.year):
            if f.statement_type.lower().startswith('income'):
                data = f.data or {}
                rev = data.get('revenue', 0)
                eb = data.get('ebitda', 0)
                
                if rev and eb:
                    revenues.append(float(rev))
                    ebitda.append(float(eb))
                    years.append(f.year)
        
        if len(revenues) >= 2:
            # Calculate CAGR
            revenue_cagr = (revenues[-1] / revenues[0]) ** (1 / len(revenues)) - 1
            ebitda_cagr = (ebitda[-1] / ebitda[0]) ** (1 / len(ebitda)) - 1
            
            # Calculate year-over-year growth rates
            rev_yoy = [(revenues[i] / revenues[i-1] - 1) for i in range(1, len(revenues))]
            ebitda_yoy = [(ebitda[i] / ebitda[i-1] - 1) for i in range(1, len(ebitda))]
            
            return {
                "revenue_cagr": float(revenue_cagr),
                "ebitda_cagr": float(ebitda_cagr),
                "revenue_yoy": [float(r) for r in rev_yoy],
                "ebitda_yoy": [float(e) for e in ebitda_yoy],
                "years": years
            }
    except Exception as e:
        print(f"Error calculating growth rates: {e}")
    
    return {
        "revenue_cagr": 0.0,
        "ebitda_cagr": 0.0,
        "revenue_yoy": [],
        "ebitda_yoy": [],
        "years": []
    }