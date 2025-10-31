from sqlalchemy import Index, text
from .models.models import Company, Financial, DealPair, Valuation

def create_indexes():
    """Create database indexes for better query performance."""
    # Company table indexes
    Index('idx_company_market_cap', Company.market_cap)
    Index('idx_company_sector_market_cap', Company.sector, Company.market_cap)
    
    # Financials table indexes
    Index('idx_financials_company_type', 
          Financial.company_id, 
          Financial.statement_type)
    Index('idx_financials_year_quarter', 
          Financial.year, 
          Financial.quarter)
    
    # DealPair table compound indexes
    Index('idx_dealpair_scores', 
          DealPair.compatibility_score.desc())
    Index('idx_dealpair_acquirer_target', 
          DealPair.acquirer_id, 
          DealPair.target_id)
    
    # Valuation table indexes
    Index('idx_valuation_scores',
          Valuation.dcf_value,
          Valuation.comps_value,
          Valuation.ensemble_value)