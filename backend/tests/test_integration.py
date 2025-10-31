"""Integration tests for the M&A analysis platform."""
import unittest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
import sys
import pytest
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.db import Base
from backend.models.models import Company, Financial, DealPair, Valuation
from backend.metrics import calculate_wacc, calculate_growth_rates
from backend.valuation import (
    calculate_base_fcf,
    project_cash_flows,
    calculate_dcf_confidence
)

# Use SQLite for testing
TEST_DB_URL = "sqlite:///./test_ma_deals.db"

class TestMAAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create test database
        cls.engine = create_engine(TEST_DB_URL)
        Base.metadata.create_all(cls.engine)
        
        # Create session
        Session = sessionmaker(bind=cls.engine)
        cls.session = Session()
        
        # Create test data
        cls.setup_test_data()
    
    @classmethod
    def tearDownClass(cls):
        cls.session.close()
        os.remove("./test_ma_deals.db")
    
    @classmethod
    def setup_test_data(cls):
        # Create test companies
        acquirer = Company(
            ticker="AAPL",
            name="Apple Inc",
            sector="Technology",
            market_cap=2500000000000.0
        )
        
        target = Company(
            ticker="SPOT",
            name="Spotify Technology SA",
            sector="Technology",
            market_cap=50000000000.0
        )
        
        cls.session.add(acquirer)
        cls.session.add(target)
        
        # Add financial data
        financials = [
            Financial(
                company=target,
                statement_type="income",
                period="annual",
                year=2022,
                data={
                    "revenue": 10000000000,
                    "operating_income": 1000000000,
                    "depreciation": 100000000,
                    "ebitda": 1100000000
                }
            ),
            Financial(
                company=target,
                statement_type="income",
                period="annual",
                year=2021,
                data={
                    "revenue": 8000000000,
                    "operating_income": 800000000,
                    "depreciation": 80000000,
                    "ebitda": 880000000
                }
            )
        ]
        
        cls.session.add_all(financials)
        
        # Create deal pair
        pair = DealPair(
            acquirer=acquirer,
            target=target,
            compatibility_score=85.5
        )
        
        cls.session.add(pair)
        cls.session.commit()
        
        # Store references
        cls.acquirer = acquirer
        cls.target = target
        cls.pair = pair
        cls.financials = financials
    
    def test_dcf_valuation(self):
        """Test DCF valuation calculation."""
        # Calculate base FCF
        base_fcf = calculate_base_fcf(self.financials)
        self.assertGreater(base_fcf, 0)
        
        # Test cash flow projection
        growth_rate = 0.05
        years = 5
        fcfs = project_cash_flows(base_fcf, growth_rate, years)
        self.assertEqual(len(fcfs), years)
        self.assertGreater(fcfs[-1], fcfs[0])
        
        # Test confidence calculation
        confidence = calculate_dcf_confidence(self.financials, growth_rate, 0.1)
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)
    
    def test_metrics_calculation(self):
        """Test financial metrics calculations."""
        # Test WACC calculation
        wacc = calculate_wacc(self.target)
        self.assertGreater(wacc, 0)
        self.assertLess(wacc, 0.2)  # Should be reasonable
        
        # Test growth rate calculations
        growth_metrics = calculate_growth_rates(self.financials)
        self.assertIn("revenue_cagr", growth_metrics)
        self.assertGreater(growth_metrics["revenue_cagr"], 0)
    
    def test_deal_scoring(self):
        """Test deal compatibility scoring."""
        self.assertGreater(self.pair.compatibility_score, 0)
        self.assertLessEqual(self.pair.compatibility_score, 100)

if __name__ == '__main__':
    unittest.main()