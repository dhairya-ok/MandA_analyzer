import uuid
from sqlalchemy import Column, String, Float, Integer, ForeignKey, UniqueConstraint, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from ..db import Base


class Company(Base):
	__tablename__ = "companies"

	id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
	ticker = Column(String(16), unique=True, nullable=False, index=True)
	name = Column(String(256), nullable=False)
	sector = Column(String(128), index=True)
	market_cap = Column(Float)

	financials = relationship("Financial", back_populates="company")
	acquirer_pairs = relationship("DealPair", foreign_keys="DealPair.acquirer_id", back_populates="acquirer")
	target_pairs = relationship("DealPair", foreign_keys="DealPair.target_id", back_populates="target")


class Financial(Base):
	__tablename__ = "financials"

	id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
	company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False, index=True)
	statement_type = Column(String(32), nullable=False)  # income, balance, cashflow
	period = Column(String(16), nullable=False)  # annual, quarterly
	year = Column(Integer, nullable=False)
	quarter = Column(Integer)
	data = Column(JSON, nullable=False)

	company = relationship("Company", back_populates="financials")
	__table_args__ = (
		UniqueConstraint("company_id", "statement_type", "period", "year", "quarter", name="uq_financials_unique_period"),
	)


class DealPair(Base):
	__tablename__ = "deal_pairs"

	id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
	acquirer_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False, index=True)
	target_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False, index=True)
	compatibility_score = Column(Float, index=True)
	metadata_json = Column(JSON, nullable=True)

	acquirer = relationship("Company", foreign_keys=[acquirer_id], back_populates="acquirer_pairs")
	target = relationship("Company", foreign_keys=[target_id], back_populates="target_pairs")
	__table_args__ = (
		UniqueConstraint("acquirer_id", "target_id", name="uq_unique_pair"),
	)


class Valuation(Base):
    __tablename__ = "valuations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pair_id = Column(UUID(as_uuid=True), ForeignKey("deal_pairs.id"), nullable=False, index=True)
    
    # Different valuation methods
    dcf_value = Column(Float)
    comps_value = Column(Float)
    precedent_value = Column(Float)
    lbo_value = Column(Float)
    ddm_value = Column(Float)
    nav_value = Column(Float)
    ensemble_value = Column(Float)
    
    # Confidence scores for each method
    confidence_scores = Column(JSON)  # {method: score}
    
    # Model inputs and assumptions
    dcf_assumptions = Column(JSON)  # growth rates, margins, wacc
    comps_assumptions = Column(JSON)  # peer group, multiples
    synergy_assumptions = Column(JSON)  # revenue/cost synergies
    
    # Sensitivity analysis
    sensitivity_grid = Column(JSON)  # wacc vs growth matrix
    scenario_analysis = Column(JSON)  # conservative/base/optimistic
    
    # Accretion/Dilution
    accretion_analysis = Column(JSON)  # eps impact years 1-3
    
    # Deal rationale and narrative
    strategic_rationale = Column(JSON)  # AI generated rationale
    risk_factors = Column(JSON)  # Key risks and mitigations
    
    # Metadata
    last_updated = Column(Float)  # timestamp
    data_completeness = Column(JSON)  # missing data flags
    
    pair = relationship("DealPair")
