import React from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from 'recharts';
import LoadingSpinner from './LoadingSpinner';
import { downloadDealBrief } from '../utils/pdf';

interface DealBriefProps {
  pairId: string;
  acquirer: string;
  target: string;
  loading: boolean;
  data: any;
}

const DealBrief: React.FC<DealBriefProps> = ({ pairId, acquirer, target, loading, data }) => {
  if (loading) {
    return (
      <div className="deal-brief loading">
        <LoadingSpinner />
        <p>Generating Deal Brief...</p>
      </div>
    );
  }

  const {
    enterprise_value,
    confidence,
    assumptions,
    projections,
    sensitivity
  } = data || {};

  const chartData = projections?.fcfs?.map((fcf: number, i: number) => ({
    year: `Year ${i + 1}`,
    fcf
  })) || [];

  return (
    <div className="deal-brief">
      <header className="brief-header">
        <div className="brief-header-content">
          <h2>Deal Brief: {acquirer} ⟶ {target}</h2>
          <div className="confidence-badge" style={{
            backgroundColor: confidence > 0.7 ? '#4CAF50' : confidence > 0.4 ? '#FFA726' : '#EF5350'
          }}>
            Confidence: {Math.round(confidence * 100)}%
          </div>
        </div>
        <button className="download-button" onClick={async () => {
          try {
            await downloadDealBrief(pairId);
          } catch (error) {
            console.error('Error downloading deal brief:', error);
          }
        }}>
          Download PDF
        </button>
      </header>

      <div className="grid-container">
        <div className="card valuation-summary">
          <h3>Valuation Summary</h3>
          <div className="value-box">
            <span className="label">Enterprise Value</span>
            <span className="value">${(enterprise_value / 1e9).toFixed(2)}B</span>
          </div>
          <div className="assumptions-list">
            <h4>Key Assumptions</h4>
            <ul>
              <li>Growth Rate: {(assumptions?.growth_rate * 100).toFixed(1)}%</li>
              <li>WACC: {(assumptions?.wacc * 100).toFixed(1)}%</li>
              <li>Terminal Growth: {(assumptions?.terminal_growth * 100).toFixed(1)}%</li>
            </ul>
          </div>
        </div>

        <div className="card projections-chart">
          <h3>Projected Free Cash Flows</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="year" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="fcf" stroke="#8884d8" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="card sensitivity-analysis">
          <h3>Sensitivity Analysis</h3>
          <div className="sensitivity-grid">
            {sensitivity?.values?.map((row: number[], i: number) => (
              <div key={i} className="sensitivity-row">
                {row.map((value, j) => (
                  <div key={j} className="sensitivity-cell" style={{
                    backgroundColor: `rgba(33, 150, 243, ${value / Math.max(...row)})`
                  }}>
                    ${(value / 1e9).toFixed(1)}B
                  </div>
                ))}
              </div>
            ))}
          </div>
          <div className="grid-labels">
            <div>Growth Rate →</div>
            <div>← WACC →</div>
          </div>
        </div>
      </div>

      <div className="strategic-fit">
        <h3>Strategic Rationale</h3>
        <ul className="rationale-list">
          <li>Market expansion opportunity</li>
          <li>Complementary product portfolios</li>
          <li>Operational synergies potential</li>
          <li>Technology integration benefits</li>
        </ul>
      </div>
    </div>
  );
};

export default DealBrief;