"""Generate professional Deal Brief PDFs."""
from typing import Dict, Any
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak
)
from reportlab.graphics.shapes import Drawing, Line
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.barcharts import VerticalBarChart
import os
import tempfile
from datetime import datetime

def format_currency(value: float) -> str:
    """Format currency values with appropriate scale (B/M)."""
    if abs(value) >= 1e9:
        return f"${value/1e9:.1f}B"
    elif abs(value) >= 1e6:
        return f"${value/1e6:.1f}M"
    else:
        return f"${value:,.0f}"

def _create_chart_style() -> dict:
    """Create consistent chart styling."""
    return {
        'colors': [
            colors.HexColor('#4f8cff'),  # Primary blue
            colors.HexColor('#8a5cff'),  # Secondary purple
            colors.HexColor('#4CAF50'),  # Success green
            colors.HexColor('#FFA726'),  # Warning orange
            colors.HexColor('#EF5350')   # Error red
        ],
        'grid': colors.HexColor('#f0f0f0'),
        'line_width': 2,
        'background': colors.HexColor('#ffffff'),
        'font': 'Helvetica',
        'font_size': 8,
        'label_color': colors.HexColor('#666666')
    }

def _format_percent(value: float) -> str:
    """Format percentage values."""
    return f"{value*100:.1f}%"

def create_deal_brief(
    deal_data: Dict[str, Any],
    output_path: str
) -> None:
    """Generate a professional Deal Brief PDF with advanced styling and charts."""
    # Set up the document with better margins and styling
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=60,
        leftMargin=60,
        topMargin=48,
        bottomMargin=48
    )
    
    # Enhanced style system
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=32,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=20
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=24,
        textColor=colors.HexColor('#34495e'),
        spaceBefore=20,
        spaceAfter=12
    )
    
    section_style = ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#2c3e50'),
        spaceBefore=16,
        spaceAfter=8
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#2c3e50')
    )
    
    label_style = ParagraphStyle(
        'Label',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#7f8c8d')
    )
    
    # Build the document content
    elements = []
    
    # Header with deal overview
    elements.append(Paragraph(
        f"M&A Deal Brief: {deal_data['acquirer']} → {deal_data['target']}",
        title_style
    ))
    
    # Deal Summary Table
    summary_data = [
        ["Enterprise Value", format_currency(deal_data['enterprise_value'])],
        ["Revenue Multiple", f"{deal_data.get('revenue_multiple', 0):.1f}x"],
        ["EBITDA Multiple", f"{deal_data.get('ebitda_multiple', 0):.1f}x"],
        ["Deal Confidence", _format_percent(deal_data['confidence'])]
    ]
    
    summary_table = Table(
        summary_data,
        colWidths=[2*inch, 1.5*inch],
        style=TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fa')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('LEFTPADDING', (0, 0), (-1, -1), 16),
            ('RIGHTPADDING', (0, 0), (-1, -1), 16),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6'))
        ])
    )
    elements.append(summary_table)
    elements.append(Spacer(1, 20))
    
    # Strategic Rationale
    elements.append(Paragraph("Strategic Rationale", subtitle_style))
    elements.append(Paragraph(
        "• Market Position: Strengthening competitive position through complementary capabilities<br/>"
        "• Revenue Synergies: Cross-selling opportunities and expanded market reach<br/>"
        "• Cost Synergies: Operational efficiency and economies of scale<br/>"
        "• Technology: Enhanced digital capabilities and innovation potential",
        body_style
    ))
    elements.append(Spacer(1, 20))
    
    # Add charts and detailed analysis sections
    chart_style = _create_chart_style()
    
    # Valuation Analysis
    elements.append(Paragraph("Valuation Analysis", subtitle_style))
    
    # Add DCF assumptions
    if 'assumptions' in deal_data:
        elements.append(Paragraph("Key Assumptions", section_style))
        assumptions = [
            ["Metric", "Value"],
            ["Growth Rate", _format_percent(deal_data['assumptions']['growth_rate'])],
            ["WACC", _format_percent(deal_data['assumptions']['wacc'])],
            ["Terminal Growth", _format_percent(deal_data['assumptions']['terminal_growth'])]
        ]
        
        assumptions_table = Table(
            assumptions,
            colWidths=[2*inch, 1.5*inch],
            style=TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f8f9fa')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('LEFTPADDING', (0, 0), (-1, -1), 12),
                ('RIGHTPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6'))
            ])
        )
        elements.append(assumptions_table)
        elements.append(Spacer(1, 20))
    
    # Cash Flow Projections Chart
    if 'projections' in deal_data and 'fcfs' in deal_data['projections']:
        elements.append(Paragraph("Projected Free Cash Flows", section_style))
        
        drawing = Drawing(400, 200)
        
        chart = HorizontalLineChart()
        chart.x = 50
        chart.y = 50
        chart.height = 125
        chart.width = 300
        
        fcfs = deal_data['projections']['fcfs']
        
        chart.data = [fcfs]
        chart.lines[0].strokeWidth = chart_style['line_width']
        chart.lines[0].strokeColor = chart_style['colors'][0]
        
        # Configure axes
        years = [str(i+1) for i in range(len(fcfs))]
        chart.categoryAxis.categoryNames = years
        chart.categoryAxis.style = 'sticks'
        chart.categoryAxis.strokeWidth = 1
        chart.categoryAxis.strokeColor = chart_style['grid']
        chart.categoryAxis.labelTextFormat = lambda x: f'Year {x}'
        
        # Format value axis
        max_fcf = max(fcfs)
        chart.valueAxis.valueMin = 0
        chart.valueAxis.valueMax = max_fcf * 1.2
        chart.valueAxis.valueStep = max_fcf / 5
        chart.valueAxis.labelTextFormat = lambda x: format_currency(x)
        chart.valueAxis.strokeWidth = 1
        chart.valueAxis.strokeColor = chart_style['grid']
        
        # Add grid
        chart.gridLines = [(0, 0, 0, 1), (0, 0, -1, 0)]
        chart.gridStrokeColor = chart_style['grid']
        chart.gridStrokeWidth = 0.5
        
        drawing.add(chart)
        elements.append(drawing)
        elements.append(Spacer(1, 20))
    
    # Sensitivity Analysis
    if 'sensitivity' in deal_data:
        elements.append(Paragraph("Valuation Sensitivity Analysis", section_style))
        elements.append(Paragraph(
            "Impact of Growth Rate and WACC on Enterprise Value",
            label_style
        ))
        
        sensitivity = deal_data['sensitivity']
        if 'values' in sensitivity and isinstance(sensitivity['values'], list):
            data = [[format_currency(v) for v in row] for row in sensitivity['values']]
            wacc_range = sensitivity.get('wacc_range', [f"WACC {i}%" for i in range(len(data[0]))])
            growth_range = sensitivity.get('growth_range', [f"Growth {i}%" for i in range(len(data))])
            
            # Add headers
            data.insert(0, [''] + [f"WACC: {w}" for w in wacc_range])
            for i, row in enumerate(data[1:], 1):
                row.insert(0, f"Growth: {growth_range[i-1]}")
            
            sensitivity_table = Table(
                data,
                style=TableStyle([
                    ('BACKGROUND', (1, 1), (-1, -1), colors.HexColor('#f8f9fa')),
                    ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f8f9fa')),
                    ('BACKGROUND', (1, 0), (-1, 0), colors.HexColor('#f8f9fa')),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6'))
                ])
            )
            elements.append(sensitivity_table)
    
    # Add confidence metrics breakdown
    if 'confidence_metrics' in deal_data:
        elements.append(PageBreak())
        elements.append(Paragraph("Valuation Confidence Analysis", subtitle_style))
        
        metrics = deal_data['confidence_metrics']
        confidence_data = [
            ["Metric", "Score", "Details"],
            ["Data Quality", _format_percent(metrics.get('data_quality', 0)), 
             "Assessment of financial data completeness and reliability"],
            ["FCF Stability", _format_percent(metrics.get('stability', 0)),
             "Historical cash flow consistency and predictability"],
            ["Growth Credibility", _format_percent(metrics.get('growth_credibility', 0)),
             "Reasonableness of growth assumptions"],
            ["Risk Assessment", _format_percent(metrics.get('risk_assessment', 0)),
             "Evaluation of key risk factors and mitigants"]
        ]
        
        confidence_table = Table(
            confidence_data,
            colWidths=[1.5*inch, 1*inch, 3*inch],
            style=TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f8f9fa')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6'))
            ])
        )
        elements.append(confidence_table)
    
    # Build the PDF
    doc.build(elements)
    
def _format_percent(value: float) -> str:
    """Format percentage values."""
    return f"{value*100:.1f}%"
def generate_deal_brief(deal_data: Dict[str, Any], output_path: str) -> None:
    """Generate a professional PDF deal brief."""
    # Create document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )

    # Setup styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Title'],
        fontSize=24,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Heading2'],
        fontSize=18,
        spaceAfter=20,
        spaceBefore=20,
        alignment=0  # Left alignment
    )
    section_style = ParagraphStyle(
        'Section',
        parent=styles['Heading3'],
        fontSize=14,
        spaceAfter=15,
        spaceBefore=15
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=10
    )
    label_style = ParagraphStyle(
        'Label',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#666666'),
        spaceAfter=5
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=20,
        spaceAfter=12
    )
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        spaceBefore=6
    )
    
    # Build content
    story = []
    
    # Header
    story.append(Paragraph(
        f"Deal Brief: {deal_data['acquirer']} → {deal_data['target']}",
        title_style
    ))
    story.append(Paragraph(
        f"Generated on {datetime.now().strftime('%B %d, %Y')}",
        styles['Italic']
    ))
    story.append(Spacer(1, 20))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    story.append(Paragraph(
        deal_data.get('executive_summary', ''),
        body_style
    ))
    
    # Strategic Rationale
    story.append(Paragraph("Strategic Rationale", heading_style))
    rationale = deal_data.get('strategic_rationale', {})
    for point in rationale.get('key_points', []):
        story.append(Paragraph(f"• {point}", body_style))
    
    # Valuation Summary
    story.append(Paragraph("Valuation Summary", heading_style))
    valuation = deal_data.get('valuation', {})
    
    # Create valuation summary table
    valuation_data = [
        ['Method', 'Value', 'Weight', 'Confidence'],
        ['DCF', format_currency(valuation.get('dcf_value', 0)), '40%', f"{valuation.get('dcf_confidence', 0)*100:.0f}%"],
        ['Comps', format_currency(valuation.get('comps_value', 0)), '40%', f"{valuation.get('comps_confidence', 0)*100:.0f}%"],
        ['Precedent', format_currency(valuation.get('precedent_value', 0)), '20%', f"{valuation.get('precedent_confidence', 0)*100:.0f}%"],
    ]
    
    t = Table(valuation_data, colWidths=[100, 100, 70, 70])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(t)
    story.append(Spacer(1, 20))
    
    # Financial Projections
    story.append(Paragraph("Financial Projections", heading_style))
    projections = deal_data.get('projections', {})
    
    # Create projections chart
    if projections.get('years') and projections.get('values'):
        drawing = Drawing(400, 200)
        chart = HorizontalLineChart()
        chart.x = 50
        chart.y = 50
        chart.height = 125
        chart.width = 300
        chart.data = [projections['values']]
        chart.categoryAxis.categoryNames = projections['years']
        drawing.add(chart)
        story.append(drawing)
    
    # Risk Factors
    story.append(Paragraph("Risk Factors", heading_style))
    risks = deal_data.get('risks', [])
    for risk in risks:
        story.append(Paragraph(f"• {risk['description']}", body_style))
        story.append(Paragraph(f"  Mitigation: {risk['mitigation']}", styles['Italic']))
    
    # Build the PDF
    doc.build(story)

def generate_deal_brief(deal_data: Dict[str, Any]) -> str:
    """Generate a Deal Brief PDF and return the file path."""
    # Create temp file
    fd, path = tempfile.mkstemp(suffix='.pdf')
    os.close(fd)
    
    # Generate PDF
    create_deal_brief(deal_data, path)
    
    return path