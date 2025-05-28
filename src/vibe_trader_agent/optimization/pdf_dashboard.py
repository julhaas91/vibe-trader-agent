"""PDF Dashboard Generator for Portfolio Optimization Results.

This module creates comprehensive PDF reports from portfolio optimization results,
combining charts, metrics, and analysis into a professional dashboard format.
"""

import base64
import datetime
import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


class PDFDashboardGenerator:
    """Generate comprehensive PDF dashboards from portfolio optimization results.
    """
    
    def __init__(self, 
                 title: str = "Portfolio Optimization Dashboard",
                 subtitle: str = "Investment Analysis Report",
                 company_name: str = "Vibe Trader",
                 logo_path: Optional[str] = None):
        """Initialize the PDF dashboard generator.
        
        Args:
            title: Main title for the dashboard
            subtitle: Subtitle for the dashboard
            company_name: Company/organization name
            logo_path: Path to logo image file (optional)
        """
        self.title = title
        self.subtitle = subtitle
        self.company_name = company_name
        self.logo_path = logo_path
        
        # Set up styling
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
        # Color scheme
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#17becf',
            'light_gray': '#f8f9fa',
            'dark_gray': '#343a40'
        }
        
    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1f77b4')
        )
        
        self.subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#666666')
        )
        
        self.section_style = ParagraphStyle(
            'SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.HexColor('#1f77b4')
        )
        
        self.metric_style = ParagraphStyle(
            'MetricText',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6
        )

    def create_enhanced_charts(self, results: Dict, output_dir: str) -> Dict[str, str]:
        """Create enhanced charts for the PDF dashboard.
        
        Args:
            results: Optimization results dictionary
            output_dir: Directory to save enhanced charts
            
        Returns:
            Dictionary mapping chart names to file paths
        """
        chart_paths = {}
        
        # Set style for professional charts
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # 1. Executive Summary Chart (Key Metrics)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Portfolio Optimization Summary', fontsize=16, fontweight='bold')
        
        # Success probability gauge
        success_prob = results['results']['success_prob']
        ax1.pie([success_prob, 1-success_prob], labels=['Success', 'Risk'], 
                colors=['#2ca02c', '#d62728'], startangle=90, autopct='%1.1f%%')
        ax1.set_title(f'Success Probability\n{success_prob:.1%}', fontweight='bold')
        
        # Risk metrics
        metrics = ['Volatility', 'Max Drawdown', 'Worst Day']
        values = [results['results']['volatility'], 
                 results['results']['avg_drawdown'],
                 results['results']['avg_worst_day']]
        limits = [results['inputs']['sigma_max'],
                 results['inputs']['max_drawdown'], 
                 results['inputs']['worst_day_limit']]
        
        x_pos = np.arange(len(metrics))
        bars = ax2.bar(x_pos, values, color=['#ff7f0e', '#d62728', '#9467bd'], alpha=0.7)
        ax2.plot(x_pos, limits, 'ro-', linewidth=2, markersize=8, label='Limits')
        ax2.set_xlabel('Risk Metrics')
        ax2.set_ylabel('Value')
        ax2.set_title('Risk Profile vs Constraints', fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(metrics, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Portfolio value projection
        start_val = results['inputs']['start_portfolio']
        target_val = results['inputs']['target_portfolio']
        avg_final = results['results']['avg_final']
        
        categories = ['Start', 'Target', 'Expected']
        values_proj = [start_val, target_val, avg_final]
        colors_proj = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        bars = ax3.bar(categories, values_proj, color=colors_proj, alpha=0.7)
        ax3.set_ylabel('Portfolio Value ($)')
        ax3.set_title('Value Projection', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Asset allocation pie chart
        weights = results['results']['weights']
        tickers = results['inputs']['tickers']
        
        # Only show assets with >1% allocation
        significant_weights = [(w, t) for w, t in zip(weights, tickers) if w > 0.01]
        if len(significant_weights) < len(weights):
            other_weight = sum(w for w, t in zip(weights, tickers) if w <= 0.01)
            significant_weights.append((other_weight, 'Other'))
        
        weights_sig, tickers_sig = zip(*significant_weights)
        ax4.pie(weights_sig, labels=tickers_sig, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Asset Allocation', fontweight='bold')
        
        plt.tight_layout()
        chart_path = os.path.join(output_dir, 'executive_summary.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths['executive_summary'] = chart_path
        
        # 2. Enhanced Asset Allocation Chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Horizontal bar chart
        y_pos = np.arange(len(tickers))
        bars = ax1.barh(y_pos, weights, color=sns.color_palette("husl", len(tickers)))
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(tickers)
        ax1.set_xlabel('Weight')
        ax1.set_title('Portfolio Weights by Asset', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add percentage labels
        for i, (bar, weight) in enumerate(zip(bars, weights)):
            ax1.text(weight + 0.01, i, f'{weight:.1%}', va='center')
        
        # Risk contribution analysis (simplified)
        vol = results['results']['volatility']
        risk_contrib = [w * vol for w in weights]  # Simplified risk contribution
        
        bars2 = ax2.barh(y_pos, risk_contrib, color=sns.color_palette("viridis", len(tickers)))
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(tickers)
        ax2.set_xlabel('Risk Contribution')
        ax2.set_title('Risk Contribution by Asset', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        chart_path = os.path.join(output_dir, 'asset_allocation_detailed.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths['asset_allocation'] = chart_path
        
        # 3. Performance Metrics Dashboard
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Performance Analysis', fontsize=16, fontweight='bold')
        
        # Return vs Risk scatter (single point for this portfolio)
        ax1.scatter(results['results']['volatility'], 
                   (avg_final/start_val)**(1/results['inputs']['horizon_years']) - 1,
                   s=200, c='red', alpha=0.7, edgecolors='black', linewidth=2)
        ax1.set_xlabel('Volatility (Annual)')
        ax1.set_ylabel('Expected Annual Return')
        ax1.set_title('Risk-Return Profile', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        
        # Constraint compliance
        constraints = ['Volatility', 'Drawdown', 'Worst Day', 'Cash Min']
        actual_values = [results['results']['volatility'],
                        results['results']['avg_drawdown'],
                        results['results']['avg_worst_day'],
                        results['results']['cash_allocation']]
        constraint_limits = [results['inputs']['sigma_max'],
                           results['inputs']['max_drawdown'],
                           results['inputs']['worst_day_limit'],
                           results['inputs']['cash_min']]
        
        compliance = []
        for actual, limit, constraint in zip(actual_values, constraint_limits, constraints):
            if constraint == 'Cash Min':
                compliance.append('Pass' if actual >= limit else 'Fail')
            else:
                compliance.append('Pass' if actual <= limit else 'Fail')
        
        colors_compliance = ['green' if c == 'Pass' else 'red' for c in compliance]
        
        y_pos = np.arange(len(constraints))
        ax2.barh(y_pos, [1]*len(constraints), color=colors_compliance, alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(constraints)
        ax2.set_xlabel('Compliance Status')
        ax2.set_title('Constraint Compliance', fontweight='bold')
        ax2.set_xlim(0, 1)
        
        # Add compliance labels
        for i, comp in enumerate(compliance):
            ax2.text(0.5, i, comp, ha='center', va='center', fontweight='bold', color='white')
        
        # Time horizon analysis
        years = np.arange(1, results['inputs']['horizon_years'] + 1)
        compound_growth = [(avg_final/start_val)**(1/y) - 1 for y in years]
        
        ax3.plot(years, compound_growth, marker='o', linewidth=2, markersize=6)
        ax3.set_xlabel('Investment Horizon (Years)')
        ax3.set_ylabel('Annualized Return')
        ax3.set_title('Expected Returns by Horizon', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        
        # Monte Carlo confidence intervals (simplified visualization)
        scenarios = [0.1, 0.25, 0.5, 0.75, 0.9]
        scenario_labels = ['10th %ile', '25th %ile', 'Median', '75th %ile', '90th %ile']
        
        # Simulate some percentiles based on normal distribution (simplified)
        mean_return = (avg_final/start_val)**(1/results['inputs']['horizon_years']) - 1
        vol_annual = results['results']['volatility']
        
        percentile_values = []
        for p in scenarios:
            # Simplified calculation using normal distribution
            z_score = np.percentile(np.random.standard_normal(10000), p*100)
            final_val = start_val * (1 + mean_return + z_score * vol_annual / np.sqrt(results['inputs']['horizon_years']))**results['inputs']['horizon_years']
            percentile_values.append(final_val)
        
        bars = ax4.bar(scenario_labels, percentile_values, color=sns.color_palette("coolwarm", len(scenarios)))
        ax4.axhline(y=target_val, color='red', linestyle='--', linewidth=2, label=f'Target: ${target_val:,.0f}')
        ax4.set_ylabel('Final Portfolio Value ($)')
        ax4.set_title('Monte Carlo Scenarios', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        chart_path = os.path.join(output_dir, 'performance_analysis.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths['performance'] = chart_path
        
        return chart_paths

    def create_summary_table(self, results: Dict) -> Table:
        """Create a summary table of key metrics."""
        data = [
            ['Metric', 'Value', 'Target/Limit', 'Status'],
            ['Success Probability', f"{results['results']['success_prob']:.1%}", 
             "Maximize", '✓'],
            ['Expected Final Value', f"${results['results']['avg_final']:,.0f}", 
             f"${results['inputs']['target_portfolio']:,.0f}", 
             '✓' if results['results']['avg_final'] >= results['inputs']['target_portfolio'] else '⚠'],
            ['Portfolio Volatility', f"{results['results']['volatility']:.1%}", 
             f"≤ {results['inputs']['sigma_max']:.1%}", 
             '✓' if results['results']['volatility'] <= results['inputs']['sigma_max'] else '⚠'],
            ['Average Drawdown', f"{results['results']['avg_drawdown']:.1%}", 
             f"≤ {results['inputs']['max_drawdown']:.1%}", 
             '✓' if results['results']['avg_drawdown'] <= results['inputs']['max_drawdown'] else '⚠'],
            ['Worst Day Drop', f"{results['results']['avg_worst_day']:.1%}", 
             f"≤ {results['inputs']['worst_day_limit']:.1%}", 
             '✓' if results['results']['avg_worst_day'] <= results['inputs']['worst_day_limit'] else '⚠'],
            ['Cash Allocation', f"{results['results']['cash_allocation']:.1%}", 
             f"≥ {results['inputs']['cash_min']:.1%}", 
             '✓' if results['results']['cash_allocation'] >= results['inputs']['cash_min'] else '⚠'],
        ]
        
        table = Table(data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch, 0.8*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        
        return table

    def create_allocation_table(self, results: Dict) -> Table:
        """Create asset allocation table."""
        weights = results['results']['weights']
        tickers = results['inputs']['tickers']
        
        data = [['Asset', 'Weight', 'Allocation ($)']]
        
        total_value = results['inputs']['start_portfolio']
        for ticker, weight in zip(tickers, weights):
            allocation = total_value * weight
            data.append([ticker, f"{weight:.1%}", f"${allocation:,.0f}"])
        
        # Add total row
        data.append(['TOTAL', '100.0%', f"${total_value:,.0f}"])
        
        table = Table(data, colWidths=[1.5*inch, 1.5*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -2), colors.beige),
            ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#f0f0f0')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ]))
        
        return table

    def generate_pdf_dashboard(self, 
                             results: Dict, 
                             output_path: str,
                             include_charts: bool = True,
                             chart_dir: Optional[str] = None) -> str:
        """Generate a comprehensive PDF dashboard.
        
        Args:
            results: Optimization results dictionary
            output_path: Path for the output PDF file
            include_charts: Whether to include enhanced charts
            chart_dir: Directory containing chart files (if None, will create new ones)
            
        Returns:
            Path to the generated PDF file
        """
        # Create enhanced charts if needed
        chart_paths = {}
        if include_charts:
            if chart_dir is None:
                chart_dir = os.path.dirname(output_path)
            chart_paths = self.create_enhanced_charts(results, chart_dir)
        
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        story = []
        
        # Title page
        story.append(Paragraph(self.title, self.title_style))
        story.append(Paragraph(self.subtitle, self.subtitle_style))
        story.append(Spacer(1, 20))
        
        # Add logo if provided
        if self.logo_path and os.path.exists(self.logo_path):
            logo = Image(self.logo_path, width=2*inch, height=1*inch)
            story.append(logo)
            story.append(Spacer(1, 20))
        
        # Report metadata
        report_date = datetime.datetime.now().strftime("%B %d, %Y")
        story.append(Paragraph(f"<b>Report Date:</b> {report_date}", self.metric_style))
        story.append(Paragraph(f"<b>Generated by:</b> {self.company_name}", self.metric_style))
        story.append(Spacer(1, 30))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", self.section_style))
        
        summary_text = f"""
        This portfolio optimization analysis was conducted for a {results['inputs']['horizon_years']}-year investment horizon 
        with an initial portfolio value of ${results['inputs']['start_portfolio']:,.0f} and a target value of 
        ${results['inputs']['target_portfolio']:,.0f}. The optimization achieved a success probability of 
        {results['results']['success_prob']:.1%} with an expected final portfolio value of 
        ${results['results']['avg_final']:,.0f}.
        
        The optimized portfolio maintains a volatility of {results['results']['volatility']:.1%}, 
        an average maximum drawdown of {results['results']['avg_drawdown']:.1%}, and an average worst-day 
        drop of {results['results']['avg_worst_day']:.1%}. The cash allocation is 
        {results['results']['cash_allocation']:.1%}.
        """
        
        story.append(Paragraph(summary_text, self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Key Metrics Table
        story.append(Paragraph("Key Performance Metrics", self.section_style))
        story.append(self.create_summary_table(results))
        story.append(Spacer(1, 20))
        
        # Asset Allocation Table
        story.append(Paragraph("Asset Allocation", self.section_style))
        story.append(self.create_allocation_table(results))
        story.append(PageBreak())
        
        # Charts section
        if include_charts and chart_paths:
            story.append(Paragraph("Visual Analysis", self.section_style))
            
            # Executive Summary Chart
            if 'executive_summary' in chart_paths:
                story.append(Paragraph("Portfolio Overview", self.styles['Heading3']))
                img = Image(chart_paths['executive_summary'], width=7*inch, height=4.5*inch)
                story.append(img)
                story.append(Spacer(1, 20))
            
            # Asset Allocation Chart
            if 'asset_allocation' in chart_paths:
                story.append(Paragraph("Detailed Asset Allocation", self.styles['Heading3']))
                img = Image(chart_paths['asset_allocation'], width=7*inch, height=3*inch)
                story.append(img)
                story.append(PageBreak())
            
            # Performance Analysis Chart
            if 'performance' in chart_paths:
                story.append(Paragraph("Performance Analysis", self.styles['Heading3']))
                img = Image(chart_paths['performance'], width=7*inch, height=4.5*inch)
                story.append(img)
                story.append(Spacer(1, 20))
        
        # Technical Details
        story.append(PageBreak())
        story.append(Paragraph("Technical Details", self.section_style))
        
        # Optimization parameters
        story.append(Paragraph("Optimization Parameters", self.styles['Heading3']))
        
        tech_details = f"""
        <b>Monte Carlo Scenarios:</b> {results['inputs']['scenarios']:,}<br/>
        <b>Rebalancing Frequency:</b> {results['inputs']['mc_freq']} times per year<br/>
        <b>Maximum Iterations:</b> {results['inputs']['max_iterations']}<br/>
        <b>Optimization Time:</b> {results['results']['elapsed_time']:.2f} seconds<br/>
        <b>Actual Iterations:</b> {results['results']['iterations']}<br/>
        <b>Upper Bounds per Asset:</b> {results['inputs']['upper_bounds']:.1%}<br/>
        """
        
        story.append(Paragraph(tech_details, self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Black-Litterman Views (if any)
        bl_views = results['inputs'].get('bl_views', {})
        if bl_views and bl_views.get('P') is not None:
            story.append(Paragraph("Black-Litterman Views", self.styles['Heading3']))
            
            P = np.array(bl_views['P'])
            Q = np.array(bl_views['Q'])
            omega = bl_views.get('omega', [])
            
            views_text = "The following market views were incorporated into the optimization:<br/><br/>"
            
            for i in range(len(Q)):
                view_assets = []
                for j, ticker in enumerate(results['inputs']['tickers']):
                    if P[i, j] != 0:
                        view_assets.append(f"{ticker} ({P[i, j]:+.1f})")
                
                views_text += f"<b>View {i+1}:</b> {' vs '.join(view_assets)} = {Q[i]:.1%}<br/>"
                if omega and len(omega) > i:
                    try:
                        omega_val = float(omega[i]) if hasattr(omega[i], '__float__') else omega[i]
                        if isinstance(omega_val, (int, float)):
                            views_text += f"&nbsp;&nbsp;&nbsp;&nbsp;Uncertainty: {omega_val:.4f}<br/>"
                    except (ValueError, TypeError):
                        pass  # Skip if can't convert to float
                views_text += "<br/>"
            
            story.append(Paragraph(views_text, self.styles['Normal']))
        
        # Risk Constraints
        story.append(Paragraph("Risk Constraints", self.styles['Heading3']))
        
        constraints_text = f"""
        <b>Maximum Portfolio Volatility:</b> {results['inputs']['sigma_max']:.1%}<br/>
        <b>Maximum Drawdown:</b> {results['inputs']['max_drawdown']:.1%}<br/>
        <b>Maximum Single-Day Loss:</b> {results['inputs']['worst_day_limit']:.1%}<br/>
        <b>Minimum Cash Allocation:</b> {results['inputs']['cash_min']:.1%}<br/>
        <b>Cash Equivalent Asset:</b> {results['inputs']['cash_ticker']}<br/>
        """
        
        story.append(Paragraph(constraints_text, self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Disclaimer
        story.append(Paragraph("Important Disclaimer", self.section_style))
        disclaimer_text = """
        This analysis is for informational purposes only and should not be considered as investment advice. 
        Past performance does not guarantee future results. All investments carry risk of loss. 
        Monte Carlo simulations are based on historical data and mathematical models that may not accurately 
        predict future market conditions. Please consult with a qualified financial advisor before making 
        investment decisions.
        """
        story.append(Paragraph(disclaimer_text, self.styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        return output_path

    def create_web_dashboard_html(self, results: Dict, output_path: str) -> str:
        """Create an HTML version of the dashboard for web display.
        
        Args:
            results: Optimization results dictionary
            output_path: Path for the output HTML file
            
        Returns:
            Path to the generated HTML file
        """
        # Create charts and encode as base64 for embedding
        chart_dir = os.path.dirname(output_path)
        chart_paths = self.create_enhanced_charts(results, chart_dir)
        
        # Convert charts to base64
        chart_data = {}
        for name, path in chart_paths.items():
            with open(path, 'rb') as f:
                chart_data[name] = base64.b64encode(f.read()).decode()
        
        # Generate HTML
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{self.title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .title {{ color: #1f77b4; font-size: 2.5em; margin-bottom: 10px; }}
                .subtitle {{ color: #666; font-size: 1.2em; }}
                .section {{ margin: 30px 0; }}
                .section h2 {{ color: #1f77b4; border-bottom: 2px solid #1f77b4; padding-bottom: 10px; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
                .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }}
                .metric-value {{ font-size: 2em; font-weight: bold; color: #1f77b4; }}
                .metric-label {{ color: #666; margin-top: 5px; }}
                .chart {{ text-align: center; margin: 20px 0; }}
                .chart img {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
                .table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .table th, .table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                .table th {{ background-color: #1f77b4; color: white; }}
                .status-pass {{ color: #2ca02c; font-weight: bold; }}
                .status-warn {{ color: #d62728; font-weight: bold; }}
                .download-btn {{ background: #1f77b4; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; margin: 10px; }}
                .download-btn:hover {{ background: #0d5aa7; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1 class="title">{self.title}</h1>
                    <p class="subtitle">{self.subtitle}</p>
                    <p>Generated on {datetime.datetime.now().strftime("%B %d, %Y")}</p>
                </div>
                
                <div class="section">
                    <h2>Key Metrics</h2>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-value">{results['results']['success_prob']:.1%}</div>
                            <div class="metric-label">Success Probability</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${results['results']['avg_final']:,.0f}</div>
                            <div class="metric-label">Expected Final Value</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{results['results']['volatility']:.1%}</div>
                            <div class="metric-label">Portfolio Volatility</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{results['results']['avg_drawdown']:.1%}</div>
                            <div class="metric-label">Average Drawdown</div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Portfolio Overview</h2>
                    <div class="chart">
                        <img src="data:image/png;base64,{chart_data.get('executive_summary', '')}" alt="Executive Summary">
                    </div>
                </div>
                
                <div class="section">
                    <h2>Asset Allocation</h2>
                    <div class="chart">
                        <img src="data:image/png;base64,{chart_data.get('asset_allocation', '')}" alt="Asset Allocation">
                    </div>
                </div>
                
                <div class="section">
                    <h2>Performance Analysis</h2>
                    <div class="chart">
                        <img src="data:image/png;base64,{chart_data.get('performance', '')}" alt="Performance Analysis">
                    </div>
                </div>
                
                <div class="section">
                    <h2>Detailed Results</h2>
                    <table class="table">
                        <thead>
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                                <th>Target/Limit</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        
        # Add table rows
        metrics_data = [
            ('Success Probability', f"{results['results']['success_prob']:.1%}", 'Maximize', 'Pass'),
            ('Expected Final Value', f"${results['results']['avg_final']:,.0f}", 
             f"${results['inputs']['target_portfolio']:,.0f}", 
             'Pass' if results['results']['avg_final'] >= results['inputs']['target_portfolio'] else 'Warning'),
            ('Portfolio Volatility', f"{results['results']['volatility']:.1%}", 
             f"≤ {results['inputs']['sigma_max']:.1%}", 
             'Pass' if results['results']['volatility'] <= results['inputs']['sigma_max'] else 'Warning'),
            ('Average Drawdown', f"{results['results']['avg_drawdown']:.1%}", 
             f"≤ {results['inputs']['max_drawdown']:.1%}", 
             'Pass' if results['results']['avg_drawdown'] <= results['inputs']['max_drawdown'] else 'Warning'),
            ('Worst Day Drop', f"{results['results']['avg_worst_day']:.1%}", 
             f"≤ {results['inputs']['worst_day_limit']:.1%}", 
             'Pass' if results['results']['avg_worst_day'] <= results['inputs']['worst_day_limit'] else 'Warning'),
            ('Cash Allocation', f"{results['results']['cash_allocation']:.1%}", 
             f"≥ {results['inputs']['cash_min']:.1%}", 
             'Pass' if results['results']['cash_allocation'] >= results['inputs']['cash_min'] else 'Warning'),
        ]
        
        for metric, value, target, status in metrics_data:
            status_class = 'status-pass' if status == 'Pass' else 'status-warn'
            status_symbol = '✓' if status == 'Pass' else '⚠'
            html_content += f"""
                            <tr>
                                <td>{metric}</td>
                                <td>{value}</td>
                                <td>{target}</td>
                                <td class="{status_class}">{status_symbol}</td>
                            </tr>
            """
        
        html_content += """
                        </tbody>
                    </table>
                </div>
                
                <div class="section">
                    <h2>Asset Allocation Details</h2>
                    <table class="table">
                        <thead>
                            <tr>
                                <th>Asset</th>
                                <th>Weight</th>
                                <th>Allocation ($)</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        
        # Add allocation table
        weights = results['results']['weights']
        tickers = results['inputs']['tickers']
        total_value = results['inputs']['start_portfolio']
        
        for ticker, weight in zip(tickers, weights):
            allocation = total_value * weight
            html_content += f"""
                            <tr>
                                <td>{ticker}</td>
                                <td>{weight:.1%}</td>
                                <td>${allocation:,.0f}</td>
                            </tr>
            """
        
        html_content += f"""
                            <tr style="font-weight: bold; background-color: #f0f0f0;">
                                <td>TOTAL</td>
                                <td>100.0%</td>
                                <td>${total_value:,.0f}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                
                <div class="section" style="text-align: center;">
                    <button class="download-btn" onclick="window.print()">Print Dashboard</button>
                </div>
                
                <div class="section">
                    <h2>Disclaimer</h2>
                    <p style="font-size: 0.9em; color: #666;">
                        This analysis is for informational purposes only and should not be considered as investment advice. 
                        Past performance does not guarantee future results. All investments carry risk of loss. 
                        Monte Carlo simulations are based on historical data and mathematical models that may not accurately 
                        predict future market conditions. Please consult with a qualified financial advisor before making 
                        investment decisions.
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path


def create_dashboard_from_results(results_dict: Dict, 
                                output_dir: str,
                                format_type: str = 'both',
                                title: str = "Portfolio Optimization Dashboard") -> Dict[str, str]:
    """Convenience function to create dashboards from optimization results.
    
    Args:
        results_dict: Results dictionary from PortfolioOptimizer.optimize()
        output_dir: Directory to save dashboard files
        format_type: 'pdf', 'html', or 'both'
        title: Dashboard title
        
    Returns:
        Dictionary with paths to generated files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    generator = PDFDashboardGenerator(title=title)
    output_files = {}
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if format_type in ['pdf', 'both']:
        pdf_path = os.path.join(output_dir, f'portfolio_dashboard_{timestamp}.pdf')
        generator.generate_pdf_dashboard(results_dict, pdf_path)
        output_files['pdf'] = pdf_path
    
    if format_type in ['html', 'both']:
        html_path = os.path.join(output_dir, f'portfolio_dashboard_{timestamp}.html')
        generator.create_web_dashboard_html(results_dict, html_path)
        output_files['html'] = html_path
    
    return output_files


if __name__ == "__main__":
    # Example usage
    from portfolio_optimizer import PortfolioOptimizer
    
    # Run optimization
    optimizer = PortfolioOptimizer(
        tickers=["SPY", "QQQ", "TLT", "GLD", "BIL"],
        scenarios=1000,  # Reduced for faster testing
        max_iterations=20
    )
    
    results = optimizer.optimize(save_outputs=False)
    
    # Create dashboards
    output_files = create_dashboard_from_results(
        results, 
        output_dir="./dashboard_output",
        format_type='both'
    )
    
    print("Generated dashboards:")
    for format_type, path in output_files.items():
        print(f"{format_type.upper()}: {path}") 
