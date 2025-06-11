"""
Tests for the visualization module.
"""

import pytest
import pandas as pd
import matplotlib.pyplot as plt
from src.visualization import LCAVisualizer, VisualizationError

@pytest.fixture
def sample_impact_data():
    """Create sample calculated impact data for testing visualization."""
    return pd.DataFrame({
        'product_id': ['P001', 'P001', 'P001', 'P002', 'P002', 'P002'],
        'product_name': ['Product1', 'Product1', 'Product1', 'Product2', 'Product2', 'Product2'],
        'life_cycle_stage': ['Manufacturing', 'Transportation', 'End-of-Life'] * 2,
        'material_type': ['steel', 'steel', 'steel', 'aluminum', 'aluminum', 'aluminum'],
        'quantity_kg': [100, 100, 100, 50, 50, 50],
        'waste_generated_kg': [5, 0, 100, 1, 0, 20],
        'recycling_rate': [0.9, 0, 0.9, 0.85, 0, 0.85],
        'landfill_rate': [0.05, 0, 0.05, 0.1, 0, 0.1],
        'incineration_rate': [0.05, 0, 0.05, 0.05, 0, 0.05],
        'carbon_impact': [180.0, 50.0, 10.0, 125.0, 30.0, 5.0],
        'energy_impact': [2000.0, 500.0, 100.0, 1250.0, 300.0, 50.0],
        'water_impact': [15000.0, 3000.0, 1000.0, 10000.0, 2000.0, 400.0]
    })

def test_plot_impact_breakdown(sample_impact_data):
    """Test impact breakdown pie chart."""
    visualizer = LCAVisualizer()
    fig = None
    try:
        # Test by material type
        fig = visualizer.plot_impact_breakdown(sample_impact_data, 'carbon_impact', 'material_type')
        assert isinstance(fig, plt.Figure)
        assert "Carbon Impact (kg CO2e) by Material Type" in fig.axes[0].get_title()
    finally:
        if fig: plt.close(fig)
    
    try:
        # Test by life cycle stage
        fig = visualizer.plot_impact_breakdown(sample_impact_data, 'energy_impact', 'life_cycle_stage')
        assert isinstance(fig, plt.Figure)
        assert "Energy Impact (kWh) by Life Cycle Stage" in fig.axes[0].get_title()
    finally:
        if fig: plt.close(fig)

def test_plot_life_cycle_impacts(sample_impact_data):
    """Test life cycle impacts bar chart (2x2 subplots)."""
    visualizer = LCAVisualizer()
    fig = None
    try:
        fig = visualizer.plot_life_cycle_impacts(sample_impact_data, 'P001')
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4  # Should create a 2x2 subplot layout
        assert 'Life Cycle Impacts for Product P001' in fig._suptitle.get_text()
    finally:
        if fig: plt.close(fig)

def test_plot_product_comparison(sample_impact_data):
    """Test product comparison radar chart."""
    visualizer = LCAVisualizer()
    fig = None
    try:
        # Aggregate data first as the visualizer expects totals per product
        total_impacts = sample_impact_data.groupby('product_id').sum(numeric_only=True).reset_index()
        fig = visualizer.plot_product_comparison(total_impacts, ['P001', 'P002'])
        assert isinstance(fig, plt.Figure)
        # Check for polar projection
        assert 'polar' in fig.axes[0].name
    finally:
        if fig: plt.close(fig)

def test_plot_end_of_life_breakdown(sample_impact_data):
    """Test end-of-life breakdown stacked bar chart."""
    visualizer = LCAVisualizer()
    fig = None
    try:
        fig = visualizer.plot_end_of_life_breakdown(sample_impact_data, 'P001')
        assert isinstance(fig, plt.Figure)
        assert 'End-of-Life Management for Product P001' in fig.axes[0].get_title()
    finally:
        if fig: plt.close(fig)

def test_plot_impact_correlation(sample_impact_data):
    """Test impact correlation heatmap."""
    visualizer = LCAVisualizer()
    fig = None
    try:
        fig = visualizer.plot_impact_correlation(sample_impact_data)
        assert isinstance(fig, plt.Figure)
        assert 'Impact Category Correlations' in fig.axes[0].get_title()
    finally:
        if fig: plt.close(fig)

def test_visualization_errors():
    """Test that plotting functions raise VisualizationError for bad input."""
    visualizer = LCAVisualizer()
    
    # Test with empty dataframe
    with pytest.raises(VisualizationError, match="data is empty"):
        visualizer.plot_impact_breakdown(pd.DataFrame(), 'carbon_impact', 'material_type')

    # Test with missing columns
    df = pd.DataFrame({'a': [1], 'b': [2]})
    with pytest.raises(VisualizationError, match="Missing required columns"):
        visualizer.plot_impact_breakdown(df, 'carbon_impact', 'material_type')
        
    # Test with no data for a specific product
    df = pd.DataFrame({
        'product_id': ['P999'],
        'life_cycle_stage': ['Manufacturing'],
        # Add other required impact columns with dummy data if needed by the function
        'carbon_impact': [0],
        'energy_impact': [0],
        'water_impact': [0],
        'waste_generated_kg': [0]
    })
    with pytest.raises(VisualizationError, match="No data found for product ID: P001"):
        visualizer.plot_life_cycle_impacts(df, 'P001')