"""
Tests for the calculations module.
"""

import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path
from src import LCACalculator

@pytest.fixture
def sample_data():
    """Create sample data for testing, including cases for name normalization."""
    return pd.DataFrame({
        'product_id': ['P001', 'P001', 'P001', 'P002', 'P002', 'P002', 'P003'],
        'product_name': ['Product1', 'Product1', 'Product1', 'Product2', 'Product2', 'Product2', 'Product3'],
        'life_cycle_stage': ['Manufacturing', 'Transportation', 'End-of-Life', 'manufacturing', 'transportation', 'EOL', 'use'],
        'material_type': ['steel', 'steel', 'steel', 'aluminum', 'aluminum', 'aluminum', 'PVC'],
        'quantity_kg': [100, 100, 100, 50, 50, 50, 20],
        'energy_consumption_kwh': [120, 20, 50, 180, 25, 20, 10],
        'transport_distance_km': [50, 100, 30, 180, 140, 35, 0],
        'transport_mode': ['Truck'] * 7,
        'waste_generated_kg': [5, 0, 100, 1, 0, 20, 0],
        'recycling_rate': [0.9, 0, 0.9, 0.85, 0, 0.85, 0],
        'landfill_rate': [0.05, 0, 0.05, 0.1, 0, 0.1, 0],
        'incineration_rate': [0.05, 0, 0.05, 0.05, 0, 0.05, 0],
        'carbon_footprint_kg_co2e': [180, 50, 10, 125, 30, 5, 2],
        'water_usage_liters': [150, 30, 10, 100, 0, 6, 1]
    })

@pytest.fixture
def impact_factors():
    """Create sample impact factors for testing, including normalized names."""
    return {
        'steel': {
            'manufacturing': {'carbon_impact': 1.8, 'energy_impact': 20, 'water_impact': 150},
            'transportation': {'carbon_impact': 0.5, 'energy_impact': 5, 'water_impact': 30},
            'disposal': {'carbon_impact': 0.1, 'energy_impact': 1, 'water_impact': 10}
        },
        'aluminum': {
            'manufacturing': {'carbon_impact': 2.5, 'energy_impact': 25, 'water_impact': 200},
            'transportation': {'carbon_impact': 0.6, 'energy_impact': 6, 'water_impact': 40},
            'disposal': {'carbon_impact': 0.1, 'energy_impact': 1, 'water_impact': 8}
        },
        'plastic': { # Normalized from 'pvc'
            'use': {'carbon_impact': 0.05, 'energy_impact': 0.1, 'water_impact': 0.2}
        }
    }

@pytest.fixture
def calculator(impact_factors, tmp_path):
    """Creates an LCACalculator instance with temp impact factors file."""
    impact_file = tmp_path / "test_impact_factors.json"
    with open(impact_file, 'w') as f:
        json.dump(impact_factors, f)
    return LCACalculator(impact_factors_path=impact_file)

def test_calculate_impacts_with_factors(calculator, sample_data):
    """Test impact calculations using impact factors."""
    results = calculator.calculate_impacts(sample_data)
    
    assert not results.empty
    assert all(col in results.columns for col in ['carbon_impact', 'energy_impact', 'water_impact'])
    assert len(results) == len(sample_data)

    # Check a specific calculation for steel manufacturing
    steel_mfg = results[(results['material_type'] == 'steel') & (results['life_cycle_stage'] == 'Manufacturing')]
    assert np.isclose(steel_mfg['carbon_impact'].iloc[0], 100 * 1.8) # 100kg * 1.8 factor

    # Check name normalization (PVC -> plastic, EOL -> disposal)
    pvc_use = results[results['material_type'] == 'PVC']
    assert np.isclose(pvc_use['carbon_impact'].iloc[0], 20 * 0.05) # 20kg * 0.05 factor
    
    alu_eol = results[(results['material_type'] == 'aluminum') & (results['life_cycle_stage'] == 'EOL')]
    assert np.isclose(alu_eol['carbon_impact'].iloc[0], 50 * 0.1) # 50kg * 0.1 factor

def test_calculate_impacts_without_factors(calculator, sample_data):
    """Test impact calculations using direct measurements."""
    results = calculator.calculate_impacts(sample_data, use_impact_factors=False)
    
    assert not results.empty
    # Results should equal the direct measurement columns
    pd.testing.assert_series_equal(results['carbon_impact'], sample_data['carbon_footprint_kg_co2e'], check_names=False)
    pd.testing.assert_series_equal(results['energy_impact'], sample_data['energy_consumption_kwh'], check_names=False)
    pd.testing.assert_series_equal(results['water_impact'], sample_data['water_usage_liters'], check_names=False)

def test_calculate_impacts_invalid_input(calculator):
    """Test that calculate_impacts raises errors for invalid input."""
    with pytest.raises(ValueError, match="Input data is empty"):
        calculator.calculate_impacts(pd.DataFrame())
        
    with pytest.raises(ValueError, match="Missing required columns"):
        calculator.calculate_impacts(pd.DataFrame({'a': [1]}))

def test_calculate_total_impacts(calculator, sample_data):
    """Test total impact calculations."""
    impacts = calculator.calculate_impacts(sample_data)
    total_impacts = calculator.calculate_total_impacts(impacts)
    
    assert len(total_impacts) == 3  # Three products
    assert 'product_name' in total_impacts.columns
    assert 'carbon_impact' in total_impacts.columns

    # Verify sum for P001
    p001_total_carbon = 100 * 1.8 + 100 * 0.5 + 100 * 0.1
    actual_p001_carbon = total_impacts[total_impacts['product_id'] == 'P001']['carbon_impact'].iloc[0]
    assert np.isclose(actual_p001_carbon, p001_total_carbon)

def test_normalize_impacts(calculator, sample_data):
    """Test impact normalization."""
    impacts = calculator.calculate_impacts(sample_data)
    total_impacts = calculator.calculate_total_impacts(impacts)
    normalized = calculator.normalize_impacts(total_impacts)
    
    assert all(col in normalized.columns for col in ['carbon_impact_normalized', 'energy_impact_normalized', 'water_impact_normalized'])
    assert all(normalized[col].max() <= 1 for col in ['carbon_impact_normalized', 'energy_impact_normalized', 'water_impact_normalized'])
    assert all(normalized[col].min() >= 0 for col in ['carbon_impact_normalized', 'energy_impact_normalized', 'water_impact_normalized'])
    # The product with the highest impact should have a normalized value of 1
    assert np.isclose(normalized['carbon_impact_normalized'].max(), 1.0)

def test_compare_alternatives(calculator, sample_data):
    """Test product comparison."""
    impacts = calculator.calculate_impacts(sample_data)
    total_impacts = calculator.calculate_total_impacts(impacts)
    comparison = calculator.compare_alternatives(total_impacts, ['P001', 'P002'])
    
    assert len(comparison) == 2
    assert all(f'{col}_relative' in comparison.columns for col in ['carbon_impact', 'energy_impact', 'water_impact'])
    
    # The product with the minimum impact should have a relative score of 0
    min_carbon_product_idx = comparison['carbon_impact'].idxmin()
    assert np.isclose(comparison.loc[min_carbon_product_idx, 'carbon_impact_relative'], 0.0)

def test_get_impact_summary(calculator, sample_data):
    """Test the impact summary generation."""
    impacts = calculator.calculate_impacts(sample_data)
    summary = calculator.get_impact_summary(impacts)

    assert isinstance(summary, dict)
    assert 'carbon_impact' in summary
    assert 'total' in summary['carbon_impact']
    assert 'mean' in summary['carbon_impact']
    assert np.isclose(summary['carbon_impact']['total'], impacts['carbon_impact'].sum())