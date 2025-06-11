"""
Tests for the data input module.
"""

import pytest
import pandas as pd
import json
from pathlib import Path
import numpy as np
from src import DataInput, DataValidationError

@pytest.fixture
def sample_data_df():
    """Create a sample valid DataFrame for testing."""
    return pd.DataFrame({
        'product_id': ['P001', 'P001', 'P001'],
        'product_name': ['Product1', 'Product1', 'Product1'],
        'life_cycle_stage': ['Manufacturing', 'Transportation', 'End-of-Life'],
        'material_type': ['steel', 'steel', 'steel'],
        'quantity_kg': [100, 100, 100],
        'energy_consumption_kwh': [120, 20, 50],
        'transport_distance_km': [50, 100, 30],
        'transport_mode': ['Truck', 'Truck', 'Truck'],
        'waste_generated_kg': [10, 0, 100],
        'recycling_rate': [0.5, 0, 0.9],
        'landfill_rate': [0.3, 0, 0.05],
        'incineration_rate': [0.2, 0, 0.05],
        'carbon_footprint_kg_co2e': [180, 50, 10],
        'water_usage_liters': [150, 30, 10]
    })

@pytest.fixture
def sample_impact_factors_dict():
    """Create a sample valid impact factors dictionary."""
    return {
        'steel': {
            'manufacturing': {'carbon_impact': 1.8, 'energy_impact': 20, 'water_impact': 150},
            'transportation': {'carbon_impact': 0.5, 'energy_impact': 5, 'water_impact': 30},
            'disposal': {'carbon_impact': 0.1, 'energy_impact': 1, 'water_impact': 10}
        }
    }

def test_read_data_csv(sample_data_df, tmp_path):
    """Test reading data from a CSV file."""
    data_input = DataInput()
    csv_file = tmp_path / "test_data.csv"
    sample_data_df.to_csv(csv_file, index=False)
    
    data = data_input.read_data(csv_file)
    assert isinstance(data, pd.DataFrame)
    pd.testing.assert_frame_equal(data, sample_data_df)

def test_read_data_json(sample_data_df, tmp_path):
    """Test reading data from a JSON file."""
    data_input = DataInput()
    json_file = tmp_path / "test_data.json"
    sample_data_df.to_json(json_file, orient='records')

    data = data_input.read_data(json_file)
    assert isinstance(data, pd.DataFrame)
    # JSON conversion might change dtypes, so compare values
    pd.testing.assert_frame_equal(data, sample_data_df, check_dtype=False)

def test_read_data_errors(tmp_path):
    """Test error handling in read_data."""
    data_input = DataInput()
    
    with pytest.raises(FileNotFoundError):
        data_input.read_data(tmp_path / "nonexistent.csv")

    bad_file = tmp_path / "bad.txt"
    bad_file.write_text("this is not a valid format")
    with pytest.raises(ValueError, match="Unsupported file format"):
        data_input.read_data(bad_file)

def test_validate_data_valid(sample_data_df):
    """Test validation with a perfectly valid DataFrame."""
    data_input = DataInput()
    result = data_input.validate_data(sample_data_df)
    assert result['is_valid'] is True
    assert not result['errors']
    assert not result['warnings']

def test_validate_data_invalid(sample_data_df):
    """Test various invalid data scenarios."""
    data_input = DataInput()
    
    # Test missing required column
    invalid_data = sample_data_df.drop('product_id', axis=1)
    result = data_input.validate_data(invalid_data)
    assert result['is_valid'] is False
    assert "Missing required columns: ['product_id']" in result['errors'][0]

    # Test invalid numeric data
    invalid_data = sample_data_df.copy()
    invalid_data['quantity_kg'] = invalid_data['quantity_kg'].astype('object')
    invalid_data.loc[0, 'quantity_kg'] = 'invalid'
    result = data_input.validate_data(invalid_data)
    assert result['is_valid'] is False
    assert "contains 1 non-numeric values" in result['errors'][0]
    
    # Test invalid rates sum
    invalid_data = sample_data_df.copy()
    invalid_data.loc[0, 'recycling_rate'] = 0.6  # Sum becomes 1.1
    result = data_input.validate_data(invalid_data)
    assert result['is_valid'] is False
    assert "rates that don't sum to 1.0" in result['errors'][0]

def test_validate_data_warnings(sample_data_df):
    """Test scenarios that should produce warnings."""
    data_input = DataInput()

    # Test negative value warning
    warn_data = sample_data_df.copy()
    warn_data.loc[0, 'quantity_kg'] = -10
    result = data_input.validate_data(warn_data)
    assert result['is_valid'] is True # Warnings don't make it invalid
    assert "contains negative values" in result['warnings'][0]

    # Test duplicate rows warning
    warn_data = pd.concat([sample_data_df, sample_data_df.head(1)], ignore_index=True)
    result = data_input.validate_data(warn_data)
    assert result['is_valid'] is True
    assert "duplicate product_id + life_cycle_stage combinations found" in result['warnings'][0]

def test_read_impact_factors_valid(sample_impact_factors_dict, tmp_path):
    """Test reading a valid impact factors JSON file."""
    data_input = DataInput()
    json_file = tmp_path / "factors.json"
    with open(json_file, 'w') as f:
        json.dump(sample_impact_factors_dict, f)
    
    factors = data_input.read_impact_factors(json_file)
    assert isinstance(factors, dict)
    assert factors == sample_impact_factors_dict

def test_read_impact_factors_invalid(tmp_path):
    """Test reading invalid impact factors JSON files."""
    data_input = DataInput()
    
    # Test invalid JSON format
    json_file = tmp_path / "bad.json"
    json_file.write_text("{'key': 'value'}") # Invalid JSON
    with pytest.raises(DataValidationError, match="Invalid JSON format"):
        data_input.read_impact_factors(json_file)
        
    # Test invalid structure (missing impact key)
    invalid_structure = {"steel": {"manufacturing": {"carbon_impact": 1.8}}}
    json_file.write_text(json.dumps(invalid_structure))
    with pytest.raises(DataValidationError, match="Invalid impact factors structure"):
        data_input.read_impact_factors(json_file)

def test_clean_data():
    """Test the data cleaning and preprocessing function."""
    data_input = DataInput()
    raw_data = pd.DataFrame({
        'quantity_kg': [10, '20', np.nan, 'bad'],
        'product_name': [' A ', 'B', ' C ', 'D'],
        'all_nan': [np.nan, np.nan, np.nan, np.nan]
    })
    
    cleaned = data_input.clean_data(raw_data)
    
    # Check NaN filling and type conversion
    expected_qty = pd.Series([10.0, 20.0, 0.0, 0.0], name='quantity_kg')
    pd.testing.assert_series_equal(cleaned['quantity_kg'], expected_qty)

    # Check string stripping
    expected_name = pd.Series(['A', 'B', 'C', 'D'], name='product_name')
    pd.testing.assert_series_equal(cleaned['product_name'], expected_name)
    
    # Check removal of all-NaN rows (if implemented, original doesn't remove by row)
    assert len(cleaned) == 4