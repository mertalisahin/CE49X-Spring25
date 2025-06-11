"""
Fixed Utilities module for LCA tool.
Contains helper functions and constants with proper error handling.
"""

import pandas as pd
import json
from typing import Dict, List, Union, Any
from pathlib import Path
import numpy as np

# Improved unit conversion system
UNIT_CONVERSIONS = {
    'mass': {
        'base': 'kg',
        'conversions': {
            'g': 0.001,      # 1 g = 0.001 kg
            'ton': 1000,     # 1 ton = 1000 kg
            'tonne': 1000,   # Alternative spelling
            'lb': 0.453592,  # 1 lb = 0.453592 kg
            'oz': 0.0283495  # 1 oz = 0.0283495 kg
        }
    },
    'volume': {
        'base': 'L',
        'conversions': {
            'mL': 0.001,     # 1 mL = 0.001 L
            'ml': 0.001,     # Alternative case
            'm3': 1000,      # 1 m³ = 1000 L
            'm³': 1000,      # Unicode version
            'gal': 3.78541,  # 1 US gal = 3.78541 L
            'gallon': 3.78541,
            'ft3': 28.3168,  # 1 ft³ = 28.3168 L
            'ft³': 28.3168
        }
    },
    'energy': {
        'base': 'MJ',
        'conversions': {
            'kJ': 0.001,     # 1 kJ = 0.001 MJ
            'J': 0.000001,   # 1 J = 0.000001 MJ
            'kWh': 3.6,      # 1 kWh = 3.6 MJ
            'kwh': 3.6,      # Alternative case
            'BTU': 0.00105506, # 1 BTU = 0.00105506 MJ
            'btu': 0.00105506,
            'cal': 0.000004184, # 1 cal = 0.000004184 MJ
            'kcal': 0.004184   # 1 kcal = 0.004184 MJ
        }
    }
}

class UnitConversionError(Exception):
    """Custom exception for unit conversion errors."""
    pass



def convert_units(value: Union[float, int], from_unit: str, to_unit: str) -> float:
    """
    Convert values between different units with proper validation.
    
    Args:
        value: Value to convert
        from_unit: Source unit
        to_unit: Target unit
        
    Returns:
        Converted value
        
    Raises:
        UnitConversionError: If units are not supported or conversion fails
        TypeError: If value is not numeric
    """
    # Input validation
    if not isinstance(value, (int, float)):
        raise TypeError(f"Value must be numeric, got {type(value)}")
    
    if not isinstance(from_unit, str) or not isinstance(to_unit, str):
        raise TypeError("Units must be strings")
    
    # Quick return for same units
    if from_unit.lower() == to_unit.lower():
        return float(value)
    
    # Normalize unit names
    from_unit = from_unit.lower().strip()
    to_unit = to_unit.lower().strip()
    
    # Find the appropriate conversion category
    conversion_category = None
    base_unit = None
    conversions = None
    
    for category, info in UNIT_CONVERSIONS.items():
        base = info['base'].lower()
        unit_conversions = {k.lower(): v for k, v in info['conversions'].items()}
        
        # Check if both units are in this category
        from_in_category = from_unit == base or from_unit in unit_conversions
        to_in_category = to_unit == base or to_unit in unit_conversions
        
        if from_in_category and to_in_category:
            conversion_category = category
            base_unit = base
            conversions = unit_conversions
            break
    
    if not conversion_category:
        raise UnitConversionError(
            f"Cannot convert between '{from_unit}' and '{to_unit}'. "
            f"Supported categories: {list(UNIT_CONVERSIONS.keys())}"
        )
    
    try:
        # Convert to base unit
        if from_unit == base_unit:
            base_value = value
        else:
            if from_unit not in conversions:
                raise UnitConversionError(f"Unknown unit: '{from_unit}'")
            base_value = value * conversions[from_unit]
        
        # Convert from base unit to target unit
        if to_unit == base_unit:
            result = base_value
        else:
            if to_unit not in conversions:
                raise UnitConversionError(f"Unknown unit: '{to_unit}'")
            result = base_value / conversions[to_unit]
        
        return float(result)
        
    except (KeyError, ZeroDivisionError) as e:
        raise UnitConversionError(f"Conversion error: {str(e)}")

def save_results(data: pd.DataFrame, file_path: Union[str, Path], 
                format: str = 'csv', **kwargs) -> None:
    """
    Save analysis results to file with validation and error handling.
    
    Args:
        data: DataFrame to save
        file_path: Path to save file
        format: File format ('csv', 'xlsx', or 'json')
        **kwargs: Additional arguments for pandas save methods
        
    Raises:
        TypeError: If data is not a DataFrame
        ValueError: If format is not supported or data is empty
        IOError: If file cannot be saved
    """
    # Input validation
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame")
    
    if data.empty:
        raise ValueError("Cannot save empty DataFrame")
    
    file_path = Path(file_path)
    
    # Create directory if it doesn't exist
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise IOError(f"Cannot create directory {file_path.parent}: {e}")
    
    # Save based on format
    try:
        if format.lower() == 'csv':
            # Default parameters for CSV
            csv_kwargs = {'index': False, 'encoding': 'utf-8'}
            csv_kwargs.update(kwargs)
            data.to_csv(file_path, **csv_kwargs)
            
        elif format.lower() == 'xlsx':
            # Default parameters for Excel
            excel_kwargs = {'index': False, 'engine': 'openpyxl'}
            excel_kwargs.update(kwargs)
            data.to_excel(file_path, **excel_kwargs)
            
        elif format.lower() == 'json':
            # Default parameters for JSON
            json_kwargs = {'orient': 'records', 'indent': 2}
            json_kwargs.update(kwargs)
            data.to_json(file_path, **json_kwargs)
            
        else:
            raise ValueError(f"Unsupported format: '{format}'. Use 'csv', 'xlsx', or 'json'")
            
        print(f"Successfully saved {len(data)} records to {file_path}")
        
    except Exception as e:
        raise IOError(f"Failed to save file '{file_path}': {str(e)}")

def load_impact_factors(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load impact factors from a JSON file with proper error handling.
    
    Args:
        file_path: Path to impact factors file
        
    Returns:
        Dictionary of impact factors
        
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If JSON is invalid or malformed
        IOError: If file cannot be read
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Impact factors file not found: {file_path}")
    
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            impact_factors = json.load(f)
        
        # Basic validation
        if not isinstance(impact_factors, dict):
            raise ValueError("Impact factors file must contain a JSON object")
        
        print(f"Loaded impact factors for {len(impact_factors)} materials")
        return impact_factors
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in impact factors file: {str(e)}")
    except Exception as e:
        raise IOError(f"Error reading impact factors file: {str(e)}")

def validate_numeric_range(value: Union[float, int], 
                         min_value: Union[float, int] = None,
                         max_value: Union[float, int] = None,
                         name: str = "value") -> bool:
    """
    Validate that a numeric value is within specified range.
    
    Args:
        value: Value to validate
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        name: Name of the value for error messages
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If value is outside range
        TypeError: If value is not numeric
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(value)}")
    
    if pd.isna(value):
        raise ValueError(f"{name} cannot be NaN")
    
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} ({value}) is below minimum ({min_value})")
    
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} ({value}) is above maximum ({max_value})")
    
    return True

def create_summary_report(data: pd.DataFrame, impacts: pd.DataFrame) -> Dict[str, Any]:
    """
    Create a comprehensive summary report of the LCA analysis.
    
    Args:
        data: Original input data
        impacts: Calculated impacts data
        
    Returns:
        Dictionary containing summary statistics
    """
    report = {
        'metadata': {
            'timestamp': pd.Timestamp.now().isoformat(),
            'input_records': len(data),
            'processed_records': len(impacts),
            'products_analyzed': data['product_id'].nunique() if 'product_id' in data.columns else 0
        },
        'data_quality': {},
        'impact_summary': {},
        'material_breakdown': {},
        'stage_breakdown': {}
    }
    
    # Data quality metrics
    if not data.empty:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        report['data_quality'] = {
            'completeness': (1 - data.isnull().sum() / len(data)).to_dict(),
            'numeric_ranges': {
                col: {
                    'min': float(data[col].min()),
                    'max': float(data[col].max()),
                    'mean': float(data[col].mean())
                } for col in numeric_cols
            }
        }
    
    # Impact summary
    if not impacts.empty:
        impact_cols = [col for col in impacts.columns if 'impact' in col]
        if impact_cols:
            report['impact_summary'] = {
                col: {
                    'total': float(impacts[col].sum()),
                    'mean': float(impacts[col].mean()),
                    'std': float(impacts[col].std())
                } for col in impact_cols
            }
    
    # Material breakdown
    if 'material_type' in data.columns:
        material_counts = data['material_type'].value_counts()
        report['material_breakdown'] = material_counts.to_dict()
    
    # Stage breakdown
    if 'life_cycle_stage' in data.columns:
        stage_counts = data['life_cycle_stage'].value_counts()
        report['stage_breakdown'] = stage_counts.to_dict()
    
    return report

def get_supported_units() -> Dict[str, List[str]]:
    """
    Get a dictionary of all supported units by category.
    
    Returns:
        Dictionary mapping categories to lists of supported units
    """
    supported = {}
    
    for category, info in UNIT_CONVERSIONS.items():
        units = [info['base']] + list(info['conversions'].keys())
        supported[category] = sorted(units)
    
    return supported