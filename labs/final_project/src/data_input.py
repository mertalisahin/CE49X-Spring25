"""
Fixed Data input module for LCA tool.
Handles reading and validating input data from various sources with proper error handling.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Union, Tuple


class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    def __init__(self, message: str, errors: List[str]):
        super().__init__(message)
        self.errors = errors

class DataInput:
    def __init__(self):
        self.supported_formats = ['.csv', '.xlsx', '.json']
        self.required_columns = [
            'product_id', 'product_name', 'life_cycle_stage', 'material_type',
            'quantity_kg', 'energy_consumption_kwh', 'transport_distance_km',
            'transport_mode', 'waste_generated_kg', 'recycling_rate',
            'landfill_rate', 'incineration_rate', 'carbon_footprint_kg_co2e',
            'water_usage_liters'
        ]
    
    def detect_file_format(self, file_path: Path) -> str:
        """Auto-detect file format based on content and extension."""
        try:
            # First try extension-based detection
            if file_path.suffix.lower() in self.supported_formats:
                return file_path.suffix.lower()
            
            # Try to detect JSON by reading first few characters
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(100).strip()
                if content.startswith('{') or content.startswith('['):
                    return '.json'
                
            # If no extension or content detection, fallback to extension
            return file_path.suffix.lower()
        except Exception as e:
            print(f"Could not detect file format: {e}")
            return file_path.suffix.lower()
    
    def read_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Read data from various file formats with enhanced error handling.
        
        Args:
            file_path: Path to the input file
            
        Returns:
            DataFrame containing the input data
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file does not exist
            DataValidationError: If data cannot be read
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Detect file format
        detected_format = self.detect_file_format(file_path)
        
        if detected_format not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {detected_format}")
        
        try:
            if detected_format == '.csv':
                # Try different encodings
                for encoding in ['utf-8', 'utf-8-sig', 'latin-1']:
                    try:
                        return pd.read_csv(file_path, encoding=encoding)
                    except UnicodeDecodeError:
                        continue
                raise ValueError("Could not read CSV file with any supported encoding")
                
            elif detected_format == '.xlsx':
                return pd.read_excel(file_path)
                
            elif detected_format == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return pd.DataFrame(data)
                    elif isinstance(data, dict):
                        return pd.DataFrame([data])
                    else:
                        raise ValueError("JSON must contain a list or dictionary")
                        
        except Exception as e:
            raise DataValidationError(f"Error reading file {file_path}: {str(e)}", [str(e)])
    
    def validate_data(self, data: pd.DataFrame) -> Dict[str, Union[bool, List[str]]]:
        """
        Validate input data structure and content with detailed error reporting.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary with validation results including errors and warnings
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check if data is empty
        if data.empty:
            validation_result['is_valid'] = False
            validation_result['errors'].append("Data is empty")
            return validation_result
        
        # Check required columns
        missing_cols = [col for col in self.required_columns if col not in data.columns]
        if missing_cols:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Missing required columns: {missing_cols}")
        
        # Validate numeric columns
        numeric_columns = [
            'quantity_kg', 'energy_consumption_kwh', 'transport_distance_km',
            'waste_generated_kg', 'recycling_rate', 'landfill_rate',
            'incineration_rate', 'carbon_footprint_kg_co2e', 'water_usage_liters'
        ]
        
        available_numeric_cols = [col for col in numeric_columns if col in data.columns]
        
        for col in available_numeric_cols:
            # Check for non-numeric values
            numeric_series = pd.to_numeric(data[col], errors='coerce')
            non_numeric_count = numeric_series.isna().sum() - data[col].isna().sum()
            
            if non_numeric_count > 0:
                validation_result['is_valid'] = False
                validation_result['errors'].append(
                    f"Column '{col}' contains {non_numeric_count} non-numeric values"
                )
            
            # Check for negative values where they shouldn't be
            if col != 'carbon_footprint_kg_co2e':  # Carbon footprint can be negative (carbon sequestration)
                if (numeric_series < 0).any():
                    validation_result['warnings'].append(
                        f"Column '{col}' contains negative values"
                    )
        
        # Validate rates sum to 1 ONLY for rows with waste generation
        rate_columns = ['recycling_rate', 'landfill_rate', 'incineration_rate']
        if all(col in data.columns for col in rate_columns + ['waste_generated_kg']):
            # Only check rates for rows where waste is generated
            waste_rows = data['waste_generated_kg'] > 0
            
            if waste_rows.any():
                waste_data = data.loc[waste_rows, rate_columns]
                rate_sums = waste_data.sum(axis=1)
                invalid_rates = ~((rate_sums - 1).abs() < 0.001)
                
                if invalid_rates.any():
                    validation_result['is_valid'] = False
                    invalid_count = invalid_rates.sum()
                    validation_result['errors'].append(
                        f"{invalid_count} rows with waste generation have rates that don't sum to 1.0"
                    )
            
            # Check for rows with zero waste but non-zero rates
            no_waste_rows = data['waste_generated_kg'] == 0
            if no_waste_rows.any():
                no_waste_data = data.loc[no_waste_rows, rate_columns]
                non_zero_rates = (no_waste_data > 0).any(axis=1)
                
                if non_zero_rates.any():
                    validation_result['warnings'].append(
                        f"{non_zero_rates.sum()} rows have zero waste but non-zero disposal rates"
                    )
        
        # Check for duplicate product_id + life_cycle_stage combinations
        if 'product_id' in data.columns and 'life_cycle_stage' in data.columns:
            duplicates = data.duplicated(['product_id', 'life_cycle_stage'])
            if duplicates.any():
                validation_result['warnings'].append(
                    f"{duplicates.sum()} duplicate product_id + life_cycle_stage combinations found"
                )
        
        return validation_result
    
    def read_impact_factors(self, file_path: Union[str, Path]) -> Dict:
        """
        Read impact factors from JSON file with enhanced error handling.
        
        Args:
            file_path: Path to the impact factors JSON file
            
        Returns:
            Dictionary containing impact factors
            
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file format is invalid
            DataValidationError: If JSON structure is invalid
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Impact factors file not found: {file_path}")
        
        if file_path.suffix.lower() != '.json':
            raise ValueError("Impact factors must be provided in JSON format")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                impact_factors = json.load(f)
            
            # Validate structure
            if not isinstance(impact_factors, dict):
                raise DataValidationError("Impact factors must be a dictionary", ["Root is not a dictionary"])
            
            # Validate nested structure
            errors = []
            for material, material_data in impact_factors.items():
                if not isinstance(material_data, dict):
                    errors.append(f"Material '{material}' data is not a dictionary")
                    continue
                
                for stage, stage_data in material_data.items():
                    if not isinstance(stage_data, dict):
                        errors.append(f"Stage '{stage}' data for material '{material}' is not a dictionary")
                        continue
                    
                    required_impacts = ['carbon_impact', 'energy_impact', 'water_impact']
                    for impact in required_impacts:
                        if impact not in stage_data:
                            errors.append(f"Missing '{impact}' in {material}.{stage}")
                        elif not isinstance(stage_data[impact], (int, float)):
                            errors.append(f"Invalid '{impact}' value in {material}.{stage}")
            
            if errors:
                raise DataValidationError("Invalid impact factors structure", errors)
            
            return impact_factors
            
        except json.JSONDecodeError as e:
            raise DataValidationError(f"Invalid JSON format: {str(e)}", [str(e)])
        except Exception as e:
            raise DataValidationError(f"Error reading impact factors: {str(e)}", [str(e)])
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess data.
        
        Args:
            data: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        cleaned = data.copy()
        
        # Remove rows with all NaN values
        cleaned = cleaned.dropna(how='all')
        
        # Fill NaN values in numeric columns with 0
        numeric_columns = [
            'quantity_kg', 'energy_consumption_kwh', 'transport_distance_km',
            'waste_generated_kg', 'recycling_rate', 'landfill_rate',
            'incineration_rate', 'carbon_footprint_kg_co2e', 'water_usage_liters'
        ]
        
        for col in numeric_columns:
            if col in cleaned.columns:
                cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce').fillna(0)
        
        # Standardize text columns
        text_columns = ['product_name', 'material_type', 'transport_mode']
        for col in text_columns:
            if col in cleaned.columns:
                cleaned[col] = cleaned[col].astype(str).str.strip()
        
        return cleaned