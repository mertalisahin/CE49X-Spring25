"""
Fixed Calculations module for LCA tool.
Handles environmental impact calculations and analysis with proper logic.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional
from pathlib import Path

class LCACalculator:
    def __init__(self, impact_factors_path: Union[str, Path] = None):
        """
        Initialize LCA Calculator with impact factors.
        
        Args:
            impact_factors_path: Path to the impact factors JSON file
        """
        self.impact_factors = self._load_impact_factors(impact_factors_path) if impact_factors_path else {}
        
        # Material name mapping to standardize input
        self.material_mapping = {
            'polyvinyl chloride': 'plastic',
            'pvc': 'plastic',
            'reinforced concrete': 'concrete',
            'structural steel': 'steel',
            'engineered wood': 'wood',
            'steel rebar': 'steel',
            'mineral wool insulation': 'mineral_wool'
        }
        
        # Life cycle stage mapping
        self.stage_mapping = {
            'end-of-life': 'disposal',
            'eol': 'disposal',
            'manufacturing': 'manufacturing',
            'transportation': 'transportation',
            'use': 'use',
            'disposal': 'disposal'
        }
        
    def _load_impact_factors(self, file_path: Union[str, Path]) -> Dict:
        """Load impact factors from JSON file with error handling."""
        try:
            from .data_input import DataInput
            data_input = DataInput()
            return data_input.read_impact_factors(file_path)
        except Exception as e:
            print(f"Could not load impact factors: {e}")
            return {}
    
    def _normalize_material_name(self, material: str) -> str:
        """Normalize material name using mapping."""
        material_lower = material.lower().strip()
        return self.material_mapping.get(material_lower, material_lower)
    
    def _normalize_stage_name(self, stage: str) -> str:
        """Normalize life cycle stage name using mapping."""
        stage_lower = stage.lower().strip()
        return self.stage_mapping.get(stage_lower, stage_lower)
    
    def calculate_impacts(self, data: pd.DataFrame, use_impact_factors: bool = True) -> pd.DataFrame:
        """
        Calculate environmental impacts for each product and life cycle stage.
        
        Args:
            data: DataFrame containing product data
            use_impact_factors: If True, use impact factors; if False, use direct measurements only
            
        Returns:
            DataFrame with calculated impacts
            
        Raises:
            ValueError: If input data is invalid
        """
        # Validate input
        if data.empty:
            raise ValueError("Input data is empty")
        
        required_cols = ['material_type', 'life_cycle_stage', 'quantity_kg']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Use vectorized operations for better performance
        data_copy = data.copy()
        
        # Normalize material and stage names
        data_copy['material_normalized'] = data_copy['material_type'].apply(self._normalize_material_name)
        data_copy['stage_normalized'] = data_copy['life_cycle_stage'].apply(self._normalize_stage_name)
        
        if use_impact_factors and self.impact_factors:
            # Calculate impacts using impact factors
            data_copy['carbon_impact'] = data_copy.apply(
                lambda row: self._calculate_single_impact(row, 'carbon_impact'), axis=1
            )
            data_copy['energy_impact'] = data_copy.apply(
                lambda row: self._calculate_single_impact(row, 'energy_impact'), axis=1
            )
            data_copy['water_impact'] = data_copy.apply(
                lambda row: self._calculate_single_impact(row, 'water_impact'), axis=1
            )
        else:
            # Use direct measurements
            data_copy['carbon_impact'] = data_copy.get('carbon_footprint_kg_co2e', 0)
            data_copy['energy_impact'] = data_copy.get('energy_consumption_kwh', 0)
            data_copy['water_impact'] = data_copy.get('water_usage_liters', 0)
        
        # Select and return relevant columns
        result_columns = [
            'product_id', 'product_name', 'life_cycle_stage', 'material_type',
            'quantity_kg', 'carbon_impact', 'energy_impact', 'water_impact',
            'waste_generated_kg', 'recycling_rate', 'landfill_rate', 'incineration_rate'
        ]
        
        # Only include columns that exist in the data
        available_columns = [col for col in result_columns if col in data_copy.columns]
        
        return data_copy[available_columns].copy()
    
    def _calculate_single_impact(self, row: pd.Series, impact_type: str) -> float:
        """Calculate single impact value for a row."""
        material = row['material_normalized']
        stage = row['stage_normalized']
        quantity = row['quantity_kg']
        
        # Get impact factors
        material_factors = self.impact_factors.get(material, {})
        stage_factors = material_factors.get(stage, {})
        impact_factor = stage_factors.get(impact_type, 0)
        
        return quantity * impact_factor
    
    def calculate_total_impacts(self, impacts: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate total impacts across all life cycle stages for each product.
        
        Args:
            impacts: DataFrame with calculated impacts
            
        Returns:
            DataFrame with total impacts per product
        """
        if impacts.empty:
            return pd.DataFrame()
        
        # Group by product and sum impacts
        grouping_cols = ['product_id']
        if 'product_name' in impacts.columns:
            grouping_cols.append('product_name')
        
        impact_cols = ['carbon_impact', 'energy_impact', 'water_impact']
        available_impact_cols = [col for col in impact_cols if col in impacts.columns]
        
        if 'waste_generated_kg' in impacts.columns:
            available_impact_cols.append('waste_generated_kg')
        
        if not available_impact_cols:
            raise ValueError("No impact columns found in data")
        
        total_impacts = impacts.groupby(grouping_cols)[available_impact_cols].sum().reset_index()
        
        return total_impacts
    
    def normalize_impacts(self, impacts: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize impacts to a common scale (0-1).
        
        Args:
            impacts: DataFrame with calculated impacts
            
        Returns:
            DataFrame with normalized impacts
        """
        if impacts.empty:
            return impacts.copy()
        
        normalized = impacts.copy()
        
        impact_columns = ['carbon_impact', 'energy_impact', 'water_impact']
        available_columns = [col for col in impact_columns if col in impacts.columns]
        
        for col in available_columns:
            max_value = impacts[col].max()
            if max_value > 0:
                normalized[f'{col}_normalized'] = impacts[col] / max_value
            else:
                normalized[f'{col}_normalized'] = 0
                
        return normalized
    
    def compare_alternatives(self, impacts: pd.DataFrame, product_ids: List[str]) -> pd.DataFrame:
        """
        Compare environmental impacts between alternative products.
        
        Args:
            impacts: DataFrame with calculated impacts
            product_ids: List of product IDs to compare
            
        Returns:
            DataFrame with comparison results
        """
        if impacts.empty:
            return pd.DataFrame()
        
        if not product_ids:
            raise ValueError("Product IDs list cannot be empty")
        
        # Filter data for specified products
        comparison = impacts[impacts['product_id'].isin(product_ids)].copy()
        
        if comparison.empty:
            print(f"No data found for products: {product_ids}")
            return pd.DataFrame()
        
        # Calculate relative differences with proper zero handling
        impact_types = ['carbon_impact', 'energy_impact', 'water_impact']
        available_types = [col for col in impact_types if col in comparison.columns]
        
        for impact_type in available_types:
            min_value = comparison[impact_type].min()
            
            if min_value == 0:
                # If minimum is zero, calculate relative to mean or set to 0
                mean_value = comparison[impact_type].mean()
                if mean_value > 0:
                    comparison[f'{impact_type}_relative'] = (
                        (comparison[impact_type] - mean_value) / mean_value * 100
                    )
                else:
                    comparison[f'{impact_type}_relative'] = 0
            else:
                comparison[f'{impact_type}_relative'] = (
                    (comparison[impact_type] - min_value) / min_value * 100
                )
            
        return comparison
    
    def get_impact_summary(self, impacts: pd.DataFrame) -> Dict[str, float]:
        """
        Get summary statistics for impacts.
        
        Args:
            impacts: DataFrame with calculated impacts
            
        Returns:
            Dictionary with summary statistics
        """
        if impacts.empty:
            return {}
        
        summary = {}
        impact_columns = ['carbon_impact', 'energy_impact', 'water_impact', 'waste_generated_kg']
        available_columns = [col for col in impact_columns if col in impacts.columns]
        
        for col in available_columns:
            summary[col] = {
                'total': impacts[col].sum(),
                'mean': impacts[col].mean(),
                'median': impacts[col].median(),
                'std': impacts[col].std(),
                'min': impacts[col].min(),
                'max': impacts[col].max()
            }
        
        return summary