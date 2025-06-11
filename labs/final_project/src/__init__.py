"""
LCA (Life Cycle Assessment) Tool Package
=========================================

A comprehensive tool for environmental impact analysis and life cycle assessment.

Modules:
    data_input: Handles reading and validation of input data from various sources
    calculations: Performs environmental impact calculations and analysis
    visualization: Creates plots and charts for impact analysis
    utils: Utility functions and helper methods

Classes:
    DataInput: Main class for data input and validation
    LCACalculator: Core calculator for environmental impacts
    LCAVisualizer: Visualization engine for creating plots and charts

Exceptions:
    DataValidationError: Custom exception for data validation errors
    UnitConversionError: Custom exception for unit conversion errors
    VisualizationError: Custom exception for visualization errors

Example:
    >>> from src import DataInput, LCACalculator, LCAVisualizer
    >>> 
    >>> # Load and validate data
    >>> data_input = DataInput()
    >>> data = data_input.read_data('input_data.csv')
    >>> validation = data_input.validate_data(data)
    >>> 
    >>> # Calculate impacts
    >>> calculator = LCACalculator()
    >>> impacts = calculator.calculate_impacts(data)
    >>> 
    >>> # Create visualizations
    >>> visualizer = LCAVisualizer()
    >>> fig = visualizer.plot_impact_breakdown(impacts, 'carbon_impact')
"""

# Import main classes
from .data_input import DataInput, DataValidationError
from .calculations import LCACalculator
from .visualization import LCAVisualizer, VisualizationError
from .utils import (
    UnitConversionError,
    convert_units,
    save_results,
    load_impact_factors,
    validate_numeric_range,
    create_summary_report,
    get_supported_units
)

# Package metadata
__version__ = "1.0.0"
__author__ = "Mert Ali Şahin"
__email__ = "mail@mertalisahin.com"
__description__ = "A comprehensive tool for environmental impact analysis and life cycle assessment"

# Define what gets imported with "from src import *"
__all__ = [
    # Main classes
    'DataInput',
    'LCACalculator', 
    'LCAVisualizer',
    
    # Exceptions
    'DataValidationError',
    'UnitConversionError',
    'VisualizationError',
    
    # Utility functions
    'convert_units',
    'save_results',
    'load_impact_factors',
    'validate_numeric_range',
    'create_summary_report',
    'get_supported_units',
    
    # Package info
    '__version__',
    '__author__',
    '__email__',
    '__description__'
]

# Package configuration
SUPPORTED_FILE_FORMATS = ['.csv', '.xlsx', '.json']
DEFAULT_IMPACT_CATEGORIES = ['carbon_impact', 'energy_impact', 'water_impact', 'waste_generated_kg']
DEFAULT_LIFE_CYCLE_STAGES = ['manufacturing', 'transportation', 'use', 'disposal']

def get_package_info():
    """
    Get package information and configuration.
    
    Returns:
        dict: Package information including version, supported formats, etc.
    """
    return {
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'supported_formats': SUPPORTED_FILE_FORMATS,
        'default_impact_categories': DEFAULT_IMPACT_CATEGORIES,
        'default_life_cycle_stages': DEFAULT_LIFE_CYCLE_STAGES
    }

