# LCA (Life Cycle Assessment) Tool

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

A comprehensive Python tool for environmental impact analysis and life cycle assessment (LCA). This tool provides a complete workflow for loading, validating, analyzing, and visualizing environmental data for various products and materials.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Input Data Format](#input-data-format)
  - [Sample Data File (`.csv`, `.xlsx`, `.json`)](#sample-data-file-csv-xlsx-json)
  - [Impact Factors File (`.json`)](#impact-factors-file-json)
- [Output Description](#output-description)
- [Module & API Reference](#module--api-reference)
  - [1. `lca_tool` Package (`src/__init__.py`)](#1-lca_tool-package-src__init__py)
  - [2. Data Input Module (`src/data_input.py`)](#2-data-input-module-srcdata_inputpy)
  - [3. Calculations Module (`src/calculations.py`)](#3-calculations-module-srccalculationspy)
  - [4. Visualization Module (`src/visualization.py`)](#4-visualization-module-srcvisualizationpy)
  - [5. Utilities Module (`src/utils.py`)](#5-utilities-module-srcutilspy)
  - [6. Main Execution Script (`main.py`)](#6-main-execution-script-mainpy)
- [Error Handling](#error-handling)

## Overview

The LCA Tool is a modular and extensible command-line application designed to perform life cycle assessments. It simplifies the process of quantifying the environmental impacts of a product, from raw material extraction through manufacturing, transportation, use, and end-of-life disposal. The tool is built with a clear separation of concerns, making it easy to extend or modify.

The core workflow is as follows:
1.  **Load Data**: Ingests product and material data from various file formats.
2.  **Validate & Clean**: Ensures data integrity and cleans it for processing.
3.  **Calculate Impacts**: Computes environmental impacts (carbon, energy, water, etc.) using customizable impact factors.
4.  **Analyze & Report**: Aggregates results, compares alternatives, and generates summary reports.
5.  **Visualize**: Creates insightful charts and plots to communicate the results effectively.

## Features

- **Flexible Data Input**: Reads data from `.csv`, `.xlsx`, and `.json` files with auto-detection.
- **Robust Validation**: Comprehensive data validation with detailed error and warning messages.
- **Advanced Calculations**:
    - Calculates impacts based on life cycle stages and material types.
    - Aggregates impacts per product.
    - Normalizes results for easy comparison.
    - Compares environmental performance of alternative products.
- **Powerful Visualizations**:
    - Pie charts for impact breakdown (e.g., carbon impact by material).
    - Bar charts for stage-by-stage impact analysis.
    - Radar charts for multi-product, multi-criteria comparison.
    - Correlation heatmaps to show relationships between different impacts.
- **Comprehensive Reporting**: Generates a detailed JSON summary report with metadata, data quality metrics, and impact statistics.
- **Utility Functions**: Includes helpers for unit conversion, secure file saving, and more.
- **Custom Exceptions**: Provides clear, specific exceptions for better error handling.

## Project Structure

The project is organized into a source package (`src`), data and results directories, and a main execution script.

```
lca_tool_project/
├── data/
│   └── raw/
│       ├── sample_data.csv        # Primary input data for products
│       └── impact_factors.json    # Environmental impact factors per material/stage
├── results/
│   ├── plots/                     # Directory for saved plots
│   ├── stage_impacts.csv          # Example output file
│   ├── total_impacts.xlsx         # Example output file
│   └── summary_report.json        # Example output file
├── src/
│   ├── __init__.py                # Initializes the 'lca_tool' package and exports components
│   ├── data_input.py              # Handles data loading, validation, and cleaning
│   ├── calculations.py            # Performs all LCA calculations
│   ├── visualization.py           # Generates all plots and charts
│   └── utils.py                   # Contains helper functions and unit conversions
├── main.py                        # Main script to run the full analysis workflow
└── requirements.txt               # Project dependencies
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd lca_tool_project
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    The project requires the following Python libraries. Install them using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
    The `requirements.txt` file should contain:
    ```
    pandas
    numpy
    matplotlib
    seaborn
    openpyxl
    ```

## Usage

The `main.py` script serves as the primary entry point for running a full analysis. It demonstrates the entire workflow from data loading to visualization.

To run the analysis, simply execute the script from the root directory of the project:

```bash
python main.py
```

The script will:
1.  Log its progress to the console.
2.  Load data from `data/raw/sample_data.csv` and `data/raw/impact_factors.json`.
3.  Validate and clean the data.
4.  Perform all calculations and print summaries to the console.
5.  Save the results (stage-level impacts, total impacts, comparison) to the `results/` directory.
6.  Generate and save all plots to the `results/plots/` directory.

## Input Data Format

### Sample Data File (`.csv`, `.xlsx`, `.json`)

The primary input data file contains information about each product at different life cycle stages. It must contain the following columns:

- `product_id`: Unique identifier for the product.
- `product_name`: Name of the product.
- `life_cycle_stage`: Stage of life cycle (e.g., `manufacturing`, `transportation`).
- `material_type`: The primary material used (e.g., `steel`, `plastic`).
- `quantity_kg`: Mass of the material in kilograms.
- `energy_consumption_kwh`: Energy used in kWh.
- `transport_distance_km`: Distance transported in kilometers.
- `transport_mode`: Mode of transport (e.g., `truck`, `ship`).
- `waste_generated_kg`: Waste produced in kilograms.
- `recycling_rate`: Proportion of waste recycled (0.0 to 1.0).
- `landfill_rate`: Proportion of waste sent to landfill (0.0 to 1.0).
- `incineration_rate`: Proportion of waste incinerated (0.0 to 1.0).
- `carbon_footprint_kg_co2e`: Direct carbon footprint measurement.
- `water_usage_liters`: Direct water usage measurement.

### Impact Factors File (`.json`)

This file provides the conversion factors used to calculate impacts from material quantities. It follows a nested JSON structure:

`{ "material_name": { "life_cycle_stage": { "impact_type": value } } }`

- **`material_name`**: The name of the material (e.g., `steel`, `plastic`), matching `material_type` in the data file.
- **`life_cycle_stage`**: The stage (e.g., `manufacturing`), matching `life_cycle_stage` in the data file.
- **`impact_type`**: The environmental impact category (e.g., `carbon_impact`, `energy_impact`).

**Example `impact_factors.json`:**
```json
{
  "steel": {
    "manufacturing": {
      "carbon_impact": 1.8,
      "energy_impact": 20,
      "water_impact": 250
    },
    "disposal": {
      "carbon_impact": 0.1,
      "energy_impact": 0.5,
      "water_impact": 10
    }
  },
  "plastic": {
    "manufacturing": {
      "carbon_impact": 2.5,
      "energy_impact": 70,
      "water_impact": 180
    }
  }
}
```

## Output Description

After running `main.py`, the `results/` directory will be populated with the following:

- **`results/stage_impacts.csv`**: Detailed impacts calculated for each product, material, and life cycle stage.
- **`results/total_impacts.xlsx`**: Aggregated total impacts for each unique product across all stages.
- **`results/comparison.json`**: A JSON file containing the comparison data for selected products.
- **`results/summary_report.json`**: A comprehensive report with metadata, data quality analysis, and overall impact summaries.
- **`results/plots/`**: This subdirectory will contain all generated visualizations as PNG images:
    - `carbon_by_material.png`: Pie chart of carbon impact breakdown.
    - `p001_lifecycle_impacts.png`: Bar chart of impacts for product P001.
    - `product_comparison_radar.png`: Radar chart comparing products.
    - `p002_eol_breakdown.png`: Stacked bar chart for end-of-life management.
    - `impact_correlation_heatmap.png`: Heatmap of correlations between impact types.

---

## Module & API Reference

### 1. `lca_tool` Package (`src/__init__.py`)

This file initializes the `lca_tool` as a Python package. It conveniently exports the main classes, functions, and exceptions from the various modules, allowing for easy imports.

**Exports:**
- **Classes**: `DataInput`, `LCACalculator`, `LCAVisualizer`
- **Exceptions**: `DataValidationError`, `UnitConversionError`, `VisualizationError`
- **Functions**: `convert_units`, `save_results`, `create_summary_report`, etc.

### 2. Data Input Module (`src/data_input.py`)

Handles all data ingestion, validation, and preprocessing tasks.

- **`class DataInput`**:
    - `read_data(file_path)`: Reads data from CSV, XLSX, or JSON files. Handles different encodings and raises `FileNotFoundError` or `DataValidationError`.
    - `validate_data(data)`: Performs extensive checks on the DataFrame, including for missing columns, non-numeric values, and logical inconsistencies (e.g., disposal rates summing to 1.0). Returns a dictionary with a validity status, errors, and warnings.
    - `clean_data(data)`: Preprocesses the data by filling null numeric values with 0, trimming whitespace from text, and removing empty rows.
    - `read_impact_factors(file_path)`: Loads and validates the structure of the impact factors JSON file.

### 3. Calculations Module (`src/calculations.py`)

The core engine for all environmental impact calculations.

- **`class LCACalculator`**:
    - `__init__(impact_factors_path)`: Initializes the calculator, loading impact factors from the specified path.
    - `calculate_impacts(data, use_impact_factors)`: Calculates impacts for each row in the input data. If `use_impact_factors` is `True`, it multiplies material quantity by the corresponding impact factor. Otherwise, it uses direct measurements from the input file.
    - `calculate_total_impacts(impacts)`: Groups the stage-level impacts by product ID to compute the total environmental footprint for each product.
    - `normalize_impacts(impacts)`: Normalizes impact data to a 0-1 scale, which is useful for comparison.
    - `compare_alternatives(impacts, product_ids)`: Filters data for a list of products and calculates their relative impact differences, providing a basis for comparison.
    - `get_impact_summary(impacts)`: Computes descriptive statistics (total, mean, min, max, etc.) for each impact category.

### 4. Visualization Module (`src/visualization.py`)

Responsible for creating all plots and charts to visualize the analysis results.

- **`class LCAVisualizer`**:
    - `plot_impact_breakdown(...)`: Creates a pie chart to show the contribution of different groups (e.g., materials) to a specific impact. Intelligently groups small slices into an "Other" category for clarity.
    - `plot_life_cycle_impacts(...)`: Generates bar charts showing the impacts of a single product across its different life cycle stages.
    - `plot_product_comparison(...)`: Creates a radar chart to compare multiple products across several normalized impact categories simultaneously.
    - `plot_end_of_life_breakdown(...)`: Visualizes the end-of-life scenario (recycling, landfill, incineration rates) for a product using a stacked bar chart.
    - `plot_impact_correlation(...)`: Generates a heatmap to show the correlation between different impact categories (e.g., if carbon impact is strongly correlated with energy impact).
    - `close_all_figures()`: A utility to close all open Matplotlib figures and free up memory.

### 5. Utilities Module (`src/utils.py`)

A collection of helper functions used across the application.

- **`convert_units(value, from_unit, to_unit)`**: A powerful unit converter for mass, volume, and energy. Raises `UnitConversionError` for invalid conversions.
- **`save_results(data, file_path, format)`**: Saves a DataFrame to disk in CSV, XLSX, or JSON format with robust error handling.
- **`load_impact_factors(file_path)`**: A standalone function to load impact factors from a JSON file.
- **`validate_numeric_range(value, min_value, max_value)`**: Checks if a numeric value falls within a specified range.
- **`create_summary_report(data, impacts)`**: Compiles a comprehensive dictionary summarizing the entire analysis.
- **`get_supported_units()`**: Returns a dictionary of all unit types and their supported units.

### 6. Main Execution Script (`main.py`)

The orchestration script that ties all the modules together to perform a complete LCA workflow. It is heavily commented and serves as an excellent example of how to use the `lca_tool` package.

## Error Handling

The application uses custom exceptions to provide specific and informative error messages:
- **`DataValidationError`**: Raised by `data_input.py` for issues with input data format or content.
- **`UnitConversionError`**: Raised by `utils.py` when a unit conversion is not possible.
- **`VisualizationError`**: Raised by `visualization.py` if a plot cannot be generated due to missing data or other issues.
- Standard Python exceptions like `FileNotFoundError` and `ValueError` are also handled gracefully.