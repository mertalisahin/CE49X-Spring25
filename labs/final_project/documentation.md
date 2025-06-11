# LCA Tool: Life Cycle Assessment Calculator

A comprehensive Python tool for environmental impact analysis and life cycle assessment (LCA). This tool provides a complete workflow for loading data, calculating impacts, generating reports, and creating insightful visualizations.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Modules Overview](#modules-overview)
- [Input Data Format](#input-data-format)
  - [Product Data (`sample_data.csv`)](#product-data-sample_datacs)
  - [Impact Factors (`impact_factors.json`)](#impact-factors-impact_factorsjson)

## Features

- **Flexible Data Input**: Read data from various formats including CSV, XLSX, and JSON.
- **Robust Data Validation**: Comprehensive checks for missing columns, data types, and logical consistency (e.g., disposal rates summing to 1.0).
- **Advanced Impact Calculation**: Calculate environmental impacts (carbon, energy, water) using customizable, material-specific impact factors.
- **In-depth Analysis**:
  - Aggregate impacts across the entire product life cycle.
  - Normalize impacts to a common scale for fair comparison.
  - Compare environmental performance between alternative products.
- **Rich Visualization**: Generate a variety of plots to analyze results:
  - Pie charts for impact breakdown by material or stage.
  - Bar charts for life cycle stage analysis.
  - Radar charts for multi-product, multi-criteria comparison.
  - Correlation heatmaps to understand relationships between impacts.
- **Comprehensive Reporting**: Generate and save summary reports and analysis results in multiple formats (CSV, XLSX, JSON).
- **Modular & Extensible**: A clean, package-based structure makes it easy to extend and maintain.

## Project Structure

The project is organized into a Python package (`lca_tool`) and directories for data and results.

```
.
├── src/
│   ├── __init__.py         # Makes 'lca_tool' a package and defines the public API
│   ├── calculations.py     # Core environmental impact calculation logic
│   ├── data_input.py       # Handles data loading, validation, and cleaning
│   ├── utils.py            # Utility functions (unit conversion, saving results)
│   └── visualization.py    # Generates all plots and charts
│
├── data/
│   └── raw/
│       ├── impact_factors.json # Customizable impact factors
│       └── sample_data.csv     # Input data for analysis
│
├── results/                  # Directory for generated output
│   ├── plots/                # Saved visualization images
│   ├── summary_report.json   # High-level analysis report
│   └── ...                   # Other saved result files
│
├── main.py                   # Main script to run the full analysis workflow
└── README.md                 # This documentation file
```

## Usage

The primary entry point for the application is `main.py`. This script runs the entire analysis workflow from start to finish.

**To run the analysis:**

Execute the script from the root directory of the project:

```bash
python main.py
```

**Expected Output:**

1.  **Console Logs**: The script will print detailed logs to the console, showing its progress through each step:
    - Data loading and validation status.
    - Sample views of calculated dataframes (stage impacts, total impacts, etc.).
    - Summary statistics.
    - Confirmation messages for saved files and plots.

2.  **Generated Files**: The script will create the following files in the `results/` directory:
    - `results/stage_impacts.csv`: Detailed impacts per product and life cycle stage.
    - `results/total_impacts.xlsx`: Aggregated total impacts for each product.
    - `results/comparison.json`: A comparison between specified alternative products.
    - `results/summary_report.json`: A high-level JSON report of the entire analysis.
    - `results/plots/`: This sub-directory will contain all the generated charts as PNG images.

## Modules Overview

- **`data_input.py`**: Contains the `DataInput` class, responsible for reading data from files, performing extensive validation, and cleaning the data for processing.
- **`calculations.py`**: Home to the `LCACalculator` class, which is the core engine for calculating environmental impacts using material and life-cycle-stage factors.
- **`visualization.py`**: The `LCAVisualizer` class handles all plotting. It takes processed dataframes and creates various charts to help interpret the results.
- **`utils.py`**: A collection of helper functions used across the package, including `convert_units`, `save_results`, and `create_summary_report`.
- **`__init__.py`**: Establishes the `src` directory as a package and defines its public API, making imports clean and organized.

## Input Data Format

The tool requires two primary input files.

### Product Data (`sample_data.csv`)

This file contains the raw data for the products being analyzed. It must include the following columns:

- `product_id`: A unique identifier for the product.
- `product_name`: The human-readable name of the product.
- `life_cycle_stage`: The stage being described (e.g., `manufacturing`, `transportation`, `use`, `disposal`).
- `material_type`: The primary material for that stage (e.g., `steel`, `plastic`).
- `quantity_kg`: The mass of the material in kilograms.
- `energy_consumption_kwh`: Direct energy consumption.
- `transport_distance_km`: Distance for the transportation stage.
- `transport_mode`: Method of transport (e.g., `truck`, `ship`).
- `waste_generated_kg`: Waste produced at the end-of-life stage.
- `recycling_rate`, `landfill_rate`, `incineration_rate`: Proportions (0.0 to 1.0) for how waste is managed. Must sum to 1.0 for a given entry.
- `carbon_footprint_kg_co2e`: Direct carbon emissions measurement.
- `water_usage_liters`: Direct water usage measurement.

### Impact Factors (`impact_factors.json`)

This JSON file provides the conversion factors used to calculate impacts from material quantities. It follows a nested structure:

`Material -> Life Cycle Stage -> Impact Type`

**Example Structure:**

```json
{
  "steel": {
    "manufacturing": {
      "carbon_impact": 1.8,
      "energy_impact": 20.0,
      "water_impact": 250.0
    },
    "disposal": {
      "carbon_impact": 0.1,
      "energy_impact": 0.5,
      "water_impact": 5.0
    }
  },
  "plastic": {
    "manufacturing": {
      "carbon_impact": 2.5,
      "energy_impact": 70.0,
      "water_impact": 180.0
    },
    "disposal": {
      "carbon_impact": 0.5,
      "energy_impact": 1.2,
      "water_impact": 10.0
    }
  }
}

