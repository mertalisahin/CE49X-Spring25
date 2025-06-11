"""
Main execution script for the Life Cycle Assessment (LCA) Tool.

This script demonstrates the full workflow of the LCA tool:
1.  Loading and validating input data and impact factors.
2.  Cleaning and preparing the data for analysis.
3.  Calculating environmental impacts based on life cycle stages.
4.  Aggregating, normalizing, and comparing impacts.
5.  Generating a comprehensive summary report.
6.  Creating and saving various visualizations.
7.  Saving the results to different file formats.
8.  Demonstrating utility functions like unit conversion.
"""

import logging
from pathlib import Path
import pandas as pd

# Import all necessary components from the lca_tool package
from src import (
    DataInput,
    LCACalculator,
    LCAVisualizer,
    DataValidationError,
    VisualizationError,
    UnitConversionError,
    convert_units,
    save_results,
    load_impact_factors,
    validate_numeric_range,
    create_summary_report,
    get_supported_units
)

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define file paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

# Input files
SAMPLE_DATA_PATH = DATA_DIR / "sample_data.csv"
IMPACT_FACTORS_PATH = DATA_DIR / "impact_factors.json"


def main():
    """Main function to run the LCA analysis workflow."""
    logging.info("Starting LCA Tool analysis workflow...")

    # Create output directories if they don't exist
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    try:
        # --- 1. INITIALIZATION ---
        logging.info("Step 1: Initializing components...")
        data_input = DataInput()
        visualizer = LCAVisualizer(style='seaborn-v0_8-whitegrid')
        # The calculator will be initialized after loading impact factors

        # --- 2. DATA LOADING & VALIDATION (from data_input.py) ---
        logging.info(f"Step 2: Loading data from {SAMPLE_DATA_PATH}")
        raw_data = data_input.read_data(SAMPLE_DATA_PATH)
        logging.info(f"Successfully loaded {len(raw_data)} records.")

        logging.info("Validating raw data structure and content...")
        validation_results = data_input.validate_data(raw_data)
        if not validation_results['is_valid']:
            logging.error("Data validation failed!")
            for error in validation_results['errors']:
                logging.error(f" - {error}")
            # Exit if data is critically invalid
            return
        if validation_results['warnings']:
            for warning in validation_results['warnings']:
                logging.warning(f" - {warning}")
        logging.info("Data validation successful.")

        logging.info("Cleaning and preprocessing data...")
        cleaned_data = data_input.clean_data(raw_data)
        logging.info("Data cleaning complete.")

        # --- 3. UTILITIES (from utils.py) ---
        logging.info("Step 3: Demonstrating utility functions...")
        
        # Demonstrate get_supported_units
        supported_units = get_supported_units()
        logging.info(f"Supported units: {supported_units}")

        # Demonstrate convert_units
        try:
            mass_kg = 1500
            mass_tons = convert_units(mass_kg, 'kg', 'ton')
            logging.info(f"Unit Conversion: {mass_kg} kg is equal to {mass_tons:.2f} tons.")
        except UnitConversionError as e:
            logging.error(f"Unit conversion failed: {e}")

        # Demonstrate validate_numeric_range
        try:
            sample_rate = 0.85
            validate_numeric_range(sample_rate, 0, 1, name="Recycling Rate")
            logging.info(f"Numeric Range Validation: Rate {sample_rate} is valid (between 0 and 1).")
        except (ValueError, TypeError) as e:
            logging.error(f"Numeric range validation failed: {e}")
            
        # Demonstrate load_impact_factors (from utils.py)
        # Note: LCACalculator also has its own loading mechanism, but we demonstrate the utility here.
        logging.info(f"Loading impact factors from {IMPACT_FACTORS_PATH} using utility function...")
        impact_factors_dict = load_impact_factors(IMPACT_FACTORS_PATH)

        # --- 4. CALCULATIONS (from calculations.py) ---
        logging.info("Step 4: Performing LCA calculations...")
        # Initialize calculator with the path to the factors file
        calculator = LCACalculator(impact_factors_path=IMPACT_FACTORS_PATH)

        logging.info("Calculating impacts per life cycle stage...")
        stage_impacts = calculator.calculate_impacts(cleaned_data)
        logging.info("Stage-level impact calculation complete.")
        print("\n--- Stage-Level Impacts (Sample) ---")
        print(stage_impacts.head())

        logging.info("Calculating total impacts per product...")
        total_impacts = calculator.calculate_total_impacts(stage_impacts)
        logging.info("Total impact calculation complete.")
        print("\n--- Total Impacts per Product ---")
        print(total_impacts)
        
        logging.info("Normalizing total impacts...")
        normalized_impacts = calculator.normalize_impacts(total_impacts)
        logging.info("Normalization complete.")
        print("\n--- Normalized Total Impacts ---")
        print(normalized_impacts)

        logging.info("Comparing alternative products...")
        products_to_compare = ['P001', 'P002']
        comparison_results = calculator.compare_alternatives(total_impacts, products_to_compare)
        logging.info(f"Comparison complete for products: {products_to_compare}.")
        print("\n--- Product Comparison Results ---")
        print(comparison_results)
        
        logging.info("Generating impact summary statistics...")
        impact_summary = calculator.get_impact_summary(stage_impacts)
        logging.info("Summary generation complete.")
        print("\n--- Impact Summary Statistics ---")
        # Pretty print the summary dictionary
        for impact, stats in impact_summary.items():
            print(f"  {impact}:")
            for stat, value in stats.items():
                print(f"    - {stat}: {value:.2f}")

        # --- 5. REPORTING & SAVING (from utils.py) ---
        logging.info("Step 5: Creating summary report and saving results...")
        
        # Create summary report
        summary_report = create_summary_report(cleaned_data, stage_impacts)
        logging.info("Summary report generated.")
        # Save the report as a JSON file
        report_path = RESULTS_DIR / "summary_report.json"
        save_results(pd.DataFrame([summary_report]), report_path, format='json')
        logging.info(f"Summary report saved to {report_path}")

        # Save results in different formats
        save_results(stage_impacts, RESULTS_DIR / "stage_impacts.csv", format='csv')
        save_results(total_impacts, RESULTS_DIR / "total_impacts.xlsx", format='xlsx')
        save_results(comparison_results, RESULTS_DIR / "comparison.json", format='json')

        # --- 6. VISUALIZATION (from visualization.py) ---
        logging.info("Step 6: Generating and saving visualizations...")

        try:
            # Plot 1: Impact Breakdown Pie Chart
            fig1 = visualizer.plot_impact_breakdown(
                stage_impacts, 'carbon_impact', group_by='material_type',
                save_path=PLOTS_DIR / "carbon_by_material.png"
            )
            logging.info("Generated: Carbon impact breakdown by material.")

            # Plot 2: Life Cycle Impacts Bar Chart for a single product
            fig2 = visualizer.plot_life_cycle_impacts(
                stage_impacts, product_id='P001',
                save_path=PLOTS_DIR / "p001_lifecycle_impacts.png"
            )
            logging.info("Generated: Life cycle impacts for product P001.")

            # Plot 3: Product Comparison Radar Chart
            fig3 = visualizer.plot_product_comparison(
                total_impacts, product_ids=['P001', 'P002', 'P003'],
                save_path=PLOTS_DIR / "product_comparison_radar.png"
            )
            logging.info("Generated: Product comparison radar chart.")

            # Plot 4: End-of-Life Breakdown
            fig4 = visualizer.plot_end_of_life_breakdown(
                stage_impacts, product_id='P002',
                save_path=PLOTS_DIR / "p002_eol_breakdown.png"
            )
            logging.info("Generated: End-of-life breakdown for product P002.")

            # Plot 5: Impact Correlation Heatmap
            fig5 = visualizer.plot_impact_correlation(
                total_impacts,
                save_path=PLOTS_DIR / "impact_correlation_heatmap.png"
            )
            logging.info("Generated: Impact correlation heatmap.")

        except VisualizationError as e:
            logging.error(f"A visualization could not be created: {e}")
        finally:
            # Close all figures to free up memory
            visualizer.close_all_figures()
            logging.info("All plot figures have been closed.")

        logging.info("LCA Tool analysis workflow completed successfully!")

    except FileNotFoundError as e:
        logging.error(f"A required file was not found: {e}")
    except DataValidationError as e:
        logging.error(f"Data validation error: {e.message}")
        for err in e.errors:
            logging.error(f" - {err}")
    except ValueError as e:
        logging.error(f"A value error occurred: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()