"""
Fixed Visualization module for LCA tool.
Handles creation of plots and charts for impact analysis with proper error handling.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Union, Tuple
import numpy as np
from pathlib import Path



class VisualizationError(Exception):
    """Custom exception for visualization errors."""
    pass

class LCAVisualizer:
    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        Initialize visualizer with modern matplotlib style.
        
        Args:
            style: Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except OSError:
            # Fallback to default style if seaborn is not available
            plt.style.use('default')
            print(f"Style '{style}' not available, using 'default'")
        
        self.default_figsize = (10, 6)
        self.impact_labels = {
            'carbon_impact': 'Carbon Impact (kg CO2e)',
            'energy_impact': 'Energy Impact (kWh)',
            'water_impact': 'Water Impact (L)',
            'waste_generated_kg': 'Waste Generated (kg)'
        }
    
    def _get_colors(self, n_colors: int) -> List:
        """Generate color palette with proper size."""
        if n_colors <= 10:
            return sns.color_palette("husl", n_colors)
        else:
            # For large number of colors, use a continuous palette
            return sns.color_palette("husl", n_colors)
    
    def _validate_plot_data(self, data: pd.DataFrame, required_columns: List[str]) -> None:
        """Validate data for plotting."""
        if data.empty:
            raise VisualizationError("Cannot create plot: data is empty")
        
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise VisualizationError(f"Missing required columns: {missing_cols}")
        
        # Check if there's actual data to plot
        for col in required_columns:
            if col in data.columns and data[col].notna().sum() == 0:
                print(f"Column '{col}' contains only NaN values")
    
    def plot_impact_breakdown(self, data: pd.DataFrame, impact_type: str, 
                            group_by: str = 'material_type',
                            title: Optional[str] = None,
                            figsize: Tuple[float, float] = None,
                            save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Create a pie chart showing impact breakdown by specified grouping.
        
        Args:
            data: DataFrame with impact data
            impact_type: Type of impact to plot
            group_by: Column to group by
            title: Optional title for the plot
            figsize: Figure size tuple
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        required_cols = [impact_type, group_by]
        self._validate_plot_data(data, required_cols)
        
        figsize = figsize or self.default_figsize
        fig, ax = plt.subplots(figsize=figsize)
        
        try:
            # Group and sum impacts
            impact_data = data.groupby(group_by)[impact_type].sum()
            
            # Remove zero values
            impact_data = impact_data[impact_data > 0]
            
            if impact_data.empty:
                ax.text(0.5, 0.5, 'No data to display', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            else:
                colors = self._get_colors(len(impact_data))
                wedges, texts, autotexts = ax.pie(impact_data.values, 
                                                 labels=impact_data.index, 
                                                 autopct='%1.1f%%',
                                                 colors=colors,
                                                 startangle=90)
                
                # Improve text readability
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
            
            # Set title
            if title:
                ax.set_title(title, fontsize=14, fontweight='bold')
            else:
                impact_label = self.impact_labels.get(impact_type, impact_type)
                group_label = group_by.replace('_', ' ').title()
                ax.set_title(f'{impact_label} by {group_label}', 
                           fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                
            return fig
            
        except Exception as e:
            plt.close(fig)
            raise VisualizationError(f"Error creating pie chart: {str(e)}")
    
    def plot_life_cycle_impacts(self, data: pd.DataFrame, 
                              product_id: str,
                              figsize: Tuple[float, float] = None,
                              save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Create a stacked bar chart showing impacts across life cycle stages.
        
        Args:
            data: DataFrame with impact data
            product_id: Product ID to analyze
            figsize: Figure size tuple
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        required_cols = ['product_id', 'life_cycle_stage']
        self._validate_plot_data(data, required_cols)
        
        # Filter data for specific product
        product_data = data[data['product_id'] == product_id]
        
        if product_data.empty:
            raise VisualizationError(f"No data found for product ID: {product_id}")
        
        figsize = figsize or (15, 12)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        impact_types = ['carbon_impact', 'energy_impact', 'water_impact', 'waste_generated_kg']
        available_impacts = [imp for imp in impact_types if imp in product_data.columns]
        
        if not available_impacts:
            plt.close(fig)
            raise VisualizationError("No impact columns found in data")
        
        try:
            for idx, impact_type in enumerate(available_impacts):
                if idx >= 4:  # Only plot first 4 impacts
                    break
                
                # Create pivot table for this impact
                try:
                    stage_data = product_data.pivot_table(
                        index='life_cycle_stage',
                        values=impact_type,
                        aggfunc='sum'
                    )
                    
                    if not stage_data.empty and stage_data[impact_type].sum() > 0:
                        colors = self._get_colors(len(stage_data))
                        stage_data.plot(kind='bar', ax=axes[idx], 
                                      color=colors[0], alpha=0.7)
                        
                        # Customize subplot
                        impact_label = self.impact_labels.get(impact_type, impact_type)
                        axes[idx].set_title(impact_label, fontweight='bold')
                        axes[idx].set_xlabel('Life Cycle Stage')
                        axes[idx].set_ylabel('Impact Value')
                        axes[idx].tick_params(axis='x', rotation=45)
                        axes[idx].grid(True, alpha=0.3)
                    else:
                        axes[idx].text(0.5, 0.5, 'No data', 
                                     horizontalalignment='center', 
                                     verticalalignment='center',
                                     transform=axes[idx].transAxes)
                
                except Exception as e:
                    print(f"Could not plot {impact_type}: {e}")
                    axes[idx].text(0.5, 0.5, f'Error: {str(e)}', 
                                 horizontalalignment='center', 
                                 verticalalignment='center',
                                 transform=axes[idx].transAxes)
            
            # Hide unused subplots
            for idx in range(len(available_impacts), 4):
                axes[idx].set_visible(False)
            
            plt.suptitle(f'Life Cycle Impacts for Product {product_id}', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            plt.close(fig)
            raise VisualizationError(f"Error creating life cycle plot: {str(e)}")
    
    def plot_product_comparison(self, data: pd.DataFrame, 
                              product_ids: List[str],
                              figsize: Tuple[float, float] = None,
                              save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Create a radar chart comparing multiple products across impact categories.
        
        Args:
            data: DataFrame with impact data
            product_ids: List of product IDs to compare
            figsize: Figure size tuple
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        required_cols = ['product_id']
        self._validate_plot_data(data, required_cols)
        
        if not product_ids:
            raise VisualizationError("Product IDs list cannot be empty")
        
        # Calculate total impacts for each product
        try:
            total_impacts = data[data['product_id'].isin(product_ids)].groupby('product_id').agg({
                col: 'sum' for col in ['carbon_impact', 'energy_impact', 'water_impact', 'waste_generated_kg']
                if col in data.columns
            })
            
            if total_impacts.empty:
                raise VisualizationError(f"No data found for products: {product_ids}")
            
            # Normalize the data (handle zero values properly)
            normalized = total_impacts.copy()
            for col in total_impacts.columns:
                max_val = total_impacts[col].max()
                if max_val > 0:
                    normalized[col] = total_impacts[col] / max_val
                else:
                    normalized[col] = 0  # All values are zero
            
            # Create radar chart
            categories = [self.impact_labels.get(col, col) for col in normalized.columns]
            num_vars = len(categories)
            
            if num_vars == 0:
                raise VisualizationError("No impact categories available for comparison")
            
            angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
            angles += angles[:1]  # Complete the circle
            
            figsize = figsize or (10, 10)
            fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
            
            colors = self._get_colors(len(product_ids))
            
            for idx, product_id in enumerate(product_ids):
                if product_id in normalized.index:
                    values = normalized.loc[product_id].values.tolist()
                    values += values[:1]  # Complete the circle
                    
                    ax.plot(angles, values, linewidth=2, 
                           label=product_id, color=colors[idx])
                    ax.fill(angles, values, alpha=0.1, color=colors[idx])
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 1)
            ax.set_title('Product Comparison Across Impact Categories', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.grid(True)
            
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            if 'fig' in locals():
                plt.close(fig)
            raise VisualizationError(f"Error creating radar chart: {str(e)}")
    
    def plot_end_of_life_breakdown(self, data: pd.DataFrame, 
                                 product_id: str,
                                 figsize: Tuple[float, float] = None,
                                 save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Create a stacked bar chart showing end-of-life management breakdown.
        
        Args:
            data: DataFrame with impact data
            product_id: Product ID to analyze
            figsize: Figure size tuple
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        required_cols = ['product_id', 'recycling_rate', 'landfill_rate', 'incineration_rate']
        self._validate_plot_data(data, required_cols)
        
        product_data = data[data['product_id'] == product_id]
        
        if product_data.empty:
            raise VisualizationError(f"No data found for product ID: {product_id}")
        
        figsize = figsize or self.default_figsize
        fig, ax = plt.subplots(figsize=figsize)
        
        try:
            eol_columns = ['recycling_rate', 'landfill_rate', 'incineration_rate']
            eol_data = product_data[eol_columns]
            
            # Only plot rows with actual waste (non-zero rates)
            mask = (eol_data.sum(axis=1) > 0)
            if mask.any():
                plot_data = eol_data[mask]
                plot_data.plot(kind='bar', stacked=True, ax=ax, 
                             color=['green', 'red', 'orange'],
                             alpha=0.7)
                
                ax.set_title(f'End-of-Life Management for Product {product_id}',
                           fontweight='bold')
                ax.set_xlabel('Life Cycle Stage Index')
                ax.set_ylabel('Rate')
                ax.set_ylim(0, 1)
                ax.legend(['Recycling', 'Landfill', 'Incineration'])
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
            else:
                ax.text(0.5, 0.5, 'No end-of-life data available', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=12)
            
            plt.tight_layout()
            
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            plt.close(fig)
            raise VisualizationError(f"Error creating end-of-life plot: {str(e)}")
    
    def plot_impact_correlation(self, data: pd.DataFrame,
                              figsize: Tuple[float, float] = None,
                              save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Create a correlation heatmap of different impact categories.
        
        Args:
            data: DataFrame with impact data
            figsize: Figure size tuple
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        impact_columns = ['carbon_impact', 'energy_impact', 'water_impact', 'waste_generated_kg']
        available_columns = [col for col in impact_columns if col in data.columns]
        
        if len(available_columns) < 2:
            raise VisualizationError("Need at least 2 impact columns for correlation analysis")
        
        figsize = figsize or (10, 8)
        fig, ax = plt.subplots(figsize=figsize)
        
        try:
            correlation = data[available_columns].corr()
            
            # Create heatmap
            mask = np.triu(np.ones_like(correlation, dtype=bool))
            sns.heatmap(correlation, mask=mask, annot=True, cmap='coolwarm', 
                       center=0, ax=ax, square=True, cbar_kws={'shrink': 0.8})
            
            ax.set_title('Impact Category Correlations', fontsize=14, fontweight='bold')
            
            # Improve labels
            labels = [self.impact_labels.get(col, col) for col in available_columns]
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_yticklabels(labels, rotation=0)
            
            plt.tight_layout()
            
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            plt.close(fig)
            raise VisualizationError(f"Error creating correlation plot: {str(e)}")
    
    def close_all_figures(self):
        """Close all matplotlib figures to prevent memory leaks."""
        plt.close('all')