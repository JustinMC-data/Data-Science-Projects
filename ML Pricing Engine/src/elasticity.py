"""
elasticity.py
-------------
Functions for calculating and analyzing price elasticity of demand.

Price elasticity measures how sensitive demand is to price changes:
- Elastic (|E| > 1): Demand changes more than price (luxury goods)
- Unit Elastic (|E| = 1): Demand changes proportionally to price
- Inelastic (|E| < 1): Demand changes less than price (necessities)

This module handles:
- Point elasticity calculation
- Arc elasticity calculation
- Demand curve simulation
- Revenue optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import warnings


def calculate_point_elasticity(model: Any,
                               base_features: pd.DataFrame,
                               price_col: str = 'AvgPrice',
                               pct_change: float = 0.01) -> float:
    """
    Calculate point elasticity at a specific price point.
    
    Point elasticity = (% change in quantity) / (% change in price)
    
    We estimate this by making a small price change and measuring
    the predicted quantity change.
    
    Parameters
    ----------
    model : Any
        Trained model with predict() method
    base_features : pd.DataFrame
        Single row of features at the base price point
    price_col : str
        Name of the price column in features
    pct_change : float
        Percentage change to use for calculation (default 1%)
        
    Returns
    -------
    float
        Point elasticity estimate
    """
    # Get base prediction
    base_price = base_features[price_col].values[0]
    base_qty = model.predict(base_features)[0]
    
    # Create slightly higher price scenario
    high_features = base_features.copy()
    high_features[price_col] = base_price * (1 + pct_change)
    high_qty = model.predict(high_features)[0]
    
    # Calculate elasticity
    pct_qty_change = (high_qty - base_qty) / base_qty if base_qty != 0 else 0
    elasticity = pct_qty_change / pct_change
    
    return elasticity


def simulate_demand_curve(model: Any,
                          base_features: pd.DataFrame,
                          price_col: str = 'AvgPrice',
                          price_range: Tuple[float, float] = None,
                          n_points: int = 50) -> pd.DataFrame:
    """
    Simulate demand at various price points to create a demand curve.
    
    Parameters
    ----------
    model : Any
        Trained model with predict() method
    base_features : pd.DataFrame
        Single row of features to use as template
    price_col : str
        Name of the price column
    price_range : Tuple[float, float]
        Min and max prices to simulate (default: 50% to 150% of base price)
    n_points : int
        Number of price points to simulate
        
    Returns
    -------
    pd.DataFrame
        Columns: Price, PredictedQuantity, PredictedRevenue, Elasticity
    """
    base_price = base_features[price_col].values[0]
    
    # Set default price range
    if price_range is None:
        price_range = (base_price * 0.5, base_price * 1.5)
    
    # Generate price points
    prices = np.linspace(price_range[0], price_range[1], n_points)
    
    results = []
    prev_qty = None
    prev_price = None
    
    for price in prices:
        # Update features with new price
        sim_features = base_features.copy()
        sim_features[price_col] = price
        
        # Also update price-relative features if they exist
        if 'PriceRelativeToAvg' in sim_features.columns:
            hist_avg = base_price  # Use base price as proxy for historical average
            sim_features['PriceRelativeToAvg'] = price / hist_avg
        
        # Predict quantity
        qty = max(0, model.predict(sim_features)[0])  # Clip to non-negative
        
        # Calculate revenue
        revenue = price * qty
        
        # Calculate arc elasticity (between this point and previous)
        if prev_qty is not None and prev_price is not None:
            avg_qty = (qty + prev_qty) / 2
            avg_price = (price + prev_price) / 2
            if avg_qty != 0 and avg_price != 0:
                elasticity = ((qty - prev_qty) / avg_qty) / ((price - prev_price) / avg_price)
            else:
                elasticity = np.nan
        else:
            elasticity = np.nan
        
        results.append({
            'Price': price,
            'PredictedQuantity': qty,
            'PredictedRevenue': revenue,
            'Elasticity': elasticity
        })
        
        prev_qty = qty
        prev_price = price
    
    return pd.DataFrame(results)


def find_optimal_price(demand_curve: pd.DataFrame) -> Dict[str, float]:
    """
    Find the revenue-maximizing price from a demand curve.
    
    Parameters
    ----------
    demand_curve : pd.DataFrame
        Output from simulate_demand_curve()
        
    Returns
    -------
    Dict[str, float]
        optimal_price, predicted_quantity, predicted_revenue
    """
    # Find the row with maximum revenue
    optimal_idx = demand_curve['PredictedRevenue'].idxmax()
    optimal_row = demand_curve.loc[optimal_idx]
    
    return {
        'optimal_price': optimal_row['Price'],
        'predicted_quantity': optimal_row['PredictedQuantity'],
        'predicted_revenue': optimal_row['PredictedRevenue'],
        'elasticity_at_optimal': optimal_row['Elasticity']
    }


def classify_elasticity(elasticity: float) -> str:
    """
    Classify elasticity into categories.
    
    Parameters
    ----------
    elasticity : float
        Elasticity value
        
    Returns
    -------
    str
        Classification: 'Highly Elastic', 'Elastic', 'Unit Elastic', 
        'Inelastic', or 'Highly Inelastic'
    """
    abs_e = abs(elasticity)
    
    if abs_e > 2:
        return 'Highly Elastic'
    elif abs_e > 1:
        return 'Elastic'
    elif abs_e > 0.8:
        return 'Unit Elastic'
    elif abs_e > 0.3:
        return 'Inelastic'
    else:
        return 'Highly Inelastic'


def analyze_product_elasticity(model: Any,
                               feature_matrix: pd.DataFrame,
                               feature_cols: List[str],
                               product_col: str = 'StockCode',
                               price_col: str = 'AvgPrice',
                               top_n: int = 50) -> pd.DataFrame:
    """
    Analyze price elasticity for multiple products.
    
    Parameters
    ----------
    model : Any
        Trained model
    feature_matrix : pd.DataFrame
        Full feature matrix with product identifiers
    feature_cols : List[str]
        Feature columns used by the model
    product_col : str
        Column identifying products
    price_col : str
        Price column name
    top_n : int
        Number of products to analyze (most frequent)
        
    Returns
    -------
    pd.DataFrame
        Product-level elasticity analysis
    """
    print(f"Analyzing elasticity for top {top_n} products...")
    
    # Get most frequent products
    top_products = feature_matrix[product_col].value_counts().head(top_n).index.tolist()
    
    results = []
    
    for product in top_products:
        # Get product data
        product_data = feature_matrix[feature_matrix[product_col] == product]
        
        if len(product_data) < 2:
            continue
        
        # Use median observation as base
        base_idx = len(product_data) // 2
        base_features = product_data[feature_cols].iloc[[base_idx]]
        base_price = base_features[price_col].values[0]
        
        # Simulate demand curve
        demand_curve = simulate_demand_curve(
            model, 
            base_features,
            price_col=price_col,
            n_points=30
        )
        
        # Find optimal price
        optimal = find_optimal_price(demand_curve)
        
        # Calculate average elasticity
        valid_elasticities = demand_curve['Elasticity'].dropna()
        avg_elasticity = valid_elasticities.mean() if len(valid_elasticities) > 0 else np.nan
        
        # Get product description if available
        description = product_data['Description'].iloc[0] if 'Description' in product_data.columns else ''
        
        results.append({
            'StockCode': product,
            'Description': description[:50] if description else '',  # Truncate long descriptions
            'CurrentAvgPrice': base_price,
            'OptimalPrice': optimal['optimal_price'],
            'PriceChangeRecommended': (optimal['optimal_price'] - base_price) / base_price * 100,
            'AvgElasticity': avg_elasticity,
            'ElasticityClass': classify_elasticity(avg_elasticity) if not np.isnan(avg_elasticity) else 'Unknown',
            'CurrentRevenue': base_price * product_data['TotalQuantity'].mean(),
            'PotentialRevenue': optimal['predicted_revenue'],
            'RevenueUplift': (optimal['predicted_revenue'] - base_price * product_data['TotalQuantity'].mean()) 
                            / (base_price * product_data['TotalQuantity'].mean()) * 100
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('AvgElasticity', ascending=True)  # Most elastic first
    
    print(f"Analyzed {len(results_df)} products")
    
    return results_df


def get_pricing_recommendation(product_analysis: pd.DataFrame,
                               min_uplift: float = 5.0) -> pd.DataFrame:
    """
    Generate actionable pricing recommendations.
    
    Parameters
    ----------
    product_analysis : pd.DataFrame
        Output from analyze_product_elasticity()
    min_uplift : float
        Minimum revenue uplift % to recommend a price change
        
    Returns
    -------
    pd.DataFrame
        Filtered recommendations with action items
    """
    # Filter to products with meaningful uplift potential
    recommendations = product_analysis[
        product_analysis['RevenueUplift'] > min_uplift
    ].copy()
    
    # Add recommendation text
    def get_recommendation(row):
        if row['PriceChangeRecommended'] > 5:
            return f"Increase price by {row['PriceChangeRecommended']:.1f}%"
        elif row['PriceChangeRecommended'] < -5:
            return f"Decrease price by {abs(row['PriceChangeRecommended']):.1f}%"
        else:
            return "Price is near optimal"
    
    recommendations['Recommendation'] = recommendations.apply(get_recommendation, axis=1)
    
    # Select relevant columns
    recommendations = recommendations[[
        'StockCode', 'Description', 'CurrentAvgPrice', 'OptimalPrice',
        'ElasticityClass', 'RevenueUplift', 'Recommendation'
    ]]
    
    return recommendations.sort_values('RevenueUplift', ascending=False)


def create_elasticity_summary(product_analysis: pd.DataFrame) -> Dict[str, Any]:
    """
    Create summary statistics for elasticity analysis.
    
    Parameters
    ----------
    product_analysis : pd.DataFrame
        Output from analyze_product_elasticity()
        
    Returns
    -------
    Dict[str, Any]
        Summary statistics
    """
    summary = {
        'total_products_analyzed': len(product_analysis),
        'elasticity_distribution': product_analysis['ElasticityClass'].value_counts().to_dict(),
        'avg_elasticity': product_analysis['AvgElasticity'].mean(),
        'median_elasticity': product_analysis['AvgElasticity'].median(),
        'products_with_uplift_potential': (product_analysis['RevenueUplift'] > 5).sum(),
        'avg_revenue_uplift': product_analysis['RevenueUplift'].mean(),
        'max_revenue_uplift': product_analysis['RevenueUplift'].max(),
        'most_elastic_product': product_analysis.iloc[0]['StockCode'] if len(product_analysis) > 0 else None,
        'least_elastic_product': product_analysis.iloc[-1]['StockCode'] if len(product_analysis) > 0 else None,
    }
    
    return summary


def print_elasticity_summary(summary: Dict[str, Any]) -> None:
    """Print formatted elasticity summary."""
    print("\n" + "=" * 50)
    print("ELASTICITY ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"\nTotal Products Analyzed: {summary['total_products_analyzed']}")
    print(f"\nElasticity Distribution:")
    for category, count in summary['elasticity_distribution'].items():
        print(f"  {category}: {count}")
    print(f"\nAverage Elasticity: {summary['avg_elasticity']:.3f}")
    print(f"Median Elasticity: {summary['median_elasticity']:.3f}")
    print(f"\nProducts with >5% Revenue Uplift Potential: {summary['products_with_uplift_potential']}")
    print(f"Average Revenue Uplift: {summary['avg_revenue_uplift']:.2f}%")
    print(f"Maximum Revenue Uplift: {summary['max_revenue_uplift']:.2f}%")


# Example usage
if __name__ == "__main__":
    print("Elasticity module loaded.")
    print("\nAvailable functions:")
    print("  - calculate_point_elasticity(model, base_features)")
    print("  - simulate_demand_curve(model, base_features)")
    print("  - find_optimal_price(demand_curve)")
    print("  - classify_elasticity(elasticity)")
    print("  - analyze_product_elasticity(model, feature_matrix, feature_cols)")
    print("  - get_pricing_recommendation(product_analysis)")
    print("  - create_elasticity_summary(product_analysis)")
