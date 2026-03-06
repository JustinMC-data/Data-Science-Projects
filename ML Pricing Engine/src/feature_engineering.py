"""
feature_engineering.py
----------------------
Functions for creating features for the demand forecasting model.

This module handles:
- Aggregating transaction data to weekly product-level observations
- Creating temporal features
- Creating price context features
- Creating lag features
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple


def aggregate_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transaction-level data to weekly product-level observations.
    
    Each row in the output represents one product's performance in one week,
    including total quantity sold, average price, and total revenue.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction-level data with columns:
        StockCode, Description, Quantity, Price, Revenue, YearWeek
        
    Returns
    -------
    pd.DataFrame
        Weekly aggregated data with columns:
        StockCode, Description, YearWeek, TotalQuantity, AvgPrice, 
        TotalRevenue, TransactionCount
    """
    print("Aggregating to weekly product-level data...")
    
    # Group by product and week
    weekly = df.groupby(['StockCode', 'YearWeek']).agg({
        'Description': 'first',  # Keep product description
        'Quantity': 'sum',       # Total units sold
        'Price': 'mean',         # Average price that week
        'Revenue': 'sum',        # Total revenue
        'Invoice': 'nunique',    # Number of transactions
        'InvoiceDate': 'min'     # First date of the week (for sorting)
    }).reset_index()
    
    # Rename columns for clarity
    weekly = weekly.rename(columns={
        'Quantity': 'TotalQuantity',
        'Price': 'AvgPrice',
        'Revenue': 'TotalRevenue',
        'Invoice': 'TransactionCount',
        'InvoiceDate': 'WeekStartDate'
    })
    
    # Sort by product and time
    weekly = weekly.sort_values(['StockCode', 'YearWeek']).reset_index(drop=True)
    
    print(f"Created {len(weekly):,} weekly product observations")
    print(f"Products: {weekly['StockCode'].nunique():,}")
    print(f"Weeks: {weekly['YearWeek'].nunique()}")
    
    return weekly


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal features to capture seasonality.
    
    Features added:
    - Month: Month of the year (1-12)
    - WeekOfYear: Week number (1-52)
    - Quarter: Quarter of the year (1-4)
    - IsHolidaySeason: Binary flag for Nov-Dec (holiday shopping)
    
    Parameters
    ----------
    df : pd.DataFrame
        Weekly aggregated data with WeekStartDate column
        
    Returns
    -------
    pd.DataFrame
        Data with temporal features added
    """
    df = df.copy()
    
    # Extract temporal components
    df['Month'] = df['WeekStartDate'].dt.month
    df['WeekOfYear'] = df['WeekStartDate'].dt.isocalendar().week.astype(int)
    df['Quarter'] = df['WeekStartDate'].dt.quarter
    df['Year'] = df['WeekStartDate'].dt.year
    
    # Holiday season indicator (November and December)
    df['IsHolidaySeason'] = df['Month'].isin([11, 12]).astype(int)
    
    # Beginning of month indicator (week 1-2 of month)
    df['DayOfMonth'] = df['WeekStartDate'].dt.day
    df['IsMonthStart'] = (df['DayOfMonth'] <= 7).astype(int)
    df = df.drop(columns=['DayOfMonth'])
    
    print("Added temporal features: Month, WeekOfYear, Quarter, Year, IsHolidaySeason, IsMonthStart")
    
    return df


def add_price_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features that provide context for the current price.
    
    Features added:
    - PriceRelativeToAvg: Current price / Product's historical average price
    - PriceRelativeToMin: Current price / Product's historical minimum price
    - PriceRelativeToMax: Current price / Product's historical maximum price
    - IsDiscounted: Binary flag if price is below historical average
    
    Parameters
    ----------
    df : pd.DataFrame
        Weekly aggregated data with AvgPrice column
        
    Returns
    -------
    pd.DataFrame
        Data with price context features added
    """
    df = df.copy()
    
    # Calculate historical price statistics per product
    price_stats = df.groupby('StockCode')['AvgPrice'].agg(['mean', 'min', 'max', 'std'])
    price_stats.columns = ['HistAvgPrice', 'HistMinPrice', 'HistMaxPrice', 'HistStdPrice']
    price_stats = price_stats.reset_index()
    
    # Merge back to main dataframe
    df = df.merge(price_stats, on='StockCode', how='left')
    
    # Calculate relative price metrics
    df['PriceRelativeToAvg'] = df['AvgPrice'] / df['HistAvgPrice']
    df['PriceRelativeToMin'] = df['AvgPrice'] / df['HistMinPrice']
    df['PriceRelativeToMax'] = df['AvgPrice'] / df['HistMaxPrice']
    
    # Price deviation in standard deviations
    df['PriceZScore'] = (df['AvgPrice'] - df['HistAvgPrice']) / df['HistStdPrice'].replace(0, 1)
    
    # Binary discount indicator
    df['IsDiscounted'] = (df['AvgPrice'] < df['HistAvgPrice']).astype(int)
    
    print("Added price context features: PriceRelativeToAvg, PriceRelativeToMin, PriceRelativeToMax, PriceZScore, IsDiscounted")
    
    return df


def add_lag_features(df: pd.DataFrame, lag_periods: List[int] = [1, 2, 4]) -> pd.DataFrame:
    """
    Add lagged features to capture historical patterns.
    
    Features added for each lag period:
    - Lag{n}_Quantity: Quantity sold n weeks ago
    - Lag{n}_Price: Average price n weeks ago
    - Lag{n}_Revenue: Revenue n weeks ago
    
    Parameters
    ----------
    df : pd.DataFrame
        Weekly aggregated data, sorted by StockCode and YearWeek
    lag_periods : List[int]
        Number of weeks to lag (default: 1, 2, 4 weeks)
        
    Returns
    -------
    pd.DataFrame
        Data with lag features added
    """
    df = df.copy()
    
    # Ensure sorted by product and time
    df = df.sort_values(['StockCode', 'YearWeek']).reset_index(drop=True)
    
    for lag in lag_periods:
        # Lag quantity
        df[f'Lag{lag}_Quantity'] = df.groupby('StockCode')['TotalQuantity'].shift(lag)
        
        # Lag price
        df[f'Lag{lag}_Price'] = df.groupby('StockCode')['AvgPrice'].shift(lag)
        
        # Lag revenue
        df[f'Lag{lag}_Revenue'] = df.groupby('StockCode')['TotalRevenue'].shift(lag)
    
    print(f"Added lag features for periods: {lag_periods}")
    
    return df


def add_rolling_features(df: pd.DataFrame, windows: List[int] = [4, 8]) -> pd.DataFrame:
    """
    Add rolling average features to smooth short-term fluctuations.
    
    Features added for each window:
    - Rolling{n}w_AvgQuantity: n-week rolling average of quantity
    - Rolling{n}w_AvgPrice: n-week rolling average of price
    
    Parameters
    ----------
    df : pd.DataFrame
        Weekly aggregated data, sorted by StockCode and YearWeek
    windows : List[int]
        Window sizes in weeks (default: 4 and 8 weeks)
        
    Returns
    -------
    pd.DataFrame
        Data with rolling features added
    """
    df = df.copy()
    
    # Ensure sorted
    df = df.sort_values(['StockCode', 'YearWeek']).reset_index(drop=True)
    
    for window in windows:
        # Rolling average quantity (excluding current week)
        df[f'Rolling{window}w_AvgQuantity'] = df.groupby('StockCode')['TotalQuantity'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        
        # Rolling average price (excluding current week)
        df[f'Rolling{window}w_AvgPrice'] = df.groupby('StockCode')['AvgPrice'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
    
    print(f"Added rolling features for windows: {windows} weeks")
    
    return df


def add_product_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add product-level features based on historical performance.
    
    Features added:
    - ProductAvgWeeklyQty: Product's average weekly quantity
    - ProductQtyVolatility: Standard deviation of weekly quantity
    - ProductPriceVolatility: Standard deviation of price
    - ProductWeeksActive: Number of weeks product has sales data
    
    Parameters
    ----------
    df : pd.DataFrame
        Weekly aggregated data
        
    Returns
    -------
    pd.DataFrame
        Data with product features added
    """
    df = df.copy()
    
    # Calculate product-level statistics
    product_stats = df.groupby('StockCode').agg({
        'TotalQuantity': ['mean', 'std'],
        'AvgPrice': 'std',
        'YearWeek': 'nunique'
    })
    
    product_stats.columns = ['ProductAvgWeeklyQty', 'ProductQtyVolatility', 
                              'ProductPriceVolatility', 'ProductWeeksActive']
    product_stats = product_stats.reset_index()
    
    # Fill NaN volatility (products with single observation) with 0
    product_stats['ProductQtyVolatility'] = product_stats['ProductQtyVolatility'].fillna(0)
    product_stats['ProductPriceVolatility'] = product_stats['ProductPriceVolatility'].fillna(0)
    
    # Merge back
    df = df.merge(product_stats, on='StockCode', how='left')
    
    print("Added product features: ProductAvgWeeklyQty, ProductQtyVolatility, ProductPriceVolatility, ProductWeeksActive")
    
    return df


def create_feature_matrix(df: pd.DataFrame, 
                          min_weeks: int = 4,
                          verbose: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create the complete feature matrix for model training.
    
    This function orchestrates all feature engineering steps and
    returns a clean feature matrix ready for modeling.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction-level data
    min_weeks : int
        Minimum weeks of data required for a product (filters sparse products)
    verbose : bool
        Print progress information
        
    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        - Feature matrix with all features
        - List of feature column names for modeling
    """
    if verbose:
        print("\n=== Feature Engineering Pipeline ===\n")
    
    # Step 1: Aggregate to weekly
    weekly = aggregate_to_weekly(df)
    
    # Step 2: Add temporal features
    weekly = add_temporal_features(weekly)
    
    # Step 3: Add price context features
    weekly = add_price_context_features(weekly)
    
    # Step 4: Add lag features
    weekly = add_lag_features(weekly)
    
    # Step 5: Add rolling features
    weekly = add_rolling_features(weekly)
    
    # Step 6: Add product features
    weekly = add_product_features(weekly)
    
    # Step 7: Filter products with insufficient data
    if verbose:
        print(f"\nFiltering products with < {min_weeks} weeks of data...")
    
    initial_products = weekly['StockCode'].nunique()
    weekly = weekly[weekly['ProductWeeksActive'] >= min_weeks]
    final_products = weekly['StockCode'].nunique()
    
    if verbose:
        print(f"Products before filter: {initial_products:,}")
        print(f"Products after filter: {final_products:,}")
    
    # Step 8: Handle missing values from lag features
    # (First few weeks won't have lag data)
    if verbose:
        print(f"\nRecords before dropping NaN: {len(weekly):,}")
    
    weekly = weekly.dropna()
    
    if verbose:
        print(f"Records after dropping NaN: {len(weekly):,}")
    
    # Define feature columns
    feature_cols = [
        # Temporal features
        'Month', 'WeekOfYear', 'Quarter', 'IsHolidaySeason', 'IsMonthStart',
        # Price features
        'AvgPrice', 'PriceRelativeToAvg', 'PriceRelativeToMin', 
        'PriceRelativeToMax', 'PriceZScore', 'IsDiscounted',
        # Lag features
        'Lag1_Quantity', 'Lag1_Price', 'Lag1_Revenue',
        'Lag2_Quantity', 'Lag2_Price', 'Lag2_Revenue',
        'Lag4_Quantity', 'Lag4_Price', 'Lag4_Revenue',
        # Rolling features
        'Rolling4w_AvgQuantity', 'Rolling4w_AvgPrice',
        'Rolling8w_AvgQuantity', 'Rolling8w_AvgPrice',
        # Product features
        'ProductAvgWeeklyQty', 'ProductQtyVolatility', 
        'ProductPriceVolatility', 'ProductWeeksActive'
    ]
    
    if verbose:
        print(f"\n=== Feature Engineering Complete ===")
        print(f"Total features: {len(feature_cols)}")
        print(f"Total records: {len(weekly):,}")
    
    return weekly, feature_cols


# Example usage
if __name__ == "__main__":
    # This would typically be called after loading data
    print("Feature engineering module loaded.")
    print("\nAvailable functions:")
    print("  - aggregate_to_weekly(df)")
    print("  - add_temporal_features(df)")
    print("  - add_price_context_features(df)")
    print("  - add_lag_features(df)")
    print("  - add_rolling_features(df)")
    print("  - add_product_features(df)")
    print("  - create_feature_matrix(df)")
