"""
data_loader.py
--------------
Functions for loading and cleaning the Online Retail II dataset.

This module handles:
- Loading raw Excel data
- Removing cancelled orders
- Filtering invalid records
- Basic data validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load the Online Retail II dataset from an Excel file.
    
    Parameters
    ----------
    filepath : str
        Path to the .xlsx file
        
    Returns
    -------
    pd.DataFrame
        Raw dataset with all original columns
        
    Example
    -------
    >>> df = load_raw_data('data/raw/online_retail_II.xlsx')
    """
    print(f"Loading data from {filepath}...")
    
    # The dataset has two sheets - load both and concatenate
    df_2009_2010 = pd.read_excel(filepath, sheet_name='Year 2009-2010')
    df_2010_2011 = pd.read_excel(filepath, sheet_name='Year 2010-2011')
    
    df = pd.concat([df_2009_2010, df_2010_2011], ignore_index=True)
    
    print(f"Loaded {len(df):,} records")
    return df


def clean_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Clean the dataset by removing invalid records.
    
    Cleaning steps:
    1. Remove cancelled orders (Invoice starts with 'C')
    2. Remove records with missing CustomerID
    3. Remove records with non-positive Quantity
    4. Remove records with non-positive Price
    5. Remove duplicate records
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset
    verbose : bool
        If True, print cleaning statistics
        
    Returns
    -------
    pd.DataFrame
        Cleaned dataset
    """
    initial_count = len(df)
    
    if verbose:
        print("\n--- Data Cleaning Report ---")
        print(f"Initial records: {initial_count:,}")
    
    # Step 1: Remove cancelled orders
    # Cancelled orders have Invoice numbers starting with 'C'
    df = df.copy()
    df['Invoice'] = df['Invoice'].astype(str)
    cancelled_mask = df['Invoice'].str.startswith('C')
    cancelled_count = cancelled_mask.sum()
    df = df[~cancelled_mask]
    
    if verbose:
        print(f"Cancelled orders removed: {cancelled_count:,}")
    
    # Step 2: Remove missing CustomerID
    missing_customer = df['Customer ID'].isna().sum()
    df = df.dropna(subset=['Customer ID'])
    
    if verbose:
        print(f"Missing CustomerID removed: {missing_customer:,}")
    
    # Step 3: Remove non-positive quantities
    invalid_qty = (df['Quantity'] <= 0).sum()
    df = df[df['Quantity'] > 0]
    
    if verbose:
        print(f"Non-positive quantities removed: {invalid_qty:,}")
    
    # Step 4: Remove non-positive prices
    invalid_price = (df['Price'] <= 0).sum()
    df = df[df['Price'] > 0]
    
    if verbose:
        print(f"Non-positive prices removed: {invalid_price:,}")
    
    # Step 5: Remove duplicates
    duplicates = df.duplicated().sum()
    df = df.drop_duplicates()
    
    if verbose:
        print(f"Duplicates removed: {duplicates:,}")
        print(f"\nFinal records: {len(df):,}")
        print(f"Retention rate: {len(df)/initial_count*100:.1f}%")
    
    return df.reset_index(drop=True)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add commonly used derived columns to the dataset.
    
    New columns:
    - Revenue: Quantity * Price
    - InvoiceDate: Converted to datetime
    - Year, Month, Week, DayOfWeek: Extracted from InvoiceDate
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset
        
    Returns
    -------
    pd.DataFrame
        Dataset with additional columns
    """
    df = df.copy()
    
    # Calculate revenue
    df['Revenue'] = df['Quantity'] * df['Price']
    
    # Ensure datetime format
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Extract temporal components
    df['Year'] = df['InvoiceDate'].dt.year
    df['Month'] = df['InvoiceDate'].dt.month
    df['Week'] = df['InvoiceDate'].dt.isocalendar().week
    df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
    df['YearWeek'] = df['InvoiceDate'].dt.strftime('%Y-%W')
    
    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Generate a summary of the dataset for reporting.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset to summarize
        
    Returns
    -------
    dict
        Summary statistics
    """
    summary = {
        'total_records': len(df),
        'date_range': (df['InvoiceDate'].min(), df['InvoiceDate'].max()),
        'unique_products': df['StockCode'].nunique(),
        'unique_customers': df['Customer ID'].nunique(),
        'unique_invoices': df['Invoice'].nunique(),
        'total_revenue': df['Revenue'].sum() if 'Revenue' in df.columns else None,
        'avg_transaction_value': df.groupby('Invoice')['Revenue'].sum().mean() if 'Revenue' in df.columns else None,
    }
    return summary


def print_data_summary(summary: dict) -> None:
    """Print a formatted data summary."""
    print("\n--- Dataset Summary ---")
    print(f"Total Records: {summary['total_records']:,}")
    print(f"Date Range: {summary['date_range'][0].date()} to {summary['date_range'][1].date()}")
    print(f"Unique Products: {summary['unique_products']:,}")
    print(f"Unique Customers: {summary['unique_customers']:,}")
    print(f"Unique Invoices: {summary['unique_invoices']:,}")
    if summary['total_revenue']:
        print(f"Total Revenue: £{summary['total_revenue']:,.2f}")
        print(f"Avg Transaction Value: £{summary['avg_transaction_value']:,.2f}")


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Convenience function to load, clean, and prepare data in one step.
    
    Parameters
    ----------
    filepath : str
        Path to the raw data file
        
    Returns
    -------
    pd.DataFrame
        Cleaned and prepared dataset
    """
    df = load_raw_data(filepath)
    df = clean_data(df)
    df = add_derived_columns(df)
    
    summary = get_data_summary(df)
    print_data_summary(summary)
    
    return df


# Example usage when running this file directly
if __name__ == "__main__":
    # Test with sample data path
    filepath = "../data/raw/online_retail_II.xlsx"
    
    if Path(filepath).exists():
        df = load_and_clean_data(filepath)
        print("\nFirst few rows:")
        print(df.head())
    else:
        print(f"Data file not found at {filepath}")
        print("Please download from UCI ML Repository and place in data/raw/")
