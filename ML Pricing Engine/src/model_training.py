"""
model_training.py
-----------------
Functions for training and evaluating demand forecasting models.

This module handles:
- Train/test splitting with time-based ordering
- Model training (Baseline, Linear Regression, Random Forest, XGBoost)
- Model evaluation with multiple metrics
- Cross-validation
- Feature importance analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor


def create_train_test_split(df: pd.DataFrame,
                            feature_cols: List[str],
                            target_col: str = 'TotalQuantity',
                            test_size: float = 0.2,
                            time_col: str = 'YearWeek') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Create time-based train/test split to prevent data leakage.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix
    feature_cols : List[str]
        Names of feature columns
    target_col : str
        Name of target column
    test_size : float
        Proportion of data for testing (based on time, not random)
    time_col : str
        Column to sort by for temporal ordering
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    # Sort by time
    df = df.sort_values(time_col).reset_index(drop=True)
    
    # Find split point
    split_idx = int(len(df) * (1 - test_size))
    
    # Split data
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df[target_col]
    y_test = test_df[target_col]
    
    print(f"Train set: {len(X_train):,} records")
    print(f"Test set: {len(X_test):,} records")
    print(f"Train period: {train_df[time_col].min()} to {train_df[time_col].max()}")
    print(f"Test period: {test_df[time_col].min()} to {test_df[time_col].max()}")
    
    return X_train, X_test, y_train, y_test


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics for regression.
    
    Parameters
    ----------
    y_true : np.ndarray
        Actual values
    y_pred : np.ndarray
        Predicted values
        
    Returns
    -------
    Dict[str, float]
        Dictionary of metric names and values
    """
    # Ensure arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE - handle zero values
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }


def print_metrics(metrics: Dict[str, float], model_name: str = "Model") -> None:
    """Print formatted metrics."""
    print(f"\n{model_name} Performance:")
    print(f"  RMSE:  {metrics['RMSE']:.2f}")
    print(f"  MAE:   {metrics['MAE']:.2f}")
    print(f"  R²:    {metrics['R2']:.4f}")
    print(f"  MAPE:  {metrics['MAPE']:.2f}%")


class BaselineModel:
    """
    Baseline model that predicts the mean quantity per product.
    
    This establishes a minimum performance threshold that
    more complex models should beat.
    """
    
    def __init__(self):
        self.product_means = None
        self.global_mean = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series, product_col: str = 'StockCode'):
        """
        Fit baseline by calculating mean quantity per product.
        
        Note: X must contain the StockCode column for fitting,
        but this column should be removed for other models.
        """
        # If StockCode is not in X, we need it from the original dataframe
        if product_col not in X.columns:
            # Fall back to global mean
            self.global_mean = y.mean()
            self.product_means = None
        else:
            # Calculate per-product means
            df_temp = X[[product_col]].copy()
            df_temp['y'] = y.values
            self.product_means = df_temp.groupby(product_col)['y'].mean().to_dict()
            self.global_mean = y.mean()
        
        return self
    
    def predict(self, X: pd.DataFrame, product_col: str = 'StockCode') -> np.ndarray:
        """Predict using product means, falling back to global mean."""
        if self.product_means is None or product_col not in X.columns:
            return np.full(len(X), self.global_mean)
        
        predictions = X[product_col].map(self.product_means)
        # Fill missing products with global mean
        predictions = predictions.fillna(self.global_mean)
        return predictions.values


def train_baseline(X_train: pd.DataFrame, y_train: pd.Series,
                   X_test: pd.DataFrame, y_test: pd.Series,
                   train_df: pd.DataFrame = None) -> Tuple[BaselineModel, Dict[str, float]]:
    """
    Train and evaluate baseline model.
    
    Parameters
    ----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    train_df : Full training dataframe with StockCode (optional)
    
    Returns
    -------
    Tuple[BaselineModel, Dict[str, float]]
        Trained model and test metrics
    """
    model = BaselineModel()
    
    # For baseline, we need StockCode which may not be in feature cols
    # This is a simple baseline, so we just use global mean
    model.global_mean = y_train.mean()
    model.product_means = None
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    metrics = calculate_metrics(y_test, y_pred)
    print_metrics(metrics, "Baseline (Mean)")
    
    return model, metrics


def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series,
                           X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[LinearRegression, Dict[str, float]]:
    """
    Train and evaluate Linear Regression model.
    
    Returns
    -------
    Tuple[LinearRegression, Dict[str, float]]
        Trained model and test metrics
    """
    print("\nTraining Linear Regression...")
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Clip negative predictions (quantity can't be negative)
    y_pred = np.clip(y_pred, 0, None)
    
    # Evaluate
    metrics = calculate_metrics(y_test, y_pred)
    print_metrics(metrics, "Linear Regression")
    
    return model, metrics


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series,
                        n_estimators: int = 100,
                        max_depth: int = 15,
                        random_state: int = 42) -> Tuple[RandomForestRegressor, Dict[str, float]]:
    """
    Train and evaluate Random Forest model.
    
    Returns
    -------
    Tuple[RandomForestRegressor, Dict[str, float]]
        Trained model and test metrics
    """
    print("\nTraining Random Forest...")
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    metrics = calculate_metrics(y_test, y_pred)
    print_metrics(metrics, "Random Forest")
    
    return model, metrics


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                  X_test: pd.DataFrame, y_test: pd.Series,
                  n_estimators: int = 100,
                  max_depth: int = 6,
                  learning_rate: float = 0.1,
                  random_state: int = 42) -> Tuple[XGBRegressor, Dict[str, float]]:
    """
    Train and evaluate XGBoost model.
    
    Returns
    -------
    Tuple[XGBRegressor, Dict[str, float]]
        Trained model and test metrics
    """
    print("\nTraining XGBoost...")
    
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    metrics = calculate_metrics(y_test, y_pred)
    print_metrics(metrics, "XGBoost")
    
    return model, metrics


def train_all_models(X_train: pd.DataFrame, y_train: pd.Series,
                     X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Tuple[Any, Dict[str, float]]]:
    """
    Train all models and return comparison results.
    
    Returns
    -------
    Dict[str, Tuple[Any, Dict[str, float]]]
        Dictionary mapping model name to (model, metrics) tuple
    """
    results = {}
    
    print("=" * 50)
    print("TRAINING ALL MODELS")
    print("=" * 50)
    
    # Baseline
    model, metrics = train_baseline(X_train, y_train, X_test, y_test)
    results['Baseline'] = (model, metrics)
    
    # Linear Regression
    model, metrics = train_linear_regression(X_train, y_train, X_test, y_test)
    results['Linear Regression'] = (model, metrics)
    
    # Random Forest
    model, metrics = train_random_forest(X_train, y_train, X_test, y_test)
    results['Random Forest'] = (model, metrics)
    
    # XGBoost
    model, metrics = train_xgboost(X_train, y_train, X_test, y_test)
    results['XGBoost'] = (model, metrics)
    
    return results


def compare_models(results: Dict[str, Tuple[Any, Dict[str, float]]]) -> pd.DataFrame:
    """
    Create a comparison dataframe of all model results.
    
    Returns
    -------
    pd.DataFrame
        Comparison table with models as rows and metrics as columns
    """
    comparison_data = []
    
    for model_name, (model, metrics) in results.items():
        row = {'Model': model_name}
        row.update(metrics)
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.set_index('Model')
    
    # Sort by R2 descending
    comparison_df = comparison_df.sort_values('R2', ascending=False)
    
    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)
    print(comparison_df.round(4).to_string())
    
    return comparison_df


def get_feature_importance(model: Any, 
                          feature_names: List[str],
                          model_type: str = 'tree') -> pd.DataFrame:
    """
    Extract feature importance from a trained model.
    
    Parameters
    ----------
    model : Any
        Trained model (RandomForest or XGBoost)
    feature_names : List[str]
        Names of features
    model_type : str
        'tree' for RandomForest/XGBoost, 'linear' for LinearRegression
        
    Returns
    -------
    pd.DataFrame
        Feature importance dataframe sorted by importance
    """
    if model_type == 'linear':
        importance = np.abs(model.coef_)
    else:
        importance = model.feature_importances_
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    importance_df = importance_df.sort_values('Importance', ascending=False)
    importance_df = importance_df.reset_index(drop=True)
    
    return importance_df


def save_model(model: Any, filepath: str) -> None:
    """Save a trained model to disk."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath: str) -> Any:
    """Load a trained model from disk."""
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model


def cross_validate_model(model_class: Any,
                         X: pd.DataFrame,
                         y: pd.Series,
                         n_splits: int = 5,
                         **model_params) -> Dict[str, List[float]]:
    """
    Perform time-series cross-validation.
    
    Parameters
    ----------
    model_class : Any
        Model class to instantiate
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    n_splits : int
        Number of CV folds
    **model_params
        Parameters to pass to model constructor
        
    Returns
    -------
    Dict[str, List[float]]
        Dictionary of metric lists across folds
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    cv_results = {
        'RMSE': [],
        'MAE': [],
        'R2': [],
        'MAPE': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        # Split
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Train
        model = model_class(**model_params)
        model.fit(X_train_fold, y_train_fold)
        
        # Predict
        y_pred = model.predict(X_val_fold)
        
        # Evaluate
        metrics = calculate_metrics(y_val_fold, y_pred)
        
        for metric_name, value in metrics.items():
            cv_results[metric_name].append(value)
    
    # Print summary
    print(f"\n{n_splits}-Fold Cross-Validation Results:")
    for metric_name, values in cv_results.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"  {metric_name}: {mean_val:.4f} (+/- {std_val:.4f})")
    
    return cv_results


# Example usage
if __name__ == "__main__":
    print("Model training module loaded.")
    print("\nAvailable functions:")
    print("  - create_train_test_split(df, feature_cols, target_col)")
    print("  - train_baseline(X_train, y_train, X_test, y_test)")
    print("  - train_linear_regression(X_train, y_train, X_test, y_test)")
    print("  - train_random_forest(X_train, y_train, X_test, y_test)")
    print("  - train_xgboost(X_train, y_train, X_test, y_test)")
    print("  - train_all_models(X_train, y_train, X_test, y_test)")
    print("  - compare_models(results)")
    print("  - get_feature_importance(model, feature_names)")
    print("  - save_model(model, filepath)")
    print("  - load_model(filepath)")
