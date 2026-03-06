"""
Smart Pricing Engine
--------------------
A machine learning-powered demand forecasting and price optimization tool.

Modules:
- data_loader: Load and clean the Online Retail II dataset
- feature_engineering: Create features for modeling
- model_training: Train and evaluate ML models
- elasticity: Calculate and analyze price elasticity
"""

from .data_loader import (
    load_raw_data,
    clean_data,
    add_derived_columns,
    load_and_clean_data,
    get_data_summary
)

from .feature_engineering import (
    aggregate_to_weekly,
    add_temporal_features,
    add_price_context_features,
    add_lag_features,
    add_rolling_features,
    add_product_features,
    create_feature_matrix
)

from .model_training import (
    create_train_test_split,
    train_baseline,
    train_linear_regression,
    train_random_forest,
    train_xgboost,
    train_all_models,
    compare_models,
    get_feature_importance,
    save_model,
    load_model
)

from .elasticity import (
    calculate_point_elasticity,
    simulate_demand_curve,
    find_optimal_price,
    classify_elasticity,
    analyze_product_elasticity,
    get_pricing_recommendation
)

__version__ = '1.0.0'
__author__ = 'Your Name'
