# Smart Pricing Engine - Technical Methodology

## ML-Powered Demand Forecasting & Price Optimization

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Data Strategy](#3-data-strategy)
4. [Data Processing Pipeline](#4-data-processing-pipeline)
5. [Feature Engineering](#5-feature-engineering)
6. [Model Training](#6-model-training)
7. [Model Evaluation](#7-model-evaluation)
8. [Price Elasticity Analysis](#8-price-elasticity-analysis)
9. [Revenue Optimization](#9-revenue-optimization)
10. [Results & Findings](#10-results--findings)
11. [Technical Architecture](#11-technical-architecture)
12. [Limitations & Future Work](#12-limitations--future-work)

---

## 1. Executive Summary

The Smart Pricing Engine is an end-to-end machine learning system that predicts product demand at various price points and identifies revenue-maximizing prices through price elasticity analysis.

**Key Results:**
- Analyzed 1,067,371 transactions across 4,000+ products
- Best model achieved R² = 0.347 (explaining 35% of demand variance)
- Identified pricing opportunities in 86% of analyzed products
- Average potential revenue uplift: +258%

---

## 2. Problem Statement

### The Business Challenge

Pricing is one of the most critical decisions for retail businesses:
- **Price too high** → Lost sales volume
- **Price too low** → Lost profit margin
- **Static pricing** → Missed opportunities from demand fluctuations

Traditional pricing approaches rely on:
- Gut instinct
- Simple markup rules (e.g., "cost + 30%")
- Competitor matching

These fail to capture the complex, non-linear relationship between price and customer demand.

### Why Machine Learning?

| Traditional Approach | ML Approach |
|---------------------|-------------|
| Assumes linear price-demand relationship | Captures non-linear patterns |
| One rule for all products | Product-specific predictions |
| Ignores seasonality | Incorporates temporal patterns |
| Manual and time-consuming | Scalable to thousands of products |
| Reactive | Predictive |

### Project Objective

Build a system that:
1. Predicts weekly product demand based on price and contextual features
2. Calculates price elasticity for individual products
3. Identifies revenue-maximizing price points
4. Provides actionable recommendations through an interactive interface

---

## 3. Data Strategy

### Data Source

**Dataset:** Online Retail II from UCI Machine Learning Repository

| Attribute | Value |
|-----------|-------|
| Source | https://archive.ics.uci.edu/dataset/502/online+retail+ii |
| Records | 1,067,371 transactions |
| Time Period | December 2009 – December 2011 |
| Format | Excel (.xlsx) |

### Data Fields

| Field | Description | Role in Analysis |
|-------|-------------|------------------|
| Invoice | 6-digit transaction ID | Identify unique transactions |
| StockCode | Product identifier | Group by product |
| Description | Product name | Display purposes |
| Quantity | Units per transaction | **Target variable** (aggregated) |
| InvoiceDate | Transaction timestamp | Temporal features |
| Price | Unit price (GBP) | **Key predictor** |
| CustomerID | Customer identifier | Data quality filter |
| Country | Customer location | Not used in v1 |

### Why This Dataset?

1. **Real transactional data** — Not synthetic or simulated
2. **Sufficient scale** — 1M+ records for robust ML training
3. **Price variation** — Products have varying prices over time
4. **Temporal depth** — 2 years enables seasonality analysis
5. **Public availability** — Reproducible research

---

## 4. Data Processing Pipeline

### Step 1: Data Loading

```
Raw Data: 1,067,371 transactions
├── Year 2009-2010: ~500K records
└── Year 2010-2011: ~500K records
```

### Step 2: Data Cleaning

| Cleaning Step | Records Removed | Rationale |
|---------------|-----------------|-----------|
| Cancelled orders (Invoice starts with 'C') | 19,494 | Returns, not actual demand |
| Missing CustomerID | 242,257 | Cannot track behavior reliably |
| Non-positive quantities | ~1,000 | Data errors or adjustments |
| Non-positive prices | 71 | Invalid pricing data |
| Duplicate records | 26,124 | Data quality |
| **Total Removed** | **287,946** | |
| **Final Clean Records** | **779,425** | **73% retention rate** |

### Step 3: Aggregation

**Critical Design Decision:** Aggregate from transaction-level to weekly product-level.

**Why Weekly Aggregation?**

| Transaction-Level | Weekly Aggregation |
|-------------------|-------------------|
| Very noisy | Smoother signal |
| 779K records | 155K records |
| Hard to see patterns | Clear demand trends |
| Doesn't match business reality | Matches pricing decision cycles |

**Aggregation Method:**

```
Group By: (StockCode, YearWeek)

Aggregations:
├── TotalQuantity = SUM(Quantity)
├── AvgPrice = MEAN(Price)
├── TotalRevenue = SUM(Quantity × Price)
└── TransactionCount = COUNT(*)
```

**Result:** 154,987 weekly product observations

---

## 5. Feature Engineering

### Overview

Created **28 features** across 5 categories to give the model rich context for predictions.

### 5.1 Temporal Features (6)

Capture "when" patterns in demand.

| Feature | Description | Why It Matters |
|---------|-------------|----------------|
| Month | Month of year (1-12) | Monthly seasonality |
| WeekOfYear | Week number (1-52) | Finer seasonal patterns |
| Quarter | Quarter (1-4) | Quarterly trends |
| Year | Year of observation | Year-over-year changes |
| IsHolidaySeason | 1 if Nov-Dec, else 0 | Holiday shopping surge |
| IsMonthStart | 1 if first week of month | Paycheck timing effects |

### 5.2 Price Context Features (6)

Capture "is this a good deal?" signals.

| Feature | Formula | Why It Matters |
|---------|---------|----------------|
| AvgPrice | Current week's mean price | Direct price signal |
| PriceRelativeToAvg | Price ÷ Historical Mean | Deviation from normal |
| PriceRelativeToMin | Price ÷ Historical Min | Distance from cheapest |
| PriceRelativeToMax | Price ÷ Historical Max | Distance from most expensive |
| PriceZScore | (Price - Mean) ÷ StdDev | Standardized deviation |
| IsDiscounted | 1 if Price < Historical Mean | Binary discount flag |

**Key Insight:** The same price means different things for different products:
- £10 for a luxury candle = discount
- £10 for paper clips = expensive

Relative features capture this context.

### 5.3 Lag Features (9)

Capture "what happened recently?" momentum.

| Feature | Description |
|---------|-------------|
| Lag1_Quantity | Units sold 1 week ago |
| Lag1_Price | Price 1 week ago |
| Lag1_Revenue | Revenue 1 week ago |
| Lag2_Quantity | Units sold 2 weeks ago |
| Lag2_Price | Price 2 weeks ago |
| Lag2_Revenue | Revenue 2 weeks ago |
| Lag4_Quantity | Units sold 4 weeks ago |
| Lag4_Price | Price 4 weeks ago |
| Lag4_Revenue | Revenue 4 weeks ago |

**Why These Lags?**
- Lag 1: Most recent momentum
- Lag 2: Short-term trend
- Lag 4: Monthly pattern (approximately)

### 5.4 Rolling Features (4)

Capture smoothed trends over time.

| Feature | Description |
|---------|-------------|
| Rolling4w_AvgQuantity | 4-week moving average of quantity |
| Rolling4w_AvgPrice | 4-week moving average of price |
| Rolling8w_AvgQuantity | 8-week moving average of quantity |
| Rolling8w_AvgPrice | 8-week moving average of price |

**Purpose:** Reduce noise and capture underlying trend direction.

### 5.5 Product Features (4)

Capture "what kind of product is this?" characteristics.

| Feature | Description |
|---------|-------------|
| ProductAvgWeeklyQty | Product's typical weekly volume |
| ProductQtyVolatility | Std dev of weekly quantity |
| ProductPriceVolatility | Std dev of price |
| ProductWeeksActive | Number of weeks with sales |

**Key Insight:** ProductAvgWeeklyQty became the #1 predictor — products that historically sell a lot tend to keep selling a lot.

---

## 6. Model Training

### Target Variable

**TotalQuantity** — Weekly units sold per product

### Train/Test Split

**Method:** Time-based split (not random)

```
Timeline: ────────────────────────────────►
          │◄──── Train (80%) ────►│◄─ Test (20%) ─►│

Train: 2010-01 to 2011-33 (123,989 records)
Test:  2011-33 to 2011-49 (30,998 records)
```

**Why Time-Based Split?**

| Random Split | Time-Based Split |
|--------------|------------------|
| Data leakage risk | No future information in training |
| Unrealistic evaluation | Simulates real deployment |
| May overestimate performance | Honest performance estimate |

### Models Trained

#### Model 1: Baseline (Mean Prediction)

```
Prediction = Mean(Training Quantity)
```

**Purpose:** Establish minimum performance bar. Any useful model must beat this.

#### Model 2: Linear Regression

```
Prediction = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ
```

**Why:** Simple, interpretable, fast. Coefficients show direct feature impact.

#### Model 3: Random Forest

```
Prediction = Average of 100 Decision Trees
```

**Configuration:**
- n_estimators: 100
- max_depth: 15
- random_state: 42

**Why:** Captures non-linear relationships and feature interactions.

#### Model 4: XGBoost

```
Prediction = Ensemble of Gradient Boosted Trees
```

**Configuration:**
- n_estimators: 100
- max_depth: 6
- learning_rate: 0.1
- random_state: 42

**Why:** Often achieves best performance on tabular data. Handles complex patterns.

---

## 7. Model Evaluation

### Metrics Used

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **RMSE** | √(Σ(y - ŷ)² / n) | Average error in units; penalizes large errors |
| **MAE** | Σ\|y - ŷ\| / n | Average absolute error; robust to outliers |
| **R²** | 1 - (SS_res / SS_tot) | Proportion of variance explained (0-1) |
| **MAPE** | (100/n) × Σ\|y - ŷ\| / y | Percentage error; interpretable but problematic for small values |

### Results

| Model | RMSE | MAE | R² | MAPE |
|-------|------|-----|-----|------|
| Baseline (Mean) | 149.56 | 68.63 | -0.003 | 882% |
| **Linear Regression** | **120.69** | 50.94 | **0.347** | 406% |
| Random Forest | 122.75 | 49.54 | 0.324 | 431% |
| XGBoost | 130.12 | **47.12** | 0.241 | **302%** |

### Key Findings

**1. All ML models significantly beat baseline**
- Baseline R² ≈ 0 (by definition)
- Best ML R² = 0.347 (explains 35% of variance)

**2. Linear Regression performed best on R²**
- Surprising result — simpler model won
- Suggests relationships are relatively linear once good features are engineered
- Reinforces importance of feature engineering over model complexity

**3. High MAPE values are misleading**
- Inflated by low-volume products
- Example: Actual = 2, Predicted = 4 → MAPE = 100%
- MAE (47-51 units) is more meaningful

### Feature Importance (XGBoost)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | ProductAvgWeeklyQty | 16.4% |
| 2 | PriceRelativeToMax | 12.8% |
| 3 | ProductWeeksActive | 10.7% |
| 4 | ProductQtyVolatility | 6.0% |
| 5 | PriceRelativeToMin | 6.0% |

**Interpretation:**
- Historical demand patterns (Product features) are strongest predictors
- Price context matters more than absolute price
- Feature engineering investment paid off

---

## 8. Price Elasticity Analysis

### What is Price Elasticity?

**Price Elasticity of Demand** measures how sensitive quantity demanded is to price changes.

```
Elasticity (E) = (% Change in Quantity) / (% Change in Price)
```

### Elasticity Classifications

| |E| Value | Classification | Interpretation | Pricing Strategy |
|-----------|----------------|----------------|-----------------|
| > 2 | Highly Elastic | Very price sensitive | Lower prices to increase volume |
| 1 - 2 | Elastic | Price sensitive | Careful price changes |
| 0.8 - 1 | Unit Elastic | Proportional response | Revenue stable |
| 0.3 - 0.8 | Inelastic | Price insensitive | Can raise prices |
| < 0.3 | Highly Inelastic | Very price insensitive | Raise prices for margin |

### Calculation Method

**Model-Based Demand Curve Simulation:**

1. Select a product
2. Get most recent feature values as baseline
3. Vary price from 50% to 150% of current price
4. For each price point:
   - Update price features (AvgPrice, PriceRelativeToAvg, etc.)
   - Predict demand using trained model
   - Calculate revenue = Price × Predicted Demand
5. Calculate elasticity at current price
6. Find price that maximizes revenue

```
Price Range: [50%, 60%, 70%, ... , 140%, 150%] × Current Price
           ↓
       [Predict Demand at Each Price]
           ↓
       [Calculate Revenue at Each Price]
           ↓
       [Find Maximum Revenue Point]
```

### Results (50 Products Analyzed)

| Elasticity Class | Count | Percentage |
|------------------|-------|------------|
| Elastic | 14 | 28% |
| Highly Elastic | 3 | 6% |
| Unit Elastic | 6 | 12% |
| Inelastic | 13 | 26% |
| Highly Inelastic | 14 | 28% |

---

## 9. Revenue Optimization

### Optimization Approach

For each product, find the price that maximizes:

```
Revenue(P) = P × Demand(P)
```

Where Demand(P) is predicted by the ML model.

### Example: Zinc Metal Heart Decoration

| Metric | Current | Optimal | Change |
|--------|---------|---------|--------|
| Price | £1.25 | £1.88 | +50% |
| Predicted Demand | 100 units | 85 units | -15% |
| Revenue | £125 | £160 | **+28%** |

**Interpretation:** This product is **inelastic** — customers buy it regardless of small price changes. Raising the price increases revenue despite lower volume.

### Aggregate Results

| Metric | Value |
|--------|-------|
| Products with >5% uplift potential | 43/50 (86%) |
| Average potential revenue uplift | +258% |
| Maximum uplift identified | +781% |

---

## 10. Results & Findings

### Summary of Achievements

| Objective | Result |
|-----------|--------|
| Predict demand using ML | ✅ R² = 0.347 (Linear Regression) |
| Engineer meaningful features | ✅ 28 features across 5 categories |
| Calculate price elasticity | ✅ 50 products classified |
| Identify optimization opportunities | ✅ 86% of products show potential |
| Build interactive application | ✅ 7-page Streamlit app |

### Key Insights

1. **Feature engineering > model complexity**
   - Linear Regression matched/beat tree models
   - Well-designed features capture most signal

2. **Historical demand is the best predictor**
   - ProductAvgWeeklyQty ranked #1
   - "What did this product do before?" is highly predictive

3. **Price context matters more than absolute price**
   - PriceRelativeToMax, PriceRelativeToMin in top 5
   - Same price means different things for different products

4. **Most products have pricing opportunities**
   - 86% showed >5% uplift potential
   - Many prices are suboptimal

5. **Elasticity varies widely**
   - No one-size-fits-all pricing strategy
   - Product-specific optimization is valuable

---

## 11. Technical Architecture

### Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.9+ |
| Data Processing | pandas, NumPy |
| Machine Learning | scikit-learn, XGBoost |
| Visualization | Plotly, Matplotlib |
| Web Application | Streamlit |
| Model Serialization | joblib |

### Project Structure

```
smart_pricing_engine/
├── app/
│   └── app.py                    # Streamlit web application
├── src/
│   ├── data_loader.py            # Data loading & cleaning
│   ├── feature_engineering.py    # Feature creation
│   ├── model_training.py         # Model training & evaluation
│   └── elasticity.py             # Price elasticity calculations
├── notebooks/
│   └── complete_workflow.py      # End-to-end pipeline
├── data/
│   ├── raw/                      # Original dataset
│   └── processed/                # Cleaned & feature-engineered data
├── models/                       # Saved trained models
├── requirements.txt
└── README.md
```

### Application Pages

| Page | Purpose |
|------|---------|
| Home | Project overview, key metrics |
| Data Explorer | Dataset exploration, distributions |
| Model Performance | Model comparison, feature importance |
| Price Optimizer | Interactive price simulation |
| About | Methodology and contact |

---

## 12. Limitations & Future Work

### Current Limitations

| Limitation | Impact | Potential Solution |
|------------|--------|-------------------|
| **Observational data** | Correlation ≠ causation; prices weren't experimentally varied | A/B testing for validation |
| **Historical dataset** | 2009-2011 data may not reflect current patterns | Periodic retraining on new data |
| **No external factors** | Missing competitor pricing, marketing, economic data | Integrate external data sources |
| **Weekly granularity** | Can't capture intra-week patterns | Daily aggregation (requires more data) |
| **Single market** | UK retailer only | Expand to multi-market |

### Future Enhancements

**Short-term:**
- Add confidence intervals to predictions
- Implement cross-validation
- Add more products to elasticity analysis

**Medium-term:**
- User file upload (apply to any dataset)
- Automated weekly retraining
- API for integration with e-commerce platforms

**Long-term:**
- Real-time pricing recommendations
- Multi-market expansion
- Competitor price monitoring integration
- Inventory-aware pricing (factor in stock levels)

---

## References

1. UCI Machine Learning Repository. (2019). Online Retail II Data Set. https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

2. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD '16.

3. Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.

4. Scikit-learn Documentation. https://scikit-learn.org/

5. Streamlit Documentation. https://docs.streamlit.io/

---

## Author

**Justin Conroy**  
M.S. Data Science — Eastern University

- GitHub: [@JustinMC-data](https://github.com/JustinMC-data)
- Website: [NousForge Systems](https://nousforgesystems.com)

---

*Document Version: 1.0*  
*Last Updated: March 2025*
