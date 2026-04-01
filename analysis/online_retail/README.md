# Case Study: Online Retail Customer Intelligence

## The Business Problem

A UK-based online retailer with over 1 million transactions needed to answer three questions:

- **Who are our most valuable customers — and who is about to leave?**
- **Which products drive the most revenue, and where are the geographic opportunities?**
- **What will demand look like over the next quarter?**

Without structured analysis, pricing decisions, marketing spend, and inventory planning were all based on intuition rather than data. The goal was to turn raw transactional records into a clear picture of the business — and specific actions to improve it.

---

## Dataset

| Property | Value |
|---|---|
| Raw transactions | 1,067,371 |
| Records after cleaning | 779,425 (73% retained) |
| Customers analyzed | 4,312 |
| Products analyzed | 3,924 |
| Countries represented | 40 |
| Date range | Dec 2009 — Dec 2011 |
| Total revenue | $6,927,819 |

---

## What I Delivered

### 1. Data Quality Assessment & Cleaning

The raw dataset required substantial preprocessing before analysis was meaningful:

- Removed 287,946 records including cancellations, missing customer IDs, and invalid prices
- Standardized date fields, corrected data types, and validated schema consistency
- Produced a cleaned dataset scored at **94/100** on automated data quality metrics

### 2. Exploratory Business Analysis

A full business intelligence report covering:

- **Revenue breakdown** — $6.9M total across 24,221 unique orders
- **Product performance** — top products by order count, quantity sold, and revenue
- **Geographic analysis** — United Kingdom represents 92.4% of transactions; EIRE, Germany, and France are the next largest markets
- **Customer behavior** — average 4.9 orders per customer, $1,606 average lifetime spend
- **Purchase frequency distribution** — identifying high-frequency buyers vs one-time purchasers

**Key finding:** The top 15 customers account for $1.54M — **22.3% of total revenue** — an extreme concentration that represents both an opportunity and a significant business risk.

### 3. Customer Segmentation (RFM Analysis)

Segmented all 4,312 customers into 11 behavioral groups using Recency, Frequency, and Monetary scoring:

| Segment | Customers | Revenue | Avg Spend | Action |
|---|---|---|---|---|
| Champions | 934 (21.7%) | $4,575,942 (66.1%) | $4,899 | Loyalty programs, early access |
| Loyal Customers | 700 (16.2%) | $1,114,757 (16.1%) | $1,593 | Upsell, tiered rewards |
| Potential Loyalists | 770 (17.9%) | $558,116 (8.1%) | $725 | Targeted nurture campaigns |
| At Risk | 74 (1.7%) | $66,104 (1.0%) | $893 | Urgent win-back outreach |
| Hibernating | 707 (16.4%) | $133,218 (1.9%) | $188 | Low-cost reactivation only |

**Pareto finding:** 6.7% of customers generate 50% of revenue. This concentration risk means losing a handful of Champions would have an outsized revenue impact.

**Churn risk alert:** 829 customers (19.2%) are flagged as at-risk, representing $237,548 in historical revenue requiring proactive retention intervention.

### 4. Demand Forecasting

Weekly demand forecasted 13 periods ahead using three methods compared by accuracy:

| Method | RMSE | MAPE | Result |
|---|---|---|---|
| Exponential Smoothing | 34,097 | 31.5% | ★ Best |
| Moving Average (window=6) | 37,952 | 34.9% | 2nd |
| Linear Trend | 39,274 | 36.3% | 3rd |

A MAPE of 31.5% is typical for weekly retail data with high short-term variability. Monthly aggregation or seasonal ARIMA modeling would improve accuracy further.

---

## Key Business Recommendations

| Finding | Recommended Action | Expected Impact |
|---|---|---|
| 22.3% of revenue from 15 customers | Dedicated account management for top 50 customers | Protect highest-risk revenue concentration |
| Champions drive 66.1% of revenue | Loyalty program with early product access and exclusive pricing | Increase retention in highest-value segment |
| 829 at-risk customers | Automated win-back email sequence with time-limited offers | Recover portion of $237K at-risk revenue |
| UK represents 92.4% of sales | Targeted expansion campaigns in EIRE, Germany, France | Diversify geographic revenue concentration |
| Hibernating segment (16.4%) | Two-touch reactivation maximum, then archive | Reduce wasted marketing spend |

---

## How to Reproduce

```bash
# Step 1 — Download the dataset
# https://archive.ics.uci.edu/dataset/502/online+retail+ii
# Place the Excel file in the project directory

# Step 2 — Clean the data
python tools/data_cleaner.py online_retail_II.xlsx --auto

# Step 3 — Run analysis
python tools/data_analyzer.py online_retail_II_cleaned.xlsx --domain retail

# Step 4 — Customer segmentation
python tools/customer_segmentation.py online_retail_II_cleaned.xlsx --auto

# Step 5 — Demand forecasting
python tools/data_forecaster.py online_retail_II_cleaned.xlsx --domain retail
```

---

## Reports Included

| Report | Contents |
|---|---|
| `online_retail_II_cleaned_analysis_report.html` | Full EDA — dataset overview, descriptive stats, revenue summary, top products, geographic breakdown, customer analysis |
| `online_retail_II_cleaned_segmentation_report.html` | RFM scores, 11 customer segments with business recommendations, Pareto analysis, churn risk alert, top 20 customers |
| `online_retail_II_cleaned_forecast_report.html` | 13-week demand forecast, 3 methods compared, accuracy metrics, interpretation guidance |
| `online_retail_II_cleaned_rfm_data.csv` | Customer-level RFM scores and segment assignments for further analysis |

All reports are self-contained HTML files — open in any browser, no software required.

---

## What This Demonstrates

| Capability | How It's Shown |
|---|---|
| Business framing | Analysis structured around questions a business owner actually asks |
| Data quality handling | 27% of raw records identified and removed with documented reasoning |
| Customer analytics | RFM segmentation with 11 named segments and specific per-segment actions |
| Revenue concentration analysis | Pareto analysis identifying 6.7% of customers driving 50% of revenue |
| Time series forecasting | 3 methods compared with accuracy metrics and interpretation |
| Stakeholder communication | Every finding paired with a specific recommended action |
| Automated pipeline | 4 reports generated from a single cleaned dataset |

---

## Acknowledgments

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/) — Online Retail II dataset
- Analysis generated using the INDAP proprietary analytics platform

