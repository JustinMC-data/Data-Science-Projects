# SQL Data Warehouse Project

A complete data warehouse built on SQL Server, following the Medallion (Bronze, Silver, Gold) architecture. Raw CRM and ERP data is ingested, cleaned, deduplicated, and modeled into a star schema for analytics and reporting.

This project is based on the structure and datasets from [Data With Baraa's SQL Data Warehouse project](https://github.com/DataWithBaraa/sql-data-warehouse-project). I built the full pipeline myself — every script, every cleaning rule, and every design decision below is my own work, including the data quality issues I found and fixed along the way.

## Architecture

![Data Architecture](docs/data_architecture.png)

- **Bronze**: raw CRM and ERP data loaded as-is from CSV files, unmodified. This is the audit trail — if something looks wrong downstream, I can always check whether it was wrong here first.
- **Silver**: cleaned, standardized, and deduplicated data. Business rules and data quality fixes are applied here.
- **Gold**: a star schema (`dim_customers`, `dim_products`, `fact_sales`) built for reporting, joined by clean surrogate keys.

## Tech Stack

- SQL Server / T-SQL
- SSMS
- Medallion architecture, star schema design, window functions, CTEs

## Data Quality Issues Found and Fixed

Profiling the raw data surfaced several real issues. Each one is documented here with the fix and the reasoning.

| Issue | Table | Fix |
|---|---|---|
| Duplicate customer records (uneven counts — 2 or 3 copies per customer) | `crm_cust_info` | Deduplicated using `ROW_NUMBER()` partitioned by customer ID, keeping only the most recent record by create date |
| Inconsistent ID formatting between source systems (`AW-00011000` vs `AW00011000`) | `erp_loc_a101` | Stripped formatting characters so IDs match across CRM and ERP for joining |
| Placeholder `0` used instead of NULL for unknown order dates | `crm_sales_details` | Converted to proper NULL rather than attempting to parse as a real date |
| Missing or inconsistent sales/price/quantity math (negative values, mismatched totals) | `crm_sales_details` | Recalculated `sales_amount` and `price` from the more reliable fields (quantity, price) when the stored values didn't reconcile |
| Cryptic single-letter category and gender codes | `crm_prd_info`, `crm_cust_info` | Mapped to readable business values (e.g., `M` → `Mountain`, `M`/`F` → `Male`/`Female`) |
| Inconsistent country naming (`US`, `USA`, `United States`, `DE`, `Germany`) | `erp_loc_a101` | Standardized to full country names |
| Compound key mixing category and product identifiers in one field | `crm_prd_info` | Split into separate `category_id` and `product_number` columns |
| Missing product end dates for historical versions | `crm_prd_info` | Calculated using `LEAD()` to infer the end date from the next version's start date |

## Star Schema

![Data Model](docs/data_model.png)

- `gold.fact_sales` — one row per sales transaction, linked by surrogate keys
- `gold.dim_customers` — one row per unique customer, combining CRM and ERP data
- `gold.dim_products` — one row per current product, combining CRM product data and ERP category data

Every sales row was validated against both dimension tables with zero orphaned records (no sale references a missing customer or product).

## Sample Insights

A few of the questions this warehouse can answer out of the box (full queries in `analysis/analytics_queries.sql`):

- Top 10 products by revenue
<img width="390" height="214" alt="Screenshot 2026-09-30 132853" src="https://github.com/user-attachments/assets/8e6eff56-cc68-4150-b7ea-e46e484e1ced" />

- Revenue and customer count by country
  <img width="299" height="157" alt="Screenshot 2026-09-30 135351" src="https://github.com/user-attachments/assets/c8fd743e-e7d9-451e-b327-065730a52919" />

- Monthly sales trend with running total revenue
  <img width="281" height="746" alt="Screenshot 2026-09-30 135437" src="https://github.com/user-attachments/assets/aeecae13-36ee-496d-94d3-e91e98621825" />

- Customer segmentation by lifetime spend (High / Mid / Low value)
  <img width="335" height="83" alt="Screenshot 2026-09-30 145419" src="https://github.com/user-attachments/assets/88dc0e71-1b11-418a-a538-95814896b491" />

- Product category performance ranked by revenue
  <img width="295" height="85" alt="Screenshot 2026-09-30 145606" src="https://github.com/user-attachments/assets/17312332-2eee-435a-8c1b-3416c807cd9d" />




## Repository Structure
├── datasets/ # Raw source CSVs (CRM and ERP)
├── scripts/
│ ├── bronze/ # Raw table DDL and load scripts
│ ├── silver/ # Cleaning and transformation scripts
│ └── gold/ # Star schema view definitions
├── analysis/ # Analytical queries
└── docs/ # Data catalog, naming conventions, data quality log, diagrams

## How to Run

1. Install [SQL Server Express](https://www.microsoft.com/en-us/sql-server/sql-server-downloads) and [SSMS](https://learn.microsoft.com/en-us/sql/ssms/download-sql-server-management-studio-ssms).
2. Clone or download this repository.
3. Update the file paths in `scripts/bronze/load_bronze.sql` to point to wherever you saved the `datasets/` folder.
4. Run the scripts in order: `scripts/bronze/` → `scripts/silver/` → `scripts/gold/`.
5. Query the `gold` schema views, or run the queries in `analysis/analytics_queries.sql`.

## Documentation

- [Data Catalog](docs/data_catalog.md) — column-level definitions for the gold layer
- [Naming Conventions](docs/naming_conventions.md)
- [Data Quality Log](docs/data_quality_log.md) — full detail on every cleaning decision and assumption

## License

MIT License. Datasets and original project structure from [Data With Baraa](https://github.com/DataWithBaraa/sql-data-warehouse-project).
