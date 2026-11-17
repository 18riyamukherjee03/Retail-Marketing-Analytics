# 🛍️ Retail & Marketing Analytics: Customer Segmentation & Sales Optimization

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Visuals-3F4F75.svg?logo=plotly&logoColor=white)](https://plotly.com/)
[![Power BI](https://img.shields.io/badge/Power%20BI-Business%20Intelligence-F2C811.svg?logo=power-bi&logoColor=black)](https://powerbi.microsoft.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Project Status](https://img.shields.io/badge/Status-Completed-success.svg)](#)

> **An enterprise-grade, end-to-end retail and marketing analytics project** that transforms 10,000 raw multi-channel transactions into actionable customer intelligence, predictive lifetime value models, unsupervised machine learning segments, and executive dashboard KPIs.

---

## 📋 Table of Contents

- [Executive Summary & Key Metrics](#-executive-summary--key-metrics)
- [Business Problem & Objectives](#-business-problem--objectives)
- [Project Architecture & Workflow](#-project-architecture--workflow)
- [Dataset Overview & Engineering](#-dataset-overview--engineering)
  - [Raw Data vs Engineered Features](#raw-data-vs-engineered-features)
  - [Data Cleaning & Outlier Treatment](#data-cleaning--outlier-treatment)
  - [Feature Engineering Highlights](#feature-engineering-highlights)
- [Exploratory Data Analysis (EDA) & Insights](#-exploratory-data-analysis-eda--insights)
  - [Category Performance](#1-product-category-performance)
  - [Regional Distribution](#2-regional-penetration--revenue)
  - [Temporal Dynamics & Seasonality](#3-temporal--seasonality-trends)
  - [Pareto Principle Analysis](#4-pareto-analysis)
- [Advanced Customer Analytics & Modeling](#-advanced-customer-analytics--modeling)
  - [RFM Segmentation (Rule-Based)](#1-rfm-segmentation-rule-based)
  - [K-Means Unsupervised Clustering](#2-k-means-machine-learning-clustering)
  - [Cohort Retention Analysis](#3-cohort-retention-analysis)
  - [Customer Lifetime Value (CLV) & Unit Economics](#4-customer-lifetime-value-clv--unit-economics)
- [Business Intelligence & Dashboard Architecture](#-business-intelligence--dashboard-architecture)
  - [Power BI 4-Page Dashboard](#power-bi-reporting-views)
  - [Interactive Plotly Visualizations](#interactive-visualizations-catalog)
- [Strategic Recommendations & Action Plan](#-strategic-recommendations--action-plan)
- [Repository Structure](#-repository-structure)
- [Installation & How to Run](#-installation--how-to-run)
- [Tech Stack & Tools](#-tech-stack--tools)
- [Author & Acknowledgments](#-author--acknowledgments)

---

## 📊 Executive Summary & Key Metrics

This project analyzes **10,000 transactions** across **1,986 unique customers** over a **416-day window** (January 1, 2022 to February 21, 2023), generating **$1.08M in gross revenue** and **$198.3K in net profit**. 

By applying both heuristic **RFM (Recency, Frequency, Monetary) segmentation** and **unsupervised K-Means clustering**, this analysis identifies two dominant customer tiers and seven micro-segments, proving that **46.5% of the customer base (VIPs) generates 66.2% of total retail revenue**.

### 📌 High-Level Performance Scorecard

| Metric Category | Key Performance Indicator (KPI) | Result / Value | Business Benchmark / Health |
| :--- | :--- | :--- | :--- |
| **Financial** | **Total Revenue** | **$1,078,670.98** | Stable growth across 14 operating months |
| **Financial** | **Total Gross Profit** | **$198,274.85** | Overall profit margin: **18.38%** |
| **Financial** | **Average Order Value (AOV)** | **$107.87** | Consistent across categories ($106 - $108) |
| **Financial** | **Average Daily Revenue** | **$2,592.96** | Steady run rate (~24.0 orders/day) |
| **Customer** | **Total Unique Customers** | **1,986** | 10,000 orders (5.04 orders/customer) |
| **Customer** | **Repeat Customer Rate** | **96.27%** | 1,912 repeat buyers vs 74 one-time buyers |
| **Customer** | **Customer Retention Rate** | **89.12%** | High baseline retention across cohorts |
| **Customer** | **Churn Rate** | **10.88%** | 216 customers inactive > 180 days |
| **Customer** | **Average Recency** | **81.6 days** | Mean elapsed days since last purchase |
| **Product** | **Total Units Sold** | **50,065 units** | Average 5.01 units per transaction |
| **Product** | **Catalog Breadth** | **499 SKUs** | Across 4 categories & 8 sub-categories |
| **Unit Economics** | **Customer Acquisition Cost (CAC)** | **$50.00** | Industry estimated baseline |
| **Unit Economics** | **Customer Lifetime Value (CLV)** | **$7,521.29** | Projected across active lifespan |
| **Unit Economics** | **CLV / CAC Ratio** | **150.43x** | Exceptional unit economics (Benchmark: >3.0x) |
| **Unit Economics** | **CAC Payback Period** | **4.4 months** | Rapid return on acquisition capital |

---

## 🎯 Business Problem & Objectives

### The Challenge
A multi-category omnichannel retail business experienced uneven category sales, rising customer drop-offs after initial purchases, untargeted promotional discounts, and lack of clarity on customer retention dynamics.

### Analytical Objectives
1. **Data Hygiene & Preprocessing:** Establish an automated cleaning pipeline to treat outliers, resolve null values, enforce strict typings, and engineer domain features.
2. **Sales & Category Optimization:** Uncover category-level, regional, and temporal drivers of revenue, AOV, and margin.
3. **Advanced Customer Segmentation:** Segment customer cohorts using statistical RFM scoring and machine learning (K-Means clustering) to differentiate high-yield buyers from churning accounts.
4. **Cohort & Retention Modeling:** Track monthly acquisition cohorts over time to calculate actual retention decay and measure customer lifetime value (CLV).
5. **Decision Intelligence & Dashboarding:** Deliver BI-ready data models and interactive visualizations for executive stakeholders to guide targeted marketing and inventory allocation.

---

## 🏗️ Project Architecture & Workflow

The analytics lifecycle is partitioned into five distinct modular phases:

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                               ANALYTICS PIPELINE                              │
└───────────────────────────────────────────────────────────────────────────────┘
                                       │
  ┌────────────────────────────────────▼─────────────────────────────────────┐
  │ 1. DATA ACQUISITION & SETUP                                              │
  │    • Ingestion from Kaggle via KaggleHub API / Raw CSVs                  │
  │    • Initial quality profiling, schema verification, memory inspection   │
  │    • Output: initial_inspection_report.txt, original_data_checkpoint.csv │
  └────────────────────────────────────┬─────────────────────────────────────┘
                                       │
  ┌────────────────────────────────────▼─────────────────────────────────────┐
  │ 2. DATA CLEANING & PREPROCESSING                                         │
  │    • Imputation of missing cells (median/mode)                           │
  │    • Outlier Winsorization via IQR capping on Sales, Quantity, Profit    │
  │    • Feature Engineering: 19 raw columns ───► 45 analytical attributes   │
  │    • Output: cleaned_retail_sales.csv, data_dictionary.csv               │
  └────────────────────────────────────┬─────────────────────────────────────┘
                                       │
  ┌────────────────────────────────────▼─────────────────────────────────────┐
  │ 3. EXPLORATORY DATA ANALYSIS (EDA)                                       │
  │    • Univariate distributions, bivariate correlations, Pareto analysis   │
  │    • Multi-temporal breakdowns (MoM, QoQ, Day of Week, Seasonality)      │
  │    • Output: 16+ figures & HTMLs, 03_eda_key_findings.txt                │
  └────────────────────────────────────┬─────────────────────────────────────┘
                                       │
  ┌────────────────────────────────────▼─────────────────────────────────────┐
  │ 4. ADVANCED CUSTOMER ANALYTICS                                           │
  │    • Rule-based RFM Quintile Scoring (7 granular segments)               │
  │    • StandardScaler + K-Means Clustering (Optimal k=2 via Silhouette)    │
  │    • Cohort Analysis (14 monthly retention matrices)                     │
  │    • CLV Modeling ($AOV × Frequency × Lifespan)                          │
  │    • Output: customer_segments.csv, rfm_analysis.csv, customer_clv.csv   │
  └────────────────────────────────────┬─────────────────────────────────────┘
                                       │
  ┌────────────────────────────────────▼─────────────────────────────────────┐
  │ 5. KPI DESIGN & EXECUTIVE REPORTING                                      │
  │    • Relational data models for Power BI / Tableau dashboards            │
  │    • Monthly & regional KPI time series tables                           │
  │    • Executive summary report & strategic action recommendations         │
  │    • Output: monthly_kpis.csv, category_kpis.csv, executive_summary.txt  │
  └──────────────────────────────────────────────────────────────────────────┘
```

---

## 🗄️ Dataset Overview & Engineering

### Raw Data vs Engineered Features
The raw transaction dataset contained **10,000 rows and 19 columns**. Through the cleaning and feature engineering pipeline, the dataset expanded to **45 columns**, enriching downstream analytical models with time-series, behavioral, and commercial dimensions.

| Dimension | Raw Columns (19) | Engineered Features (+26, Total 45) |
| :--- | :--- | :--- |
| **Transaction** | `Order_ID`, `Order_Date`, `Ship_Date`, `Order_Priority` | `Delivery_Days`, `Delivery_Category`, `Season` |
| **Customer** | `Customer_ID`, `Customer_Name`, `Segment`, `Region` | `Customer_Order_Count`, `Is_Repeat_Customer` |
| **Product** | `Product_ID`, `Product_Category`, `Product_Sub_Category`, `Product_Name` | `Product_Total_Sales`, `Product_Avg_Sales`, `Product_Order_Count` |
| **Financial** | `Sales`, `Quantity`, `Discount`, `Profit`, `Shipping_Cost` | `Unit_Price`, `Revenue`, `Discount_Amount`, `Net_Revenue`, `Profit_Margin`, `Profit_Ratio`, `Sales_Category`, `Sales_Original`, `Profit_Original` |
| **Calendar** | *None* | `Year`, `Month`, `Month_Name`, `Quarter`, `Day`, `Day_of_Week`, `Day_Name`, `Week_of_Year`, `Is_Weekend`, `Is_Month_Start`, `Is_Month_End` |

### Data Cleaning & Outlier Treatment
- **Missing Value Imputation:** 80 missing records (0.04% of total dataset) identified and imputed using column medians for continuous numerical fields and modes for categorical fields.
- **Deduplication:** Confirmed 0 duplicate rows; 100% data integrity retained.
- **Outlier Treatment:** Outliers in `Sales`, `Quantity`, and `Profit` were detected using the **Interquartile Range (IQR)** approach ($[Q_1 - 1.5 \times IQR, Q_3 + 1.5 \times IQR]$) and treated using **Winsorization (capping)** to avoid loss of valid transactional records while mitigating distortion in linear and distance-based clustering algorithms.
- **Memory Optimization:** Optimized storage formats (e.g., `category` for low-cardinality strings, `int32`/`uint32` for discrete counts), slashing memory footprint from 7.32 MB to 6.48 MB.

---

## 🔍 Exploratory Data Analysis (EDA) & Insights

### 1. Product Category Performance
Analysis of 4 main categories across 499 distinct SKUs reveals balanced transactional volume but distinct revenue and profit contributions:

| Product Category | Total Revenue | Revenue Share | Total Orders | Order Share | Units Sold | SKUs | Avg Order Value (AOV) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Electronics** | **$328,422.71** | **30.45%** | 3,029 | 30.29% | 15,179 | 496 | $108.43 |
| **Office Supplies** | **$321,280.86** | **29.78%** | 2,949 | 29.49% | 14,746 | 499 | $108.95 |
| **Furniture** | **$214,958.03** | **19.93%** | 2,011 | 20.11% | 9,997 | 494 | $106.89 |
| **Clothing** | **$214,009.37** | **19.84%** | 2,011 | 20.11% | 10,143 | 484 | $106.42 |

> **Key Finding:** Electronics and Office Supplies drive **60.23%** of total revenue. Average order values remain tight ($106.42 to $108.95), suggesting that revenue differences stem primarily from order frequency and basket volume rather than unit pricing disparity.

### 2. Regional Penetration & Revenue
Regional performance indicates strong customer market penetration across four designated territories:

| Region | Total Revenue | Revenue Share | Order Count | Unique Customers | Customer Penetration | Units Sold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **East** | **$318,581.37** | **29.53%** | 2,968 | 1,553 | **78.20%** | 14,704 |
| **Central** | **$275,688.18** | **25.56%** | 2,545 | 1,435 | **72.26%** | 12,702 |
| **West** | **$272,330.96** | **25.25%** | 2,512 | 1,436 | **72.31%** | 12,703 |
| **South** | **$212,070.47** | **19.66%** | 1,975 | 1,257 | **63.29%** | 9,956 |

> **Key Finding:** The **East** region leads all markets in both gross revenue ($318.6K) and customer penetration (78.2%). The **South** represents an untapped growth market with lower customer penetration (63.3%), pinpointing an immediate opportunity for targeted customer acquisition.

### 3. Temporal & Seasonality Trends
- **Quarterly Peaks:** Q1 is historically the strongest quarter, capturing **34.2%** of full-year sales volume.
- **Monthly Revenue:** Stable monthly revenue averaging **$76,000 - $82,000**, with peak revenue observed in January ($82.7K), August ($82.2K), and December ($82.2K).
- **Day of Week Effect:** Weekend sales exhibit a **+0.9% higher order value** compared to weekdays, demonstrating consistent weekly shopping cadences.

### 4. Pareto Analysis
- The top **20% of products generate 27.3% of revenue**, showing healthy catalog diversification without extreme reliance on a fragile set of single products.
- Conversely, customer distribution adheres strictly to value concentration: **the top 46.5% of customers generate 66.2% of total business revenue**.

---

## 👥 Advanced Customer Analytics & Modeling

### 1. RFM Segmentation (Rule-Based)
Each of the 1,986 customers was evaluated on three core behavioral dimensions using quintile ranking ($1 - 5$ score):
- **Recency (R):** Elapsed calendar days between the snapshot cutoff and the customer's last recorded purchase.
- **Frequency (F):** Total number of distinct completed orders.
- **Monetary (M):** Cumulative lifetime spend ($).

```
                      RFM SEGMENT DISTRIBUTION (1,986 CUSTOMERS)
                      
   About to Sleep  ████████████████████████████  446 (22.5%)
  Loyal Customers  █████████████████████████  401 (20.2%)
        Champions  ██████████████████████  353 (17.8%)
          At Risk  █████████████████████  344 (17.3%)
    New Customers  ██████████████████  289 (14.6%)
        Promising  ██████  93 (4.7%)
Potential Loyalists  ████  60 (3.0%)
```

| Segment | Customer Count | Share (%) | Behavioral Profile | Marketing Strategy |
| :--- | :---: | :---: | :--- | :--- |
| **About to Sleep** | 446 | 22.46% | Below-average recency and frequency | Personalized re-engagement incentives & surveys |
| **Loyal Customers** | 401 | 20.19% | High frequency, steady monetary, good recency | Cross-sell higher-tier categories, loyalty perks |
| **Champions** | 353 | 17.77% | Highest frequency, top monetary, lowest recency | VIP rewards, early access to new releases, advocacy |
| **At Risk** | 344 | 17.32% | High past spend but haven't bought in >100 days | Urgent win-back discounts & personalized outreach |
| **New Customers** | 289 | 14.55% | High recency (bought recently) but low frequency | Automated onboarding sequence & welcome coupon |
| **Promising** | 93 | 4.68% | Recent buyers with moderate spend | Product recommendations and bundle offers |
| **Potential Loyalists** | 60 | 3.02% | High initial basket value with repeat purchases | Tiered loyalty qualification, upsell bundles |

---

### 2. K-Means Machine Learning Clustering
To discover organic clusters without subjective threshold rules, unsupervised **K-Means clustering** was executed on standardized RFM features using `StandardScaler`.

#### Model Optimization
- **Elbow Method:** Evaluated inertia reductions across $k \in [2, 10]$.
- **Silhouette Analysis:** Peak silhouette score confirmed an optimal boundary at **$k = 2$ macro-clusters**:

```
                              MACRO SEGMENT COMPARISON
                              
                 ┌───────────────────────────┬───────────────────────────┐
                 │    VIP CUSTOMERS (46.5%)  │    AT-RISK / LOW (53.5%)  │
                 ├───────────────────────────┼───────────────────────────┤
  Customer Count │ 924 customers             │ 1,062 customers           │
  Revenue Total  │ $713,663.99 (66.2%)       │ $365,006.99 (33.8%)       │
  Avg Recency    │ 45.8 days                 │ 112.8 days                │
  Avg Frequency  │ 6.89 orders               │ 3.42 orders               │
  Avg Monetary   │ $772.36                   │ $343.70                   │
                 └───────────────────────────┴───────────────────────────┘
```

> **Strategic Takeaway:** The VIP segment is more than **2x as active** (6.89 vs 3.42 orders) and spends **2.25x as much** ($772 vs $344) with **less than half the recency gap** (45.8 vs 112.8 days). Protecting this 924-customer core is critical to enterprise margin.

---

### 3. Cohort Retention Analysis
Acquisition cohorts were established based on each customer's first purchase month (from 2022-01 through 2023-02) and tracked over successive monthly periods:

- **Initial Drop-off:** Month 1 retention across cohorts stabilizes between **26.8% and 42.9%** (e.g., 2022-01 cohort retention at Month 1 is 26.8%; 2022-06 cohort is 42.0%; 2022-08 cohort is 42.9%).
- **Long-Tail Stability:** Unlike standard consumer e-commerce curves that drop to zero, retention rates plateau between **24% and 35% through Month 12**, demonstrating exceptionally strong organic product-market fit and repeat buyer loyalty.
- **Repeat Rate:** Overall repeat customer rate reached **96.27%** (1,912 out of 1,986 customers completed 2 or more lifetime purchases).

---

### 4. Customer Lifetime Value (CLV) & Unit Economics
Customer Lifetime Value was estimated using the standard transaction model:

$$\text{CLV} = \text{Average Order Value} \times \text{Purchase Frequency (Annualized)} \times \text{Customer Lifespan (Years)}$$

- **Mean CLV:** **$7,521.29** across the active customer base.
- **CLV Distribution Quartiles:**
  - **Low CLV (Q1):** \$253.80 – \$1,701.37 (497 customers)
  - **Medium CLV (Q2):** \$1,701.37 – \$2,340.98 (496 customers)
  - **High CLV (Q3):** \$2,340.98 – \$3,334.64 (496 customers)
  - **Very High CLV (Q4):** \$3,334.64 – \$301,695.48 (497 customers)
- **Unit Economics Health:**
  - Standard customer acquisition cost benchmark: **$50.00**.
  - **CLV / CAC Ratio:** **150.43x**, indicating exceptional capital efficiency and high profitability per acquired account.
  - **CAC Payback Period:** **4.4 months**, enabling rapid reinvestment cycles into acquisition channels.

---

## 📊 Business Intelligence & Dashboard Architecture

### Power BI Reporting Views
The project provides complete tabular and relational modeling configurations for a 4-page Power BI executive reporting suite:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       POWER BI 4-PAGE DASHBOARD DESIGN                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PAGE 1: EXECUTIVE SUMMARY                                                  │
│  ├── KPI Metric Cards: Gross Revenue, Total Profit, AOV, Gross Margin       │
│  ├── Line Chart: 14-Month Revenue & Order Trajectory                        │
│  ├── Treemap: Revenue Contribution by Product Category                      │
│  └── Filled Shape Map: Geographic Revenue & Penetration by Region           │
│                                                                             │
│  PAGE 2: CUSTOMER SEGMENTATION & BEHAVIOR                                   │
│  ├── Donut Chart: Customer Distribution across RFM Segments                 │
│  ├── Scatter Plot: RFM 3D Distribution (Recency vs Frequency, Size=Monetary)│
│  ├── Heatmap Matrix: Monthly Cohort Retention Percentages                   │
│  └── Table Visual: Top 10 High-Value Customers & Recent Activity            │
│                                                                             │
│  PAGE 3: PRODUCT & MERCHANDISING PERFORMANCE                                │
│  ├── Dual-Axis Bar & Line: Top 20 SKUs by Revenue vs Quantity Sold          │
│  ├── Scatter Matrix: Category Orders vs Average Order Value                 │
│  └── Clustered Column: Product Category Sales by Geographic Region          │
│                                                                             │
│  PAGE 4: MARKETING INTELLIGENCE & CLV                                       │
│  ├── Histogram Visual: Distribution of Customer Lifetime Value              │
│  ├── Gauge Charts: Retention Rate (89.1%) vs Churn Rate (10.9%)             │
│  ├── Funnel Chart: New Customer Acquisition to Multi-Order Repeat           │
│  └── Card Grid: CAC Payback, Profit per Customer, CLV/CAC Ratio             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Interactive Visualizations Catalog
The `outputs/figures/` directory houses **27 ready-to-inspect visual artifacts** (10 high-resolution static PNGs and 17 interactive standalone HTML Plotly dashboards):

| Output File | Type | Description |
| :--- | :---: | :--- |
| `01_missing_values.png` | PNG | Data quality heatmap prior to cleaning |
| `03_outliers_before_treatment.png` | PNG | Boxplot distribution of raw sales, quantities, and profits |
| `04_outliers_after_treatment.png` | PNG | Distribution confirmation following IQR Winsorization |
| `05_numerical_distributions.png` | PNG | Multi-panel histograms of all continuous numerical features |
| `07_sales_by_category.html` | Interactive | Interactive bar breakdown of sales across 4 categories |
| `08_sales_by_region.html` | Interactive | Donut chart showing regional share of wallet |
| `09_correlation_matrix.png` | PNG | Pearson correlation matrix across numerical variables |
| `10_monthly_sales_trend.html` | Interactive | Dynamic timeline of monthly sales velocity |
| `17_rfm_segments.html` | Interactive | Donut chart of customer proportions across 7 RFM tiers |
| `18_optimal_clusters.png` | PNG | Dual-axis Elbow Curve & Silhouette Score evaluation |
| `19_customer_segments_pca.html` | Interactive | 2D PCA projection of K-Means customer clusters |
| `20_customer_segments_3d.html` | Interactive | 3D interactive scatter plot of customers in RFM space |
| `23_cohort_retention.png` | PNG | Color-coded triangle matrix of monthly cohort retention rates |
| `24_clv_distribution.html` | Interactive | Histogram of Customer Lifetime Value tiers |
| `25_monthly_revenue_trend_detailed.html` | Interactive | Granular monthly KPI time series |
| `27_segment_revenue_sunburst.html` | Interactive | Hierarchical sunburst of Segments ➔ Categories ➔ Revenue |

---

## 💡 Strategic Recommendations & Action Plan

Based on the quantitative findings, an operational roadmap is recommended across three execution horizons:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          STRATEGIC EXECUTION ROADMAP                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  HORIZON 1: IMMEDIATE ACTIONS (0 - 30 DAYS)                                 │
│  ├── 1. Institutionalize VIP Concierge Program                              │
│  │      Lock in the 924 VIP customers (66.2% of revenue) with dedicated     │
│  │      support, exclusive early access, and personalized thank-you perks.  │
│  ├── 2. Launch Automated "About to Sleep" Reactivation Trigger              │
│  │      Deploy targeted 15% discount sequences to 446 at-risk accounts whose│
│  │      purchase gap has exceeded 80 days.                                  │
│  └── 3. Protect High-Velocity Inventory in Electronics & Office Supplies    │
│         Prioritize supply chain restocking for the top 20 revenue-driving   │
│         SKUs to prevent stock-outs during peak quarter starts.              │
│                                                                             │
│  HORIZON 2: MEDIUM-TERM INITIATIVES (2 - 3 MONTHS)                          │
│  ├── 1. Regional Expansion Campaign in the South                            │
│  │      Address lower penetration (63.3%) in the South through localized    │
│  │      promotions and regionally tailored catalog assortments.             │
│  ├── 2. Basket-Building Cross-Sell Bundles                                  │
│  │      Bundle lower-margin Clothing/Furniture items with Electronics to    │
│  │      lift overall basket sizes above the current 5.01 items/order.       │
│  └── 3. Cohort Onboarding Nurture Track                                     │
│         Capitalize on high initial 30-day engagement with an automated      │
│         3-part educational email sequence to drive second-order conversion. │
│                                                                             │
│  HORIZON 3: LONG-TERM TRANSFORMATION (6 - 12 MONTHS)                        │
│  ├── 1. Machine Learning Predictive Churn Scoring                           │
│  │      Transition from retrospective RFM recency to proactive predictive   │
│  │      churn classification (Random Forest / XGBoost) trained on frequency.│
│  ├── 2. Dynamic Real-Time Personalization Engine                            │
│  │      Implement real-time product recommendations on digital storefronts  │
│  │      based on customer sub-category affinity.                            │
│  └── 3. Projected Financial Impact                                          │
│         Target: +15-20% revenue expansion, +25-30% retention improvement,   │
│         and a 20-25% reduction in dormant churned accounts.                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
retail-marketing-analytics/
├── .gitignore                                      # Production Git ignore rules
├── requirements.txt                                # Project dependencies
├── README.md                                       # Comprehensive project documentation
├── Capstone2_Retail_and_Marketing_Analytics...ipynb# Consolidated master Jupyter notebook
├── PowerBI_doc.docx                                # Power BI dashboard specifications & setup
├── PowerBI_doc_Readme.docx                         # Power BI modeling reference guide
├── retail_marketing_project.md                     # Comprehensive project syllabus & reference
│
├── data/
│   ├── raw/
│   │   ├── retail_sales_data.csv                   # Raw source transactions (10,000 rows × 19 cols)
│   │   └── original_data_checkpoint.csv            # Backup raw checkpoint
│   └── processed/
│       ├── cleaned_retail_sales.csv                # Cleaned, enriched dataset (10,000 rows × 45 cols)
│       ├── customer_segments.csv                   # RFM & K-Means clustered customer records
│       ├── rfm_analysis.csv                        # RFM scores & customer segment tags
│       ├── customer_clv.csv                        # Customer lifetime value & frequency metrics
│       └── monthly_kpis.csv                        # Monthly aggregated KPI time series
│
├── docs/
│   └── data_dictionary.csv                         # Full 45-column data dictionary & definitions
│
├── notebooks/
│   ├── retail_notebook_01.py                       # Part 1: Acquisition, environment & inspection
│   ├── retail_notebook_02.py                       # Part 2: Data cleaning, imputation & feature engineering
│   ├── retail_notebook_03.py                       # Part 3: Univariate, bivariate & time series EDA
│   ├── retail_notebook_04.py                       # Part 4: RFM segmentation, K-Means, cohorts & CLV
│   └── retail_notebook_05.py                       # Part 5: KPI framework design & dashboard prep
│
├── outputs/
│   ├── figures/                                    # 27 visual analytics outputs
│   │   ├── 01_missing_values.png                   # Initial null pattern visualization
│   │   ├── 02_data_types_distribution.png          # Variable type breakdown
│   │   ├── 03_outliers_before_treatment.png        # Raw outlier distributions
│   │   ├── 04_outliers_after_treatment.png         # Post-Winsorization boxplots
│   │   ├── 05_numerical_distributions.png          # Numerical feature histograms
│   │   ├── 06_categorical_distributions.png        # Categorical count distributions
│   │   ├── 07_sales_by_category.html               # Category sales bar visual (interactive)
│   │   ├── 08_sales_by_region.html                 # Regional revenue donut (interactive)
│   │   ├── 09_correlation_matrix.png               # Correlation heatmap
│   │   ├── 10_monthly_sales_trend.html             # Monthly revenue trend (interactive)
│   │   ├── 11_quarterly_comparison.html            # Quarterly performance comparison (interactive)
│   │   ├── 12_sales_by_day.html                    # Day of week sales profile (interactive)
│   │   ├── 13_customer_frequency.png               # Order frequency histogram
│   │   ├── 14_top_customers.html                   # High-value customer ranking (interactive)
│   │   ├── 15_top_products.html                    # Top revenue SKUs (interactive)
│   │   ├── 16_category_performance.html            # Category order vs AOV scatter (interactive)
│   │   ├── 17_rfm_segments.html                    # RFM segment donut chart (interactive)
│   │   ├── 18_optimal_clusters.png                 # Elbow method & Silhouette curve
│   │   ├── 19_customer_segments_pca.html           # 2D PCA cluster visualization (interactive)
│   │   ├── 20_customer_segments_3d.html            # 3D RFM space scatter plot (interactive)
│   │   ├── 21_segment_distribution.html            # Customer volume per segment (interactive)
│   │   ├── 23_cohort_retention.png                 # Triangle cohort retention heatmap
│   │   ├── 24_clv_distribution.html                # CLV histogram (interactive)
│   │   ├── 25_monthly_revenue_trend_detailed.html  # Detailed monthly performance trend (interactive)
│   │   ├── 26_mom_kpi_comparison.html              # Month-over-month growth comparisons (interactive)
│   │   ├── 27_segment_revenue_sunburst.html        # Hierarchical segment sunburst (interactive)
│   │   └── category_kpis_treemap.html              # Category performance treemap (interactive)
│   └── reports/
│       ├── 01_initial_inspection_report.txt        # Initial data hygiene report
│       ├── 02_cleaning_report.txt                  # Preprocessing & engineering audit
│       ├── 03_eda_key_findings.txt                 # Core exploratory takeaways
│       ├── category_kpis.csv                       # Category performance export
│       ├── cohort_retention.csv                    # Monthly retention matrix data
│       ├── executive_summary.txt                   # C-suite executive briefing
│       ├── kpi_summary.csv                         # Comprehensive 30+ KPI metrics catalog
│       ├── project_completion_summary.txt          # Verification of project milestones
│       └── regional_kpis.csv                       # Regional revenue and customer metrics
│
└── scripts/
    └── test.ipynb                                  # Kaggle API integration test script
```

---

## 🚀 Installation & How to Run

### 1. Prerequisites
- Python 3.9 or higher
- Git
- Microsoft Power BI Desktop (Optional, for `.pbix` dashboard authoring)

### 2. Environment Setup
Clone the repository and initialize a virtual environment:

```bash
# Clone the repository
git clone https://github.com/18riyamukherjee03/Retail-Marketing-Analytics.git
cd Retail-Marketing-Analytics

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Executing the Analysis Pipeline
You can reproduce the entire end-to-end analytics workflow either by running the modular Python scripts sequentially or by opening the consolidated Jupyter notebook:

#### Option A: Running Modular Python Scripts
```bash
# Step 1: Data ingestion and initial profiling
python notebooks/retail_notebook_01.py

# Step 2: Data cleaning, outlier treatment, and feature engineering
python notebooks/retail_notebook_02.py

# Step 3: Exploratory data analysis and visual generation
python notebooks/retail_notebook_03.py

# Step 4: RFM scoring, K-Means clustering, cohorts, and CLV
python notebooks/retail_notebook_04.py

# Step 5: KPI synthesis, executive reporting, and dashboard data export
python notebooks/retail_notebook_05.py
```

#### Option B: Consolidated Jupyter Notebook
```bash
jupyter notebook Capstone2_Retail_and_Marketing_Analytics_Project.ipynb
```

### 4. Viewing Interactive Reports
All interactive Plotly dashboards are saved directly as `.html` files in `outputs/figures/`. Simply open any HTML file in any modern web browser:
```bash
# Example: Open interactive 3D customer clusters
start outputs/figures/20_customer_segments_3d.html        # Windows
open outputs/figures/20_customer_segments_3d.html         # macOS
xdg-open outputs/figures/20_customer_segments_3d.html     # Linux
```

### 5. Loading into Power BI
1. Launch **Power BI Desktop**.
2. Click **Get Data** ➔ **Text/CSV**.
3. Load the pre-processed datasets from `data/processed/`:
   - `cleaned_retail_sales.csv`
   - `customer_segments.csv`
   - `rfm_analysis.csv`
   - `monthly_kpis.csv`
4. Establish 1-to-many relationships linking `Customer_ID` across tables as detailed in [PowerBI_doc.docx](file:///c:/Users/amarj/OneDrive/Desktop/retail-marketing-analytics-main/PowerBI_doc.docx).

---

## 🛠️ Tech Stack & Tools

- **Programming Language:** Python 3.9+
- **Data Manipulation & Processing:** `pandas`, `numpy`, `scipy`
- **Machine Learning & Statistical Modeling:** `scikit-learn` (StandardScaler, KMeans, PCA, Silhouette Score)
- **Data Visualization & Analytics Reporting:** `matplotlib`, `seaborn`, `plotly` (Graph Objects & Express)
- **Business Intelligence & Dashboards:** Microsoft Power BI, Plotly HTML
- **Development Environment:** Jupyter Notebook, VS Code
- **Version Control & Documentation:** Git, Markdown, KaTeX

---

## 👤 Author & Acknowledgments

- **Author:** [Riya Mukherjee](https://github.com/18riyamukherjee03)
- **Project:** Retail & Marketing Analytics Capstone Project
- **Dataset Attribution:** [Retail Sales Dataset with Seasonal Trends & Marketing (Kaggle)](https://www.kaggle.com/datasets/abdullah0a/retail-sales-data-with-seasonal-trends-and-marketing)
- **License:** Distributed under the [MIT License](LICENSE). Feel free to use this project for learning, portfolio display, or production adaptation.

---

<p align="center">
  <b>⭐ If you found this project insightful or helpful for your portfolio, consider giving it a star! ⭐</b>
</p>
