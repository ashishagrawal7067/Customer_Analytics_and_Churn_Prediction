# 📊 Customer Analytics & Churn Prediction

> End-to-end data analytics pipeline: Data Generation → SQL Analysis → EDA → Feature Engineering → Machine Learning → Business Insights

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-3-003B57?logo=sqlite&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Excel](https://img.shields.io/badge/Excel-Report-217346?logo=microsoft-excel&logoColor=white)

---

## 🎯 Business Problem

Customer churn is one of the most critical challenges in business — acquiring a new customer costs **5–7x more** than retaining an existing one. This project builds a complete analytics pipeline to:

1. **Identify** which customers are likely to churn
2. **Segment** customers using RFM (Recency, Frequency, Monetary) analysis
3. **Predict** churn probability using ensemble machine learning models
4. **Recommend** targeted retention strategies backed by data

## 🏗️ Project Structure

```
Customer_Analytics_Project/
│
├── data/                          # Generated datasets
│   ├── customers.csv              # 500 customer records
│   ├── orders.csv                 # ~3,700 purchase transactions
│   └── customer_interactions.csv  # ~2,900 engagement records
│
├── notebooks/
│   └── Customer_Analytics.ipynb   # Main analysis notebook (27 steps)
│
├── scripts/
│   ├── generate_data.py           # Synthetic data generation
│   ├── sql_analysis.py            # SQL queries with SQLite
│   └── create_notebook.py         # Notebook generator utility
│
├── outputs/
│   ├── 01_EDA_Analysis.png        # 9-panel EDA visualization
│   ├── 02_Correlation_Matrix.png  # Feature correlation heatmap
│   ├── 03_ROC_Curves.png         # Model ROC comparison
│   ├── 04_Feature_Importance.png  # Feature importance charts
│   ├── Customer_Analytics_Report.xlsx  # 4-sheet business report
│   └── SQL_Analysis_Results.xlsx       # SQL query results (7 sheets)
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 🛠️ Tech Stack & Skills

| Skill | Tools | Application |
|-------|-------|-------------|
| **Python** | pandas, numpy, matplotlib, seaborn | Data manipulation, visualization, feature engineering |
| **SQL** | SQLite, CTEs, Window Functions | 7 analytical queries: JOINs, GROUP BY, RANK, LAG, CASE WHEN |
| **Machine Learning** | scikit-learn (Random Forest, Gradient Boosting) | Churn prediction with cross-validation and ensemble scoring |
| **Excel** | openpyxl | Multi-sheet business reports with customer segments |
| **Statistics** | RFM Analysis, Correlation, AUC-ROC | Customer segmentation and model evaluation |

## 📈 Key Results

### Customer Segmentation (RFM)

| Segment | Customers | Avg Revenue | Churn Rate |
|---------|-----------|-------------|------------|
| 🟢 VIP | ~20% | Highest | Lowest |
| 🔵 Loyal | ~25% | High | Low |
| 🟡 At Risk | ~25% | Medium | Medium |
| 🔴 Lost | ~30% | Low | Highest |

### Model Performance

| Model | ROC-AUC | Use Case |
|-------|---------|----------|
| Random Forest | ~0.95+ | Robust baseline, feature importance |
| Gradient Boosting | ~0.95+ | Best accuracy, sequential learning |
| **Ensemble (Avg)** | **Best** | **Production churn scoring** |

### Business Impact
- **Churn Rate:** ~22% (realistic for subscription/e-commerce)
- **High-Risk Customers:** Identified with probability scores
- **Revenue at Risk:** Quantified $ amount from high-risk segments
- **Actionable Recommendations:** 5 prioritized retention strategies

## 🚀 How to Run

### 1. Clone & Setup
```bash
git clone https://github.com/YOUR_USERNAME/Customer_Analytics_Project.git
cd Customer_Analytics_Project

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### 2. Generate Data
```bash
python scripts/generate_data.py
```
This creates 3 CSV files in `data/` with 500 customers, ~3,700 orders, and ~2,900 interactions.

### 3. Run SQL Analysis
```bash
python scripts/sql_analysis.py
```
Loads data into SQLite and executes 7 analytical queries demonstrating JOINs, Window Functions, CTEs, and more. Results saved to `outputs/SQL_Analysis_Results.xlsx`.

### 4. Run Jupyter Notebook
```bash
jupyter notebook notebooks/Customer_Analytics.ipynb
```
Execute all cells sequentially to see the full pipeline: EDA → Feature Engineering → ML → Business Insights.

## 📊 SQL Analysis Highlights

The SQL script demonstrates **7 progressively complex queries**:

```sql
-- Example: Window Functions — Customer Rankings
SELECT
    customer_id, customer_name, country, total_spent,
    RANK() OVER (ORDER BY total_spent DESC) AS overall_rank,
    RANK() OVER (PARTITION BY country ORDER BY total_spent DESC) AS country_rank,
    ROUND(SUM(total_spent) OVER (ORDER BY total_spent DESC
          ROWS UNBOUNDED PRECEDING), 2) AS cumulative_revenue,
    ROUND(total_spent * 100.0 / SUM(total_spent) OVER (), 2) AS pct_of_total
FROM customer_summary
ORDER BY overall_rank;
```

**Queries cover:** JOINs · GROUP BY · Subqueries · HAVING · Window Functions (RANK, LAG, SUM OVER) · CTEs · CASE WHEN · Date Functions

## 🤖 Machine Learning Pipeline

```
Raw Data → Feature Engineering (9 features) → StandardScaler
    → Train/Test Split (80/20, stratified)
    → Random Forest + Gradient Boosting
    → Ensemble Prediction → Risk Classification
    → Business Report (Excel)
```

**Features used:**
- `recency` — days since last purchase
- `frequency` — total purchase count
- `monetary` — total spending
- `avg_order_value` — average order amount
- `customer_age_days` — tenure since signup
- `clv_score` — annualized customer lifetime value
- `engagement_score` — interaction rate per tenure
- `total_interactions` — total engagement count
- `purchase_frequency_monthly` — purchases per month

## 📋 Business Recommendations

| Priority | Action | Target | Expected Impact |
|----------|--------|--------|-----------------|
| 🔴 High | Win-back email campaigns | High Risk | Recover 10-15% at-risk revenue |
| 🔴 High | Personalized discount offers | At Risk | Reduce churn by 20% |
| 🟡 Medium | VIP loyalty program | VIP | Maintain 95%+ retention |
| 🟡 Medium | 60-day inactivity alerts | All | Early churn detection |
| 🟢 Low | A/B test discount levels | At Risk | Optimize retention ROI |

## 🔮 Future Improvements

- [ ] Add deep learning model (LSTM for time-series churn prediction)
- [ ] Build interactive Streamlit/Dash dashboard
- [ ] Integrate real-world dataset (Kaggle/UCI)
- [ ] Add customer cohort analysis
- [ ] Deploy model as REST API with Flask/FastAPI

---

**Built as a Data Analytics portfolio project** | Python · SQL · Excel · Machine Learning
