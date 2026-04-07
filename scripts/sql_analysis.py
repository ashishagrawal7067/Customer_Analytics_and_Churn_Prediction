"""
SQL Analysis Script — Customer Analytics Project
===================================================
Demonstrates SQL proficiency using SQLite on the customer analytics dataset.
Covers: JOINs, GROUP BY, Subqueries, Window Functions, CTEs, CASE WHEN

Usage: python scripts/sql_analysis.py
"""

import sqlite3
import pandas as pd
from pathlib import Path

# ============================================================
# Setup: Load CSVs into SQLite Database
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'outputs'
DB_PATH = DATA_DIR / 'customer_analytics.db'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("SQL ANALYSIS — CUSTOMER ANALYTICS")
print("=" * 70)

# Create / connect to SQLite database
conn = sqlite3.connect(str(DB_PATH))

# Load CSVs into database tables
print("\n📥 Loading data into SQLite database...")
for csv_name, table_name in [
    ('customers.csv', 'customers'),
    ('orders.csv', 'orders'),
    ('customer_interactions.csv', 'interactions')
]:
    df = pd.read_csv(DATA_DIR / csv_name)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    print(f"   ✓ Loaded {table_name} ({len(df)} rows)")

print(f"\n   Database: {DB_PATH}")


# ============================================================
# Helper: Run & Display a Query
# ============================================================
def run_query(title, query, conn):
    """Execute a SQL query and display results."""
    print(f"\n{'─' * 70}")
    print(f"📊 {title}")
    print(f"{'─' * 70}")
    print(f"\nSQL:\n{query.strip()}\n")
    result = pd.read_sql_query(query, conn)
    print(f"Results ({len(result)} rows):")
    print(result.to_string(index=False))
    print()
    return result


results = {}

# ============================================================
# Query 1: JOIN + GROUP BY — Customer Purchase Summary
# ============================================================
results['customer_summary'] = run_query(
    "Query 1: Customer Purchase Summary (JOIN + GROUP BY)",
    """
    SELECT
        c.customer_id,
        c.customer_name,
        c.country,
        COUNT(o.order_id)                       AS total_orders,
        COALESCE(ROUND(SUM(o.order_amount), 2), 0) AS total_spent,
        COALESCE(ROUND(AVG(o.order_amount), 2), 0) AS avg_order_value,
        MAX(o.order_date)                       AS last_order_date
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.customer_id, c.customer_name, c.country
    ORDER BY total_spent DESC
    LIMIT 15;
    """,
    conn
)

# ============================================================
# Query 2: GROUP BY + Correlated Subquery — Revenue by Country
# ============================================================
results['revenue_by_country'] = run_query(
    "Query 2: Revenue by Country & Top Product (GROUP BY + Subquery)",
    """
    SELECT
        c.country,
        COUNT(DISTINCT c.customer_id) AS num_customers,
        COUNT(o.order_id)             AS total_orders,
        ROUND(SUM(o.order_amount), 2) AS total_revenue,
        ROUND(AVG(o.order_amount), 2) AS avg_order_value,
        (
            SELECT o2.product_category
            FROM orders o2
            JOIN customers c2 ON o2.customer_id = c2.customer_id
            WHERE c2.country = c.country
            GROUP BY o2.product_category
            ORDER BY SUM(o2.order_amount) DESC
            LIMIT 1
        ) AS top_category
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.country
    ORDER BY total_revenue DESC;
    """,
    conn
)

# ============================================================
# Query 3: CASE WHEN — Customer Segmentation
# ============================================================
results['spending_segments'] = run_query(
    "Query 3: Customer Segmentation (CASE WHEN + Aggregation)",
    """
    SELECT
        spending_segment,
        COUNT(*)                      AS num_customers,
        ROUND(AVG(total_spent), 2)    AS avg_spending,
        ROUND(AVG(order_count), 1)    AS avg_orders,
        ROUND(MIN(total_spent), 2)    AS min_spent,
        ROUND(MAX(total_spent), 2)    AS max_spent
    FROM (
        SELECT
            c.customer_id,
            COALESCE(SUM(o.order_amount), 0) AS total_spent,
            COUNT(o.order_id)                AS order_count,
            CASE
                WHEN COALESCE(SUM(o.order_amount), 0) > 5000 THEN 'Premium'
                WHEN COALESCE(SUM(o.order_amount), 0) > 2000 THEN 'High Value'
                WHEN COALESCE(SUM(o.order_amount), 0) > 500  THEN 'Medium Value'
                WHEN COALESCE(SUM(o.order_amount), 0) > 0    THEN 'Low Value'
                ELSE 'No Purchase'
            END AS spending_segment
        FROM customers c
        LEFT JOIN orders o ON c.customer_id = o.customer_id
        GROUP BY c.customer_id
    )
    GROUP BY spending_segment
    ORDER BY avg_spending DESC;
    """,
    conn
)

# ============================================================
# Query 4: HAVING + Subquery — Above-Average Spenders
# ============================================================
results['above_avg_spenders'] = run_query(
    "Query 4: Customers Spending Above Average (Subquery + HAVING)",
    """
    SELECT
        c.customer_id,
        c.customer_name,
        c.country,
        c.age,
        ROUND(SUM(o.order_amount), 2) AS total_spent,
        COUNT(o.order_id)             AS num_orders
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.customer_id
    HAVING total_spent > (
        SELECT AVG(customer_total)
        FROM (
            SELECT SUM(order_amount) AS customer_total
            FROM orders
            GROUP BY customer_id
        )
    )
    ORDER BY total_spent DESC
    LIMIT 15;
    """,
    conn
)

# ============================================================
# Query 5: Window Functions — Ranking & Running Totals
# ============================================================
results['rankings'] = run_query(
    "Query 5: Customer Rankings (RANK, SUM OVER, Percent of Total)",
    """
    SELECT
        customer_id,
        customer_name,
        country,
        total_spent,
        RANK() OVER (ORDER BY total_spent DESC) AS overall_rank,
        RANK() OVER (
            PARTITION BY country ORDER BY total_spent DESC
        ) AS country_rank,
        ROUND(
            SUM(total_spent) OVER (
                ORDER BY total_spent DESC
                ROWS UNBOUNDED PRECEDING
            ), 2
        ) AS cumulative_revenue,
        ROUND(
            total_spent * 100.0 / SUM(total_spent) OVER (),
            2
        ) AS pct_of_total
    FROM (
        SELECT
            c.customer_id,
            c.customer_name,
            c.country,
            ROUND(COALESCE(SUM(o.order_amount), 0), 2) AS total_spent
        FROM customers c
        LEFT JOIN orders o ON c.customer_id = o.customer_id
        GROUP BY c.customer_id, c.customer_name, c.country
    )
    WHERE total_spent > 0
    ORDER BY overall_rank
    LIMIT 20;
    """,
    conn
)

# ============================================================
# Query 6: CTE + LAG — Monthly Revenue Trends
# ============================================================
results['monthly_trends'] = run_query(
    "Query 6: Monthly Revenue Trends (CTE + LAG Window Function)",
    """
    WITH monthly_revenue AS (
        SELECT
            SUBSTR(order_date, 1, 7)     AS order_month,
            COUNT(order_id)              AS num_orders,
            ROUND(SUM(order_amount), 2)  AS monthly_revenue,
            COUNT(DISTINCT customer_id)  AS unique_customers
        FROM orders
        WHERE order_status = 'Completed'
        GROUP BY order_month
    )
    SELECT
        order_month,
        num_orders,
        monthly_revenue,
        unique_customers,
        ROUND(monthly_revenue / unique_customers, 2) AS revenue_per_customer,
        ROUND(
            (monthly_revenue - LAG(monthly_revenue) OVER (ORDER BY order_month))
            / LAG(monthly_revenue) OVER (ORDER BY order_month) * 100,
            2
        ) AS mom_growth_pct
    FROM monthly_revenue
    ORDER BY order_month;
    """,
    conn
)

# ============================================================
# Query 7: CTE + CASE WHEN + JOIN — Churn Analysis
# ============================================================
results['churn_analysis'] = run_query(
    "Query 7: Churn Analysis (CTE + CASE WHEN + JULIANDAY)",
    """
    WITH customer_activity AS (
        SELECT
            c.customer_id,
            c.customer_name,
            c.country,
            c.signup_date,
            MAX(o.order_date)                           AS last_order_date,
            COUNT(o.order_id)                           AS total_orders,
            COALESCE(ROUND(SUM(o.order_amount), 2), 0)  AS total_spent,
            JULIANDAY('2025-06-30') - JULIANDAY(MAX(o.order_date))
                                                         AS days_since_last
    FROM customers c
        LEFT JOIN orders o ON c.customer_id = o.customer_id
        GROUP BY c.customer_id
    )
    SELECT
        CASE
            WHEN days_since_last IS NULL   THEN 'Never Purchased'
            WHEN days_since_last > 180     THEN 'Churned (6+ months)'
            WHEN days_since_last > 90      THEN 'At Risk (3-6 months)'
            WHEN days_since_last > 30      THEN 'Cooling (1-3 months)'
            ELSE 'Active (< 1 month)'
        END AS customer_status,
        COUNT(*)                   AS num_customers,
        ROUND(AVG(total_spent), 2) AS avg_lifetime_value,
        ROUND(AVG(total_orders), 1) AS avg_orders
    FROM customer_activity
    GROUP BY customer_status
    ORDER BY num_customers DESC;
    """,
    conn
)

# ============================================================
# Save All Results to Excel
# ============================================================
print(f"\n{'=' * 70}")
print("💾 SAVING RESULTS TO EXCEL")
print(f"{'=' * 70}")

excel_path = OUTPUT_DIR / 'SQL_Analysis_Results.xlsx'
with pd.ExcelWriter(str(excel_path), engine='openpyxl') as writer:
    for sheet_name, df in results.items():
        df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        print(f"   ✓ Saved sheet: {sheet_name}")

print(f"\n   📄 Excel Report: {excel_path}")

# Cleanup
conn.close()

print(f"\n{'=' * 70}")
print("✅ SQL ANALYSIS COMPLETED SUCCESSFULLY!")
print(f"{'=' * 70}")
print("""
Skills Demonstrated:
  ✓ JOINs (LEFT JOIN, INNER JOIN)
  ✓ GROUP BY with aggregation (SUM, AVG, COUNT, MAX)
  ✓ Subqueries (correlated and non-correlated)
  ✓ Window Functions (RANK, SUM OVER, LAG, PARTITION BY)
  ✓ Common Table Expressions (CTEs)
  ✓ CASE WHEN conditional logic
  ✓ Date functions (JULIANDAY, SUBSTR)
  ✓ HAVING clause for filtered aggregation
""")
