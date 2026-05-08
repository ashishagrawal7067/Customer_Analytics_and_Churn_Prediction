"""
Data Generation Script — Customer Analytics Project
=====================================================
Generates realistic synthetic customer, order, and interaction data.

KEY DESIGN PRINCIPLES (v2 — fixed):
  1. churn_flag is pre-assigned and SAVED to customers.csv (ground truth label)
  2. Order amounts follow log-normal distribution (realistic e-commerce)
  3. Classes are NOT perfectly separated — ~20% of churned customers
     have some recent activity (noise), ~10% of active customers
     have gaps in their purchase history (noise).
  4. Performance window = last 90 days (PERF_WINDOW_START to REFERENCE_DATE)
     Observation window = everything BEFORE PERF_WINDOW_START
     Features in notebook are built ONLY from observation window.

Usage: python scripts/generate_data.py
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from pathlib import Path

# ============================================================
# Configuration
# ============================================================
REFERENCE_DATE = datetime(2025, 6, 30)
PERF_WINDOW_DAYS = 90                       # performance window length
PERF_WINDOW_START = REFERENCE_DATE - timedelta(days=PERF_WINDOW_DAYS)
NUM_CUSTOMERS = 600
CHURN_RATE = 0.28                           # 28% base churn rate
NOISE_CHURNED_ACTIVE = 0.18                 # 18% of churned customers have >= 1 recent order
NOISE_ACTIVE_SILENT = 0.10                  # 10% of active customers have no recent purchase
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'data'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Name Pools
# ============================================================
FIRST_NAMES = [
    'Aarav', 'Aditi', 'Aisha', 'Amit', 'Ananya', 'Arjun', 'Deepa', 'Dev',
    'Diya', 'Gaurav', 'Isha', 'Kabir', 'Kavya', 'Kiran', 'Meera', 'Neha',
    'Nikhil', 'Pooja', 'Priya', 'Rahul', 'Riya', 'Rohit', 'Sakshi', 'Sanjay',
    'Sara', 'Shreya', 'Siddharth', 'Sneha', 'Tanvi', 'Varun', 'Vikram', 'Zara',
    'James', 'Emma', 'Liam', 'Olivia', 'Noah', 'Sophia', 'Ethan', 'Isabella',
    'Mason', 'Mia', 'Lucas', 'Charlotte', 'Oliver', 'Amelia', 'Aiden', 'Harper',
    'Carlos', 'Maria', 'Hans', 'Yuki', 'Hiroshi', 'Sakura', 'Pierre', 'Marie',
    'Lars', 'Anna', 'Diego', 'Lucia', 'Chen', 'Wei', 'Fatima', 'Omar'
]
LAST_NAMES = [
    'Sharma', 'Patel', 'Singh', 'Kumar', 'Gupta', 'Reddy', 'Verma', 'Joshi',
    'Mishra', 'Das', 'Mehta', 'Kapoor', 'Nair', 'Rao', 'Iyer', 'Malhotra',
    'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Davis', 'Miller', 'Wilson',
    'Garcia', 'Martinez', 'Lopez', 'Hernandez', 'Gonzalez', 'Rodriguez',
    'Mueller', 'Schmidt', 'Schneider', 'Fischer', 'Weber', 'Tanaka', 'Sato',
    'Watanabe', 'Dubois', 'Martin', 'Bernard', 'Wang', 'Li', 'Zhang', 'Chen'
]

COUNTRIES = ['USA', 'UK', 'Canada', 'Australia', 'Germany',
             'France', 'India', 'Japan', 'Mexico', 'Brazil']
PRODUCTS = ['Electronics', 'Fashion', 'Home & Garden', 'Books', 'Sports',
            'Beauty', 'Toys', 'Food', 'Furniture', 'Technology']
PAYMENTS = ['Credit Card', 'Debit Card', 'PayPal', 'Apple Pay', 'Google Pay']
STATUSES = ['Completed', 'Pending', 'Cancelled', 'Returned']
INTERACTIONS = ['website_visit', 'email_open', 'support_ticket', 'phone_call', 'chat']
CHANNELS = ['Website', 'Email', 'Support', 'Phone', 'Chat']


def log_normal_amount(mean=200, sigma=0.8, min_val=15.0, max_val=2500.0):
    """
    Sample order amount from log-normal distribution.
    Real e-commerce: many small orders, few large ones.
    mean=200 gives realistic median ~$130.
    """
    mu = np.log(mean) - (sigma ** 2) / 2
    amount = np.random.lognormal(mean=mu, sigma=sigma)
    return round(float(np.clip(amount, min_val, max_val)), 2)


# ============================================================
# Step 1: Generate Customers with pre-assigned churn flags
# ============================================================
print("=" * 65)
print("CUSTOMER ANALYTICS — DATA GENERATION (v2, noise-aware)")
print("=" * 65)
print(f"\n  Reference Date:        {REFERENCE_DATE.strftime('%Y-%m-%d')}")
print(f"  Performance Window:    {PERF_WINDOW_START.strftime('%Y-%m-%d')} "
      f"→ {REFERENCE_DATE.strftime('%Y-%m-%d')}")
print(f"  Target Customers:      {NUM_CUSTOMERS}")
print(f"  Base Churn Rate:       {CHURN_RATE*100:.0f}%")
print(f"  Noise (churn→active):  {NOISE_CHURNED_ACTIVE*100:.0f}% of churned have recent orders")
print(f"  Noise (active→silent): {NOISE_ACTIVE_SILENT*100:.0f}% of active have no recent orders\n")

# Pre-assign ground truth churn flag
churn_flags = np.random.choice(
    [0, 1], size=NUM_CUSTOMERS, p=[1 - CHURN_RATE, CHURN_RATE]
)

customers_data = []
for i, cid in enumerate(range(1001, 1001 + NUM_CUSTOMERS)):
    # Ensure at least 150 days of history (need enough observation window data)
    signup_date = REFERENCE_DATE - timedelta(days=random.randint(150, 1095))
    customers_data.append({
        'customer_id': cid,
        'customer_name': f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
        'email': f"customer{cid}@email.com",
        'signup_date': signup_date.date(),
        'country': random.choice(COUNTRIES),
        'age': random.randint(18, 68),
        'gender': random.choice(['M', 'F']),
        'churn_flag': int(churn_flags[i])   # ← SAVED: ground truth label, independent of recency
    })

customers_df = pd.DataFrame(customers_data)
customers_df.to_csv(OUTPUT_DIR / 'customers.csv', index=False)
print(f"✓ customers.csv  ({len(customers_df)} records, churn_flag included)")

# ============================================================
# Step 2: Generate Orders
# ============================================================
orders_data = []
order_id = 10001

for i, row in customers_df.iterrows():
    cid = row['customer_id']
    is_churned = row['churn_flag']
    signup = pd.Timestamp(row['signup_date'])
    days_tenure = (REFERENCE_DATE - signup.to_pydatetime()).days
    obs_days = (PERF_WINDOW_START - signup.to_pydatetime()).days  # observation window length

    if obs_days < 1:
        obs_days = 1  # safety

    # ── Observation window orders (before PERF_WINDOW_START) ──
    # Both churned and active customers have activity in the observation window.
    # Churned: tend to have FEWER and older orders (but not zero — adds noise)
    # Active:  tend to have MORE and more-spread orders
    if is_churned:
        num_obs_orders = random.randint(1, 7)   # churned customers had some history
    else:
        num_obs_orders = random.randint(3, 14)  # active customers are more engaged

    for _ in range(num_obs_orders):
        days_after_signup = random.randint(0, obs_days - 1)
        order_date = signup + timedelta(days=days_after_signup)
        # Keep order date strictly in observation window
        if order_date >= PERF_WINDOW_START:
            order_date = PERF_WINDOW_START - timedelta(days=random.randint(1, 30))
        orders_data.append({
            'order_id': order_id,
            'customer_id': cid,
            'order_date': order_date.date(),
            'order_amount': log_normal_amount(),
            'product_category': random.choice(PRODUCTS),
            'payment_method': random.choice(PAYMENTS),
            'order_status': random.choices(STATUSES, weights=[78, 10, 7, 5], k=1)[0]
        })
        order_id += 1

    # ── Performance window orders (last 90 days) ──
    # This determines actual churn label (did they buy recently?)
    # We add NOISE so it doesn't perfectly match churn_flag:

    if is_churned:
        # Most churned customers: NO orders in performance window
        # But ~18% (noise) have 1 recent order — they look active but we know they'll churn
        if random.random() < NOISE_CHURNED_ACTIVE:
            num_perf_orders = 1
        else:
            num_perf_orders = 0
    else:
        # Most active customers: have orders in performance window
        # But ~10% (noise) have no recent orders — they look churned but are actually active
        if random.random() < NOISE_ACTIVE_SILENT:
            num_perf_orders = 0
        else:
            num_perf_orders = random.randint(1, 6)

    for _ in range(num_perf_orders):
        days_in_perf = random.randint(0, PERF_WINDOW_DAYS - 1)
        order_date = PERF_WINDOW_START + timedelta(days=days_in_perf)
        orders_data.append({
            'order_id': order_id,
            'customer_id': cid,
            'order_date': order_date.date(),
            'order_amount': log_normal_amount(),
            'product_category': random.choice(PRODUCTS),
            'payment_method': random.choice(PAYMENTS),
            'order_status': random.choices(STATUSES, weights=[78, 10, 7, 5], k=1)[0]
        })
        order_id += 1

orders_df = pd.DataFrame(orders_data)
orders_df.to_csv(OUTPUT_DIR / 'orders.csv', index=False)
print(f"✓ orders.csv     ({len(orders_df)} records, log-normal amounts)")

# ============================================================
# Step 3: Generate Interactions (observation window only)
# ============================================================
interactions_data = []
interaction_id = 5001

for i, row in customers_df.iterrows():
    cid = row['customer_id']
    is_churned = row['churn_flag']
    signup = pd.Timestamp(row['signup_date'])
    obs_days = max(1, (PERF_WINDOW_START - signup.to_pydatetime()).days)

    # Interactions are also noisier: churned customers can still have
    # support tickets or website visits even before churning
    if is_churned:
        num_interactions = random.randint(0, 6)
    else:
        num_interactions = random.randint(1, 14)

    for _ in range(num_interactions):
        # Place interactions in observation window only
        days_after_signup = random.randint(0, obs_days - 1)
        int_date = signup + timedelta(days=days_after_signup)
        interactions_data.append({
            'interaction_id': interaction_id,
            'customer_id': cid,
            'interaction_date': int_date.date(),
            'interaction_type': random.choice(INTERACTIONS),
            'duration_minutes': random.randint(1, 90),
            'channel': random.choice(CHANNELS)
        })
        interaction_id += 1

interactions_df = pd.DataFrame(interactions_data)
interactions_df.to_csv(OUTPUT_DIR / 'customer_interactions.csv', index=False)
print(f"✓ customer_interactions.csv ({len(interactions_df)} records)")

# ============================================================
# Summary
# ============================================================
actual_churn = churn_flags.sum()
print(f"\n{'=' * 65}")
print("DATASETS CREATED SUCCESSFULLY")
print(f"{'=' * 65}")
print(f"  Customers:    {len(customers_df):,}")
print(f"  Orders:       {len(orders_df):,}")
print(f"  Interactions: {len(interactions_df):,}")
print(f"  Churn Rate:   {actual_churn}/{NUM_CUSTOMERS} ({actual_churn/NUM_CUSTOMERS*100:.1f}%)")
print(f"\n  Output Dir:   {OUTPUT_DIR}")
print(f"\n  NOTE: churn_flag in customers.csv is the GROUND TRUTH label.")
print(f"  It is independent of recency — no data leakage.")
print(f"  Performance window: {PERF_WINDOW_START.date()} → {REFERENCE_DATE.date()}")
