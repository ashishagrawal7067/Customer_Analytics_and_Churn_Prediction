"""
Data Generation Script — Customer Analytics Project
=====================================================
Generates synthetic but realistic customer, order, and interaction data
for the analytics pipeline. All dates are consistent with REFERENCE_DATE.

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
NUM_CUSTOMERS = 500
RANDOM_SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'data'

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Realistic Name Pools
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

# ============================================================
# Generate Customers
# ============================================================
print("=" * 60)
print("CUSTOMER ANALYTICS — DATA GENERATION")
print("=" * 60)
print(f"\nReference Date: {REFERENCE_DATE.strftime('%Y-%m-%d')}")
print(f"Target Customers: {NUM_CUSTOMERS}\n")

countries = ['USA', 'UK', 'Canada', 'Australia', 'Germany',
             'France', 'India', 'Japan', 'Mexico', 'Brazil']
genders = ['M', 'F']

# Pre-assign churn flags (~22% churn rate)
churn_flags = np.random.choice([0, 1], size=NUM_CUSTOMERS, p=[0.78, 0.22])

customers_data = []
for i, customer_id in enumerate(range(1001, 1001 + NUM_CUSTOMERS)):
    # Signup between 91 and 912 days before reference (ensures 3+ months tenure)
    signup_date = REFERENCE_DATE - timedelta(days=random.randint(91, 912))

    customers_data.append({
        'customer_id': customer_id,
        'customer_name': f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
        'email': f"customer{customer_id}@email.com",
        'signup_date': signup_date.date(),
        'country': random.choice(countries),
        'age': random.randint(18, 65),
        'gender': random.choice(genders)
    })

customers_df = pd.DataFrame(customers_data)
customers_df.to_csv(OUTPUT_DIR / 'customers.csv', index=False)
print(f"✓ Created customers.csv ({len(customers_df)} records)")

# ============================================================
# Generate Orders
# ============================================================
products = ['Electronics', 'Fashion', 'Home & Garden', 'Books', 'Sports',
            'Beauty', 'Toys', 'Food', 'Furniture', 'Technology']
payment_methods = ['Credit Card', 'Debit Card', 'PayPal', 'Apple Pay', 'Google Pay']
order_statuses = ['Completed', 'Pending', 'Cancelled', 'Returned']

orders_data = []
order_id = 10001

for i, customer_id in enumerate(range(1001, 1001 + NUM_CUSTOMERS)):
    is_churned = churn_flags[i]
    customer_signup = pd.to_datetime(
        customers_df[customers_df['customer_id'] == customer_id]['signup_date'].values[0]
    )
    days_since_signup = (REFERENCE_DATE - customer_signup.to_pydatetime()).days

    if is_churned:
        # Churned: fewer orders, last order > 91 days before reference
        num_orders = random.randint(0, 5)
        latest_allowed = REFERENCE_DATE - timedelta(days=91)
    else:
        # Active: more orders, recent activity
        num_orders = random.randint(3, 15)
        latest_allowed = REFERENCE_DATE

    for j in range(num_orders):
        if is_churned:
            max_days = max(1, (latest_allowed - customer_signup.to_pydatetime()).days)
            days_after_signup = random.randint(0, max_days)
        else:
            if j == 0:
                # Guarantee at least one order in last 90 days for active customers
                recent_start = max(0, days_since_signup - 90)
                days_after_signup = random.randint(recent_start, days_since_signup)
            else:
                days_after_signup = random.randint(0, days_since_signup)

        order_date = customer_signup + timedelta(days=days_after_signup)

        orders_data.append({
            'order_id': order_id,
            'customer_id': customer_id,
            'order_date': order_date.date(),
            'order_amount': round(random.uniform(50, 1000), 2),
            'product_category': random.choice(products),
            'payment_method': random.choice(payment_methods),
            'order_status': random.choices(order_statuses, weights=[80, 10, 5, 5], k=1)[0]
        })
        order_id += 1

orders_df = pd.DataFrame(orders_data)
orders_df.to_csv(OUTPUT_DIR / 'orders.csv', index=False)
print(f"✓ Created orders.csv ({len(orders_df)} records)")

# ============================================================
# Generate Interactions
# ============================================================
interaction_types = ['website_visit', 'email_open', 'support_ticket', 'phone_call', 'chat']
channels = ['Website', 'Email', 'Support', 'Phone', 'Chat']

interactions_data = []
interaction_id = 5001

for i, customer_id in enumerate(range(1001, 1001 + NUM_CUSTOMERS)):
    is_churned = churn_flags[i]
    customer_signup = pd.to_datetime(
        customers_df[customers_df['customer_id'] == customer_id]['signup_date'].values[0]
    )
    days_since_signup = (REFERENCE_DATE - customer_signup.to_pydatetime()).days

    if is_churned:
        num_interactions = random.randint(0, 4)
    else:
        num_interactions = random.randint(2, 12)

    for _ in range(num_interactions):
        days_after_signup = random.randint(0, days_since_signup)
        interaction_date = customer_signup + timedelta(days=days_after_signup)

        interactions_data.append({
            'interaction_id': interaction_id,
            'customer_id': customer_id,
            'interaction_date': interaction_date.date(),
            'interaction_type': random.choice(interaction_types),
            'duration_minutes': random.randint(1, 120),
            'channel': random.choice(channels)
        })
        interaction_id += 1

interactions_df = pd.DataFrame(interactions_data)
interactions_df.to_csv(OUTPUT_DIR / 'customer_interactions.csv', index=False)
print(f"✓ Created customer_interactions.csv ({len(interactions_df)} records)")

# ============================================================
# Summary
# ============================================================
actual_churn = churn_flags.sum()
print(f"\n{'=' * 60}")
print("✅ ALL DATASETS CREATED SUCCESSFULLY!")
print(f"{'=' * 60}")
print(f"   Customers:    {len(customers_df):,}")
print(f"   Orders:       {len(orders_df):,}")
print(f"   Interactions: {len(interactions_df):,}")
print(f"   Churn Rate:   {actual_churn}/{NUM_CUSTOMERS} ({actual_churn/NUM_CUSTOMERS*100:.1f}%)")
print(f"\n   Output Dir:   {OUTPUT_DIR}")