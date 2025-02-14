"""
churn_data_generation.py

Generates a synthetic user dataset with funnel events, subscription activity,
and a 'churned' label. Saves the dataset to 'churn_dataset.csv'.
"""

import numpy as np
import pandas as pd
import random
import datetime

def generate_churn_data(num_users=5000, start_date="2023-01-01", end_date="2023-06-30"):
    """
    Creates a synthetic dataset modeling user journeys with churn.
    - Each user has random features: sessions, total_spend, funnel_completion, ...
    - 'churned' label determined by some random logic.
    """
    np.random.seed(42)
    random.seed(42)

    start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end_dt - start_dt).days + 1

    user_ids = range(10001, 10001 + num_users)

    data = {
        "user_id": [],
        "signup_date": [],
        "last_active_date": [],
        "num_sessions": [],
        "avg_session_length": [],
        "funnel_completion_pct": [],   # from 0 to 100
        "total_spend": [],
        "device_preference": [],
        "primary_channel": [],
        "churned": [],
    }

    devices = ["Mobile", "Desktop", "Tablet"]
    channels = ["Website", "Mobile App", "Email", "Social Media"]

    for user in user_ids:
        # Random sign-up date
        signup_offset = random.randint(0, total_days - 30)  # at least 30 days of usage
        signup_dt = start_dt + datetime.timedelta(days=signup_offset)

        # Last active date between signup_dt and end_date
        days_active = random.randint(0, (end_dt - signup_dt).days)
        last_active_dt = signup_dt + datetime.timedelta(days=days_active)

        num_sess = np.random.randint(1, 50)
        avg_sess_len = round(np.random.uniform(3.0, 15.0), 2)  # minutes
        funnel_pct = round(np.random.uniform(0, 100), 2)
        spend = round(np.random.exponential(100), 2)  # random spend
        dev_pref = random.choice(devices)
        channel_pref = random.choice(channels)

        # Heuristic for churn label:
        # Let's say user is "churned" if last_active_date < end_dt - 14 days,
        # or if funnel_completion < 30 and spend < 50, etc.
        # We'll combine multiple factors with random weighting to label churn.
        inactivity_limit = end_dt - datetime.timedelta(days=14)
        churn_probability = 0.0

        # If user hasn't been active recently => higher churn probability
        if last_active_dt < inactivity_limit:
            churn_probability += 0.5
        # Low funnel completion => more likely to churn
        if funnel_pct < 30:
            churn_probability += 0.2
        # Low spending => more likely to churn
        if spend < 50:
            churn_probability += 0.2

        # Add some random noise
        churn_probability += random.uniform(-0.1, 0.1)
        churn_probability = max(0.0, min(1.0, churn_probability))

        churn_label = 1 if random.random() < churn_probability else 0

        data["user_id"].append(user)
        data["signup_date"].append(signup_dt.strftime("%Y-%m-%d"))
        data["last_active_date"].append(last_active_dt.strftime("%Y-%m-%d"))
        data["num_sessions"].append(num_sess)
        data["avg_session_length"].append(avg_sess_len)
        data["funnel_completion_pct"].append(funnel_pct)
        data["total_spend"].append(spend)
        data["device_preference"].append(dev_pref)
        data["primary_channel"].append(channel_pref)
        data["churned"].append(churn_label)

    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    df_churn = generate_churn_data(num_users=5000)
    df_churn.to_csv("churn_dataset.csv", index=False)
    print("Generated churn_dataset.csv with", df_churn.shape[0], "rows.")
