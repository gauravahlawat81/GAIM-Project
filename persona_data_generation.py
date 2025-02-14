"""
persona_data_generation.py

Generates a synthetic dataset modeling user behavior for persona clustering.
Saves to 'persona_dataset.csv'.
"""

import numpy as np
import pandas as pd
import random
import datetime

def generate_persona_data(num_users=3000, start_date="2023-01-01", end_date="2023-03-31"):
    """
    Creates a dataset simulating user journeys:
    - pages_visited: total # of pages visited
    - avg_time_on_site: average seconds (or minutes) spent
    - purchase_count: total # of purchases
    - total_spend: total money spent
    - device: "Desktop", "Mobile", "Tablet"
    - primary_channel: "Website", "Email", "Social Media", "Mobile App"
    - funnel_stage_most_often: "Landing", "ProductView", "Cart", "Checkout"
    """
    np.random.seed(42)
    random.seed(42)

    start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end_dt - start_dt).days + 1

    devices = ["Desktop", "Mobile", "Tablet"]
    channels = ["Website", "Email", "Social Media", "Mobile App"]
    funnel_stages = ["Landing", "ProductView", "Cart", "Checkout"]

    data = {
        "user_id": [],
        "signup_date": [],
        "pages_visited": [],
        "avg_time_on_site": [],
        "purchase_count": [],
        "total_spend": [],
        "device": [],
        "primary_channel": [],
        "funnel_stage_most_often": [],
    }

    for user_id in range(10001, 10001 + num_users):
        signup_offset = np.random.randint(0, total_days)
        signup_dt = start_dt + datetime.timedelta(days=int(signup_offset))

        pages = np.random.randint(5, 200)  # total pages visited
        avg_time = round(np.random.uniform(30, 600), 2)  # seconds, or up to 10 min
        purchases = np.random.randint(0, 10)  # total # of purchases
        spend = round((np.random.exponential(100) * purchases), 2)  # random spend scaled by purchase count
        dev = random.choice(devices)
        chan = random.choice(channels)
        fstage = random.choice(funnel_stages)

        data["user_id"].append(user_id)
        data["signup_date"].append(signup_dt.strftime("%Y-%m-%d"))
        data["pages_visited"].append(pages)
        data["avg_time_on_site"].append(avg_time)
        data["purchase_count"].append(purchases)
        data["total_spend"].append(spend)
        data["device"].append(dev)
        data["primary_channel"].append(chan)
        data["funnel_stage_most_often"].append(fstage)

    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    df_persona = generate_persona_data(num_users=3000)
    df_persona.to_csv("persona_dataset.csv", index=False)
    print("Generated persona_dataset.csv with", df_persona.shape[0], "rows.")
