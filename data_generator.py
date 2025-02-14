import numpy as np
import pandas as pd
import random
import datetime

def generate_complex_user_data(num_users=500000, start_date="2023-01-01", end_date="2023-02-28"):
    """
    Generate a large, complex dataset modeling multi-channel user journeys.
    
    :param num_users: number of user sessions or events
    :param start_date: earliest date in the dataset
    :param end_date: latest date in the dataset
    :return: pd.DataFrame
    """
    np.random.seed(42)
    random.seed(42)
    
    # Date range
    start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end_dt - start_dt).days + 1
    
    # Channels, devices, funnel stages, product categories, etc.
    channels = ["Website", "Mobile App", "Social Media", "Email"]
    devices = ["Desktop", "Mobile", "Tablet"]
    funnel_stages = ["Landing Page", "Product View", "Add to Cart", "Checkout", "Payment", "Confirmation"]
    product_categories = ["Electronics", "Fashion", "Home", "Beauty", "Sports", "Toys"]
    
    data = {
        "event_id": [],
        "user_id": [],
        "event_date": [],
        "channel": [],
        "device": [],
        "funnel_stage": [],
        "product_category": [],
        "spent_time_seconds": [],  # how long the user spent on that stage
        "action_outcome": [],      # "progress", "drop_off", or "purchase"
    }
    
    for i in range(num_users):
        # Each row is effectively a user *event*, not necessarily unique user.
        # We'll randomize user_id distribution.
        user_id = random.randint(10000, 99999)
        
        # Random date
        days_offset = random.randint(0, total_days-1)
        event_dt = start_dt + datetime.timedelta(days=days_offset)
        
        # Random pick for channel, device, category
        ch = random.choice(channels)
        dv = random.choice(devices)
        cat = random.choice(product_categories)
        
        # Let's pick a funnel stage
        # We'll randomly assign stages but with weighted progression
        # to mimic real dropoffs
        stage = np.random.choice(funnel_stages, p=[0.35, 0.25, 0.15, 0.10, 0.10, 0.05])
        
        # Time spent
        spent_time = np.random.randint(5, 300)  # 5s to 5min
        
        # Action outcome
        # if stage is early funnel, higher chance of "drop_off"
        # if stage is later funnel, better chance of "purchase"
        if stage in ["Landing Page", "Product View"]:
            outcome_probs = [0.75, 0.20, 0.05]  # mostly progress or drop-off, rarely purchase
        elif stage in ["Add to Cart", "Checkout"]:
            outcome_probs = [0.50, 0.40, 0.10]
        elif stage in ["Payment", "Confirmation"]:
            outcome_probs = [0.30, 0.20, 0.50]  # more purchases happen here
        outcome = np.random.choice(["progress", "drop_off", "purchase"], p=outcome_probs)
        
        data["event_id"].append(i+1)
        data["user_id"].append(user_id)
        data["event_date"].append(event_dt.strftime("%Y-%m-%d"))
        data["channel"].append(ch)
        data["device"].append(dv)
        data["funnel_stage"].append(stage)
        data["product_category"].append(cat)
        data["spent_time_seconds"].append(spent_time)
        data["action_outcome"].append(outcome)
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    df_complex = generate_complex_user_data(num_users=50000)  # large dataset
    # Save to CSV
    df_complex.to_csv("complex_user_data.csv", index=False)
    print("Data generation complete! 'complex_user_data.csv' created.")
