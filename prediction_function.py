import pickle
import numpy as np

def predict_revenue(
    facebook_spend=0,
    google_spend=0,
    tiktok_spend=0,
    instagram_spend=0,
    snapchat_spend=0,
    average_price=100,
    emails_send=120000,
    sms_send=25000,
    promotions=0,
    social_followers=80000
):
    # Load trained models and scalers
    with open("mmm_results/mmm_models.pkl", "rb") as fmodel:
        objs = pickle.load(fmodel)
    
    # Feature engineering for provided input
    features = {
        "facebook_spend_log": np.log1p(facebook_spend),
        "google_spend_log": np.log1p(google_spend),
        "tiktok_spend_log": np.log1p(tiktok_spend),
        "instagram_spend_log": np.log1p(instagram_spend),
        "snapchat_spend_log": np.log1p(snapchat_spend),
        "facebook_active": 1 if facebook_spend > 0 else 0,
        "google_active": 1 if google_spend > 0 else 0,
        "tiktok_active": 1 if tiktok_spend > 0 else 0,
        "instagram_active": 1 if instagram_spend > 0 else 0,
        "snapchat_active": 1 if snapchat_spend > 0 else 0,
        "total_social_spend_log": np.log1p(
            facebook_spend + tiktok_spend + instagram_spend + snapchat_spend
        ),
        "average_price": average_price,
        "price_log": np.log(average_price),
        "price_deviation": average_price - 99.96,  # adjust baseline as needed
        "emails_send": emails_send,
        "sms_send": sms_send,
        "promotions": promotions,
        "social_followers": social_followers,
        "followers_growth": 0,     # Default, or set based on user input/context
        "days_since_start": 700,   # Default, or estimate actual week offset
        "sin_week": 0,             # Default seasonality, can be customized per week
        "cos_week": 0,
        "sin_month": 0,
        "cos_month": 0,
        "quarter": 4,              # Default, customize as needed
        "year": 2025               # Default, customize as needed
        # Add lag and moving average features if you want, else defaults to 0
    }

    # Ensure features are ordered
    input_order = objs["features"]
    values = [features.get(key, 0) for key in input_order]
    X = np.array(values).reshape(1, -1)

    # Scale and predict
    X_scaled = objs["scaler2"].transform(X)
    log_rev = objs["revenue_model"].predict(X_scaled)[0]
    return np.exp(log_rev) - 1  # Inverse transform log1p

