from prediction_function import predict_revenue

result = predict_revenue(
    facebook_spend=5000,
    google_spend=10000,
    tiktok_spend=3000,
    instagram_spend=6000,
    snapchat_spend=2000,
    average_price=115,
    emails_send=120000,
    sms_send=22000,
    promotions=2,
    social_followers=85000
)
print("Predicted Revenue:", result)