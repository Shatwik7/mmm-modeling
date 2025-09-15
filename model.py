import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

warnings.filterwarnings('ignore')



# Load and Preprocess Data

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["week"] = pd.to_datetime(df["week"])
    df = df.sort_values("week").reset_index(drop=True)
    return df



#  Feature Engineering

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # Time features
    df["year"] = df["week"].dt.year
    df["quarter"] = df["week"].dt.quarter
    df["month"] = df["week"].dt.month
    df["week_of_year"] = df["week"].dt.isocalendar().week
    df["days_since_start"] = (df["week"] - df["week"].min()).dt.days

    # Spend logs and active indicators
    spend_cols = ["facebook_spend", "google_spend", "tiktok_spend", "instagram_spend", "snapchat_spend"]
    for col in spend_cols:
        df[f"{col}_active"] = (df[col] > 0).astype(int)
        df[f"{col}_log"] = np.log1p(df[col])

    # Total social media
    df["total_social_spend"] = df[["facebook_spend","tiktok_spend","instagram_spend","snapchat_spend"]].sum(axis=1)
    df["total_social_spend_log"] = np.log1p(df["total_social_spend"])

    # Adstock transformation
    adstock_rate = 0.5
    for col in spend_cols:
        adstock_col = f"{col}_adstock"
        df[adstock_col] = 0.0
        for i in range(len(df)):
            df.loc[i, adstock_col] = df.loc[i, col] if i == 0 else df.loc[i, col] + adstock_rate * df.loc[i-1, adstock_col]

    # Lags and moving averages
    for col in ["facebook_spend", "total_social_spend", "emails_send", "sms_send"]:
        df[f"{col}_lag1"] = df[col].shift(1).fillna(0)
        df[f"{col}_lag2"] = df[col].shift(2).fillna(0)
    for col in ["facebook_spend", "total_social_spend", "average_price"]:
        df[f"{col}_ma4"] = df[col].rolling(window=4, min_periods=1).mean()

    # Price elasticity
    avg_price = df["average_price"].mean()
    df["price_deviation"] = df["average_price"] - avg_price
    df["price_log"] = np.log(df["average_price"])

    # Followers growth
    df["followers_growth"] = df["social_followers"].diff().fillna(0)

    # Fourier (sin/cos cyclical features)
    df["sin_week"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["cos_week"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

    # Revenue cleaning
    df["revenue_clean"] = df["revenue"].replace(1.0, np.nan)
    df["revenue_log"] = np.log1p(df["revenue_clean"])
    return df



# repare Inputs

def prepare_inputs(df: pd.DataFrame):
    social_features = [
        'facebook_spend_log','tiktok_spend_log','instagram_spend_log','snapchat_spend_log',
        'facebook_adstock','tiktok_adstock','instagram_adstock','snapchat_adstock',
        'total_social_spend_log','facebook_active','tiktok_active','instagram_active','snapchat_active'
    ]
    direct_features = [
        'google_spend_log','google_adstock','google_active','emails_send','sms_send',
        'average_price','price_log','price_deviation', 'promotions','social_followers','followers_growth'
    ]
    time_features = ['days_since_start','sin_week','cos_week','sin_month','cos_month','quarter','year']
    lag_features = [col for col in df.columns if "_lag" in col or "_ma4" in col]

    all_features = social_features + direct_features + time_features + lag_features
    df_model = df.dropna(subset=["revenue_log"])

    X = df_model[[f for f in all_features if f in df_model.columns]].copy()
    y = df_model["revenue_log"]
    return X, y, social_features, direct_features, time_features



# Two-Stage Modeling

def train_models(X, y, social_features, direct_features, time_features):
    # Stage 1: Predict Google spend
    X_google = X[[f for f in social_features if f in X.columns] + [f for f in time_features if f in X.columns]]
    y_google = X['google_spend_log']
    scaler1 = RobustScaler()
    X_google_scaled = scaler1.fit_transform(X_google)
    google_model = Ridge(alpha=1.0).fit(X_google_scaled, y_google)
    google_pred = google_model.predict(X_google_scaled)

    # Stage 2: Predict revenue
    X_revenue = X[[f for f in direct_features if f in X.columns] + [f for f in time_features if f in X.columns] + [f for f in social_features if f in X.columns]].copy()
    X_revenue["google_spend_predicted"] = google_pred

    scaler2 = StandardScaler()
    X_revenue_scaled = scaler2.fit_transform(X_revenue)
    revenue_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42).fit(X_revenue_scaled, y)

    return google_model, scaler1, revenue_model, scaler2, X_revenue



# Cross-Validation

def evaluate_model(revenue_model, scaler2, X_revenue, y):
    tscv = TimeSeriesSplit(n_splits=5, test_size=12)
    rmse_list, mae_list, mape_list, r2_list = [], [], [], []

    for train_idx, test_idx in tscv.split(X_revenue):
        X_train, X_test = X_revenue.iloc[train_idx], X_revenue.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        X_train_scaled = scaler2.fit_transform(X_train)
        X_test_scaled = scaler2.transform(X_test)

        revenue_model.fit(X_train_scaled, y_train)
        y_pred = revenue_model.predict(X_test_scaled)

        rmse_list.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae_list.append(mean_absolute_error(y_test, y_pred))
        mape_list.append(mean_absolute_percentage_error(y_test, y_pred))
        r2_list.append(r2_score(y_test, y_pred))

    print("\nPerformance (5-fold time series CV):")
    print(f"RMSE: {np.mean(rmse_list):.3f} ± {np.std(rmse_list):.3f}")
    print(f"MAE: {np.mean(mae_list):.3f} ± {np.std(mae_list):.3f}")
    print(f"MAPE: {np.mean(mape_list):.3f} ± {np.std(mape_list):.3f}")
    print(f"R2: {np.mean(r2_list):.3f} ± {np.std(r2_list):.3f}")



# Saving Artifacts

def save_artifacts(google_model, scaler1, revenue_model, scaler2, X_revenue, df):
    os.makedirs("mmm_results", exist_ok=True)

    with open('mmm_results/mmm_models.pkl', 'wb') as f:
        pickle.dump({
            'google_model': google_model,
            'scaler1': scaler1,
            'revenue_model': revenue_model,
            'scaler2': scaler2,
            'features': X_revenue.columns.tolist()
        }, f)

    df.to_csv("mmm_results/processed_data.csv", index=False)
    print("\nArtifacts saved in mmm_results/: models, processed data.")



# Pipeline Runner

def run_pipeline(data_path="Assessment-2-MMM-Weekly.csv"):
    df = load_data(data_path)
    df = add_features(df)
    X, y, social_features, direct_features, time_features = prepare_inputs(df)

    google_model, scaler1, revenue_model, scaler2, X_revenue = train_models(X, y, social_features, direct_features, time_features)

    evaluate_model(revenue_model, scaler2, X_revenue, y)
    save_artifacts(google_model, scaler1, revenue_model, scaler2, X_revenue, df)
    


if __name__ == "__main__":
    run_pipeline()
