# Marketing Mix Modeling (MMM) Pipeline

This project implements a Marketing Mix Modeling (MMM) pipeline in Python to analyze the impact of multiple marketing channels on revenue.

---

## Pipeline Stages

### **Stage 1: Ridge Regression**
- **Purpose:**  
    Predict Google Ads spend based on social media spend features.
- **Library:**  
    `sklearn.linear_model.Ridge`
- **Hyperparameters:**  
    - `alpha=1.0` (controls regularization strength)
- **Preprocessing:**  
    - `RobustScaler` applied before model training for outlier resistance.

---

### **Stage 2: ElasticNet Regression**
- **Purpose:**  
    Predict log-transformed revenue using direct marketing features, predicted Google spend, and social features.
- **Library:**  
    `sklearn.linear_model.ElasticNet`
- **Hyperparameters:**  
    - `alpha=0.1` (regularization strength)  
    - `l1_ratio=0.5` (balance between L1 and L2 penalties)  
    - `random_state=42` (for reproducibility)
- **Preprocessing:**  
    - `StandardScaler` applied before model training for feature scaling.

---

## How to Run

1. **Install dependencies:**
        ```
        pip install pandas numpy scikit-learn
        ```

2. **Run the pipeline script:**
        ```
        python model.py
        ```

3. **Use the example prediction function:**
        ```
        python predict.py
        ```

---