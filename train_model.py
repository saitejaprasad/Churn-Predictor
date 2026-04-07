"""
train_model.py — Customer Churn Prediction Model
Run once: python train_model.py
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

np.random.seed(42)
N = 10000

print("📦 Generating Telco-style customer churn dataset (10,000 records)...")

tenure          = np.random.randint(1, 73, N)
monthly_charges = np.round(np.random.uniform(18.0, 118.0, N), 2)
total_charges   = np.round(tenure * monthly_charges * np.random.uniform(0.85, 1.0, N), 2)
contract        = np.random.choice(["Month-to-month","One year","Two year"], N, p=[0.55,0.25,0.20])
internet_service= np.random.choice(["DSL","Fiber optic","No"], N, p=[0.34,0.44,0.22])
payment_method  = np.random.choice(["Electronic check","Mailed check","Bank transfer","Credit card"], N, p=[0.34,0.23,0.22,0.21])
gender          = np.random.choice(["Male","Female"], N)
senior_citizen  = np.random.choice([0,1], N, p=[0.84,0.16])
partner         = np.random.choice(["Yes","No"], N)
dependents      = np.random.choice(["Yes","No"], N, p=[0.30,0.70])
phone_service   = np.random.choice(["Yes","No"], N, p=[0.90,0.10])
multiple_lines  = np.random.choice(["Yes","No","No phone service"], N, p=[0.42,0.48,0.10])
online_security = np.random.choice(["Yes","No","No internet service"], N, p=[0.28,0.50,0.22])
online_backup   = np.random.choice(["Yes","No","No internet service"], N, p=[0.34,0.44,0.22])
device_protect  = np.random.choice(["Yes","No","No internet service"], N, p=[0.34,0.44,0.22])
tech_support    = np.random.choice(["Yes","No","No internet service"], N, p=[0.29,0.49,0.22])
streaming_tv    = np.random.choice(["Yes","No","No internet service"], N, p=[0.38,0.40,0.22])
streaming_movies= np.random.choice(["Yes","No","No internet service"], N, p=[0.39,0.39,0.22])
paperless       = np.random.choice(["Yes","No"], N, p=[0.59,0.41])

df = pd.DataFrame({
    "tenure": tenure, "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
    "Contract": contract, "InternetService": internet_service, "PaymentMethod": payment_method,
    "gender": gender, "SeniorCitizen": senior_citizen, "Partner": partner,
    "Dependents": dependents, "PhoneService": phone_service, "MultipleLines": multiple_lines,
    "OnlineSecurity": online_security, "OnlineBackup": online_backup,
    "DeviceProtection": device_protect, "TechSupport": tech_support,
    "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies, "PaperlessBilling": paperless,
})

risk = (
    0.30 * (1 - tenure / 72) +
    0.20 * (monthly_charges / 118) +
    0.20 * (contract == "Month-to-month").astype(int) +
    0.10 * (internet_service == "Fiber optic").astype(int) +
    0.08 * (payment_method == "Electronic check").astype(int) +
    0.05 * senior_citizen +
    0.04 * (online_security == "No").astype(int) +
    0.03 * (tech_support == "No").astype(int)
)
prob_churn = 1 / (1 + np.exp(-7 * (risk - 0.45)))
df["Churn"] = (np.random.rand(N) < prob_churn).astype(int)
print(f"✅ Dataset: {N:,} records | Churn rate: {df['Churn'].mean():.1%}")

# ── Feature engineering (self-contained, no external function) ────────────────
def build_features(df):
    d = df.copy()
    d["gender"]          = (d["gender"] == "Female").astype(int)
    d["Partner"]         = (d["Partner"] == "Yes").astype(int)
    d["Dependents"]      = (d["Dependents"] == "Yes").astype(int)
    d["PhoneService"]    = (d["PhoneService"] == "Yes").astype(int)
    d["PaperlessBilling"]= (d["PaperlessBilling"] == "Yes").astype(int)
    d["Contract_Month"]  = (d["Contract"] == "Month-to-month").astype(int)
    d["Contract_OneYear"]= (d["Contract"] == "One year").astype(int)
    d["Internet_Fiber"]  = (d["InternetService"] == "Fiber optic").astype(int)
    d["Internet_DSL"]    = (d["InternetService"] == "DSL").astype(int)
    d["Payment_Echeck"]  = (d["PaymentMethod"] == "Electronic check").astype(int)
    d["Payment_MailedCheck"]   = (d["PaymentMethod"] == "Mailed check").astype(int)
    d["MultipleLines_Yes"]     = (d["MultipleLines"] == "Yes").astype(int)
    d["OnlineSecurity_Yes"]    = (d["OnlineSecurity"] == "Yes").astype(int)
    d["OnlineBackup_Yes"]      = (d["OnlineBackup"] == "Yes").astype(int)
    d["DeviceProtection_Yes"]  = (d["DeviceProtection"] == "Yes").astype(int)
    d["TechSupport_Yes"]       = (d["TechSupport"] == "Yes").astype(int)
    d["StreamingTV_Yes"]       = (d["StreamingTV"] == "Yes").astype(int)
    d["StreamingMovies_Yes"]   = (d["StreamingMovies"] == "Yes").astype(int)
    d["AvgMonthlySpend"]  = d["TotalCharges"] / (d["tenure"] + 1)
    svc_cols = ["MultipleLines_Yes","OnlineSecurity_Yes","OnlineBackup_Yes",
                "DeviceProtection_Yes","TechSupport_Yes","StreamingTV_Yes","StreamingMovies_Yes"]
    d["ChargePerService"] = d["MonthlyCharges"] / (d[svc_cols].sum(axis=1) + 1)
    drop_cols = ["Contract","InternetService","PaymentMethod","MultipleLines",
                 "OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport",
                 "StreamingTV","StreamingMovies"]
    return d.drop(columns=drop_cols)

df_enc = build_features(df)
feature_cols = [c for c in df_enc.columns if c != "Churn"]
X = df_enc[feature_cols]
y = df_enc["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print("\n🌲 Training models...")
models = {
    "Random Forest":      RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1),
    "Gradient Boosting":  GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42),
    "Logistic Regression":LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
}

best_model, best_name, best_f1 = None, "", 0
results = {}

for name, mdl in models.items():
    if name == "Logistic Regression":
        mdl.fit(X_train_sc, y_train)
        y_pred = mdl.predict(X_test_sc)
        y_prob = mdl.predict_proba(X_test_sc)[:,1]
    else:
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)
        y_prob = mdl.predict_proba(X_test)[:,1]

    f1 = f1_score(y_test, y_pred)
    results[name] = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1,
        "roc_auc":   roc_auc_score(y_test, y_prob),
    }
    print(f"   {name}: Acc={results[name]['accuracy']:.3f} | F1={f1:.3f} | AUC={results[name]['roc_auc']:.3f}")
    if f1 > best_f1:
        best_f1, best_model, best_name = f1, mdl, name

print(f"\n🏆 Best: {best_name} (F1={best_f1:.3f})")
preds = best_model.predict(X_test if best_name != "Logistic Regression" else X_test_sc)
print(classification_report(y_test, preds, target_names=["Not Churned","Churned"]))

if hasattr(best_model, "feature_importances_"):
    importances = dict(zip(feature_cols, best_model.feature_importances_))
else:
    importances = dict(zip(feature_cols, abs(best_model.coef_[0])))

Path("model").mkdir(exist_ok=True)
with open("model/churn_model.pkl","wb") as f:
    pickle.dump({
        "model":        best_model,
        "model_name":   best_name,
        "scaler":       scaler,
        "feature_cols": feature_cols,
        "metrics":      results[best_name],
        "all_metrics":  results,
        "importances":  importances,
        "churn_rate":   df["Churn"].mean(),
    }, f)

print("\n✅ Model saved → model/churn_model.pkl")
print("🚀 Run: streamlit run app.py")