"""
analyze_and_visualize_players.py

Generates a synthetic game player dataset, performs multiple analytical insights,
and visualizes results using matplotlib and seaborn.

Insights:
1. Churn Prediction (Random Forest)
2. Lifetime Value Segmentation
3. Engagement Pattern Analysis
4. Funnel Drop-off Simulation
5. Player Clustering

All outputs saved to ./player_insights/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Create directories
base_dir = "player_insights"
plot_dir = os.path.join(base_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

# -------------------------------
# Step 1: Generate Synthetic Dataset
# -------------------------------
np.random.seed(42)
n = 20000

df = pd.DataFrame({
    "player_id": [f"player_{i}" for i in range(1, n+1)],
    "total_playtime_hours": np.round(np.random.gamma(2.0, 20.0, n), 2),
    "number_of_sessions": np.random.poisson(50, n),
    "avg_session_time_minutes": np.round(np.clip(np.random.normal(45, 10, n), 5, 120), 2),
    "purchases_made": np.random.poisson(2, n),
    "level_achieved": np.random.randint(1, 101, n),
    "days_since_last_login": np.random.randint(0, 60, n)
})

# Churn logic (simulated)
churn_prob = (
    0.3 * (df["days_since_last_login"] / 60) +
    0.2 * (1 - df["total_playtime_hours"] / df["total_playtime_hours"].max()) +
    0.5 * (1 - df["number_of_sessions"] / df["number_of_sessions"].max())
)
df["churned"] = np.random.binomial(1, np.clip(churn_prob, 0, 1))

# Save dataset
df.to_csv(os.path.join(base_dir, "player_data.csv"), index=False)

# -------------------------------
# Step 2: Churn Prediction
# -------------------------------
X = df[["total_playtime_hours", "number_of_sessions", "avg_session_time_minutes",
        "purchases_made", "level_achieved", "days_since_last_login"]]
y = df["churned"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
report["roc_auc"] = roc_auc_score(y_test, y_proba)
report.to_csv(os.path.join(base_dir, "churn_prediction_report.csv"))

# Visualization
fig1, ax1 = plt.subplots(figsize=(8, 5))
report.iloc[:-1][["precision", "recall", "f1-score"]].plot(kind="bar", ax=ax1)
plt.title("Churn Prediction - Classification Metrics")
plt.ylabel("Score")
plt.tight_layout()
fig1.savefig(os.path.join(plot_dir, "churn_classification_scores.png"))

# -------------------------------
# Step 3: LTV Segmentation
# -------------------------------
df["approx_ltv"] = df["purchases_made"] * df["avg_session_time_minutes"]
df["ltv_segment"] = pd.qcut(df["approx_ltv"], 4, labels=["Low", "Medium", "High", "Very High"])
df[["player_id", "approx_ltv", "ltv_segment"]].to_csv(os.path.join(base_dir, "player_ltv_segments.csv"), index=False)

fig2, ax2 = plt.subplots(figsize=(6, 4))
df["ltv_segment"].value_counts().sort_index().plot(kind="bar", color="skyblue", ax=ax2)
plt.title("LTV Segment Distribution")
plt.xlabel("Segment")
plt.ylabel("Player Count")
plt.tight_layout()
fig2.savefig(os.path.join(plot_dir, "ltv_segment_distribution.png"))

# -------------------------------
# Step 4: Engagement Analysis
# -------------------------------
engagement = df.groupby("churned")[["number_of_sessions", "avg_session_time_minutes"]].mean()
engagement.to_csv(os.path.join(base_dir, "session_engagement_analysis.csv"))

fig3, ax3 = plt.subplots(figsize=(6, 4))
engagement.plot(kind="bar", ax=ax3)
plt.title("Engagement by Churn Status")
plt.xlabel("Churned (0 = Retained, 1 = Churned)")
plt.ylabel("Average")
plt.tight_layout()
fig3.savefig(os.path.join(plot_dir, "engagement_pattern.png"))

# -------------------------------
# Step 5: Funnel Drop-Off
# -------------------------------
df["viewed_item"] = df["level_achieved"] > 10
df["browsed_store"] = df["purchases_made"] > 0
funnel = pd.DataFrame({
    "Stage": ["Viewed Item", "Browsed Store", "Did Not Churn"],
    "Players": [df["viewed_item"].sum(), df["browsed_store"].sum(), (df["churned"] == 0).sum()]
})
funnel.to_csv(os.path.join(base_dir, "purchase_funnel.csv"), index=False)

fig4, ax4 = plt.subplots(figsize=(6, 4))
sns.barplot(x="Stage", y="Players", data=funnel, ax=ax4)
plt.title("Funnel Drop-Off")
plt.tight_layout()
fig4.savefig(os.path.join(plot_dir, "funnel_analysis.png"))

# -------------------------------
# Step 6: Player Clustering
# -------------------------------
features = ["total_playtime_hours", "number_of_sessions", "avg_session_time_minutes", "purchases_made"]
X_scaled = StandardScaler().fit_transform(df[features])
kmeans = KMeans(n_clusters=4, random_state=42)
df["player_cluster"] = kmeans.fit_predict(X_scaled)

cluster_summary = df.groupby("player_cluster")[features + ["churned"]].mean()
cluster_summary.to_csv(os.path.join(base_dir, "player_clusters.csv"))

fig5, ax5 = plt.subplots(figsize=(8, 5))
sns.heatmap(cluster_summary, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax5)
plt.title("Player Cluster Profiles")
plt.tight_layout()
fig5.savefig(os.path.join(plot_dir, "cluster_profile_heatmap.png"))

print("✅ All analysis complete. Results saved in:", base_dir)

