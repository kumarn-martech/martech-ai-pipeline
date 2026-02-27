import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1) Simulated CDP-style dataset
# -----------------------------
df = pd.DataFrame({
    "contact_key": [f"C{i:03d}" for i in range(1, 31)],
    "email_opens_30d": [2,5,0,9,4,1,7,3,8,2,  0,1,9,8,7,2,3,4,6,1,  8,9,0,1,2,6,7,4,3,5],
    "web_visits_30d":  [1,3,0,8,2,1,6,2,7,1,  0,1,9,7,5,2,2,3,4,1,  6,8,0,1,2,5,6,3,2,4],
    "purchases_90d":   [0,1,0,3,1,0,2,1,2,0,  0,0,4,3,2,0,1,1,2,0,  3,4,0,0,1,2,2,1,1,1]
})

# -----------------------------
# 2) Feature engineering
# -----------------------------
df["engagement_score"] = (
    df["email_opens_30d"] * 0.4 +
    df["web_visits_30d"] * 0.4 +
    df["purchases_90d"] * 0.2
)

# -----------------------------
# 3) Segmentation (K-Means)
# -----------------------------
features = ["email_opens_30d", "web_visits_30d", "purchases_90d", "engagement_score"]
X_scaled = StandardScaler().fit_transform(df[features])

kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
df["segment_id"] = kmeans.fit_predict(X_scaled)

# -----------------------------
# 4) Business-friendly segment names
# (based on avg purchases; you can change the logic later)
# -----------------------------
seg_profile = (
    df.groupby("segment_id")[["email_opens_30d", "web_visits_30d", "purchases_90d", "engagement_score"]]
    .mean()
    .sort_values("purchases_90d")
)
order = list(seg_profile.index)

name_map = {
    order[0]: "Low Intent / Nurture",
    order[1]: "Warm / Educate",
    order[2]: "High Intent / Convert"
}
df["segment_name"] = df["segment_id"].map(name_map)

# -----------------------------
# 5) Activation output (CDP → SFMC Journey-like)
# -----------------------------
def next_best_action(segment_name: str) -> str:
    mapping = {
        "High Intent / Convert": "Journey: Conversion Offer",
        "Warm / Educate": "Journey: Education Series",
        "Low Intent / Nurture": "Journey: Nurture Basics",
    }
    return mapping.get(segment_name, "Journey: General")

activation = df[["contact_key", "segment_name"]].copy()
activation["next_best_action"] = activation["segment_name"].apply(next_best_action)

# -----------------------------
# 6) Save outputs
# -----------------------------
df.to_csv("outputs/customers_with_segments.csv", index=False)
activation.to_csv("outputs/activation.csv", index=False)

print("✅ Done. Files created:")
print(" - outputs/customers_with_segments.csv")
print(" - outputs/activation.csv")
print("\nSegment profile (averages):")
print(seg_profile.round(2))