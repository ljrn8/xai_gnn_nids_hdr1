import sys

sys.path.append("../")
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder

root = Path("raw/de2c6f75dd50d933_MOHANAD_A4706/data/")
flows = pd.read_csv(root / "NF-BoT-IoT.csv")
features = pd.read_csv(root / "NetFlow_v1_Features.csv")

flows["src"] = flows["IPV4_SRC_ADDR"].astype(str) + ":" + flows.L4_SRC_PORT.astype(str)
flows["dst"] = flows["IPV4_DST_ADDR"].astype(str) + ":" + flows.L4_DST_PORT.astype(str)
flows.drop(
    ["Label", "IPV4_DST_ADDR", "IPV4_SRC_ADDR", "L4_SRC_PORT", "L4_DST_PORT"],
    axis=1,
    inplace=True,
)

cateogrical = ["PROTOCOL"]

numerical = [
    "TCP_FLAGS",
    "L7_PROTO",
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS",
]

# clamping
numerical_df = flows[numerical]
for feature in numerical_df.columns:
    if (
        numerical_df[feature].max() > 10 * numerical_df[feature].median()
        and numerical_df[feature].max() > 10
    ):
        flows[feature] = np.where(
            flows[feature] < flows[feature].quantile(0.95),
            flows[feature],
            flows[feature].quantile(0.95),
        )

# remove inf
for c in flows.columns:
    flows = flows[flows[c] != np.inf]

# normalization, standrdization, OHE
ss = StandardScaler()
le = LabelEncoder()
flows[numerical] = ss.fit_transform(flows[numerical])

# for c in cateogrical:
# flows[c] = le.fit_transform(flows[c])

# OHE all categories
flows = pd.get_dummies(flows, columns=cateogrical, drop_first=False)

# print examples of rows from the processed dataset
print("Sample rows from the processed dataset:")
print(flows.head(10))

# print metadata about the dataset
meta = {
    "num_flows": len(flows),
    "num_features": flows.shape[1] - 3,  # excluding src, dst, Attack
    "num_classes": flows.Attack.nunique(),
    "class_distribution": flows.Attack.value_counts().to_dict(),
}
print(meta)

# print all dataframe info
print("\nDataFrame Info:")
print(flows.describe())
print(flows.info())

# print max and min values for each column
print("\nMax and Min values for each column:")
for c in flows.columns:
    print(f"{c}: min={flows[c].min()}, max={flows[c].max()}")

flows.to_csv("./interm/BotIoT_v1_processed.csv", index=False)

test_split = 0.2
train_flows, test_flows = (
    flows.iloc[: int(len(flows) * (1 - test_split))],
    flows.iloc[int(len(flows) * (1 - test_split)) :],
)

test_flows.to_csv("./interm/BotIoT_v1_processed_test.csv", index=False)
train_flows.to_csv("./interm/BotIoT_v1_processed_train.csv", index=False)
