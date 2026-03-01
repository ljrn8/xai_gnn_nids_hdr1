import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder

root = Path("./raw/02934b58528a226b_NFV3DATA-A11964_A11964/data")
print("loading data")
flows = pd.read_csv(root / "NF-ToN-IoT-v3.csv")
features = pd.read_csv(root / "NetFlow_v3_Features.csv")
print("done loading data")

flows.drop("Label", axis=1, inplace=True)  # inferred from Attack

flows["src"] = (
    flows["IPV4_SRC_ADDR"].astype(str) + ":" + flows["L4_SRC_PORT"].astype(str)
)
flows["dst"] = (
    flows["IPV4_DST_ADDR"].astype(str) + ":" + flows["L4_DST_PORT"].astype(str)
)
flows.drop(
    ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT"],
    axis=1,
    inplace=True,
)

OHE_columns = []
for c in flows.columns:
    if len(np.unique(flows[c])) < 10 and c not in ("src", "dst", "Attack"):
        OHE_columns.append(c)


# clamping
numerical = [
    c
    for c in flows.columns
    if c not in OHE_columns and c not in ("src", "dst", "Attack")
]
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

flows = flows.sort_values(by="FLOW_START_MILLISECONDS")


# normalization, standrdization, OHE
ss = StandardScaler()
le = LabelEncoder()
flows[numerical] = ss.fit_transform(flows[numerical])

for categorical in OHE_columns:
    flows[categorical] = le.fit_transform(flows[categorical])
flows = pd.get_dummies(flows, columns=OHE_columns, drop_first=False)

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

flows.to_csv("./interm/ToNIoT_processed.csv", index=False)

test_split = 0.2
train_flows, test_flows = (
    flows.iloc[: int(len(flows) * (1 - test_split))],
    flows.iloc[int(len(flows) * (1 - test_split)) :],
)

test_flows.to_csv("./interm/ToNIoT_processed_test.csv", index=False)
train_flows.to_csv("./interm/ToNIoT_processed_train.csv", index=False)
