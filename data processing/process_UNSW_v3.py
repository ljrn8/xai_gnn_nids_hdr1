import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder

root = Path("raw/unsw/data")
flows = pd.read_csv(
    "../raw/f7546561558c07c5_NFV3DATA-A11964_A11964/data/NF-UNSW-NB15-v3.csv"
)

flows["src"] = (
    flows["IPV4_SRC_ADDR"].astype(str) + ":" + flows["L4_SRC_PORT"].astype(str)
)
flows["dst"] = (
    flows["IPV4_DST_ADDR"].astype(str) + ":" + flows["L4_DST_PORT"].astype(str)
)

flows.drop(["Label", "L4_SRC_PORT", "L4_DST_PORT"], axis=1, inplace=True)

flows.drop(
    ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"],
    axis=1,
    inplace=True,
)

# clamping
non_ordinal = [
    "PROTOCOL",
    "L7_PROTO",
    "ICMP_TYPE",
    "ICMP_IPV4_TYPE",
    "DNS_QUERY_ID",
    "DNS_QUERY_TYPE",
    "DNS_TTL_ANSWER",
    "FTP_COMMAND_RET_CODE",
]

high_count_categorical = [
    "ICMP_TYPE",
    "DNS_QUERY_ID",
    "PROTOCOL",
    "L7_PROTO",
    "DNS_TTL_ANSWER",
    "ICMP_IPV4_TYPE",
]

# clamping
numerical = high_count_categorical
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

for c in high_count_categorical:
    non_ordinal.remove(c)
low_count_categorical = non_ordinal

numerical = [
    c
    for c in flows.columns
    if c not in low_count_categorical
    and c not in high_count_categorical
    and c not in ("Label", "Attack", "src", "dst")
]

# remove inf
for c in flows.columns:
    flows = flows[flows[c] != np.inf]


flows = flows.sort_values(by="FLOW_START_MILLISECONDS")


# normalization, standrdization, OHE
ss = StandardScaler()
le = LabelEncoder()
flows[numerical] = ss.fit_transform(flows[numerical])

for categorical in low_count_categorical:
    flows[categorical] = le.fit_transform(flows[categorical])
flows = pd.get_dummies(flows, columns=low_count_categorical, drop_first=False)

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

# print unique values for categorical columns
print("\nUnique values for categorical columns:")
for c in [
    c
    for c in flows.columns
    if c not in numerical
    and c not in ("src", "dst")
    and c not in high_count_categorical
]:
    print(f"{c}: {flows[c].unique()}")

flows.to_csv("../interm/unsw_nb15_processed.csv", index=False)

test_split = 0.2
train_flows, test_flows = (
    flows.iloc[: int(len(flows) * (1 - test_split))],
    flows.iloc[int(len(flows) * (1 - test_split)) :],
)

test_flows.to_csv("../interm/unsw_nb15_processed_test.csv", index=False)
train_flows.to_csv("../interm/unsw_nb15_processed_train.csv", index=False)
