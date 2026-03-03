import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm

print("Loading data")
flows = pd.read_csv(
    "raw/f78acbaa2afe1595_NFV3DATA-A11964_A11964/data/NF-CICIDS2018-v3.csv"
)
print(f"Data loaded successfully, {len(flows)} flows")

flows.drop(["Label", "L4_SRC_PORT", "L4_DST_PORT"], axis=1, inplace=True)

flows["src"] = flows["IPV4_SRC_ADDR"].astype(str)
flows["dst"] = flows["IPV4_DST_ADDR"].astype(str)

flows.drop(
    ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"],
    axis=1,
    inplace=True,
)

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
print("Clamping outliers in numerical features")
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
print("Normalizing and encoding features")
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


chunk_size = 10_000
print(
    f"chunking the data into [{chunk_size}] random contingous chunks to equalize class distribution between splits and maintain temporal consistency"
)

chunks = len(flows) // chunk_size
test_chunks_list = np.random.choice(chunks, size=int(chunks * 0.2), replace=False)
train_chunks_list = [i for i in range(chunks) if i not in test_chunks_list]
test_flows = pd.DataFrame()
train_flows = pd.DataFrame()

for i in tqdm(range(chunks)):
    chunk = flows.iloc[i * chunk_size : (i + 1) * chunk_size]
    if i in test_chunks_list:
        test_flows = pd.concat([test_flows, chunk])
    else:
        train_flows = pd.concat([train_flows, chunk])

# log the percentage occurance of each lcass in 'attack' for train and test
print("\nClass distribution in training set:")
print(train_flows.Attack.value_counts(normalize=True))
print("\nClass distribution in test set:")
print(test_flows.Attack.value_counts(normalize=True))

# add metadata about chunking size
train_flows.contiguous_chunk_size = (
    f"{chunk_size}, !! keep windowing a multiple of this size"
)
test_flows.contiguous_chunk_size = (
    f"{chunk_size}, !! keep windowing a multiple of this size"
)

print("writing partitions")
test_flows.to_csv("./interm/cicids_processed_test.csv", index=False)
train_flows.to_csv("./interm/cicids_processed_train.csv", index=False)
