import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm
from loguru import logger

# -- setup

print("Loading data")
flows = pd.read_csv(
    "../raw/02934b58528a226b_NFV3DATA-A11964_A11964/data/NF-ToN-IoT-v3.csv",
    # nrows=1_000_000,
)
print(f"Data loaded successfully, {len(flows)} flows")

# ignoring port information
flows["src"] = flows["IPV4_SRC_ADDR"].astype(str)
flows["dst"] = flows["IPV4_DST_ADDR"].astype(str)

flows.drop(["Label"], axis=1, inplace=True)
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


# barely any flows in these
flows = flows[flows["Attack"] != "ransomware"]
flows = flows[flows["Attack"] != "mitm"]

# -- clamping

print("Clamping outliers in numerical features")
numerical = high_count_categorical
numerical_df = flows[numerical]
for feature in numerical_df.columns:
    if (
        numerical_df[feature].max() > 10 * numerical_df[feature].median()
        and numerical_df[feature].max() > 10
    ):
        flows[feature] = np.where(
            flows[feature] < flows[feature].quantile(0.99),
            flows[feature],
            flows[feature].quantile(0.99),
        )


# -- numerical encoding

for c in high_count_categorical:
    non_ordinal.remove(c)
low_count_categorical = non_ordinal

# split high_count_categorical into truly continuous vs categorical
high_count_truly_categorical = [
    "PROTOCOL",
    "ICMP_TYPE",
    "ICMP_IPV4_TYPE",
    "DNS_QUERY_TYPE",
]
high_count_truly_numerical = ["DNS_QUERY_ID", "DNS_TTL_ANSWER", "L7_PROTO"]

numerical = [
    c
    for c in flows.columns
    if c not in low_count_categorical
    and c not in high_count_categorical
    and c not in ("Label", "Attack", "src", "dst")
] + high_count_truly_numerical  # these are fine to standardize

# remove inf
for c in flows.columns:
    flows = flows[flows[c] != np.inf]

flows = flows.sort_values(by="FLOW_START_MILLISECONDS")

print("Normalizing and encoding features")
ss = StandardScaler()
le = LabelEncoder()
flows[numerical] = ss.fit_transform(flows[numerical])

# label encode the high count categoricals (too many values for OHE)
for categorical in high_count_truly_categorical:
    flows[categorical] = le.fit_transform(flows[categorical].astype(str))

# OHE the low count categoricals
for categorical in low_count_categorical:
    flows[categorical] = le.fit_transform(flows[categorical])
flows = pd.get_dummies(flows, columns=low_count_categorical, drop_first=False)


# -- log basic information

print("Sample rows from the processed dataset:")
print(flows.head(10))

meta = {
    "num_flows": len(flows),
    "num_features": flows.shape[1] - 3,  # excluding src, dst, Attack
    "num_classes": flows.Attack.nunique(),
    "class_distribution": flows.Attack.value_counts().to_dict(),
}
print(meta)

print("\nDataFrame Info:")
print(flows.describe())
print(flows.info())

chunk_size = 10_000
print(
    f"chunking the data into [{chunk_size}] random contingous chunks to equalize class distribution between splits and maintain temporal consistency"
)


# -- Continguous chunk downsmapling

print("RESAMPLING")
downsample_rate = 0.3


# take the first 30% benign flows
benign_flows = flows[flows.Attack == "Benign"].iloc[
    : int(downsample_rate * len(flows[flows.Attack == "Benign"]))
]
# take 10% flows from EACH attack category, being careful to keep the attack category in the dataframe
resampled_attack_flows = pd.concat(
    [
        flows[flows.Attack == attack].iloc[
            : int(downsample_rate * len(flows[flows.Attack == attack]))
        ]
        for attack in flows.Attack.unique()
        if attack != "Benign"
    ]
)

# combine with benign flows and randomize
new_flows = (
    pd.concat([benign_flows, resampled_attack_flows])
    .sample(frac=1, random_state=0)
    .reset_index(drop=True)
)

# split into test and train
test_split = 0.2
train_flows, test_flows = (
    new_flows.iloc[: int(len(new_flows) * (1 - test_split))],
    new_flows.iloc[int(len(new_flows) * (1 - test_split)) :],
)

# # randomly selected chunks thats make up 10% of flows
# chunk_idxs = np.arange(0, len(flows), chunk_size)
# sampled_chunk_idxs = np.random.choice(chunk_idxs, size=int(downsample_rate * len(chunk_idxs)), replace=False)
# chunks = [flows[i : i + chunk_size] for i in sampled_chunk_idxs]

# # sample randomly 8% chunkinto train sn 2% into test to maintain temporal consistency and class distribution
# n_train = int(0.8 * len(chunks))
# train_idx = np.random.choice(len(chunks), size=n_train, replace=False)
# test_idx = [i for i in range(len(chunks)) if i not in train_idx]
# train_flows = pd.concat([chunks[i] for i in train_idx]).reset_index(drop=True)
# test_flows = pd.concat([chunks[i] for i in test_idx]).reset_index(drop=True)

# log the percentage occurance of each class in 'attack' for train and test
print("resampled size for train and test:")
print(f"Train size: {len(train_flows)}, Test size: {len(test_flows)}")
print("\resampled Class distribution in training set:")
print(train_flows.Attack.value_counts(normalize=True))
print("\resampled Class distribution in test set:")
print(test_flows.Attack.value_counts(normalize=True))

# add metadata about chunking size
train_flows.contiguous_chunk_size = (
    f"{chunk_size}, !! keep windowing a factor of this size"
)
test_flows.contiguous_chunk_size = (
    f"{chunk_size}, !! keep windowing a factor of this size"
)

print("writing partitions")
test_flows.to_csv("../interm/NoT-IoT_processed_test.csv", index=False)
train_flows.to_csv("../interm/NoT-IoT_processed_train.csv", index=False)
