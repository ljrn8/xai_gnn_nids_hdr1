import numpy as np
import pandas as pd
from pathlib import Path

root = Path('raw/unsw/data')
flows = pd.read_csv('raw/f7546561558c07c5_NFV3DATA-A11964_A11964/data/NF-UNSW-NB15-v3.csv')

non_ordinal = [
    'IPV4_SRC_ADDR',
    'IPV4_DST_ADDR',
    'L4_SRC_PORT',
    'L4_DST_PORT',
    'PROTOCOL',
    'L7_PROTO',
    'ICMP_TYPE',
    'ICMP_IPV4_TYPE',
    'DNS_QUERY_ID',
    'DNS_QUERY_TYPE',
    'DNS_TTL_ANSWER',
    'FTP_COMMAND_RET_CODE',
]

high_count_categorical = [
    'L4_SRC_PORT', 'L4_DST_PORT', 'ICMP_TYPE', 'DNS_QUERY_ID',
    'PROTOCOL', 'L7_PROTO', 'DNS_TTL_ANSWER', 'ICMP_IPV4_TYPE'
]

for c in high_count_categorical:
    non_ordinal.remove(c)
low_count_categorical = non_ordinal

targets = ('Label', 'Attack')

numerical = [c for c in flows.columns 
             if c not in low_count_categorical 
             and c not in high_count_categorical
             and c not in targets]

# remove inf
for c in flows.columns:
    flows = flows[flows[c] != np.inf]


flows = flows.sort_values(by='FLOW_START_MILLISECONDS')
flows.drop('Label', axis=1, inplace=True) # inferred from Attack

flows['src'] = flows['IPV4_SRC_ADDR'].astype(str) + ':' + flows['L4_SRC_PORT'].astype(str)
flows['dst'] = flows['IPV4_DST_ADDR'].astype(str) + ':' + flows['L4_DST_PORT'].astype(str)
flows.drop(['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'L4_SRC_PORT', 'L4_DST_PORT'], axis=1, inplace=True)


flows.to_csv('./interm/unsw_nb15_processed.csv', index=False)

test_split = 0.2
train_flows, test_flows = (
    flows.iloc[:int(len(flows) * (1 - test_split))], 
    flows.iloc[int(len(flows) * (1 - test_split)):]
)

test_flows.to_csv('./interm/unsw_nb15_processed_test.csv', index=False)
train_flows.to_csv('./interm/unsw_nb15_processed_train.csv', index=False)
