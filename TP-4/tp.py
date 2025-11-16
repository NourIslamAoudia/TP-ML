import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




data = {
    'duration': [7, 2, 10, 0, 5, 3, 12, 1, 8, 4],
    'src_bytes': [100, 200, 5000, 50, 120, 300, 7000, 80, 150, 90],
    'dst_bytes': [50, 120, 80, 8000, 10, 40, 60, 1000, 200, 150],
    'protocol_type': ['TCP', 'TCP', 'UDP', 'TCP', 'ICMP', 'UDP', 'TCP', 'ICMP', 'TCP', 'UDP'],
    'flag': ['SF', 'SF', 'S0', 'REJ', 'SF', 'S0', 'S0', 'SF', 'SF', 'SF'],
    'label': ['normal', 'normal', 'attack', 'attack', 'normal', 'attack', 'attack', 'attack', 'normal', 'normal']
}

df = pd.DataFrame(data)
print(df)


numic_cols = ['duration', 'src_bytes', 'dst_bytes']


#describe numerical columns
describe_numic = df[numic_cols].describe()
print("\nDescriptive statistics for numerical columns:\n", describe_numic)


#to observation outliers, we can use boxplots for each numerical column
plt.figure(figsize=(8,5))
plt.boxplot([df[df['label']=='normal']['src_bytes'],
             df[df['label']=='attack']['src_bytes']],
            labels=['Normal','Attack'])
plt.title('Boxplot of Source Bytes by Label')
plt.ylabel('src_bytes')
plt.show()


#to observe distribution, we can use histograms for each numerical column
plt.figure(figsize=(8,5))
plt.hist(df[df['label']=='normal']['src_bytes'], bins=5, alpha=0.7, label='Normal')
plt.hist(df[df['label']=='attack']['src_bytes'], bins=5, alpha=0.7, label='Attack')
plt.title('Histogram of Source Bytes')
plt.xlabel('src_bytes')
plt.ylabel('frequency')
plt.legend()
plt.show()


colors = {'normal':'blue', 'attack':'red'}
plt.figure(figsize=(8,5))
for label in df['label'].unique():
    subset = df[df['label']==label]
    plt.scatter(subset['duration'], subset['src_bytes'], c=colors[label], label=label, s=100)
plt.xlabel('Duration (seconds)')
plt.ylabel('Source Bytes')
plt.title('Scatter Plot: Duration vs Source Bytes')
plt.legend()
plt.show()


protocol_counts = pd.crosstab(df['protocol_type'], df['label'])
protocol_counts.plot(kind='bar', figsize=(8,5))
plt.title('Protocol Type Counts by Label')
plt.ylabel('Count')
plt.show()