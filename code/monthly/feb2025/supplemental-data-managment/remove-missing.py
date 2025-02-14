import pandas as pd

file_path = 'code/monthly/feb2025/supplemental-data-managment/Supplemental Data Selection Management.csv'

df = pd.read_csv(file_path)
names = df['Name'].to_list()

names = ['-'.join(name.split('-')[:2]) for name in names]

names_set = list(set(names))

from collections import Counter

name_counts = Counter(names)

filtered_names = [name for name, count in name_counts.items() if count != 4]

print(filtered_names)