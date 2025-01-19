import os 
import pandas as pd

theshold_dir = '/Users/tkiker/Downloads/threshold'

df = pd.read_csv('/Users/tkiker/Downloads/Threshold Masked Images - Sheet1.csv')
names = list(df['Name'])

for name in names: 
    found = False
    for filename in os.listdir(theshold_dir): 
        if name in filename: 
            found = True
    
    if not found: 
        print(name)

for i in range(0, len(names)):
    a = names[i]
    j = i+1 
    while (j<len(names)): 
        b = names[j]
        if a==b: 
            print(a)

        j += 1

for filename in os.listdir(theshold_dir):
    if '.png' in filename: 
        object_name = filename.split('.')[0].split('-cloud')[0]
        found = False
        for name in names: 
            if name in object_name:
                found = True
        if not found: 
            print(object_name)