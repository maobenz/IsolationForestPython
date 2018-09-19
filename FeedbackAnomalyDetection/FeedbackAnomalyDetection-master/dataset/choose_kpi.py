import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
''''
files = os.listdir('yahoo_ad_data/A1Benchmark/')
for i in files:
    if '.csv' in i:
df = pd.read_csv('yahoo_ad_data/A1Benchmark/'+i)
'''''
df = pd.read_csv('real_19.csv')
value = df['value'].values
label = df['is_anomaly'].values
#index = np.where(label == 0)
plt.figure()
plt.plot(value,color='black')
index = np.where(label==1)
plt.plot(index[0],value[index],color='red')
plt.show()
#plt.savefig('fig/'+i[:-4]+'.png')
#print(i,len(index[0]))
