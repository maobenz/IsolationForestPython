from FeatureExtractor.diff import DifferencingValue
from FeatureExtractor.EWMA import EWMA
from FeatureExtractor.holt_winter import HoltWinters
from FeatureExtractor.MovingAverage import MovingAverage
#from FeatureExtractor.tsd import TimeSeriesDecomposition
from FeatureExtractor.WMA import WMA
import pandas as pd

def extract_feature(input_file,output_file):
    df = pd.read_csv(input_file)
    value = df['value'].values   #值
    ts = df['timestamp'].values   #时间
    label = df['label'].values    #标记结果

    feature_WMA = []
    wma = WMA(window=12)
    for i, j in zip(value, ts):
        feature_WMA.append(wma.extract(j, i))

    feature_MA = []
    ma = MovingAverage(window=12)
    for i, j in zip(value, ts):
        feature_MA.append(ma.extract(j, i))

    feature_EWMA = []
    ewma = EWMA(weight=0.2)
    for i, j in zip(value, ts):
        feature_EWMA.append(ewma.extract(j, i))
    print(feature_EWMA)
    exit(0)

    feature_ht = []
    HT = HoltWinters(season=24, a=0.2, b=0.2, c=0.2)
    for i, j in zip(value, ts):
        feature_ht.append(HT.extract(j, i))

    feature_diff = []
    diff = DifferencingValue(season=24)
    for i, j in zip(value, ts):
        feature_diff.append(diff.extract(j, i))

    result = {'timestamp': ts, 'label': label, 'value': value, 'WMA': feature_WMA, 'MA': feature_MA,
              'EWMA': feature_EWMA, 'Holt-winter': feature_ht, 'Diff': feature_diff}
    result_df = pd.DataFrame(data=result)
    result_df.to_csv(output_file,index=False)

df = pd.read_csv('dataset/real_17.csv')
length = len(df)
ts = range(1514736000,1514736000+length*3600,3600)#创建了一个整数列表
df['timestamp'] = ts
df.to_csv('dataset/real_17.csv',index=False)
extract_feature('dataset/real_17.csv','real_17_feature.csv')