# -*- coding: utf-8 -*-
'''
    EWMA
'''
import pandas as pd

class EWMA():
    def __init__(self,weight):
        self.weight = float(weight)
        self.dataList = []
        self.predict = None

    def extract(self,ts,value):
        '''
            Given input data, and return 2 features
        '''
        feature = None

        # if len(self.dataList) < 2:
        #     self.dataList.append(value)
        # else:

        #     df = pd.DataFrame(self.dataList)
        #     predictV = df.ewm(com=self.weight).mean()
        #     feature = abs(predictV[0][0]-value)

        #     self.dataList.append(value)
        #     del self.dataList[0]

        # return feature


        if self.predict == None:
            self.predict = value
        else:
            feature = abs(self.predict-value)
            self.predict = self.weight*value + (1-self.weight )*self.predict

        return feature