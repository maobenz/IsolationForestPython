# -*- coding: utf-8 -*-
'''
    Holt Winters
'''
import math


class HoltWinters():
    def __init__(self, season, a, b, c):

        self.season = int(season)
        self.a = a
        self.b = b
        self.c = c
        self.valueList = []
        self.seasonList = []
        self.residualList = []
        self.base = 0
        self.trend = 0

    def extract(self, ts, value):
        '''
            Given input data, and return a feature
        '''
        feature = None
        predictValue = None
        self.valueList.append(value)

        if len(self.valueList) < self.season:
            self.seasonList.append(0)
            self.residualList.append(None)
        elif len(self.valueList) == self.season:
            self.seasonList.append(0)  # Now the seasonList add to season length
            self.residualList.append(None)

            # Initialize item
            self.base = float(sum(self.valueList)) / self.season
            for i in range(0, self.season):
                self.seasonList[i] = self.valueList[i] - self.base
            self.trend = 0
        else:
            # Predict
            predictValue = self.base + self.trend + self.seasonList[0]
            if self.residualList[0] != None:
                feature = abs(value - predictValue)

            # Update item
            oldBase = self.base
            oldTrend = self.trend
            self.base = self.a * (value - self.seasonList[0]) + (1 - self.a) * (oldBase + oldTrend)
            self.trend = self.b * (self.base - oldBase) + (1 - self.b) * oldTrend
            self.seasonList.append(self.c * (value - self.base) + (1 - self.c) * self.seasonList[0])

            if self.residualList[0] == None:
                self.residualList.append(abs(value - predictValue))
            else:
                self.residualList.append(self.c * abs(value - predictValue) + (1 - self.c) * self.residualList[0])

            del self.valueList[0]
            del self.seasonList[0]
            del self.residualList[0]

        return feature
