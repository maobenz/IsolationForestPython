# -*- coding: utf-8 -*-
'''
    Weighted Moving average
'''
import math


class WMA():
    def __init__(self, window):

        self.window = window
        self.valueList = []
        self.weightedNum = 0
        for i in range(0, window):
            self.weightedNum += (i + 1)

    def extract(self, ts, value):
        feature = None

        if len(self.valueList) < self.window:
            pass
        else:
            weightedSum = 0
            for i in range(0, len(self.valueList)):
                weightedSum += (i + 1) * self.valueList[i]

            average = weightedSum / float(self.weightedNum)
            feature = abs(value - average)

            del self.valueList[0]

        self.valueList.append(value)
        return feature
