# -*- coding: utf-8 -*-
'''
    Moving average
'''
import math


class MovingAverage():
    def __init__(self,window):

        self.window = window
        self.valueList = []

    def extract(self,ts,value):
        feature = None

        if len(self.valueList) < self.window:
            pass
        else:
            average = sum(self.valueList)/len(self.valueList)
            feature = abs((value - average))

            del self.valueList[0]

        self.valueList.append(value)

        return feature