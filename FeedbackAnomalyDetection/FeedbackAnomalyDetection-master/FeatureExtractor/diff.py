# -*- coding: utf-8 -*-

class DifferencingValue():
    def __init__(self,season):
        self.valueList = []
        self.season  = season

    def extract(self,ts,value):

        feature = None

        if len(self.valueList) < self.season:
            pass
        else:
            feature = abs(value - self.valueList[0])
            del self.valueList[0]
        self.valueList.append(value)

        return feature
