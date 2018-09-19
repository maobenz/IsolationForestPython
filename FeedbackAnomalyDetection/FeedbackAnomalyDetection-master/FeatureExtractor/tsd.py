#import statsmodels.api as sm
import numpy as np
import math


# def tsd(value, season):
#     res = sm.tsa.seasonal_decompose(value, freq=season, model='additive')
#     predict = list(res.trend + res.seasonal)
#     predict = np.nan_to_num(predict)
#     feature = np.zeros(len(value))
#     for i in range(len(predict)):
#         feature[i] = abs(predict[i]-value[i])
#
#     return feature
# -*- coding: utf-8 -*-
'''
    Time series decomposition detection
'''

from .RunningMedian import RunningMedian
from itertools import islice
import math

featureDefinition = 0


class TimeSeriesDecomposition():
    def __init__(self, season, medianFlag, historicalCycleNum):
        self.season = int(season)
        self.medianFlag = medianFlag
        self.timeSeries = []
        self.offset = 0
        self.trendMedian = None
        self.noiseMedian = None
        self.noiseMAD = None
        self.noiseMean = None
        self.noiseSTD = None
        self.madToNormRate = 1.4826
        self.maxLength = historicalCycleNum * season

    def extract(self, ts, value):
        '''
            Given input data, and return 4 features
        '''
        returnValue = None  # noise, or noise deviation
        noiseDeviation = None
        trendValue = None
        seasonValue = None
        noiseValue = None
        self.timeSeries.append({'v': value})

        tlen = len(self.timeSeries)
        pos = tlen - 1
        if tlen < int(self.season) + self.offset:
            pass
        elif tlen == int(self.season) + self.offset:  # The data of 1st season is ready

            if self.medianFlag:
                # [Calculate trend using running median]
                medianList = []
                for i in range(0, int(self.season)):
                    medianList.append(self.timeSeries[i]['v'])
                it = iter(medianList)
                self.trendMedian = RunningMedian(islice(it, int(self.season)))
                firstSeasonMedian = self.trendMedian.getMedian()
                for i in range(0, int(self.season)):
                    self.timeSeries[i]['t'] = firstSeasonMedian
            else:
                # [Calculate trend using running mean]
                total = 0
                for i in range(0, int(self.season)):
                    total += self.timeSeries[i]['v']
                firstSeasonMean = float(total) / int(self.season)
                for i in range(0, int(self.season)):
                    self.timeSeries[i]['t'] = firstSeasonMean



        else:  # 2nd season begin
            if self.medianFlag:
                # [Update trend using median]
                self.timeSeries[pos]['t'] = self.trendMedian.update(self.timeSeries[pos]['v'])
            else:
                # [Update trend using mean]
                self.timeSeries[pos]['t'] = (self.timeSeries[pos - 1]['t'] * int(self.season) + self.timeSeries[pos][
                    'v'] - self.timeSeries[pos - self.season]['v']) / self.season

            # Calculate season and noise from the 2nd season
            seasonPos = pos
            seasonList = []
            seasonCount = 0
            while seasonPos >= 0:
                seasonList.append(self.timeSeries[seasonPos]['v'] - self.timeSeries[seasonPos]['t'])
                seasonCount += 1
                seasonPos -= int(self.season)
            if self.medianFlag:
                self.timeSeries[pos]['s'] = self.getMedian(seasonList)
            else:
                self.timeSeries[pos]['s'] = self.getMean(seasonList)

            self.timeSeries[pos]['n'] = self.timeSeries[pos]['v'] - self.timeSeries[pos]['t'] - self.timeSeries[pos][
                's']

            trendValue = self.timeSeries[pos]['t']
            seasonValue = self.timeSeries[pos]['s']
            noiseValue = self.timeSeries[pos]['n']

            # Calculate the deviation of noise
            if self.medianFlag:
                if self.noiseMedian != None:
                    noiseDeviation = (self.noiseMedian - self.timeSeries[pos]['n'])
                    if featureDefinition == 0:
                        # noiseDeviation = abs(noiseDeviation)
                        noiseDeviation = noiseDeviation
                    else:
                        if abs(float(float(value) - noiseDeviation)) > 0:
                            noiseDeviation = abs(noiseDeviation) / abs(float(value) - noiseDeviation)
                        else:
                            if abs(float(value)) > 0:
                                noiseDeviation = abs(noiseDeviation) / abs(float(value))
                            else:
                                noiseDeviation = 0
            else:
                if self.noiseMean != None:
                    noiseDeviation = (self.noiseMean - self.timeSeries[pos]['n'])
                    if featureDefinition == 0:
                        # noiseDeviation = abs(noiseDeviation)
                        noiseDeviation = noiseDeviation
                    else:
                        if abs(float(float(value) - noiseDeviation)) > 0:
                            noiseDeviation = abs(noiseDeviation) / abs(float(value) - noiseDeviation)
                        else:
                            if abs(float(value)) > 0:
                                noiseDeviation = abs(noiseDeviation) / abs(float(value))
                            else:
                                noiseDeviation = 0

            # Recalculate the median and MAD of noise
            if (pos - self.offset + 1) % int(
                    self.season) == 0:  # When a season has just passed (the first passed season is the 2nd season)
                startPos = int(self.season) + self.offset  # Skip the first season, where noise is not calcualted
                if startPos < 0:
                    startPos = 0
                endPos = pos
                self.calNoiseThreshold(startPos, endPos)

            # Remove extra data point
            if tlen > self.maxLength:
                del self.timeSeries[0]
                self.offset -= 1

        if noiseDeviation == None:
            return None
        else:
            return abs(noiseDeviation)

    def getMedian(self, l):
        l.sort()
        win = len(l)
        median = 0
        if win % 2 == 0:
            median = (l[int(win / 2)] + l[int(win / 2) - 1]) / 2
        else:
            median = l[int((win - 1) / 2)]
        return median

    def getMean(self, l):
        return float(sum(l)) / len(l)

    def calNoiseThreshold(self, startPos, endPos):

        tempList = []

        for i in range(startPos, endPos + 1):
            tempList.append(self.timeSeries[i]['n'])
        win = len(tempList)

        if self.medianFlag:
            self.noiseMedian = self.getMedian(tempList)
            devList = []

            for i in range(0, win):
                devList.append(abs(tempList[i] - self.noiseMedian))
            self.noiseMAD = self.getMedian(devList)

            # To avoid divison by zero
            if self.noiseMAD == 0:
                self.noiseMAD = 0.0001
            else:
                self.noiseMAD = self.noiseMAD * self.madToNormRate
        else:
            self.noiseMean = self.getMean(tempList)
            tmp = 0
            for i in range(0, win):
                tmp += math.pow((tempList[i] - self.noiseMean), 2)
            self.noiseSTD = math.sqrt(float(tmp) / win)
            if self.noiseSTD == 0:
                self.noiseSTD = 0.0001


