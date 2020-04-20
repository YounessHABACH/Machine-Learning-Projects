# upper confidence bound

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

dataset = pd.read_csv("../../datasets/Ads_CTR_Optimisation.csv")

# Implement UCB from scratch
N = 10000
d = 10
adsSelected = []
numberOfSelections = [0] * d
sumsOfRewards = [0] * d
for n in range(0, N):
    ad = 0
    maxUpperBound = 0
    for i in range(0, d):
        if(numberOfSelections > 0):
            averageReward = sumsOfRewards[i] / numberOfSelections[i]
            delta_i = math.sqrt(3/2 * math.log(n+1) / numberOfSelections[i])
            upperBound = averageReward + delta_i
        else:
            upperBound = 1e400
        if(upperBound > maxUpper):
            maxUpperBound = upperBound
            ad = i
    adsSelected.append(ad)
    numberOfSelections[ad] += 1