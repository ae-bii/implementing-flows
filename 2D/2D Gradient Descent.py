import sys
sys.path.append("../implementing-flows")

import numpy as np
import random
import math
import numdifftools as nd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import functions
import cProfile, pstats
import time

start = time.time()


random.seed(0)
np.random.seed(0)

e = math.e
pi = math.pi

# Our own norm method because the np.norm was too slow
def norm(i):
    return np.sqrt(sum(np.square(i)))


#REVISED:
def BetaGradient(): # Newton's method (Experimental)
    Proportion = -0.5
    xSummation = np.zeros(NumFs)
    ySummation = np.zeros(NumFs)
    Beta = np.zeros(NumFs)
    for f in range(NumFs):
        xSummation[f] = sum(np.apply_along_axis(PotentialFs[f],1,MixtureSample))
        ySummation[f] = sum(np.apply_along_axis(PotentialFs[f],1,TargetSample))
        Beta[f] = (1/len(MixtureSample)) * xSummation[f] -(1/len(TargetSample)) * ySummation[f]
    return Beta * Proportion

def u(x, Beta):
    F_eval = []
    for f in range(NumFs):
        F_eval.append(PotentialFs[f](x))
    return (np.dot(x, x) / 2) + np.dot(Beta, F_eval)/NumFs

def uConjugate(y, Beta):
    ConvexCandidate = []
    for i in range(len(MixtureSample)):
        ConjugateValue = np.dot(MixtureSample[i], y) - u(MixtureSample[i], Beta)
        ConvexCandidate.append(ConjugateValue)
    return max(ConvexCandidate)

def D(Beta):
    xSummation = 0
    ySummation = 0
    for i in range(len(MixtureSample)):
        xSummation += u(MixtureSample[i], Beta)
        ySummation += uConjugate(TargetSample[i], Beta)

    LL = 1/len(MixtureSample) * xSummation + 1 / \
        len(TargetSample) * ySummation

    return LL

def SamplesUpdate(OldMixtureSample):
    NewMixtureSample = []
    for i in range(0, len(OldMixtureSample)):
        F_eval_x = np.zeros(NumFs)
        F_eval_y = np.zeros(NumFs)
        for f in range(0,NumFs):
            gradient = nd.Gradient(PotentialFs[f])(OldMixtureSample[i])
            F_eval_x[f] = gradient[0]
            F_eval_y[f] = gradient[1]

        xval = OldMixtureSample[i][0] + np.dot(Beta, F_eval_x)
        yval = OldMixtureSample[i][1] + np.dot(Beta, F_eval_y)
        NewMixtureSample.append([xval,yval])
    NewMixtureSample = np.array(NewMixtureSample)
    return NewMixtureSample

def MixtureSampleGenerator():
    mean1 = [1, 2]
    cov1 = [[0.5, 0], [0, 0.5]]
    mean2 = [2, 1]
    cov2 = [[0.5, 0], [0, 0.5]]
    x = np.random.multivariate_normal(mean1, cov1, SampleSize)
    y = np.random.multivariate_normal(mean2, cov2, SampleSize)
    MixtureSample = []
    for i in range(SampleSize):
        RandomSelector = random.random()
        if RandomSelector > 0.7:
            MixtureSample.append(x[i])
        else:
            MixtureSample.append(y[i])
    MixtureSample = np.array(MixtureSample)
    return MixtureSample

def StandardNormalGenerator():
    Sample = []
    x = np.random.standard_normal(500)
    y = np.random.standard_normal(500)
    for i in range(500):
        Sample.append([x[i], y[i]])
    return Sample


#------------------------------------------------------------------ TESTING (change to heatmap, add animtation)------------------------------------------------------------
TargetSample = np.loadtxt("C:/Users/Laptop/OneDrive - Grinnell College/Desktop/Research/PolyMath 2022/implementing-flows/SampleMoon.csv", delimiter=",")
TargetSample = random.sample(list(TargetSample), 250)
SampleSize = len(TargetSample)
print(SampleSize)
MixtureSample = MixtureSampleGenerator()
CenterGeneratorList = TargetSample

print(MixtureSample[1])
PotentialFs = [functions.Giulio_F(alpha=1.5),
                functions.Gaussian_F(alpha=1.5, constant=1),
                functions.Bump_F(alpha = 1.5, constant=1),
                functions.InverseQuadratic_F(alpha=1.5, constant=1),
                functions.InverseMultiquadric_F(alpha=1.5, constant=1)]
NumFs = len(PotentialFs)

plt.subplot(1,3,3)
plt.title("Target")
plt.scatter(*zip(*TargetSample), color = 'r', alpha = 0.2)
plt.xlim(-4, 4)
plt.ylim(-4, 4)

plt.subplot(1,3,1)
plt.title("Initial")
plt.scatter(*zip(*MixtureSample), color = 'b', alpha = 0.2)
plt.xlim(-4, 4)
plt.ylim(-4, 4)

DValue = 0
Iteration = 0
Beta = 0

# Profiling code
# profiler = cProfile.Profile()
# profiler.enable()

while True: # Maybe there is a problem of overfitting
    #print("Iteration " + str(i))
    Iteration += 1
    # if Iteration >= 10:
    #     CenterGeneratorList = MixtureSample
    CenterList = []
    TooClose = 1
    for i in range(0,NumFs):
        while TooClose == 1:
            c = CenterGeneratorList[random.randint(0, len(CenterGeneratorList) - 1)]
            if CenterList == []:
                TooClose = 0
            elif abs(c - np.median(CenterList))>1:
                TooClose = 0
        CenterList.append(c)
        PotentialFs[i].setCenter(c)
    OldBeta = Beta
    Beta = BetaGradient()
    OldD = DValue
    DValue = D(Beta)
    print(DValue)
    MixtureSample = SamplesUpdate(MixtureSample)
    if norm(OldBeta - Beta) < 0.0001 or Iteration > 50:
        break
    
# profiler.disable()
# stats = pstats.Stats(profiler).sort_stats('tottime')
# stats.strip_dirs()
# stats.dump_stats("newtoncrescent.prof")


plt.subplot(1,3,2)
plt.title("Optimal Transport")
plt.scatter(*zip(*MixtureSample), color = 'g', alpha = 0.2)
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.show()

end = time.time()
print(end - start)