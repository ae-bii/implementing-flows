import numpy as np
import random
import math
import numdifftools as nd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from family_of_f import PotentialF

random.seed(0)
np.random.seed(0)

e = math.e
pi = math.pi

def distance(z1, z2):
    sum = 0
    for i in range(0,len(z1)-1):
        sum += (z1[i] - z2[i]) ** 2
    return math.sqrt(sum)

def F_1(z):
    r = distance(z, center)
    alpha = 1.5 # Consistent with the density (As this gets larger, less samples are moved close to 0)
    return r * math.erf(r/alpha) + (alpha/math.sqrt(pi)) * math.pow(e, -(r/alpha) ** 2)

def F_2(z):
    r = distance(z, center)
    alpha = 1.5
    return alpha + r - alpha * math.log(abs(alpha + r))

def BetaCalculation(F):
    Proportion = 0.5
    xSummationDerivative = 0
    ySummationDerivative = 0
    for j in range(0, len(StandardNormal)):
        ySummationDerivative += F(StandardNormal[j])
    for i in range(0, len(MixtureSample)):
        xSummationDerivative += F(MixtureSample[i])
    Beta = (-1/len(MixtureSample)) * xSummationDerivative + \
        (1/len(StandardNormal)) * ySummationDerivative
    return Beta * Proportion

def u(x, Beta_1, Beta_2):
    return (((x[0] ** 2) + (x[1] ** 2)) / 2) + Beta_1 * F_1(x) + Beta_2 * F_2(x)

def uConjugate(y, Beta_1, Beta_2):
    ConvexCandidate = []
    for i in range(0, len(MixtureSample)):
        ConvexCandidate.append(np.dot(MixtureSample[i], y) - u(MixtureSample[i], Beta_1, Beta_2))
    return max(ConvexCandidate)

def D(Beta_1, Beta_2):
    xSummation = 0
    ySummation = 0
    for i in range(0, len(MixtureSample)):
        xSummation += u(MixtureSample[i], Beta_1, Beta_2)
    for j in range(0, len(StandardNormal)):
        ySummation += uConjugate(StandardNormal[j], Beta_1, Beta_2)

    LL = 1/len(MixtureSample) * xSummation + 1 / \
        len(StandardNormal) * ySummation

    return LL



def SamplesUpdate(OldMixtureSample):
    NewMixtureSample = []
    for i in range(0, len(OldMixtureSample)):
        xval = OldMixtureSample[i][0] + Beta_1 * nd.Gradient(F_1)(OldMixtureSample[i])[0] + Beta_2 * nd.Gradient(F_2)(OldMixtureSample[i])[0]
        yval = OldMixtureSample[i][1] + Beta_1 * nd.Gradient(F_1)(OldMixtureSample[i])[1] + Beta_2 * nd.Gradient(F_2)(OldMixtureSample[i])[1]
        NewMixtureSample.append([xval,yval])
    NewMixtureSample = np.array(NewMixtureSample)
    return NewMixtureSample

def MixtureSampleGenerator():
    mean1 = [1, -1]
    cov1 = [[0.5, 0], [0, 0.5]]
    mean2 = [-1, 1]
    cov2 = [[0.5, 0], [0, 0.5]]
    x = np.random.multivariate_normal(mean1, cov1, 500)
    y = np.random.multivariate_normal(mean2, cov2, 500)
    MixtureSample = []
    for i in range(500):
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
MixtureSample = MixtureSampleGenerator()
StandardNormal = StandardNormalGenerator()
CenterGeneratorList = MixtureSample + StandardNormal

plt.subplot(1,3,3)
plt.title("Target")
plt.scatter(*zip(*StandardNormal), color = 'r', alpha = 0.2)
plt.xlim(-4, 4)
plt.ylim(-4, 4)

plt.subplot(1,3,1)
plt.title("Initial")
plt.scatter(*zip(*MixtureSample), color = 'b', alpha = 0.2)
plt.xlim(-4, 4)
plt.ylim(-4, 4)

DValue = 0
while True: # Maybe there is a problem of overfitting
    #print("Iteration " + str(i))
    center = CenterGeneratorList[random.randint(0, len(CenterGeneratorList) - 1)]
    Beta_1 = BetaCalculation(F_1)
    Beta_2 = BetaCalculation(F_2)
    OldD = DValue
    DValue = D(Beta_1, Beta_2)
    print(DValue)
    MixtureSample = SamplesUpdate(MixtureSample)
    if abs(DValue - OldD) < 0.001:
        break

plt.subplot(1,3,2)
plt.title("Optimal Transport")
plt.scatter(*zip(*MixtureSample), color = 'g', alpha = 0.2)
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.show()
