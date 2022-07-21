import numpy as np
import random
import math
import numdifftools as nd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

random.seed(0)
np.random.seed(0)

e = math.e
pi = math.pi

def distance(z1, z2):
    sum = 0
    for i in range(0,len(z1)-1):
        sum += (z1[i] - z2[i]) ** 2
    return math.sqrt(sum)

def F_0(z):
    r = distance(z, center0)
    alpha = 1.5 # Consistent with the density (As this gets larger, less samples are moved close to 0)
    return r * math.erf(r/alpha) + (alpha/math.sqrt(pi)) * math.pow(e, -(r/alpha) ** 2)

def F_1(z):
    r = distance(z, center1)
    alpha = 1.5
    return alpha + r - alpha * math.log(abs(alpha + r))

def BetaCalculation():
    Proportion = 0.5
    xSummationDerivative = [0, 0]
    ySummationDerivative = [0, 0]
    Beta = [0, 0]
    for j in range(0, len(StandardNormal)):            
        ySummationDerivative[0] += F_0(StandardNormal[j])
        ySummationDerivative[1] += F_1(StandardNormal[j])
        #ySummationDerivative[2] += F_2(StandardNormal[i])
    for i in range(0, len(MixtureSample)):
        xSummationDerivative[0] += F_0(MixtureSample[i])
        xSummationDerivative[1] += F_1(MixtureSample[i])
        #xSummationDerivative[2] += F_2(MixtureSample[i])
    Beta[0] = (-1/len(MixtureSample)) * xSummationDerivative[0] + \
        (1/len(StandardNormal)) * ySummationDerivative[0]
    Beta[1] = (-1/len(MixtureSample)) * xSummationDerivative[1] + \
        (1/len(StandardNormal)) * ySummationDerivative[1]
    return np.multiply(Beta, Proportion)

def u(x, Beta):
    return np.dot(x, x)/2 + Beta[0] * F_0(x) + Beta[1] * F_1(x)

def uConjugate(y, Beta):
    ConvexCandidate = []
    for i in range(0, len(MixtureSample)):
        ConvexCandidate.append(np.dot(MixtureSample[i], y) - u(MixtureSample[i], Beta))
    return max(ConvexCandidate)

def D(Beta):
    xSummation = 0
    ySummation = 0
    for i in range(0, len(MixtureSample)):
        xSummation += u(MixtureSample[i], Beta)
    for j in range(0, len(StandardNormal)):
        ySummation += uConjugate(StandardNormal[j], Beta)
    LL = 1/len(MixtureSample) * xSummation + 1 / \
        len(StandardNormal) * ySummation
    return LL

def SamplesUpdate(OldMixtureSample):
    NewMixtureSample = []
    for i in range(0, len(OldMixtureSample)):
        cur = []
        for j in range(dim):
            cur.append(OldMixtureSample[i][j] + Beta[0] * nd.Gradient(F_0)(OldMixtureSample[i])[j] + Beta[1] * nd.Gradient(F_1)(OldMixtureSample[i])[j])
        NewMixtureSample.append(np.array(cur))
    NewMixtureSample = np.array(NewMixtureSample)
    return NewMixtureSample

def MixtureSampleGenerator():
    mean1 = []
    mean2 = []
    mean3 = []
    for i in range(dim):
        mean1.append(random.random()*2)
        mean2.append(random.random()*2)
        mean3.append(random.random()*2)
    mean1, mean2, mean3 = np.array(mean1), np.array(mean2), np.array(mean3)
    cov = np.array([0.5]*dim)
    cov = np.diag(cov**dim)
    x = np.random.multivariate_normal(mean1, cov, 200)
    y = np.random.multivariate_normal(mean2, cov, 200)
    z = np.random.multivariate_normal(mean3, cov, 200)
    MixtureSample = []
    for i in range(200):
        RandomSelector = random.random()
        if RandomSelector > 0.7:
            MixtureSample.append(x[i])
        elif RandomSelector < 0.3:
            MixtureSample.append(z[i])
        else:
            MixtureSample.append(y[i])
    MixtureSample = np.array(MixtureSample)
    return MixtureSample

def StandardNormalGenerator():
    Sample = []
    Normals = []
    for i in range(dim):
        Normals.append(np.random.standard_normal(200))
    for j in range(200):
        cur = []
        for k in range(dim):
            cur.append(Normals[k][j])
        Sample.append(np.array(cur))
    return np.array(Sample)

dim = 3

MixtureSample = MixtureSampleGenerator()
StandardNormal = StandardNormalGenerator()
CenterGeneratorList = MixtureSample + StandardNormal

DValue = 0
iteration = 0
Proportion = 0.5
while True:
    while TooClose == 1:
        center0 = CenterGeneratorList[random.randint(0, len(CenterGeneratorList) - 1)]
        center1 = CenterGeneratorList[random.randint(0, len(CenterGeneratorList) - 1)]
        if distance(center0, center1) >= 0.75:
            TooClose = 0
    Beta = BetaCalculation()
    OldD = DValue
    DValue = D(Beta)
    print(DValue)
    MixtureSample = SamplesUpdate(MixtureSample)
    iteration += 1
    if abs(DValue - OldD) < 0.0001 or iteration > 30:
        break
