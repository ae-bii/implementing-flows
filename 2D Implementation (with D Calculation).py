
import numpy as np
import random
import math
import numdifftools
import matplotlib.pyplot as plt
import matplotlib.animation as animation


e = math.e
pi = math.pi

def F(z):
    rx = abs(z[0] - center[0])
    ry = abs(z[1] - center[1])# Random point in the cluster
    alpha = 1.5 # Consistent with the density (As this gets larger, less samples are moved close to 0)
    xval = rx * math.erf(rx/alpha) + (alpha/math.sqrt(pi)) * math.pow(e, -(rx/alpha) ** 2)
    yval = ry * math.erf(ry/alpha) + (alpha/math.sqrt(pi)) * math.pow(e, -(ry/alpha) ** 2)
    return [xval, yval]

def F_1(z):
    r = abs(z - center[0])# Random point in the cluster
    alpha = 1.5 # Consistent with the density (As this gets larger, less samples are moved close to 0)
    return r * math.erf(r/alpha) + (alpha/math.sqrt(pi)) * math.pow(e, -(r/alpha) ** 2)

def F_2(z):
    r = abs(z - center[1])# Random point in the cluster
    alpha = 1.5 # Consistent with the density (As this gets larger, less samples are moved close to 0)
    return r * math.erf(r/alpha) + (alpha/math.sqrt(pi)) * math.pow(e, -(r/alpha) ** 2)

def BetaCalculation():
    Proportion = 0.5
    xSummationDerivative = [0, 0]
    ySummationDerivative = [0, 0]
    Beta = []
    for j in range(0, len(StandardNormal)):
        #CurTuple = F(StandardNormal[j])
        ySummationDerivative[0] += F(StandardNormal[j])[0]
        ySummationDerivative[1] += F(StandardNormal[j])[1]
    for i in range(0, len(MixtureSample)):
        xSummationDerivative[0] += F(MixtureSample[i])[0]
        xSummationDerivative[1] += F(MixtureSample[i])[1]
    Beta.append((-1/len(MixtureSample)) * xSummationDerivative[0] + \
        (1/len(StandardNormal)) * ySummationDerivative[0])
    Beta.append((-1/len(MixtureSample)) * xSummationDerivative[1] + \
        (1/len(StandardNormal)) * ySummationDerivative[1])
    return np.multiply(Beta, Proportion)

def u(x, Beta):
    xval = (x[0] ** 2 / 2) + Beta[0] * F(x)[0] + Beta[0] * F(x)[0]
    yval = (x[1] ** 2 / 2) + Beta[1] * F(x)[1] + Beta[0] * F(x)[1]
    return (xval, yval)

def uConjugate(y, Beta):
    ConvexCandidate = []
    for i in range(0, len(MixtureSample)):
        ConjugateVector = np.subtract(np.dot(MixtureSample[i], y), u(MixtureSample[i], Beta))
        ConvexCandidate.append(ConjugateVector)
    SumList = []
    for i in range(len(ConvexCandidate)):
        SumList.append(np.linalg.norm(ConvexCandidate[i]))
    index = SumList.index(max(SumList))
    return ConvexCandidate[index]

def D(Beta): # Experimental
    xvalSummationOfx = 0
    yvalSummationOfx = 0
    xvalSummationOfy = 0
    yvalSummationOfy = 0
    for i in range(0, len(MixtureSample) - 1):
        xvalSummationOfx += u(MixtureSample[i], Beta)[0]
        yvalSummationOfx += u(MixtureSample[i], Beta)[1]
    for j in range(0, len(StandardNormal) - 1):
        xvalSummationOfy += uConjugate(StandardNormal[j], Beta)[0]
        yvalSummationOfy += uConjugate(StandardNormal[j], Beta)[1]
    D = np.linalg.norm([(1/len(MixtureSample)) * xvalSummationOfx + (1/len(StandardNormal)) * xvalSummationOfy, (1/len(MixtureSample)) * yvalSummationOfx + (1/len(StandardNormal)) * yvalSummationOfy])
    return D



def SamplesUpdate(OldMixtureSample):
    NewMixtureSample = []
    for i in range(0, len(OldMixtureSample)):
        xval = OldMixtureSample[i][0] + Beta[0] * numdifftools.Gradient(F_1)([OldMixtureSample[i][0]])
        yval = OldMixtureSample[i][1] + Beta[1] * numdifftools.Gradient(F_2)([OldMixtureSample[i][1]])
        NewMixtureSample.append([xval, yval])
    NewMixtureSample = np.array(NewMixtureSample)
    return NewMixtureSample

def MixtureSampleGenerator():
    mean1 = [1, -1]
    cov1 = [[0.5, 0], [0, 0.5]]
    mean2 = [-1, 1]
    cov2 = [[0.5, 0], [0, 0.5]]
    x = np.random.multivariate_normal(mean1, cov1, 100)
    y = np.random.multivariate_normal(mean2, cov2, 100)
    MixtureSample = []
    for i in range(100):
        RandomSelector = random.random()
        if RandomSelector > 0.7:
            MixtureSample.append(x[i])
        else:
            MixtureSample.append(y[i])
    MixtureSample = np.array(MixtureSample)
    return MixtureSample

def StandardNormalGenerator():
    Sample = []
    x = np.random.standard_normal(100)
    y = np.random.standard_normal(100)
    for i in range(100):
        Sample.append([x[i], y[i]])
    return Sample


#--------------------------------------------------------------------- TESTING ------------------------------------------------------------------
MixtureSample = MixtureSampleGenerator()
StandardNormal = StandardNormalGenerator()

plt.subplot(1,2,1)
plt.scatter(*zip(*StandardNormal), color = 'r', alpha = 0.2)
plt.xlim(-4, 4)
plt.ylim(-4, 4)

plt.subplot(1,2,2)
plt.scatter(*zip(*MixtureSample), color = 'b', alpha = 0.2)

for i in range(20): # Maybe there is a problem of overfitting
    center = StandardNormal[random.randint(0, len(StandardNormal) - 1)]
    Beta = BetaCalculation()
    DValue = D(Beta)
    print(DValue)
    MixtureSample = SamplesUpdate(MixtureSample)

plt.scatter(*zip(*MixtureSample), color = 'g', alpha = 0.2)
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.show()