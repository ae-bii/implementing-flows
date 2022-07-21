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

def F_1(z):
    r = distance(z, center_1)
    alpha = 0.75 # Consistent with the density (As this gets larger, less samples are moved close to 0)
    return r * math.erf(r/alpha) + (alpha/math.sqrt(pi)) * math.pow(e, -(r/alpha) ** 2)

def F_2(z):
    r = distance(z, center_2)
    alpha = 0.75
    return alpha + r - alpha * math.log(abs(alpha + r))

def BetaNewton(): # Newton's method (Experimental)
    xSummationGradient_1 = 0
    ySummationGradient_1 = 0
    xSummationGradient_2 = 0
    ySummationGradient_2 = 0
    for i in range(0, len(MixtureSample)):
        xSummationGradient_1 += F_1(MixtureSample[i])
        xSummationGradient_2 += F_2(MixtureSample[i])
    for j in range(0, len(StandardNormal)):
        ySummationGradient_1 += F_1(StandardNormal[j])
        ySummationGradient_2 += F_2(StandardNormal[j])
    G_1 = (1/len(MixtureSample)) * xSummationGradient_1 - (1/len(StandardNormal)) * ySummationGradient_1
    G_2 = (1/len(MixtureSample)) * xSummationGradient_2 - (1/len(StandardNormal)) * ySummationGradient_2
    G = np.array([G_1, G_2])
    yHessian_11 = 0
    yHessian_12 = 0
    yHessian_21 = 0
    yHessian_22 = 0
    for l in range(0, len(StandardNormal)):
        yHessian_11 += np.dot(nd.Gradient(F_1)([StandardNormal[l]]), nd.Gradient(F_1)(StandardNormal[l]))
        yHessian_12 += np.dot(nd.Gradient(F_1)([StandardNormal[l]]), nd.Gradient(F_2)(StandardNormal[l]))
        yHessian_21 += np.dot(nd.Gradient(F_2)([StandardNormal[l]]), nd.Gradient(F_1)(StandardNormal[l]))
        yHessian_22 += np.dot(nd.Gradient(F_2)([StandardNormal[l]]), nd.Gradient(F_2)(StandardNormal[l]))
    H_11 = (1/len(StandardNormal)) * yHessian_11
    H_12 = (1/len(StandardNormal)) * yHessian_12
    H_21 = (1/len(StandardNormal)) * yHessian_21
    H_22 = (1/len(StandardNormal)) * yHessian_22
    H = np.array([[H_11,H_12],
                 [H_21,H_22]])
    HInverseNeg = (-1) * np.linalg.inv(H)
    Beta = np.matmul(HInverseNeg, G)
    LearningRate = 0.01 # Not sure how to choose this value
    ParameterList = [1, LearningRate/np.linalg.norm(Beta)]
    return Beta * min(ParameterList) # min(ParameterList) can be understood as similar to the "Proportion" in gradient descent

def u(x, Beta_1, Beta_2):
    return (((x[0] ** 2) + (x[1] ** 2)) / 2) + Beta_1 * F_1(x) + Beta_2 * F_2(x)

def uConjugate(y, Beta_1, Beta_2):
    ConvexCandidate = []
    for i in range(0, len(MixtureSample)):
        ConjugateValue = np.dot(MixtureSample[i], y) - u(MixtureSample[i], Beta_1, Beta_2)
        ConvexCandidate.append(ConjugateValue)
    return max(ConvexCandidate)

def D(Beta):
    xSummation = 0
    ySummation = 0
    for i in range(0, len(MixtureSample)):
        xSummation += u(MixtureSample[i], Beta[0], Beta[1])
    for j in range(0, len(StandardNormal)):
        ySummation += uConjugate(StandardNormal[j], Beta[0], Beta[1])

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
Iteration = 0
while True: # Maybe there is a problem of overfitting
    #print("Iteration " + str(i))
    Iteration += 1
    center_1 = CenterGeneratorList[random.randint(0, len(CenterGeneratorList) - 1)]
    center_2 = CenterGeneratorList[random.randint(0, len(CenterGeneratorList) - 1)]
    while Iteration <= 5 and distance(center_1, center_2) <= 0.75:
        center_1 = MixtureSample[random.randint(0, len(MixtureSample) - 1)]
        center_2 = MixtureSample[random.randint(0, len(MixtureSample) - 1)]
    while distance(center_1, center_2) <= 0.75:
        center_1 = CenterGeneratorList[random.randint(0, len(CenterGeneratorList) - 1)]
    Beta = BetaNewton()
    print(Beta)
    Beta_1 = Beta[0]
    Beta_2 = Beta[1]
    OldD = DValue
    DValue = D(Beta)
    print(DValue)
    MixtureSample = SamplesUpdate(MixtureSample)
    if abs(DValue - OldD) < 0.0001 or Iteration > 25:
        break

plt.subplot(1,3,2)
plt.title("Optimal Transport")
plt.scatter(*zip(*MixtureSample), color = 'g', alpha = 0.2)
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.show()
