import sys
sys.path.append("../implementing-flows")

import numpy as np
import random
import math
import numdifftools as nd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import functions

random.seed(0)
np.random.seed(0)

e = math.e
pi = math.pi


# REVISED:
def BetaNewton(): # Newton's method (Experimental)
    xSummationGradient = np.zeros(NumFs)
    ySummationGradient = np.zeros(NumFs)
    G = np.zeros(NumFs)
    for i in range(0, len(MixtureSample)):
        for f in range(0,NumFs):
            xSummationGradient[f] += PotentialFs[f](MixtureSample[i])
    for j in range(0, len(StandardNormal)):
        for f in range(0,NumFs):
            ySummationGradient[f] += PotentialFs[f](StandardNormal[i])
    for k in range(0, NumFs):
        G[k] = (1/len(MixtureSample)) * xSummationGradient[k] - (1/len(StandardNormal)) * ySummationGradient[k]
    G = np.array(G)
    yHessian = np.zeros([NumFs,NumFs])
    for l in range(0, len(StandardNormal)):
        F_gradient = []
        for f in range(0,NumFs):
            F_gradient.append(nd.Gradient(PotentialFs[f])(StandardNormal[l]))
        for m in range(0, NumFs):
            for n in range(0, NumFs):
                yHessian[m][n] += np.dot(F_gradient[m], F_gradient[n])
    
    H = np.multiply(yHessian, 1/len(StandardNormal))
    HInverseNeg = (-1) * np.linalg.inv(H)
    Beta = np.matmul(HInverseNeg, G)
    LearningRate = 0.5 # Not sure how to choose this value
    ParameterList = [1, LearningRate/np.linalg.norm(Beta)]
    return Beta * min(ParameterList) # min(ParameterList) can be understood as similar to the "Proportion" in gradient descent

def u(x, Beta):
    F_eval = []
    for f in range(0,NumFs):
        F_eval.append(PotentialFs[f](x))
    return (((x[0] ** 2) + (x[1] ** 2)) / 2) + np.dot(Beta, F_eval)

def uConjugate(y, Beta):
    ConvexCandidate = []
    for i in range(0, len(MixtureSample)):
        ConjugateValue = np.dot(MixtureSample[i], y) - u(MixtureSample[i], Beta)
        ConvexCandidate.append(ConjugateValue)
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


PotentialFs = [functions.Giulio_F(alpha=0.75),
                functions.Gaussian_F(alpha=0.75, constant=1),
                functions.Multiquadric_F(alpha=0.75, constant=1),
                functions.InverseQuadratic_F(alpha=0.75, constant=1),
                functions.InverseMultiquadric_F(alpha=0.75, constant=1)]
NumFs = len(PotentialFs)

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
    CenterList = []
    for i in range(0,NumFs):
        c = CenterGeneratorList[random.randint(0, len(CenterGeneratorList) - 1)]
        CenterList.append(c)
        PotentialFs[i].setCenter(c)
    Beta = BetaNewton()
    OldD = DValue
    DValue = D(Beta)
    print(DValue)
    MixtureSample = SamplesUpdate(MixtureSample)
    if abs(DValue - OldD) < 0.0001 or Iteration > 10:
        break

plt.subplot(1,3,2)
plt.title("Optimal Transport")
plt.scatter(*zip(*MixtureSample), color = 'g', alpha = 0.2)
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.show()
