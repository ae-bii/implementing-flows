import sys
sys.path.append("./")

import numpy as np
import random
import math
import numdifftools as nd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import family_of_f

random.seed(0)
np.random.seed(0)

e = math.e
pi = math.pi


# REVISED:
def BetaNewton(): # Newton's method (Experimental)
    xSummationGradient = [0, 0, 0, 0, 0]
    ySummationGradient = [0, 0, 0, 0, 0]
    G = [0, 0, 0, 0, 0]
    for i in range(0, len(MixtureSample)):
        xSummationGradient[0] += PotentialF_1.giulio_F(MixtureSample[i])
        xSummationGradient[1] += PotentialF_2.gaussian_F(MixtureSample[i])
        xSummationGradient[2] += PotentialF_3.multiquadric_F(MixtureSample[i])
        xSummationGradient[3] += PotentialF_4.inverseQuadratic_F(MixtureSample[i])
        xSummationGradient[4] += PotentialF_5.inverseMultiquadric_F(MixtureSample[i])
    for j in range(0, len(StandardNormal)):
        ySummationGradient[0] += PotentialF_1.giulio_F(StandardNormal[j])
        ySummationGradient[1] += PotentialF_2.gaussian_F(StandardNormal[j])
        ySummationGradient[2] += PotentialF_3.multiquadric_F(StandardNormal[j])
        ySummationGradient[3] += PotentialF_4.inverseQuadratic_F(StandardNormal[j])
        ySummationGradient[4] += PotentialF_5.inverseMultiquadric_F(StandardNormal[j])
    for k in range(0, 5):
        G[k] = (1/len(MixtureSample)) * xSummationGradient[k] - (1/len(StandardNormal)) * ySummationGradient[k]
    G = np.array(G)
    yHessian = np.zeros([5,5])
    for l in range(0, len(StandardNormal)):
        F_gradient = [nd.Gradient(PotentialF_1.giulio_F)([StandardNormal[l]]),
        nd.Gradient(PotentialF_2.gaussian_F)(StandardNormal[l]),
        nd.Gradient(PotentialF_3.multiquadric_F)(StandardNormal[l]),
        nd.Gradient(PotentialF_4.inverseQuadratic_F)(StandardNormal[l]),
        nd.Gradient(PotentialF_5.inverseMultiquadric_F)(StandardNormal[l])]
        for m in range(0, 5):
            for n in range(0, 5):
                yHessian[m][n] += np.dot(F_gradient[m], F_gradient[n])
    
    H = np.multiply(yHessian, 1/len(StandardNormal))
    HInverseNeg = (-1) * np.linalg.inv(H)
    Beta = np.matmul(HInverseNeg, G)
    LearningRate = 1 # Not sure how to choose this value
    ParameterList = [1, LearningRate/np.linalg.norm(Beta)]
    return Beta * min(ParameterList) # min(ParameterList) can be understood as similar to the "Proportion" in gradient descent

def u(x, Beta):
    return (((x[0] ** 2) + (x[1] ** 2)) / 2) + Beta[0] * PotentialF_1.giulio_F(x) + Beta[1] * PotentialF_2.gaussian_F(x) + Beta[2] * PotentialF_3.multiquadric_F(x) + Beta[3] * PotentialF_4.inverseQuadratic_F(x) + Beta[4] * PotentialF_5.inverseMultiquadric_F(x)

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
        xval = OldMixtureSample[i][0] + Beta[0] * nd.Gradient(PotentialF_1.giulio_F)(OldMixtureSample[i])[0] + Beta[1] * nd.Gradient(PotentialF_2.gaussian_F)(OldMixtureSample[i])[0] + Beta[2] * nd.Gradient(PotentialF_3.multiquadric_F)(OldMixtureSample[i])[0] + Beta[3] * nd.Gradient(PotentialF_4.inverseQuadratic_F)(OldMixtureSample[i])[0] + Beta[4] * nd.Gradient(PotentialF_5.inverseMultiquadric_F)(OldMixtureSample[i])[0]
        yval = OldMixtureSample[i][1] + Beta[0] * nd.Gradient(PotentialF_1.giulio_F)(OldMixtureSample[i])[1] + Beta[1] * nd.Gradient(PotentialF_2.gaussian_F)(OldMixtureSample[i])[1] + Beta[2] * nd.Gradient(PotentialF_3.multiquadric_F)(OldMixtureSample[i])[1] + Beta[3] * nd.Gradient(PotentialF_4.inverseQuadratic_F)(OldMixtureSample[i])[1] + Beta[4] * nd.Gradient(PotentialF_5.inverseMultiquadric_F)(OldMixtureSample[i])[1]
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

PotentialF_1 = family_of_f.PotentialF()
PotentialF_2 = family_of_f.PotentialF()
PotentialF_3 = family_of_f.PotentialF()
PotentialF_4 = family_of_f.PotentialF()
PotentialF_5 = family_of_f.PotentialF()


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
    for i in range(0,5):
        CenterList.append(CenterGeneratorList[random.randint(0, len(CenterGeneratorList) - 1)])
    PotentialF_1.set_center(CenterList[0])
    PotentialF_2.set_center(CenterList[1])
    PotentialF_3.set_center(CenterList[2])
    PotentialF_4.set_center(CenterList[3])
    PotentialF_5.set_center(CenterList[4])
    PotentialF_1.set_alpha(0.75)
    PotentialF_2.set_alpha(0.75)
    PotentialF_3.set_alpha(0.75)
    PotentialF_4.set_alpha(0.75)
    PotentialF_5.set_alpha(0.75)
    PotentialF_2.set_constant(1)
    PotentialF_3.set_constant(1)
    PotentialF_4.set_constant(1)
    PotentialF_5.set_constant(1)
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
