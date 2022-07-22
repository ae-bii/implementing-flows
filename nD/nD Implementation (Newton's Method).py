import sys
sys.path.append("./")

import numpy as np
import time
import random
import math
import numdifftools as nd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import family_of_f
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

random.seed(0)
np.random.seed(0)

e = math.e
pi = math.pi

def distance(z1, z2):
    sum = 0
    for i in range(0,len(z1)-1):
        sum += (z1[i] - z2[i]) ** 2
    return math.sqrt(sum)


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
    LearningRate = 0.01 # Not sure how to choose this value
    ParameterList = [1, LearningRate/np.linalg.norm(Beta)]
    return Beta * min(ParameterList) # min(ParameterList) can be understood as similar to the "Proportion" in gradient descent

def u(x, Beta):
    return np.dot(x, x)/2 + Beta[0] * PotentialF_1.giulio_F(x) + Beta[1] * PotentialF_2.gaussian_F(x) + Beta[2] * PotentialF_3.multiquadric_F(x) + Beta[3] * PotentialF_4.inverseQuadratic_F(x) + Beta[4] * PotentialF_5.inverseMultiquadric_F(x)

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
        cur = []
        for j in range(dim):
            cur.append(OldMixtureSample[i][j] + Beta[0] * nd.Gradient(PotentialF_1.giulio_F)(OldMixtureSample[i])[j] + Beta[1] * nd.Gradient(PotentialF_2.gaussian_F)(OldMixtureSample[i])[j] + Beta[2] * nd.Gradient(PotentialF_3.multiquadric_F)(OldMixtureSample[i])[j] + Beta[3] * nd.Gradient(PotentialF_4.inverseQuadratic_F)(OldMixtureSample[i])[j] + Beta[4] * nd.Gradient(PotentialF_5.inverseMultiquadric_F)(OldMixtureSample[i])[j])
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


#------------------------------------------------------------------ TESTING ------------------------------------------------------------
dim = 3

# Testing and Plot:
MixtureSample = MixtureSampleGenerator()
StandardNormal = StandardNormalGenerator()
CenterGeneratorList = MixtureSample + StandardNormal

PotentialF_1 = family_of_f.PotentialF()
PotentialF_2 = family_of_f.PotentialF()
PotentialF_3 = family_of_f.PotentialF()
PotentialF_4 = family_of_f.PotentialF()
PotentialF_5 = family_of_f.PotentialF()

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(1, 2, 1, projection='3d')
plt.title("Target")
ax.scatter3D(*zip(*StandardNormal), color = 'r', alpha = 0.2)

ax = fig.add_subplot(1, 2, 2, projection='3d')
plt.title("Inital")
ax.scatter3D(*zip(*MixtureSample), color = 'r', alpha = 0.2)

DValue = 0
Iteration = 0

# Converts time in seconds to hours, minutes, seconds
def time_convert(sec):
  mins = sec // 60
  sec = sec % 60
  hours = mins // 60
  mins = mins % 60
  print("Time Lapsed = {0}:{1}:{2}".format(int(hours),int(mins),sec))

start_time = time.time()

for i in range(0, 10): # Maybe there is a problem of overfitting
    Iteration += 1
    CenterList = []
    for j in range(0,5):
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

end_time = time.time()

time_convert(end_time-start_time)

plt.title("Optimal Transport")
ax.scatter3D(*zip(*MixtureSample), color = 'r', alpha = 0.2)

plt.show()


