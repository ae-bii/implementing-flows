import sys
sys.path.append("./")

import numpy as np
import time
import random
import math
import numdifftools as nd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import functions
import SampleGeneratorND

random.seed(0)
np.random.seed(0)

e = math.e
pi = math.pi

def distance(z1, z2):
    return np.sqrt(sum(np.square(np.subtract(z1,z2))))

def norm(i):
    return np.sqrt(sum(np.square(i)))

def GradientApprox(VariableList):
    Gradient = []
    delta = 1e-8
    for f in range(NumFs):
        grad = []
        for i in range(dim):
            temp = np.zeros(dim)
            temp[i] = delta/2
            grad.append(((PotentialFsVectorized[f](VariableList + temp) - (PotentialFsVectorized[f](VariableList - temp)))/delta))
        Gradient.append(np.squeeze(np.transpose(grad)))

    return Gradient


def BetaNewton(): # Newton's method (Experimental)
    xSummationGradient = np.zeros(NumFs)
    ySummationGradient = np.zeros(NumFs)
    G = np.zeros(NumFs)
    xSummationGradient = [sum(PotentialFsVectorized[f](Initial)) for f in range(NumFs)]
    ySummationGradient = [sum(PotentialFsVectorized[f](Target)) for f in range(NumFs)]
    G = [(1/len(Initial)) * xSummationGradient[k] - (1/len(Target)) * ySummationGradient[k] for k in range(NumFs)]
    G = np.array(G)
    yHessian = np.zeros([NumFs,NumFs])
    F_gradient = GradientApprox(Target)
    for m in range(0, NumFs):
        for n in range(0, NumFs):
            yHessian[m][n] = sum((F_gradient[m]*F_gradient[n]).sum(axis = 1))

    H = np.multiply(yHessian, 1/len(Target))
    HInverseNeg = (-1) * np.linalg.inv(H)
    Beta = np.matmul(HInverseNeg, G)
    LearningRate = 0.5 # Not sure how to choose this value
    ParameterList = [1, LearningRate/norm(Beta)]
    return Beta * min(ParameterList)

def u(x, Beta):
    F_eval = [(PotentialFs[f](x)) for f in range(NumFs)]
    return (np.dot(x,x) / 2) + np.dot(Beta, F_eval)

def uVectorized(x):
    F_eval = [PotentialFsVectorized[f](x) for f in range(NumFs)]
    return np.array([np.dot(x[z],x[z]) for z in range(len(x))])/2 + (np.transpose(F_eval) * Beta).sum(axis = 1)

def uConjugate(y, Beta):
    temp = Initial * y
    sum = np.zeros(len(temp))
    for i in range(dim):
        sum += temp[:,i]
    ConvexVector = sum - uVectorized(Initial)
    return max(ConvexVector)

def uConjugateVec(y):
    ConvexMatrix = np.array(np.matmul(y, np.transpose(Initial))) - uVectorized(Initial)
    return ConvexMatrix.max(axis = 1)

def D():
    xSummation = sum(uVectorized(Initial))
    ySummation = sum(uConjugateVec(Target))

    LL = 1/len(Initial) * xSummation + 1 / \
        len(Target) * ySummation
    return LL

def SamplesUpdate(OldMixtureSample):
    NewMixtureSample = []
    F_eval = [[None] * NumFs for i in range(dim)]
    for f in range(0,NumFs):
        gradient = GradientApprox(OldMixtureSample)[f]
        for i in range(dim):
            F_eval[i][f] = (gradient[:,i])
    F_eval = np.array(F_eval)
    vals = [OldMixtureSample[:,i] + (np.multiply(np.transpose(F_eval[i]), Beta)).sum(axis = 1) for i in range(dim)]
    NewMixtureSample = np.array(vals)
    NewMixtureSample = np.transpose(NewMixtureSample)

    return NewMixtureSample

def MixtureSampleGenerator():
    mean1 = []
    mean2 = []
    mean3 = []
    for i in range(dim):
        mean1.append(random.random()*0.25)
        mean2.append(random.random()*0.25)
        mean3.append(random.random()*0.25)
    mean1, mean2, mean3 = np.array(mean1), np.array(mean2), np.array(mean3)
    cov = np.array([0.5]*dim)
    cov = np.diag(cov**dim)
    x = np.random.multivariate_normal(mean1, cov, len(Target))
    y = np.random.multivariate_normal(mean2, cov, len(Target))
    z = np.random.multivariate_normal(mean3, cov, len(Target))
    MixtureSample = []
    for i in range(len(Target)):
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
        Normals.append(np.random.standard_normal(500))
    for j in range(500):
        cur = []
        for k in range(dim):
            cur.append(Normals[k][j])
        Sample.append(np.array(cur))
    return np.array(Sample)


#------------------------------------------------------------------ TESTING ------------------------------------------------------------
dim = 3

# Testing and Plot:
Target = SampleGeneratorND.JointSampleGenerator()
Initial = SampleGeneratorND.IndependentCouplingGenerator(Target, 1000)
CenterGeneratorList = Target

PotentialFs = [functions.Giulio_F(),
                functions.Gaussian_F(),
                functions.Multiquadric_F(),
                functions.InverseQuadratic_F(),
                functions.InverseMultiquadric_F(),
                functions.PolyharmonicSpline_F(),
                functions.ThinPlateSpline_F()]

PotentialFsVectorized = [functions.Giulio_F_Vectorized(),
                        functions.Gaussian_F_Vectorized(),
                        functions.Multiquadric_F_Vectorized(),
                        functions.InverseQuadratic_F_Vectorized(),
                        functions.InverseMultiquadric_F_Vectorized(),
                        functions.PolyharmonicSpline_F_Vectorized(),
                        functions.ThinPlateSpline_F_Vectorized()]
NumFs = len(PotentialFsVectorized)


fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(1, 3, 3, projection='3d')
plt.title("Target")
ax.scatter3D(*zip(*Target), color = 'r', alpha = 0.2)
ax.set_xlim3d(-2,2)
ax.set_ylim3d(-2,2)
ax.set_zlim3d(-2,2)

ax = fig.add_subplot(1, 3, 1, projection='3d')
plt.title("Inital")
ax.scatter3D(*zip(*Initial), color = 'r', alpha = 0.2)
ax.set_xlim3d(-2,2)
ax.set_ylim3d(-2,2)
ax.set_zlim3d(-2,2)

DValue = 0
Iteration = 0
Beta = 0

# Converts time in seconds to hours, minutes, seconds
# def time_convert(sec):
#   mins = sec // 60
#   sec = sec % 60
#   hours = mins // 60
#   mins = mins % 60
#   print("Time Lapsed = {0}:{1}:{2}".format(int(hours),int(mins),sec))

# start_time = time.time()

for i in range(500): # Maybe there is a problem of overfitting
    Iteration += 1
    if Iteration >= 10:
        CenterGeneratorList = np.concatenate((Initial,Target), axis=0)
    CenterList = []
    
    for i in range(0,NumFs):
        c = CenterGeneratorList[random.randint(0, len(CenterGeneratorList) - 1)]
        CenterList.append(c + 1e-5)
        PotentialFsVectorized[i].setCenter(c + 1e-5)
    OldBeta = Beta
    Beta = BetaNewton()
    OldD = DValue
    DValue = D()
    print(DValue)
    Initial = SamplesUpdate(Initial)

# end_time = time.time()

# time_convert(end_time-start_time)

ax = fig.add_subplot(1, 3, 2, projection='3d')
plt.title("Optimal Transport")
ax.scatter3D(*zip(*Initial), color = 'r', alpha = 0.2)
ax.set_xlim3d(-2,2)
ax.set_ylim3d(-2,2)
ax.set_zlim3d(-2,2)
plt.show()


