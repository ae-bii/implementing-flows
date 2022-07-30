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


# REVISED:
def BetaNewton(): # Newton's method (Experimental)
    xSummationGradient = np.zeros(NumFs)
    ySummationGradient = np.zeros(NumFs)
    G = np.zeros(NumFs)
    xSummationGradient = [sum(PotentialFsVectorized[f](MixtureSample)) for f in range(NumFs)]
    ySummationGradient = [sum(PotentialFsVectorized[f](CrescentSample)) for f in range(NumFs)]
    G = [(1/len(MixtureSample)) * xSummationGradient[k] - (1/len(CrescentSample)) * ySummationGradient[k] for k in range(NumFs)]
    G = np.array(G)
    yHessian = np.zeros([NumFs,NumFs])
    F_gradient = [np.apply_along_axis(nd.Gradient(PotentialFs[f]),1,CrescentSample) for f in range(NumFs)]
    for m in range(0, NumFs):
        for n in range(0, NumFs):
            yHessian[m][n] = sum(np.apply_along_axis(sum,1,F_gradient[m]*F_gradient[n]))
    
    H = np.multiply(yHessian, 1/len(CrescentSample))
    HInverseNeg = (-1) * np.linalg.inv(H)
    Beta = np.matmul(HInverseNeg, G)
    LearningRate = 0.5 # Not sure how to choose this value
    ParameterList = [1, LearningRate/norm(Beta)]
    return Beta * min(ParameterList) # min(ParameterList) can be understood as similar to the "Proportion" in gradient descent

def u(x):
    F_eval = [(PotentialFs[f](x)) for f in range(NumFs)]
    return (((x[0] ** 2) + (x[1] ** 2)) / 2) + np.dot(Beta, F_eval)

def uVectorized(x):
    F_eval = [PotentialFsVectorized[f](x) for f in range(NumFs)]
    return ((np.square(x[:,0])) + (np.square(x[:,1])))/2 + (np.transpose(F_eval) * Beta).sum(axis = 1)

def uConjugate(y):
    ConvexVector = ((MixtureSample * y)[:,0] + (MixtureSample * y)[:,1]) - uVectorized(MixtureSample)
    return max(ConvexVector)

def D():

    xSummation = sum(uVectorized(MixtureSample))
    ySummation = sum(np.apply_along_axis(uConjugate, 1, CrescentSample))

    LL = 1/len(MixtureSample) * xSummation + 1 / \
        len(CrescentSample) * ySummation

    return LL

def SamplesUpdate(OldMixtureSample):
    NewMixtureSample = []
    F_eval_x = [0,0,0,0,0]
    F_eval_y = [0,0,0,0,0]
    for f in range(0,NumFs):
        gradient = np.apply_along_axis(nd.Gradient(PotentialFs[f]),1,(OldMixtureSample))
        F_eval_x[f] = (gradient[:,0])
        F_eval_y[f] = (gradient[:,1])
    F_eval_x = np.array(F_eval_x)
    F_eval_y= np.array(F_eval_y)

    xVal = OldMixtureSample[:,0] + (np.multiply(np.transpose(F_eval_x), Beta)).sum(axis = 1)
    yVal = OldMixtureSample[:,1] + (np.multiply(np.transpose(F_eval_y), Beta)).sum(axis = 1)
    NewMixtureSample = np.array([xVal, yVal])
    NewMixtureSample = np.transpose(NewMixtureSample)

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
CrescentSample = np.loadtxt("implementing-flows/SampleMoon.csv", delimiter=",")
CenterGeneratorList = CrescentSample


PotentialFs = [functions.Giulio_F(alpha=1),
                functions.Gaussian_F(alpha=1, constant=1),
                functions.Multiquadric_F(alpha=1, constant=1),
                functions.InverseQuadratic_F(alpha=1, constant=1),
                functions.InverseMultiquadric_F(alpha=1, constant=1)]
NumFs = len(PotentialFs)
PotentialFsVectorized = [functions.Giulio_F_Vectorized(alpha = 1),
                        functions.Gaussian_F_Vectorized(alpha=1, constant=1),
                        functions.Multiquadric_F_Vectorized(alpha=1, constant=1),
                        functions.InverseQuadratic_F_Vectorized(alpha=1, constant=1),
                        functions.InverseMultiquadric_F_Vectorized(alpha=1, constant=1)]

plt.subplot(1,3,3)
plt.title("Target")
plt.scatter(*zip(*CrescentSample), color = 'r', alpha = 0.2)
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
profiler = cProfile.Profile()
profiler.enable()

for i in range(25): # Maybe there is a problem of overfitting
    #print("Iteration " + str(i))
    Iteration += 1
    if Iteration >= 10:
        CenterGeneratorList = MixtureSample + CrescentSample
    CenterList = []
    DistanceMixture = np.zeros([500,5])
    DistanceTarget = np.zeros([500,5])
    for i in range(0,NumFs):
        c = CenterGeneratorList[random.randint(0, len(CenterGeneratorList) - 1)]
        CenterList.append(c)
        PotentialFs[i].setCenter(c)
        PotentialFsVectorized[i].setCenter(c)
    OldBeta = Beta
    Beta = BetaNewton()
    OldD = DValue
    DValue = D()
    print(DValue)
    MixtureSample = SamplesUpdate(MixtureSample)

    
profiler.disable()
stats = pstats.Stats(profiler).sort_stats('tottime')
stats.strip_dirs()
stats.dump_stats("newtonvectorized.prof")


plt.subplot(1,3,2)
plt.title("Optimal Transport")
plt.scatter(*zip(*MixtureSample), color = 'g', alpha = 0.2)
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.show()

end = time.time()
print(end - start)