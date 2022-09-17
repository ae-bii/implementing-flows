from re import X
from socket import create_server
from statistics import median
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
import SampleGenerator as sg
import MMDFunctions 
import torch
start = time.time()


random.seed(0)
np.random.seed(0)

e = math.e
pi = math.pi

# Our own norm method because the np.norm was too slow
def norm(i):
    return np.sqrt(sum(np.square(i)))

def GradientApprox(VariableList):
    Gradient = []
    delta = 1e-8
    for f in range(NumFs):
        GradientX = ((PotentialFsVectorized[f](VariableList + [delta/2, 0]) - (PotentialFsVectorized[f](VariableList - [delta/2, 0])))/delta)
        GradientY = ((PotentialFsVectorized[f](VariableList + [0, delta/2]) - (PotentialFsVectorized[f](VariableList - [0, delta/2])))/delta)
        Gradient.append(np.squeeze(np.transpose([GradientX, GradientY])))

    return Gradient

    
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
    F_gradient = GradientApprox(CrescentSample)
    for m in range(0, NumFs):
        for n in range(0, NumFs):
            yHessian[m][n] = sum((F_gradient[m]*F_gradient[n]).sum(axis = 1))
    
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

def uConjugateVec(y):
    ConvexMatrix = np.array(np.matmul(y, np.transpose(MixtureSample))) - uVectorized(MixtureSample)
    return ConvexMatrix.max(axis = 1)

def D():

    xSummation = sum(uVectorized(MixtureSample))
    ySummation = sum(uConjugateVec(CrescentSample))

    LL = 1/len(MixtureSample) * xSummation + 1 / \
        len(CrescentSample) * ySummation

    return LL

def SamplesUpdate(OldMixtureSample):
    NewMixtureSample = []
    F_eval_x = [0,0,0,0,0]
    F_eval_y = [0,0,0,0,0]
    for f in range(0,NumFs):
        gradient = GradientApprox(OldMixtureSample)[f]
        F_eval_x[f] = (gradient[:,0])
        F_eval_y[f] = (gradient[:,1])
    F_eval_x = np.array(F_eval_x)
    F_eval_y= np.array(F_eval_y)

    xVal = OldMixtureSample[:,0]
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

def SigCalculation(X,Y):
    Z = X + Y
    DistList = []
    for i in range(len(Z)):
        for j in range(len(Z)):
            DistList.append(functions.distance(Z[i],Z[j]))

    return median(DistList)



    


#------------------------------------------------------------------ TESTING (change to heatmap, add animtation)------------------------------------------------------------
CrescentSample = sg.JointSampleGenerator()
MixtureSample = sg.IndependentCouplingGenerator(CrescentSample, len(CrescentSample))
CenterGeneratorList = CrescentSample
  
steps = [MixtureSample]

PotentialFs = [functions.Giulio_F(alpha=1),
                functions.Gaussian_F(alpha=1, constant=1),
                functions.Multiquadric_F(alpha=1, constant=1),
                functions.InverseQuadratic_F(alpha=1, constant=1),
                functions.InverseMultiquadric_F(alpha=1, constant=1)]
NumFs = len(PotentialFs)
PotentialFsVectorized = [functions.Giulio_F_Vectorized(alpha = 1),
                        functions.Gaussian_F_Vectorized(alpha=1, constant=0),
                        functions.Multiquadric_F_Vectorized(alpha=1, constant=0),
                        functions.InverseQuadratic_F_Vectorized(alpha=1, constant=0),
                        functions.InverseMultiquadric_F_Vectorized(alpha=1, constant=0)]


DValue = 0
Iteration = 0
Beta = 0
MMDList = []

plt.rc('axes', titlesize=15) 

plt.subplot(2,3,1)
plt.title("Initial (Independent Coupling)")
plt.scatter(*zip(*MixtureSample), color = 'b', alpha = 0.2)
plt.xlim(-4, 4)
plt.ylim(-1, 6)
CenterList = []

plt.subplot(2,3,6)
plt.title("Target (Joint Samples)")
plt.scatter(*zip(*CrescentSample), color = 'r', alpha = 0.2)
plt.xlim(-4, 4)
plt.ylim(-1, 6)

# Profiling code
profiler = cProfile.Profile()
profiler.enable()
SamplesSaved = []
for i in range(1000): # Maybe there is a problem of overfitting
    #print("Iteration " + str(i))
    Iteration += 1
    if Iteration >= 10:
        CenterGeneratorList = MixtureSample + CrescentSample
    CenterList = []
    # DistanceMixture = np.zeros([500,5])
    # DistanceTarget = np.zeros([500,5])
    for f in range(0,NumFs):
        c = CenterGeneratorList[random.randint(0, len(CenterGeneratorList) - 1)]
        CenterList.append(c)
        PotentialFs[f].setCenter(c)
        PotentialFsVectorized[f].setCenter(c)
    OldBeta = Beta
    Beta = BetaNewton()
    OldD = DValue
    DValue = D()
    print(DValue)
    MixtureSample = SamplesUpdate(MixtureSample)
    if i % 5 == 0:
        Sigma = SigCalculation(MixtureSample, CrescentSample)
        X = torch.from_numpy(MixtureSample)
        Y = torch.from_numpy(CrescentSample)
        MMDList.append((MMDFunctions.mmd(X, Y, Sigma)).numpy())
    if i == 249 or i == 499 or i == 749 or i == 999:
        SamplesSaved.append(MixtureSample)
    steps.append(MixtureSample)

    
plt.subplot(2,3,2)
plt.title("Flow Transport (250 iterations)")
plt.scatter(*zip(*SamplesSaved[0]), color = 'g', alpha = 0.2)
plt.xlim(-4, 4)
plt.ylim(-1, 6)

plt.subplot(2,3,3)
plt.title("Flow Transport (500 iterations)")
plt.scatter(*zip(*SamplesSaved[1]), color = 'g', alpha = 0.2)
plt.xlim(-4, 4)
plt.ylim(-1, 6)

plt.subplot(2,3,4)
plt.title("Flow Transport (750 iterations)")
plt.scatter(*zip(*SamplesSaved[2]), color = 'g', alpha = 0.2)
plt.xlim(-4, 4)
plt.ylim(-1, 6)

plt.subplot(2,3,5)
plt.title("Flow Transport (1000 iterations)")
plt.scatter(*zip(*SamplesSaved[3]), color = 'g', alpha = 0.2)
plt.xlim(-4, 4)
plt.ylim(-1, 6)

plt.show()

X = torch.from_numpy(SamplesSaved[3])
Y = torch.from_numpy(CrescentSample)

Sigma = SigCalculation(SamplesSaved[3], CrescentSample)
MMD = MMDFunctions.mmd(X, Y, Sigma)
print("MMD Value:",MMD)
TestValue = MMDFunctions.TestThreshold(X, Y, Sigma)
print("Critical Value (5% Significance Level):",TestValue)

if MMD < TestValue:
    print("Null hypothesis not rejected, distributions are the same")
else:
    print("Null hypothesis rejected, distributions failed to be the same")

Step = np.arange(0, len(MMDList), 1)
Step = np.multiply(Step,5)
plt.plot(Step,MMDList)
plt.title("MMD Values vs. Iteration Steps")
plt.ylabel("MMD Values (Recorded every 5 iterations)")
plt.xlabel("Iteration Steps")

plt.show(block=True)
time.sleep(360)