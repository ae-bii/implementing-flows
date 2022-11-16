from re import X
from socket import create_server
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
        GradientX = PotentialGradVectorized[f](VariableList,wrt=0)
        GradientY = PotentialGradVectorized[f](VariableList,wrt=1)
        Gradient.append(np.squeeze(np.transpose([GradientX, GradientY])))

    return Gradient

    
# REVISED:
def BetaNewton(): # Newton's method (Experimental)
    xSummationGradient = np.zeros(NumFs)
    ySummationGradient = np.zeros(NumFs)
    GVec = np.zeros(NumFs)
    xSummationGradient = [sum(PotentialFsVectorized[f](MixtureSample)) for f in range(NumFs)]
    ySummationGradient = [sum(PotentialFsVectorized[f](CrescentSample)) for f in range(NumFs)]
    GVec = [(1/len(MixtureSample)) * xSummationGradient[k] - (1/len(CrescentSample)) * ySummationGradient[k] for k in range(NumFs)]
    GVec = np.array(GVec)
    yHessian = np.zeros([NumFs,NumFs])
    F_gradient = GradientApprox(CrescentSample)
    for m in range(0, NumFs):
        for n in range(0, NumFs):
            yHessian[m][n] = sum((F_gradient[m]*F_gradient[n]).sum(axis = 1))
    
    HMat = (1/len(CrescentSample)) * yHessian
    HInverseNeg = (-1) * np.linalg.inv(HMat)
    Beta = np.dot(HInverseNeg, GVec)
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
    F_eval_x = [0,0]
    F_eval_y = [0,0]
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

PotentialFsVectorized = [
                        functions.Gaussian_F_Vectorized(alpha=1.5, constant=0),
                        functions.InverseQuadratic_F_Vectorized(alpha=1.5, constant=0)]

PotentialGradVectorized = [
                          functions.Gaussian_fgrad_Vectorized(alpha=1.5, constant=0),
                          functions.InverseQuadratic_fgrad_Vectorized(alpha=1.5, constant=0)]

NumFs = len(PotentialFsVectorized)

DValue = 0
Iteration = 0
Beta = 0

plt.rc('axes', titlesize=15) 

plt.subplot(1,3,1)
plt.title("Initial (Independent Coupling)")
plt.scatter(*zip(*MixtureSample), color = 'b', alpha = 0.2)
plt.xlim(-4, 4)
plt.ylim(-1, 6)
CenterList = []

plt.subplot(1,3,3)
plt.title("Target (Joint Samples)")
plt.scatter(*zip(*CrescentSample), color = 'r', alpha = 0.2)
plt.xlim(-4, 4)
plt.ylim(-1, 6)

# Profiling code
profiler = cProfile.Profile()
profiler.enable()
SamplesSaved = []
for i in range(200): # Maybe there is a problem of overfitting
    #print("Iteration " + str(i))
    if i > 10:
        CenterGeneratorList = np.concatenate((CrescentSample,MixtureSample))
    CenterList = []
    # DistanceMixture = np.zeros([500,5])
    # DistanceTarget = np.zeros([500,5])
    for f in range(0,NumFs):
        DistFlag = False # Test whether two centers are too close
        c = CenterGeneratorList[random.randint(0, len(CenterGeneratorList) - 1)]
        while f > 0 and DistFlag == False:
            if all([functions.distance(c,CenterList[k]) >= 4 for k in range(0,f)])== True: # If too close, generate another center until distance between centers > 2
                DistFlag = True
            else:
                c = CenterGeneratorList[random.randint(0, len(CenterGeneratorList) - 1)]

        CenterList.append(c)
        PotentialFsVectorized[f].setCenter(c) # Once a center is chosen, set it to the center of a RBF and its partial derivatives
        PotentialGradVectorized[f].setCenter(c)
    OldBeta = Beta
    Beta = BetaNewton()
    OldD = DValue
    DValue = D()
    MixtureSample = SamplesUpdate(MixtureSample)

    
plt.subplot(1,3,2)
plt.title("Flow Transport (1000 iterations)")
plt.scatter(*zip(*MixtureSample), color = 'g', alpha = 0.2)
plt.xlim(-4, 4)
plt.ylim(-1, 6)

plt.show()