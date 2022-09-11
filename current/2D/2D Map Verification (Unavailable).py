from cmath import sqrt
from re import M, X
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
from mpl_toolkits.mplot3d import Axes3D
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
    xSummationGradient = [sum(PotentialFsVectorized[f](JointSample)) for f in range(NumFs)]
    ySummationGradient = [sum(PotentialFsVectorized[f](IndependentSample)) for f in range(NumFs)]
    G = [(1/len(JointSample)) * xSummationGradient[k] - (1/len(IndependentSample)) * ySummationGradient[k] for k in range(NumFs)]
    G = np.array(G)
    yHessian = np.zeros([NumFs,NumFs])
    F_gradient = GradientApprox(IndependentSample)
    for m in range(0, NumFs):
        for n in range(0, NumFs):
            yHessian[m][n] = sum((F_gradient[m]*F_gradient[n]).sum(axis = 1))
    
    H = np.multiply(yHessian, 1/len(IndependentSample))
    HInverseNeg = (-1) * np.linalg.inv(H)
    Beta = np.matmul(HInverseNeg, G)
    LearningRate = 0.2 # Not sure how to choose this value
    ParameterList = [1, LearningRate/norm(Beta)]
    return Beta * min(ParameterList) # min(ParameterList) can be understood as similar to the "Proportion" in gradient descent

def u(x):
    F_eval = [(PotentialFs[f](x)) for f in range(NumFs)]
    return (((x[0] ** 2) + (x[1] ** 2)) / 2) + np.dot(Beta, F_eval)

def uVectorized(x):
    F_eval = [PotentialFsVectorized[f](x) for f in range(NumFs)]
    return ((np.square(x[:,1])))/2 + (np.transpose(F_eval) * Beta).sum(axis = 1)

def uConjugate(y):
    ConvexVector = ((JointSample * y)[:,0] + (JointSample * y)[:,1]) - uVectorized(JointSample)
    return max(ConvexVector)

def uConjugateVec(y):
    ConvexMatrix = np.array(np.matmul(y, np.transpose(JointSample))) - uVectorized(JointSample)
    return ConvexMatrix.max(axis = 1)

def D():

    xSummation = sum(uVectorized(JointSample))
    ySummation = sum(uConjugateVec(IndependentSample))

    LL = 1/len(JointSample) * xSummation + 1 / \
        len(IndependentSample) * ySummation

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

def Verification():
    fig = plt.figure()

    ax = fig.add_subplot(1,2,1, projection = "3d")
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    X2,Y2 = np.meshgrid(np.arange(-5,5,0.1),np.arange(-5,5,0.1))
    Z2 = ((1/(math.sqrt(2 * pi))) * (e ** (-(X2 ** 2)/2))) * (((1/(math.sqrt(2 * pi))) * (e ** (-((((X2 ** 2) + Y2) - (X2 ** 2)) ** 2)/2))))
    ax.plot_surface(X2,Y2,Z2, rstride = 1, cstride = 1, cmap = "viridis")

    ax = fig.add_subplot(1,2,2, projection = "3d")
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    X3,Y3 = np.meshgrid(np.arange(-5,5,0.2),np.arange(-5,5,0.2))
    
    def VerificationInner(X3,Y3):
        TempSample = []
        for i in range(50):
            for j in range(50):
                TempSample.append([X3[i][j], Y3[i][j]])
        TempSample = np.array(TempSample)
        for m in range(500):
            c = CenterList[m]
            for f in range(0,NumFs):
                PotentialFsVectorized[f].setCenter(c[f])
            BetaTemp = BetaList[m]
            TempSample = SamplesUpdate(TempSample, BetaTemp)
        TempSampleSplit = np.array_split(TempSample, 50)
        for n in range(50):
            Y3[n] = TempSampleSplit[n][:,1]
        return np.array(Y3)

    Z3 = ((1/(math.sqrt(2 * pi))) * (e ** (-(X3 ** 2)/2))) * (((1/(math.sqrt(2 * pi))) * (e ** (-((VerificationInner(X3,Y3) - (X3 ** 2)) ** 2)/2))))
    ax.plot_surface(X3,Y3,Z3, rstride = 1, cstride = 1, cmap = "viridis",antialiased=True)

# An alternative way of plotting:
#   X3 = JointSampleSaved[:,0]
#   for m in range(500):
#        c = CenterList[m]
#       for i in range(0,NumFs):
#            PotentialFsVectorized[i].setCenter(c[i])
#        BetaTemp = BetaList[m]
#        TempSample = SamplesUpdate(JointSampleSaved, BetaTemp)
#    Y3 = TempSample[:,1]
#    Z3 = ((1/(math.sqrt(2 * pi))) * (e ** (-(X3 ** 2)/2))) * (((1/(math.sqrt(2 * pi))) * (e ** (-((Y3 - (X3 ** 2)) ** 2)/2))))
#    ax.plot_trisurf(X3,Y3,Z3, cmap = "viridis",antialiased=True)
#    plt.show()

    plt.show()


#------------------------------------------------------------------ TESTING (change to heatmap, add animtation)------------------------------------------------------------

# For the Verification function to work properly:
# The following code should be used to replace whatever is inside the function sg.JointSampleGenerator()：
#   x = np.random.standard_normal(500)
#   JointSample = []
#    for i in range(len(x)):
#        y = np.float64(np.random.normal((x[i]) ** 2, 1, 1))
#        JointSample.append([x[i], y])

JointSample = sg.JointSampleGenerator()
IndependentSample = sg.IndependentCouplingGenerator(JointSample,len(JointSample))

CenterGeneratorList = JointSample
JointSampleSaved = JointSample  
steps = [JointSample]

PotentialFs = [functions.Giulio_F(alpha=1),
                functions.Gaussian_F(alpha=1, constant=0),
                functions.Multiquadric_F(alpha=1, constant=0),
                functions.InverseQuadratic_F(alpha=1, constant=0),
                functions.InverseMultiquadric_F(alpha=1, constant=0)]
NumFs = len(PotentialFs)
PotentialFsVectorized = [functions.Giulio_F_Vectorized(alpha = 1),
                        functions.Gaussian_F_Vectorized(alpha=1, constant=0),
                        functions.Multiquadric_F_Vectorized(alpha=1, constant=0),
                        functions.InverseQuadratic_F_Vectorized(alpha=1, constant=0),
                        functions.InverseMultiquadric_F_Vectorized(alpha=1, constant=0)]


DValue = 0
Iteration = 0
Beta = 0
BetaList = [0]
# Profiling code
# profiler = cProfile.Profile()
# profiler.enable()

plt.rc('axes', titlesize=15) 
plt.subplot(2,3,1)
plt.title("Initial (Independent Coupling)")
plt.scatter(*zip(*IndependentSample), color = 'b', alpha = 0.2)
plt.xlim(-4, 4)
plt.ylim(-1, 6)
CenterList = []
SamplesSaved = []

plt.subplot(2,3,6)
plt.title("Target (Joing Samples)")
plt.scatter(*zip(*JointSample), color = 'r', alpha = 0.2)
plt.xlim(-4, 4)
plt.ylim(-1, 6)

for i in range(1000): # Maybe there is a problem of overfitting
    #print("Iteration " + str(i))
    Iteration += 1
    if Iteration >= 10:
        CenterGeneratorList = IndependentSample + JointSample
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
    IndependentSample = SamplesUpdate(IndependentSample)
    if i == 249 or i == 499 or i == 749 or i == 999:
        SamplesSaved.append(IndependentSample)
    steps.append(IndependentSample)
        


CenterList = np.array(CenterList)



# profiler.disable()
# stats = pstats.Stats(profiler).sort_stats('tottime')
# stats.strip_dirs()
# stats.dump_stats("newtonvectorized.prof")



plt.subplot(2,3,2)
plt.title("Flow Transport (500 iterations)")
plt.scatter(*zip(*SamplesSaved[0]), color = 'g', alpha = 0.2)
plt.xlim(-4, 4)
plt.ylim(-1, 6)

plt.subplot(2,3,3)
plt.title("Flow Transport (1000 iterations)")
plt.scatter(*zip(*SamplesSaved[1]), color = 'g', alpha = 0.2)
plt.xlim(-4, 4)
plt.ylim(-1, 6)

plt.subplot(2,3,4)
plt.title("Flow Transport (1500 iterations)")
plt.scatter(*zip(*SamplesSaved[2]), color = 'g', alpha = 0.2)
plt.xlim(-4, 4)
plt.ylim(-1, 6)

plt.subplot(2,3,5)
plt.title("Flow Transport (2000 iterations)")
plt.scatter(*zip(*SamplesSaved[3]), color = 'g', alpha = 0.2)
plt.xlim(-4, 4)
plt.ylim(-1, 6)

plt.show()
# Verification()
