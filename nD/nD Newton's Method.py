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
dim = 10
SampleSize = 1000000

# Our own norm method because the np.norm was too slow
def norm(i):
    return np.sqrt(sum(np.square(i)))

def GradientApprox(VariableList):
    Gradient = []
    delta = 1e-8
    deltaList = []
    for i in range(dim):
        deltaList.append([0]*dim)
    for j in range(dim):
        deltaList[j][j] = delta/2
    Gradient = []
    for f in range(NumFs):
        cur = []
        for k in range(dim):
            cur.append((PotentialFsVectorized[f](VariableList + deltaList[k]) - (PotentialFsVectorized[f](VariableList - deltaList[k])))/delta)
        Gradient.append(np.squeeze(np.transpose(cur)))

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
    sum = [0]*len(x)
    for i in range(dim):
        sum[:] += np.square(x[:,i])


    return np.array(sum)/2 + (np.transpose(F_eval) * Beta).sum(axis = 1)

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
    F_eval = []
    [F_eval.append([0,0,0,0,0]) for i in range(dim)]
    for f in range(0,NumFs):
        for j in range(dim):
            gradient = GradientApprox(OldMixtureSample)[f]
            F_eval[j][f] = gradient[:,j]
    vals = []
    for i in range(dim):
        curVal = OldMixtureSample[:,i] + (np.multiply(np.transpose(np.array(F_eval[i])), Beta)).sum(axis = 1)
        vals.append(curVal)
    NewMixtureSample = np.array(vals)
    NewMixtureSample = np.transpose(NewMixtureSample)

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
    x = np.random.multivariate_normal(mean1, cov, SampleSize)
    y = np.random.multivariate_normal(mean2, cov, SampleSize)
    z = np.random.multivariate_normal(mean3, cov, SampleSize)
    MixtureSample = []
    for i in range(SampleSize):
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
    for j in range(SampleSize):
        cur = []
        for k in range(dim):
            cur.append(Normals[k][j])
        Sample.append(np.array(cur))
    return np.array(Sample)

def nCr(n,r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)

def visualize(data, dim, col):
    numVisuals = nCr(dim, 2)
    k = 1
    for i in range(dim):
        for j in range(dim - i - 1):
            x = int(i+1)
            y = int(i+j+1)
            cur = zip([x[i] for x in data], [x[i+j] for x in data])
            plt.subplot(1, numVisuals, k)
            plt.title("Cross Section")
            plt.scatter(*zip(*data),  color = col, alpha = 0.2)
            k += 1
    plt.show()


#------------------------------------------------------------------ TESTING (change to heatmap, add animtation)------------------------------------------------------------
MixtureSample = MixtureSampleGenerator()
#CrescentSample = np.loadtxt("C:/Users/Laptop/OneDrive - Grinnell College/Desktop/Research/PolyMath 2022/implementing-flows/SampleMoon.csv", delimiter=",")
CrescentSample = StandardNormalGenerator()
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

DValue = 0
Iteration = 0
Beta = 0

# Profiling code
#profiler = cProfile.Profile()
#profiler.enable()

for i in range(500): # Maybe there is a problem of overfitting
    #print("Iteration " + str(i))
    Iteration += 1
    if Iteration >= 300:
        CenterGeneratorList = MixtureSample + CrescentSample
    CenterList = []
    DistanceMixture = np.zeros([SampleSize],5])
    DistanceTarget = np.zeros([SampleSize,5])
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

    
# profiler.disable()
# stats = pstats.Stats(profiler).sort_stats('tottime')
# stats.strip_dirs()
# stats.dump_stats("newtonvectorized.prof")

end = time.time()
print(end - start)

#visualize(MixtureSample, dim, 'r')