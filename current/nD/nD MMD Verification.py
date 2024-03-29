import sys
from turtle import color
sys.path.append("./")
from statistics import median
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
import torch
import MMDFunctions
random.seed(0)
np.random.seed(0)

Start = time.time()

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
    vals = [OldMixtureSample[:,0]]
    F_eval = [[None] * NumFs for i in range(dim)]
    for f in range(0,NumFs):
        gradient = GradientApprox(OldMixtureSample)[f]
        for i in range(dim):
            F_eval[i][f] = (gradient[:,i])
    F_eval = np.array(F_eval)
    for i in range(1,dim):
        vals.append(OldMixtureSample[:,i] + (np.multiply(np.transpose(F_eval[i]), Beta)).sum(axis = 1))
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

def SigCalculation(X,Y):
    Z = X + Y
    DistList = []
    for i in range(len(Z)):
        for j in range(len(Z)):
            DistList.append(functions.distance(Z[i],Z[j]))

    return median(DistList)


#------------------------------------------------------------------ TESTING ------------------------------------------------------------
dim = 4

# Testing and Plot:
Target = SampleGeneratorND.JointSampleGenerator()
Initial = SampleGeneratorND.IndependentCouplingGenerator(Target, len(Target))
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
                        functions.InverseMultiquadric_F_Vectorized()
]
NumFs = len(PotentialFsVectorized)




DValue = 0
Iteration = 0
Beta = 0
SamplesSaved = []
MMDList = []
# Converts time in seconds to hours, minutes, seconds
# def time_convert(sec):
#   mins = sec // 60
#   sec = sec % 60
#   hours = mins // 60
#   mins = mins % 60
#   print("Time Lapsed = {0}:{1}:{2}".format(int(hours),int(mins),sec))

# start_time = time.time()

for i in range(1000): # Maybe there is a problem of overfitting
    Iteration += 1
    if Iteration >= 10:
        CenterGeneratorList = np.concatenate((Initial,Target), axis=0)
    CenterList = []
    
    for f in range(0,NumFs):
        c = CenterGeneratorList[random.randint(0, len(CenterGeneratorList) - 1)]
        CenterList.append(c)
        PotentialFsVectorized[f].setCenter(c)
    OldBeta = Beta
    Beta = BetaNewton()
    OldD = DValue
    DValue = D()
    # print(DValue)
    if i % 5 == 0:
        Sigma = SigCalculation(Initial, Target)
        X = torch.from_numpy(Initial)
        Y = torch.from_numpy(Target)
        MMDList.append((MMDFunctions.mmd(X, Y, Sigma)).numpy())
    Initial = SamplesUpdate(Initial)
    if i == 249 or i == 499 or i == 749 or i == 999:
        SamplesSaved.append(Initial)

end = time.time()
print(end - Start)
X = torch.from_numpy(SamplesSaved[3])
Y = torch.from_numpy(Target)

Sigma = SigCalculation(SamplesSaved[3], Target)
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
