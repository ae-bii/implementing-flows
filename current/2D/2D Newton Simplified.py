import numpy as np
import random
import matplotlib.pyplot as plt
import math
from scipy.special import erf
import SampleGenerator as sg


def radialBasisFunc(center, varList):
    rList = np.linalg.norm(np.subtract(varList,center), axis=1)
    return np.multiply(rList,erf(np.divide(rList, alpha))) + np.multiply((alpha/np.sqrt(np.pi)), np.exp(-np.square(np.divide(rList, alpha))))

def radialBasisGrad(center, varList):
    Gradient = []
    delta = 1e-8
    GradientX = ((radialBasisFunc(center, np.add(varList,[delta/2, 0])) - radialBasisFunc(center, np.subtract(varList,[delta/2, 0])))/delta)
    GradientY = ((radialBasisFunc(center, np.add(varList,[0, delta/2])) - radialBasisFunc(center, np.subtract(varList,[0, delta/2])))/delta)
    Gradient = np.c_[GradientX,GradientY]
    return Gradient

def betaNewton(initialSample,targetSample,center):
    grad = (1/len(initialSample)) * np.sum(radialBasisFunc(initialSample,center)) - (1/len(targetSample)) * np.sum(radialBasisFunc(targetSample,center))
    hess = (1/len(targetSample)) * np.sum((radialBasisGrad(targetSample,center) * radialBasisGrad(targetSample,center)).sum(axis = 1))
    betaRaw = -(1/hess) * grad
    learningRate = [1, 0.5/abs(betaRaw)]
    betaAdjusted = betaRaw * min(learningRate)
    print(betaAdjusted)
    return betaAdjusted

def sampleUpdate(center, varList,beta):    
    rbfGrad = radialBasisGrad(center, varList)
    rbfEvalX = np.array(rbfGrad[:,0])
    rbfEvalY = np.array(rbfGrad[:,1])
    xVal = varList[:,0] 
    yVal = varList[:,1] + (np.multiply(rbfEvalY, beta))
    newVarList = np.c_[xVal,yVal]
    return newVarList

targetSample = sg.JointSampleGenerator()
initialSample = sg.IndependentCouplingGenerator(targetSample, len(targetSample))

CenterGeneratorList = targetSample.tolist()

plt.rc('axes', titlesize=15) 

plt.subplot(1,3,1)
plt.title("Initial (Independent Coupling)")
plt.scatter(*zip(*initialSample), color = 'b', alpha = 0.2)
plt.xlim(-4, 4)
plt.ylim(-1, 6)
CenterList = []

plt.subplot(1,3,3)
plt.title("Target (Joint Samples)")
plt.scatter(*zip(*targetSample), color = 'r', alpha = 0.2)
plt.xlim(-4, 4)
plt.ylim(-1, 6)

for i in range(750): # Maybe there is a problem of overfitting
    center = CenterGeneratorList[random.randint(0, len(CenterGeneratorList) - 1)]
    CenterGeneratorList.remove(center)
    if len(CenterGeneratorList) == 0:
        CenterGeneratorList = targetSample.tolist()
    if center[0]<2 and center[0]>-2 and center[1]<5 and center[1]>4:
        maybeDensity = 0.8
    else:
        maybeDensity = 0.2
    alpha = ((15/(len(initialSample) + len(targetSample))) * (1/(1/30) + 1/maybeDensity)) ** (1/2)
    beta = betaNewton(initialSample, targetSample, center)
    initialSample = sampleUpdate(center, initialSample, beta)

plt.subplot(1,3,2)
plt.title("Flow Transport (1000 iterations)")
plt.scatter(*zip(*initialSample), color = 'g', alpha = 0.2)
plt.xlim(-4, 4)
plt.ylim(-1, 6)

plt.show()
