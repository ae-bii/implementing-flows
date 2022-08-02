from re import M
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import random

pi = math.pi
e = math.e
def JointSampleGenerator(): # x is distributed uniformmly and y follows a normal distribution whose mean is dependent on x
    x = np.random.uniform(0,1,500)
    JointSample = []
    for i in range(0,len(x)):
        y = np.float64((np.random.normal(2 * x[i] + 1, 0.2, 1)))
        JointSample.append([x[i],y])
    return np.array(JointSample)

def IndependentCouplingGenerator():
    Sample = []
    x = JointSampleGenerator()[:,0]
    y = JointSampleGenerator()[:,1]
    for i in range(0, len(x)):
        Sample.append([x[i], y[random.randint(0,len(x) - 1)]])
    return np.array(Sample)


def RejectionSampling(Formula):
    TargetPDF = Formula
    Sample = []
    PlotPoint = []
    while len(Sample) < 500:
        SampleCandidate = [np.float64(np.random.uniform(0,1,1)), np.float64(np.random.uniform(-10,10,1))] 
        CandidateDistance = np.random.uniform(0,5,1) # This is equivalent to randomly choosing a point in this space: 0 < x < 1, -10 < y < 10 0 < z < 5
        if CandidateDistance <= TargetPDF(SampleCandidate):
            Sample.append(SampleCandidate) # Accept the point if it falls under the surface which represents the probability density function
            PlotPoint.append(SampleCandidate + [CandidateDistance])
    PlotPoint = np.array(PlotPoint)
    return np.array(Sample)

def main():
    SampleIndependent = IndependentCouplingGenerator()
    SampleJoint  = JointSampleGenerator()


    plt.subplot(1,2,1)
    plt.title("IndependentCoupling")
    plt.scatter(*zip(*SampleIndependent), color = 'b', alpha = 0.2)
    plt.xlim(-2, 6)
    plt.ylim(0, 4)

    plt.subplot(1,2,2)
    plt.title("JointDistribution")
    plt.scatter(*zip(*SampleJoint), color = 'r', alpha = 0.2)
    plt.xlim(-2, 2)
    plt.ylim(0, 4)
    plt.show()


