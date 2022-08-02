from re import M
from statistics import mean
from tkinter import Variable
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
        y = np.float64((np.random.normal(2 * x[i] + 1, 2 * x[i] + 1, 1)))
        z = np.float64((np.random.normal(abs(x[i] + y), abs(x[i] + y), 1)))
        JointSample.append([x[i],y,z])
    return np.array(JointSample)

def IndependentCouplingGenerator():
    dim = len(JointSampleGenerator()[1])
    Sample = [JointSampleGenerator()[:,i] for i in range(dim)]
    for j in range(1,dim):
        random.shuffle(Sample[j])
    return np.array(np.transpose(Sample))


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


    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    plt.title("Independent Coupling")
    ax.scatter3D(*zip(*SampleIndependent), color = 'r', alpha = 0.2)
    ax.set_xlim3d(0,8)
    ax.set_ylim3d(0,8)
    ax.set_zlim3d(0,8)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    plt.title("Joint Distribution")
    ax.scatter3D(*zip(*SampleJoint), color = 'r', alpha = 0.2)
    ax.set_xlim3d(0,8)
    ax.set_ylim3d(0,8)
    ax.set_zlim3d(0,8)
    plt.show()


