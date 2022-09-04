
import numpy as np
import matplotlib.pyplot as plt

def RejectionSampling():
    Formula = lambda x: 4 - 2*(x ** 2)/5
    Sample = []

    while len(Sample) < 500:
        SampleCandidate = [np.float64(np.random.uniform(-4,4,1)), np.float64(np.random.uniform(-6,6,1))] 
        if ((SampleCandidate[1])**2)/7 < Formula(SampleCandidate[0]) and ((SampleCandidate[1] + 7/5)**2)/7 > Formula(SampleCandidate[0]):
            Sample.append(SampleCandidate) # Accept the point if it falls under the surface which represents the probability density function

    return np.array(Sample)

def main():
    TestSample = RejectionSampling()

    plt.title("Test")
    plt.scatter(*zip(*TestSample), color = 'b', alpha = 0.2)
    plt.xlim(-4, 4)
    plt.ylim(-1, 6)
    plt.show()

    np.savetxt("EllipseMoon.csv", TestSample, delimiter = ",")