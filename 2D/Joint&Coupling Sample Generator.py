from re import M
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math


pi = math.pi
e = math.e
def JointDistributionGenerator(): # x is distributed uniformmly and y follows a normal distribution whose mean is dependent on x
    x = np.random.uniform(0,1,500)
    JointSample = []
    for i in range(0,len(x)):
        y = np.float64((np.random.normal(2 * x[i] + 1, 0.2, 1)))
        JointSample.append([x[i],y])
    return JointSample

def IndependentCouplingGenerator():
    # MarginalX = 1
    MarginalY = lambda xy: 1/4 * (math.erf((5/math.sqrt(2)) * (xy[1] - 1)) - math.erf((5/math.sqrt(2)) * (xy[1] - 3))) # Obtain by integrating their joint density with respect to x
    return RejectionSampling(MarginalY)


def RejectionSampling(Formula):
    TargetPDF = Formula
    Sample = []
    PlotPoint = []
    while len(Sample) < 500:
        SampleCandidate = [np.random.uniform(0,1,1), np.random.uniform(-10,10,1)] 
        CandidateDistance = np.random.uniform(0,5,1) # This is equivalent to randomly choosing a point in this space: 0 < x < 1, -10000 < y < 10000 0 < z < 5
        if CandidateDistance <= TargetPDF(SampleCandidate):
            Sample.append(SampleCandidate) # Accept the point if it falls under the surface which represents the probability density function
            PlotPoint.append(SampleCandidate + [CandidateDistance])
    PlotPoint = np.array(PlotPoint)
    return [np.array(Sample), PlotPoint]

SampleIndependent = IndependentCouplingGenerator()
SampleJoint  = JointDistributionGenerator()


plt.subplot(1,2,1)
plt.title("IndependentCoupling")
plt.scatter(*zip(*SampleIndependent[0]), color = 'b', alpha = 0.2)
plt.xlim(-2, 6)
plt.ylim(0, 4)

plt.subplot(1,2,2)
plt.title("JointDistribution")
plt.scatter(*zip(*SampleJoint), color = 'r', alpha = 0.2)
plt.xlim(-2, 2)
plt.ylim(0, 4)
plt.show()


fig = plt.figure(figsize = (12,10))

x = np.arange(0, 1.1, 0.2)
y = np.arange(0, 4, 0.2)

X, Y = np.meshgrid(x, y)

Z1 = 1/4 * (np.vectorize(math.erf)(10 * math.sqrt(2) * (Y - 1)) - np.vectorize(math.erf)(10 * math.sqrt(2) * (Y - 3)))




PlotPoint = SampleIndependent[1]

ax = fig.add_subplot(121, projection='3d')
ax.scatter(PlotPoint[:,0],PlotPoint[:,1],PlotPoint[:,2])
ax.plot_surface(X, Y, Z1, cmap = plt.cm.cividis)
ax.set_xlabel('x', labelpad=20)
ax.set_ylabel('y', labelpad=20)
ax.set_zlabel('z', labelpad=20)
ax.set_title('Independent Coupling Samples')

PlotPoint = SampleJoint[1]



plt.show() # If most of the points are inside the surface, then Rejection Sampling is successful
