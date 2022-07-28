import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Currently looking for better functions that could make larger, more visible difference and easy to calculate (relatively) at the same time


JointDistribution = lambda xy: 12/5 * xy[0] * (2 - xy[0] - xy[1]) # From "A First Course in Probability 9th Edition Page 251"
IndependentCoupling = lambda xy: 144/25 * (3/2 * xy[0] - (xy[0]) ** 2) * (2/3 - (xy[1])/2) # Obtained by multiplying the marginal distributions of the joint distribution above
# Not sure if the method used to obtain Independent Coupling is correct

def RejectionSampling(Formula):
    TargetPDF = Formula
    Sample = []
    global PlotPoint
    PlotPoint = []
    while len(Sample) < 500:
        SampleCandidate = [np.random.uniform(0,1,1), np.random.uniform(0,1,1)] 
        CandidateDistance = np.random.uniform(0,2.5,1) # This is equivalent to randomly choosing a point in this space: 0 < x < 1, 0 < y < 1 0 < z < 2.5
        if CandidateDistance <= TargetPDF(SampleCandidate):
            Sample.append(SampleCandidate) # Accept the point if it falls under the surface which represents the probability density function
            PlotPoint.append(SampleCandidate + [CandidateDistance])
    PlotPoint = np.array(PlotPoint)
    return [np.array(Sample), PlotPoint]

SampleIndependent = RejectionSampling(IndependentCoupling)
SampleJoint  = RejectionSampling(JointDistribution)

plt.subplot(1,2,1)
plt.title("IndependentCoupling")
plt.scatter(*zip(*SampleIndependent[0]), color = 'b', alpha = 0.2)
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)

plt.subplot(1,2,2)
plt.title("JointDistribution")
plt.scatter(*zip(*SampleJoint[0]), color = 'r', alpha = 0.2)
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.show()


fig = plt.figure(figsize = (12,10))

x = np.arange(-0.1, 1.1, 0.2)
y = np.arange(-0.1, 1.1, 0.2)

X, Y = np.meshgrid(x, y)
Z1 = 144/25 * (3/2 * X - (X) ** 2) * (2/3 - (Y)/2)
Z2 = 12/5 * X * (2 - X - Y)



PlotPoint = SampleIndependent[1]

ax = fig.add_subplot(121, projection='3d')
ax.scatter(PlotPoint[:,0],PlotPoint[:,1],PlotPoint[:,2])
ax.plot_surface(X, Y, Z1, cmap = plt.cm.cividis)
ax.set_xlabel('x', labelpad=20)
ax.set_ylabel('y', labelpad=20)
ax.set_zlabel('z', labelpad=20)
ax.set_title('Independent Coupling Samples')

PlotPoint = SampleJoint[1]

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(PlotPoint[:,0],PlotPoint[:,1],PlotPoint[:,2])
ax2.plot_surface(X, Y, Z2, cmap = plt.cm.cividis)
ax2.set_xlabel('x', labelpad=20)
ax2.set_ylabel('y', labelpad=20)
ax2.set_zlabel('z', labelpad=20)
ax2.set_title('Joint Distribution Samples')

plt.show() # If most of the points are inside the surface, then Rejection Sampling is successful
