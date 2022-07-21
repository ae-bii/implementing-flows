import math
import numpy as np
from scipy.special import expi

e = math.e
pi = math.pi


def distance(z1, z2):
    sum = 0
    for i in range(0, len(z1)-1):
        sum += (z1[i] - z2[i]) ** 2
    return math.sqrt(sum)


class Potentialf:
    def bump_f(z, center, alpha):
        value = 0
        r = distance(z, center)
        if r < 1/alpha:
            value = math.pow(e, -(1/(1-(alpha*r)**2)))
        return value

    def giulio_f(z, center, alpha):
        r = distance(z, center)
        return math.erf(r/alpha) / r

    def gaussian_f(z, center, alpha):
        r = distance(z, center)
        return math.pow(e, -((alpha*r)**2))

    def multiquadric_f(z, center, alpha):
        r = distance(z, center)
        return math.sqrt(1+(alpha*r)**2)

    def inverseQuadratic_f(z, center, alpha):
        r = distance(z, center)
        return 1/(1+(alpha*r)**2)

    def inverseMultiquadric_f(z, center, alpha):
        r = distance(z, center)
        return 1/math.sqrt(1+(alpha*r)**2)

    def polyharmonicSpline_f(z, center, k):
        r = distance(z, center)
        if k % 2 == 0:
            return r**(k-1)*np.log(r**r)
        else:
            return r**k

    def thinPlateSpline_f(z, center):
        r = distance(z, center)
        return r**2*np.log(r)


class PotentialF:
    def __init__(self, center=[], alpha=1, constant=0, k=0):
        self._center = center
        self._alpha = alpha
        self._constant = constant
        self._k = k
        
    def set_center(self, center):
        self._center = center
        
    def set_alpha(self, alpha):
        self._alpha = alpha
        
    def set_constant(self, constant):
        self._constant = constant
        
    def set_k(self, k):
        self._k = k
    
    def bump_F(z, center, alpha, constant):
        r = distance(z, center)
        if r < 1/alpha:
            return (math.exp(1 / (alpha**2 * r**2 - 1)) * (alpha**2 * r**2 - 1) - expi(1 / (alpha**2 * r**2 - 1))) / (2 * alpha**2) + constant
        return r + constant

    # Testing if setter method works
    def giulio_F(self, z):
        r = distance(z, self._center)
        return r * math.erf(r/self._alpha) + (self._alpha/math.sqrt(pi)) * math.pow(e, -(r/self._alpha) ** 2)

    def gaussian_F(z, center, alpha, constant):
        r = distance(z, center)
        return -math.pow(e, -alpha**2 * r**2) / (2 * alpha**2) + constant

    def multiquadric_F(z, center, alpha, constant):
        r = distance(z, center)
        return math.pow(alpha**2 * r**2 + 1, 3/2) / (3 * alpha**2) + constant

    def inverseQuadratic_F(z, center, alpha, constant):
        r = distance(z, center)
        return math.log(alpha**2 * r**2 + 1) / (2 * alpha**2) + constant

    def inverseMultiquadric_F(z, center, alpha, constant):
        r = distance(z, center)
        return math.sqrt(alpha**2 * r**2 + 1) / (alpha**2) + constant

    def polyharmonicSpline_F(z, center, constant, k):
        r = distance(z, center)
        if k % 2 == 0:
            # Since k and r are positive
            return (r**(k+1) * ((k+2) * math.log(r) - 1)) / ((k+2)**2) + constant
        else:
            return r**(k+2) / (k+2) + constant

    def thinPlateSpline_F(z, center, constant):
        r = distance(z, center)
        return r**4 * (math.log(r)/4 - 1/16) + constant
