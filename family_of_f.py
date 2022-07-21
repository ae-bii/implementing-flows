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
    def __init__(self, center=[], alpha=1, k=0):
        self._center = center
        self._alpha = alpha
        self._k = k

    def set_center(self, center):
        self._center = center

    def set_alpha(self, alpha):
        self._alpha = alpha

    def set_k(self, k):
        self._k = k

    def bump_f(self, z):
        value = 0
        r = distance(z, self._center)
        if r < 1/self._alpha:
            value = math.pow(e, -(1/(1-(self._alpha*r)**2)))
        return value

    def giulio_f(self, z):
        r = distance(z, self._center)
        return math.erf(r/self._alpha) / r

    def gaussian_f(self, z):
        r = distance(z, self._center)
        return math.pow(e, -((self._alpha*r)**2))

    def multiquadric_f(self, z):
        r = distance(z, self._center)
        return math.sqrt(1+(self._alpha*r)**2)

    def inverseQuadratic_f(self, z):
        r = distance(z, self._center)
        return 1/(1+(self._alpha*r)**2)

    def inverseMultiquadric_f(self, z):
        r = distance(z, self._center)
        return 1/math.sqrt(1+(self._alpha*r)**2)

    def polyharmonicSpline_f(self, z):
        r = distance(z, self._center)
        if self._k % 2 == 0:
            return r**(self._k-1)*np.log(r**r)
        else:
            return r**self._k

    def thinPlateSpline_f(self, z):
        r = distance(z, self._center)
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

    def bump_F(self, z):
        r = distance(z, self._center)
        if r < 1/self._alpha:
            return (math.exp(1 / (self._alpha**2 * r**2 - 1)) * (self._alpha**2 * r**2 - 1) - expi(1 / (self._alpha**2 * r**2 - 1))) / (2 * self._alpha**2) + self._constant
        return r + self._constant

    def giulio_F(self, z):
        r = distance(z, self._self._center)
        return r * math.erf(r/self._alpha) + (self._alpha/math.sqrt(pi)) * math.pow(e, -(r/self._alpha) ** 2)

    def gaussian_F(self, z):
        r = distance(z, self._center)
        return -math.pow(e, -self._alpha**2 * r**2) / (2 * self._alpha**2) + self._constant

    def multiquadric_F(self, z):
        r = distance(z, self._center)
        return math.pow(self._alpha**2 * r**2 + 1, 3/2) / (3 * self._alpha**2) + self._constant

    def inverseQuadratic_F(self, z):
        r = distance(z, self._center)
        return math.log(self._alpha**2 * r**2 + 1) / (2 * self._alpha**2) + self._constant

    def inverseMultiquadric_F(self, z):
        r = distance(z, self._center)
        return math.sqrt(self._alpha**2 * r**2 + 1) / (self._alpha**2) + self._constant

    def polyharmonicSpline_F(self, z):
        r = distance(z, self._center)
        if self._k % 2 == 0:
            # Since k and r are positive
            return (r**(self._k+1) * ((self._k+2) * math.log(r) - 1)) / ((self._k+2)**2) + self._constant
        else:
            return r**(self._k+2) / (self._k+2) + self._constant

    def thinPlateSpline_F(self, z):
        r = distance(z, self._center)
        return r**4 * (math.log(r)/4 - 1/16) + self._constant
