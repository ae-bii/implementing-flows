import math
import numpy as np
from scipy.special import expi

e = math.e
pi = math.pi


def distance(z1, z2):
    return np.linalg.norm(np.subtract(z1,z2))

class Bump_F:
    def __init__(self, alpha=1, constant=0):
        self._alpha = alpha
        self._constant = constant
    def setCenter(self, center):
        self._center = center
    def __call__(self, z):
        r = distance(z, self._center)
        if r < 1/self._alpha:
            return (math.exp(1 / (self._alpha**2 * r**2 - 1)) * (self._alpha**2 * r**2 - 1) - expi(1 / (self._alpha**2 * r**2 - 1))) / (2 * self._alpha**2) + self._constant
        return r + self._constant

class Giulio_F:
    def __init__(self, alpha=1):
        self._alpha = alpha
    def setCenter(self, center):
        self._center = center
    def __call__(self, z):
        r = distance(z, self._center)
        return r * math.erf(r/self._alpha) + (self._alpha/math.sqrt(pi)) * math.pow(e, -(r/self._alpha) ** 2)

class Gaussian_F:
    def __init__(self, alpha=1, constant=0):
        self._alpha = alpha
        self._constant = constant
    def setCenter(self, center):
        self._center = center
    def __call__(self, z):
        r = distance(z, self._center)
        return -math.pow(e, -self._alpha**2 * r**2) / (2 * self._alpha**2) + self._constant

class Multiquadric_F:
    def __init__(self, alpha=1, constant=0):
        self._alpha = alpha
        self._constant = constant
    def setCenter(self, center):
        self._center = center
    def __call__(self, z):
        r = distance(z, self._center)
        return math.pow(self._alpha**2 * r**2 + 1, 3/2) / (3 * self._alpha**2) + self._constant

class InverseQuadratic_F:
    def __init__(self, alpha=1, constant=0):
        self._alpha = alpha
        self._constant = constant
    def setCenter(self, center):
        self._center = center
    def __call__(self, z):
        r = distance(z, self._center)
        return math.log(self._alpha**2 * r**2 + 1) / (2 * self._alpha**2) + self._constant

class InverseMultiquadric_F:
    def __init__(self, alpha=1, constant=0):
        self._alpha = alpha
        self._constant = constant
    def setCenter(self, center):
        self._center = center
    def __call__(self, z):
        r = distance(z, self._center)
        return math.sqrt(self._alpha**2 * r**2 + 1) / (self._alpha**2) + self._constant

class PolyharmonicSpline_F:
    def __init__(self, constant=0, k=0):
        self._k = k
        self._constant = constant
    def setCenter(self, center):
        self._center = center
    def __call__(self, z):
        r = distance(z, self._center)
        if self._k % 2 == 0:
            # Since k and r are positive
            return (r**(self._k+1) * ((self._k+2) * math.log(r) - 1)) / ((self._k+2)**2) + self._constant
        else:
            return r**(self._k+2) / (self._k+2) + self._constant

class ThinPlateSpline_F:
    def __init__(self, constant=0):
        self._constant = constant
    def setCenter(self, center):
        self._center = center
    def __call__(self, z):
        r = distance(z, self._center)
        return r**4 * (math.log(r)/4 - 1/16) + self._constant
