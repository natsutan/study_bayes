import numpy as np
import random
# ディレクレ分布の可視化
# http://blog.bogatron.net/blog/2014/02/02/visualizing-dirichlet-distributions/

import matplotlib.pyplot as plt
import matplotlib.tri as tri
from functools import reduce

DICE_THRESH = [0.33, 0.80]
MAX_DICE = 150

corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
triangle = tri.Triangulation(corners[:, 0], corners[:, 1])

refiner = tri.UniformTriRefiner(triangle)
trimesh = refiner.refine_triangulation(subdiv=4)

plt.figure(figsize=(8, 4))

midpoints = [(corners[(i + 1) % 3] + corners[(i + 2) % 3]) / 2.0 \
             for i in range(3)]



def xy2bc(xy, tol=1.e-3):
    '''Converts 2D Cartesian coordinates to barycentric.'''
    s = [(corners[i] - midpoints[i]).dot(xy - midpoints[i]) / 0.75 \
         for i in range(3)]
    return np.clip(s, tol, 1.0 - tol)


class Dirichlet(object):
    def __init__(self, alpha):
        from math import gamma
        from operator import mul
        self._alpha = np.array(alpha)
        self._coef = gamma(np.sum(self._alpha)) / reduce(mul, [gamma(a) for a in self._alpha])

    def pdf(self, x):
        '''Returns pdf value for `x`.'''
        from operator import mul
        return self._coef * reduce(mul, [xx ** (aa - 1)
                                         for (xx, aa) in zip(x, self._alpha)])

def draw_pdf_contours(dist, fname, nlevels=200, subdiv=8, **kwargs):
    import math

    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    pvals = [dist.pdf(xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]

    plt.tricontourf(trimesh, pvals, nlevels, **kwargs)
    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 0.75**0.5)
    plt.axis('off')

    plt.savefig('img/%s' % fname)
    plt.clf()

def throw_dice():
    v = random.random()

    if v < DICE_THRESH[0]:
        return 0
    elif v < DICE_THRESH[1]:
        return 1
    else:
        return 2



def main():
    xs = []
    thetas = [1, 1, 1]

    draw_pdf_contours(Dirichlet(thetas), "output00.png")
    for i in range(MAX_DICE):
        v = throw_dice()
        thetas[v] += 1
        if i % 5 == 0:
            fname = "output%02d.png" % ((i // 5) + 1)
            print(fname)
            draw_pdf_contours(Dirichlet(thetas), fname)


if __name__ == '__main__':
    main()