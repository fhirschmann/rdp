import numpy as np


def pldist(x0, x1, x2):
    return np.divide(np.linalg.norm(np.linalg.det([x2 - x1, x1 - x0])),
                     np.linalg.norm(x2 - x1))


def rdp(points, epsilon=0, dist=pldist):
    if epsilon <= 0:
        return points

    if len(points) < 3:
        return points

    dmax = 0.0
    index = -1
    for i in xrange(1, len(points)):
        d = pldist(np.array(points[i]), np.array(points[0]), np.array(points[-1]))
        if d > dmax:
            index = i
            dmax = d
    if dmax > epsilon:
        r1 = rdp(points[:index + 1], epsilon)
        r2 = rdp(points[index:], epsilon)
        return r1[:-1] + r2
    else:
        return [points[0], points[-1]]
