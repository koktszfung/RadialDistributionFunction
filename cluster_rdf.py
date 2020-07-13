import os
import numpy as np
from scipy.spatial.distance import cdist
from itertools import product
from pattern import Pattern
import time


def write_dists(n, nr, group_name):
    p = Pattern(n)
    p.group(group_name)
    p.to_fractional_coordinate()
    lattice_vectors = np.array([p.a1, p.a2])
    center_cell_fracts = np.mod(p.xys + .5, 1) - .5
    rmax = np.max([np.linalg.norm(np.dot(ij, lattice_vectors)) for ij in product((-1, 1), repeat=2)])

    num_layer = 36
    x = range(-num_layer, num_layer + 1)
    I, J = np.meshgrid(x, x)
    I, J = I.flatten(), J.flatten()
    xys = I[:, None]*p.a1 + J[:, None]*p.a2
    mask = xys[:, 0]**2 + xys[:, 1]**2 <= rmax*rmax*4
    I, J = I[mask], J[mask]
    image_cell_fracts = np.concatenate([center_cell_fracts + np.array(ij) for ij in zip(I, J)])

    center_cell_carts = np.array([np.dot(fract, lattice_vectors) for fract in center_cell_fracts])
    image_cell_carts = np.array([np.dot(fract, lattice_vectors) for fract in image_cell_fracts])

    dists = cdist(center_cell_carts, image_cell_carts)
    dists = dists[np.logical_and(dists > 0, dists <= rmax)]

    bins = np.linspace(0, rmax, nr)
    hist = np.histogram(dists, bins)
    counts = hist[0]

    rho = (len(center_cell_carts)/np.linalg.norm(np.cross(p.a1, p.a2)))
    return counts/counts.sum()/rho/(2*np.pi*bins[1:])/(bins[1] - bins[0])


def main():
    out_dir = "data/cluster_1/"

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    start_time = time.time()
    nums = [1, 2, 2, 2, 2, 4, 4, 4, 4, 4, 8, 8, 3, 6, 6, 6, 12]
    for num, group_name in zip(nums, Pattern.group_names):
        print(group_name)
        np.savez(out_dir + group_name + ".npz", [write_dists(int(24/num), 501, group_name) for _ in range(1200)])

    print(time.time() - start_time)


if __name__ == '__main__':
    main()
