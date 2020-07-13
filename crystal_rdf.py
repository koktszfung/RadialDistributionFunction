import os
import numpy as np
from itertools import product, permutations
from pattern import Pattern
import time


def write_dists(n, nr, group_name):
    p = Pattern(n)
    p.group(group_name)
    p.to_fractional_coordinate()
    lattice_vectors = np.array([p.a1, p.a2])
    fracts = np.mod(p.xys+.5, 1)-.5
    rmax = np.max([np.linalg.norm(np.dot(ij, lattice_vectors)) for ij in product((-1, 1), repeat=2)])

    fract_disps = np.array([i - j for i, j in permutations(fracts, r=2)])
    fract_disps = np.mod(fract_disps+.5, 1)-.5
    cart_disps = np.array([np.dot(ij, lattice_vectors) for ij in fract_disps])
    dists = np.linalg.norm(cart_disps, axis=1)
    dists = dists[np.logical_and(dists > 1e-10, dists <= rmax)]

    bins = np.linspace(0, rmax, nr)
    hist = np.histogram(dists, bins)
    counts = hist[0]

    rho = (len(fracts)/np.linalg.norm(np.cross(p.a1, p.a2)))
    return counts/counts.sum()/rho/(2*np.pi*bins[1:])/(bins[1]-bins[0])


def main():
    out_dir = "data/crystal_2/"

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    start_time = time.time()
    for group_name in Pattern.group_names:
        print(group_name)
        np.savez(os.path.join(out_dir, group_name + ".npz"), [write_dists(5, 501, group_name) for _ in range(1200)])

    print(time.time() - start_time)


if __name__ == '__main__':
    main()
