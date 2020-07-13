import numpy as np
import matplotlib.pyplot as plt
from itertools import product, permutations
from pattern import Pattern


def main():
    for group_name in Pattern.group_names:
        print(group_name)
        p = Pattern(10)
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

        bins = np.linspace(0, rmax, 500)
        hist = np.histogram(dists, bins)
        counts = hist[0]

        rho = (fracts.size/np.linalg.norm(np.cross(p.a1, p.a2)))
        rdf = counts/counts.sum()/rho/(2*np.pi*bins[1:])/(bins[1]-bins[0])

        plt.figure(figsize=(16, 8))
        plt.subplot(121)
        p.translational(n=1)
        p.to_cartesian_coordinate()
        plt.scatter(p.xys[:, 0], p.xys[:, 1])
        plt.axis("equal")
        plt.subplot(122)
        plt.plot(bins[1:], rdf)
        plt.ylim(0, rdf.max()*1.2)
        plt.xlim(-.1, rmax+1)
        plt.show()


if __name__ == '__main__':
    main()
