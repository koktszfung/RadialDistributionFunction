import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from itertools import product
from pattern import Pattern


def main():
    for group_name in Pattern.group_names:
        p = Pattern(10)
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
        trunc_image_cell_carts = image_cell_carts[np.logical_or.reduce(dists <= rmax, 0)]
        dists = dists[np.logical_and(dists > 0, dists <= rmax)]

        bins = np.linspace(0, rmax, 500)
        hist = np.histogram(dists, bins)
        counts = hist[0]

        rho = (center_cell_carts.size/np.linalg.norm(np.cross(p.a1, p.a2)))
        rdf = counts/counts.sum()/rho/(2*np.pi*bins[1:])/(bins[1]-bins[0])

        plt.figure(figsize=(16, 8))
        plt.subplot(121)
        plt.scatter(image_cell_carts[:, 0], image_cell_carts[:, 1])
        plt.scatter(trunc_image_cell_carts[:, 0], trunc_image_cell_carts[:, 1])
        plt.scatter(center_cell_carts[:, 0], center_cell_carts[:, 1])
        plt.axis("equal")
        plt.subplot(122)
        plt.plot(bins[1:], rdf)
        plt.ylim(0, rdf.max()*1.2)
        plt.xlim(-.1, rmax+1)
        plt.show()


if __name__ == '__main__':
    main()
