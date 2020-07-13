import numpy as np
import matplotlib.pyplot as plt
from itertools import product


class Pattern:
    group_names = ["p1", "p2", "pm", "pg", "cm", "pmm", "pmg", "pgg", "cmm",
                   "p4", "p4m", "p4g", "p3", "p3m1", "p31m", "p6", "p6m"]

    def __init__(self, n=0):
        self.n = n
        self.N = 100
        self.a1 = None
        self.a2 = None
        self.xys = None
        self.d = 1e-1
        if n != 0:
            self.reset()

    def random(self, shape=None):
        return np.random.random(shape)*(1 - self.d*2) + self.d

    def reset(self):
        self.xys = self.random((self.n, 2))

    def oblique_basis(self):
        self.a1 = np.array([1, 0])
        r = self.random()
        a = self.random()*np.pi
        self.a2 = np.array([r*np.cos(a), r*np.sin(a)])

    def rectangular_basis(self):
        self.a1 = np.array([1, 0])
        self.a2 = np.array([0, self.random()*.5+.5])

    def rhombic_basis(self):
        a = self.random()*np.pi*.5
        c, s = np.cos(a), np.sin(a)
        self.a1, self.a2 = np.array([c, s]), np.array([-c, s])

    def square_basis(self):
        self.a1 = np.array([1, 0])
        self.a2 = np.array([0, 1])

    def hexagonal_basis(self):
        self.a1 = np.array([1, 0])
        self.a2 = np.array([.5, np.sqrt(3)*.5])

    def rotational(self, fold):
        if len(self.xys) > self.N:
            raise RuntimeWarning("Too many points")
        ras = [(np.linalg.norm([x, y]), np.arctan2(y, x)) for x, y in self.xys]
        das = np.linspace(0, 2*np.pi, fold + 1)[:-1]
        self.xys = np.array([(r*np.cos(a + da), r*np.sin(a + da)) for da in das for r, a in ras])

    def reflectional(self, h: bool):
        if len(self.xys) > self.N:
            raise RuntimeWarning("Too many points")
        image_xys = self.xys.copy()
        if h:
            image_xys[:, 1] = -image_xys[:, 1]
        else:
            image_xys[:, 0] = -image_xys[:, 0]
        self.xys = np.concatenate([image_xys, self.xys])

    def glide_relfectional(self, t, my=0.):
        if len(self.xys) > self.N:
            raise RuntimeWarning("Too many points")
        image_xys = self.xys.copy()
        if t[1] == 0:
            image_xys[:, 1] = -image_xys[:, 1] + my*2
            image_xys[:, 0] -= t[0]*.5
        else:
            image_xys[:, 0] = -image_xys[:, 0]
            image_xys[:, 1] -= t[1]*.5
        self.xys = np.concatenate([image_xys, self.xys])

    def to_fractional_coordinate(self):
        det = self.a1[0]*self.a2[1] - self.a1[1]*self.a2[0]
        self.xys = np.array([((self.a2[1]*x - self.a2[0]*y), (-self.a1[1]*x + self.a1[0]*y)) for x, y in self.xys])/det

    def to_cartesian_coordinate(self):
        self.xys = np.array([(self.a1[0]*x + self.a2[0]*y, self.a1[1]*x + self.a2[1]*y) for x, y in self.xys])

    def translational(self, n=2, cartesian: bool = False):
        if cartesian:
            self.to_fractional_coordinate()
            self.xys = np.mod(self.xys + .5, 1) - .5
            self.xys = np.concatenate([self.xys + np.array(ij) for ij in product(range(-n, n + 1), repeat=2)])
            self.to_cartesian_coordinate()
        else:
            self.xys = np.concatenate([self.xys + np.array(ij) for ij in product(range(-n, n + 1), repeat=2)])

    def group(self, group_name):
        try:
            getattr(self, group_name)()
        except AttributeError:
            print(f"group {group_name} does not exist, fall back to group p1")
            self.p1()

    def p1(self):
        self.oblique_basis()

    def p2(self):
        self.oblique_basis()
        self.rotational(2)

    def pm(self):
        self.rectangular_basis()
        self.reflectional(h=True)

    def pg(self):
        self.rectangular_basis()
        self.glide_relfectional(self.a2)

    def cm(self):
        self.rhombic_basis()
        self.reflectional(h=True)

    def pmm(self):
        self.rectangular_basis()
        self.reflectional(h=True)
        self.reflectional(h=False)

    def pmg(self):
        self.rectangular_basis()
        self.reflectional(h=True)
        self.glide_relfectional(self.a2)

    def pgg(self):
        self.rectangular_basis()
        self.glide_relfectional(self.a1)
        self.glide_relfectional(self.a2)

    def cmm(self):
        self.rhombic_basis()
        self.reflectional(h=True)
        self.reflectional(h=False)

    def p4(self):
        self.square_basis()
        self.rotational(4)

    def p4m(self):
        self.square_basis()
        self.rotational(4)
        self.reflectional(h=True)

    def p4g(self):
        self.square_basis()
        self.rotational(4)
        self.glide_relfectional(self.a1, 0.25)

    def p3(self):
        self.hexagonal_basis()
        self.rotational(3)

    def p3m1(self):
        self.hexagonal_basis()
        self.rotational(3)
        self.reflectional(h=False)

    def p31m(self):
        self.hexagonal_basis()
        self.rotational(3)
        self.reflectional(h=True)

    def p6(self):
        self.hexagonal_basis()
        self.rotational(6)

    def p6m(self):
        self.hexagonal_basis()
        self.rotational(6)
        self.reflectional(h=True)

    def plot(self):
        plt.scatter(self.xys[:, 0], self.xys[:, 1])
        plt.axis("equal")

    def show(self):
        self.plot()
        plt.show()


def main():
    np.random.seed(0)
    p = Pattern(10)
    for group_name in Pattern.group_names:
        print(group_name)
        p.group(group_name)
        print(len(p.xys))
        p.translational()
        p.plot()
        plt.arrow(0, 0, p.a1[0], p.a1[1], head_width=0.05, color="k")
        plt.arrow(0, 0, p.a2[0], p.a2[1], head_width=0.05, color="k")
        plt.show()
        p.reset()


if __name__ == '__main__':
    main()
