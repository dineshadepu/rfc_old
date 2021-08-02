"""Numerical simulation of interactions between free surface and rigid body
using a robust SPH method, Pengnan Sun, 2015

3.1.3 Water entry of 2-D cylinder

"""
import numpy as np

from pysph.base.kernels import CubicSpline
# from pysph.base.utils import get_particle_array_rigid_body
from pysph.base.utils import get_particle_array
from pysph.sph.integrator import EPECIntegrator
from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser

from rigid_fluid_coupling import RigidFluidCouplingScheme

from pysph.examples.solid_mech.impact import add_properties
from pysph.examples.rigid_body.sphere_in_vessel_akinci import (create_boundary,
                                                               create_fluid,
                                                               create_sphere)
from pysph.tools.geometry import get_2d_block
from geometry import hydrostatic_tank_2d, create_tank_2d_from_block_2d
from pysph.sph.equation import Equation, Group


def create_circle_1(diameter=1, spacing=0.05, center=None):
    dx = spacing
    x = [0.0]
    y = [0.0]
    r = spacing
    nt = 0
    radius = diameter / 2.

    tmp_dist = radius - spacing/2.
    i = 0
    while tmp_dist > spacing/2.:
        perimeter = 2. * np.pi * tmp_dist
        no_of_points = int(perimeter / spacing) + 1
        theta = np.linspace(0., 2. * np.pi, no_of_points)
        for t in theta[:-1]:
            x.append(tmp_dist * np.cos(t))
            y.append(tmp_dist * np.sin(t))
        i = i + 1
        tmp_dist = radius - spacing/2. - i * spacing

    x = np.array(x)
    y = np.array(y)
    x, y = (t.ravel() for t in (x, y))
    if center is None:
        return x, y
    else:
        return x + center[0], y + center[1]


class SpongeLayerDamping(Equation):
    def __init__(self, dest, sources, sponge_width):
        self.sponge_width = sponge_width

        super(SpongeLayerDamping, self).__init__(dest, sources)

    def initialize(self, d_idx, d_x, d_y, d_arho,
                   d_ap, d_au, d_av, d_aw,
                   d_wall_left,
                   d_wall_right,
                   d_wall_bottom,
                   dt):
        if d_x[d_idx] < d_wall_left[0] + self.sponge_width:
            d = abs(d_x[d_idx] - d_wall_left[0])
            lmbda = d / self.sponge_width
            tmp = 0.9**(50. * lmbda)
            fac = 1. - 100.**(-tmp)

            d_arho[d_idx] = fac * d_arho[d_idx]
            d_ap[d_idx] = fac * d_ap[d_idx]
            d_au[d_idx] = fac * d_au[d_idx]
            d_av[d_idx] = fac * d_av[d_idx]
            d_aw[d_idx] = fac * d_aw[d_idx]

        elif d_x[d_idx] > d_wall_right[0] - self.sponge_width:
            d = abs(d_wall_right[0] - d_x[d_idx])
            lmbda = d / self.sponge_width

            tmp = 0.9**(50. * lmbda)
            fac = 1. - 100.**(-tmp)

            d_arho[d_idx] = fac * d_arho[d_idx]
            d_ap[d_idx] = fac * d_ap[d_idx]
            d_au[d_idx] = fac * d_au[d_idx]
            d_av[d_idx] = fac * d_av[d_idx]
            d_aw[d_idx] = fac * d_aw[d_idx]

        elif d_y[d_idx] < d_wall_bottom[0] + self.sponge_width:
            d = abs(d_y[d_idx] - d_wall_bottom[0])
            lmbda = d / self.sponge_width

            tmp = 0.9**(50. * lmbda)
            fac = 1. - 100.**(-tmp)

            d_arho[d_idx] = fac * d_arho[d_idx]
            d_ap[d_idx] = fac * d_ap[d_idx]
            d_au[d_idx] = fac * d_au[d_idx]
            d_av[d_idx] = fac * d_av[d_idx]
            d_aw[d_idx] = fac * d_aw[d_idx]


class RigidFluidCoupling(Application):
    def initialize(self):
        spacing = 0.005
        self.hdx = 1.0

        # the fluid dimensions are
        # x-dimension (this is in the plane of the paper going right)
        self.L = 0.7
        self.sponge_width = 0.1
        self.fluid_length = 1. * self.L
        # y-dimension (this is in the plane of the paper going up)
        self.fluid_height = 1. * self.L

        self.fluid_density = 1000.
        self.fluid_spacing = spacing

        self.tank_length = self.fluid_length
        self.tank_height = 1.4 * self.L
        self.tank_spacing = spacing
        self.tank_layers = 3

        self.body_diameter = 0.11
        self.body_density = 500
        self.body_spacing = spacing
        self.body_h = self.hdx * self.body_spacing

        self.h = self.hdx * self.fluid_spacing

        # self.solid_rho = 500
        # self.m = 1000 * self.dx * self.dx
        self.co = 10 * np.sqrt(2 * 9.81 * self.fluid_height)
        self.p0 = self.fluid_density * self.co**2.
        self.c0 = self.co
        self.alpha = 0.1
        self.gy = -9.81
        self.dim = 2

    def create_particles(self):
        xf, yf, xt, yt = hydrostatic_tank_2d(self.fluid_length,
                                             self.fluid_height,
                                             self.tank_height,
                                             self.tank_layers,
                                             self.fluid_spacing,
                                             self.fluid_spacing)

        m_fluid = self.fluid_density * self.fluid_spacing**self.dim

        fluid = get_particle_array(x=xf,
                                   y=yf,
                                   m=m_fluid,
                                   h=self.h,
                                   rho=self.fluid_density,
                                   name="fluid")
        # set the pressure of the fluid
        fluid.p[:] = - self.fluid_density * self.gy * (max(fluid.y) -
                                                       fluid.y[:])

        tank = get_particle_array(x=xt,
                                  y=yt,
                                  m=m_fluid,
                                  m_fluid=m_fluid,
                                  h=self.h,
                                  rho=self.fluid_density,
                                  rad_s=self.fluid_spacing/2.,
                                  name="tank",
                                  constants={
                                      'E': 69 * 1e9,
                                      'poisson_ratio': 0.3,
                                  })
        tank.add_property('dem_id', type='int', data=1)

        # Translate the tank and fluid so that fluid starts at 0
        min_xf = abs(np.min(xf))

        fluid.x += min_xf
        tank.x += min_xf

        # Create the rigid body
        # print("body spacing", self.body_spacing)
        # print("fluid spacing", self.fluid_spacing)
        xb, yb = create_circle_1(self.body_diameter,
                                 self.body_spacing)
        xb -= np.min(xb) - np.min(fluid.x)
        xb += 65 * 1e-3 - self.body_spacing/2.
        m = self.body_density * self.body_spacing**self.dim
        body = get_particle_array(name='body',
                                  x=xb,
                                  y=yb,
                                  h=self.body_h,
                                  m=m,
                                  rho=self.body_density,
                                  m_fluid=m_fluid,
                                  rad_s=self.body_spacing / 2.,
                                  constants={
                                      'E': 69 * 1e9,
                                      'poisson_ratio': 0.3,
                                  })
        body_id = np.zeros(len(xb), dtype=int)
        body.add_property('body_id', type='int', data=body_id)
        body.add_constant('max_tng_contacts_limit', 30)
        body.add_property('dem_id', type='int', data=0)

        # move the body to the appropriate position
        body.y[:] += max(fluid.y) - min(body.y) + self.fluid_spacing
        # body.y[:] -= 0.25 * self.L
        # body.y[:] -= self.fluid_spacing/2.
        body.x[:] -= min(body.x) - min(fluid.x)
        body.x[:] += 0.5 * self.L
        body.x[:] -= self.body_diameter/2.

        self.scheme.setup_properties([fluid, tank, body])

        self.scheme.scheme.set_linear_velocity(
            body, np.array([0.0, -2.955, 0.]))

        # Remove the fluid particles which are intersecting the gate and
        # gate_support
        # collect the indices which are closer to the stucture
        indices = []
        min_xs = min(body.x)
        max_xs = max(body.x)
        min_ys = min(body.y)
        max_ys = max(body.y)

        xf = fluid.x
        yf = fluid.y
        fac = 1. * self.fluid_spacing
        for i in range(len(fluid.x)):
            if xf[i] < max_xs + fac and xf[i] > min_xs - fac:
                if yf[i] < max_ys + fac and yf[i] > min_ys - fac:
                    indices.append(i)

        fluid.remove_particles(indices)

        # body.y[:] += 0.5
        body.m_fsi[:] += self.fluid_density * self.body_spacing**self.dim
        body.rho_fsi[:] = self.fluid_density

        # add properties for sponge layer
        tmp = (self.tank_layers - 1.) * self.fluid_spacing
        fluid.add_constant('wall_left', min(tank.x) + tmp)
        fluid.add_constant('wall_right', max(tank.x) - tmp)
        fluid.add_constant('wall_bottom', min(tank.y) + tmp)

        return [fluid, tank, body]

    def create_scheme(self):
        rfc = RigidFluidCouplingScheme(rigid_bodies=["body"],
                                       fluids=['fluid'],
                                       boundaries=['tank'],
                                       dim=2,
                                       rho0=self.fluid_density,
                                       p0=self.p0,
                                       c0=self.c0,
                                       gy=self.gy,
                                       nu=0.,
                                       h=None)
        s = SchemeChooser(default='rfc', rfc=rfc)
        return s

    def configure_scheme(self):
        scheme = self.scheme
        scheme.configure(h=self.h)

        dt = 0.25 * self.fluid_spacing * self.hdx / (self.co * 1.1)
        print("DT: %s" % dt)
        tf = 0.6

        self.scheme.configure_solver(dt=dt, tf=tf, pfreq=100)

    def create_equations(self):
        eqns = self.scheme.get_equations()

        # Apply external force
        sponge_eqs = []
        sponge_eqs.append(
            SpongeLayerDamping("fluid", sources=None,
                               sponge_width=self.sponge_width))

        eqns.groups[-1].append(Group(sponge_eqs))

        return eqns

    def customize_output(self):
        self._mayavi_config('''
        particle_arrays['bg'].visible = False
        if 'wake' in particle_arrays:
            particle_arrays['wake'].visible = False
        if 'ghost_inlet' in particle_arrays:
            particle_arrays['ghost_inlet'].visible = False
        for name in ['fluid', 'inlet', 'outlet']:
            b = particle_arrays[name]
            b.scalar = 'p'
            b.range = '-1000, 1000'
            b.plot.module_manager.scalar_lut_manager.lut_mode = 'seismic'
        for name in ['fluid', 'solid']:
            b = particle_arrays[name]
            b.point_size = 2.0
        ''')

    def post_process(self, fname):
        """This function will run once per time step after the time step is
        executed. For some time (self.wall_time), we will keep the wall near
        the cylinders such that they settle down to equilibrium and replicate
        the experiment.
        By running the example it becomes much clear.
        """
        if len(self.output_files) == 0:
            return

        from pysph.solver.utils import iter_output
        from pysph.solver.utils import get_files, load
        import os
        files = self.output_files

        data = load(files[0])
        arrays = data['arrays']
        body = arrays['body']
        xcm_initial = body.xcm[1]

        t = []
        system_x = []
        system_y = []
        for sd, array in iter_output(files[::10], 'body'):
            _t = sd['t']
            t.append(_t)
            system_y.append(xcm_initial - array.xcm[1])

        import matplotlib.pyplot as plt
        t = np.asarray(t)

        # data = np.loadtxt('x_com_zhang.csv', delimiter=',')
        # tx, xcom_zhang = data[:, 0], data[:, 1]

        # plt.plot(tx, xcom_zhang, "s--", label='Experimental')
        plt.plot(t, system_y, "s-", label='Simulated PySPH')
        plt.xlabel("time")
        plt.ylabel("displacement")
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "displacement.png")
        plt.savefig(fig, dpi=300)
        plt.clf()


if __name__ == '__main__':
    app = RigidFluidCoupling()
    app.run()
    app.post_process(app.info_filename)
