"""A cube translating and rotating freely without the influence of gravity.
This is used to test the rigid body dynamics equations.
"""

import numpy as np

from pysph.base.kernels import CubicSpline
# from pysph.base.utils import get_particle_array_rigid_body
from pysph.base.utils import get_particle_array
from pysph.sph.integrator import EPECIntegrator
from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser
from rigid_body_3d import RigidBody3DScheme

from pysph.examples.solid_mech.impact import add_properties


class Case0(Application):
    def initialize(self):
        self.rho0 = 10.0
        self.hdx = 1.3
        self.dx = 0.02
        self.dy = 0.02
        self.kn = 1e4
        self.mu = 0.5
        self.en = 1.0
        self.dim = 2

        self.dt = 1e-3
        self.tf = 10

    def create_scheme(self):
        rb3d = RigidBody3DScheme(rigid_bodies=['body'], boundaries=None, dim=2)
        s = SchemeChooser(default='rb3d', rb3d=rb3d)
        return s

    def configure_scheme(self):
        dt = self.dt
        tf = self.tf
        self.scheme.configure_solver(dt=dt, tf=tf, pfreq=100)

    def create_particles(self):
        nx, ny = 10, 10
        dx = self.dx
        x, y = np.mgrid[0:1:nx * 1j, 0:1:ny * 1j]
        x = x.flat
        y = y.flat
        m = np.ones_like(x) * dx * dx * self.rho0
        h = np.ones_like(x) * self.hdx * dx
        # radius of each sphere constituting in cube
        rad_s = np.ones_like(x) * dx
        body = get_particle_array(name='body', x=x, y=y, h=h, m=m, rad_s=rad_s,
                                       constants={
                                           'E': 69 * 1e9,
                                           'poisson_ratio': 0.3,
                                           'initial_spacing0': self.dx
                                       })
        body_id = np.zeros(len(x), dtype=int)
        dem_id = np.zeros(len(x), dtype=int)
        body.add_property('body_id', type='int', data=body_id)
        body.add_property('dem_id', type='int', data=dem_id)
        body.add_constant('total_no_bodies', [1])

        # setup the properties
        self.scheme.setup_properties([body])

        body.add_property('contact_force_is_boundary')
        body.contact_force_is_boundary[:] = body.is_boundary[:]

        self.scheme.scheme.set_linear_velocity(body, np.array([0.5, 0.5, 0.]))
        self.scheme.scheme.set_angular_velocity(body, np.array([0., 0., 1.]))

        # body.vcm[0] = 0.5
        # body.vcm[1] = 0.5
        # body.omega[2] = 1.

        return [body]

    def post_process(self, fname):
        if len(self.output_files) == 0:
            return

        from pysph.solver.utils import iter_output

        files = self.output_files
        files = files[3:]
        t, total_energy = [], []
        x, y = [], []
        for sd, body in iter_output(files, 'body'):
            _t = sd['t']
            t.append(_t)
            total_energy.append(0.5 * np.sum(body.m[:] * (body.u[:]**2. +
                                                          body.v[:]**2.)))
            x.append(body.xcm[0])
            y.append(body.xcm[1])

        import matplotlib
        import os
        # matplotlib.use('Agg')

        from matplotlib import pyplot as plt

        # res = os.path.join(self.output_dir, "results.npz")
        # np.savez(res, t=t, amplitude=amplitude)

        # gtvf data
        # data = np.loadtxt('./oscillating_plate.csv', delimiter=',')
        # t_gtvf, amplitude_gtvf = data[:, 0], data[:, 1]

        plt.clf()

        # plt.plot(t_gtvf, amplitude_gtvf, "s-", label='GTVF Paper')
        plt.plot(t, total_energy, "-", label='Simulated')

        plt.xlabel('t')
        plt.ylabel('total energy')
        plt.legend()
        fig = os.path.join(self.output_dir, "total_energy_vs_t.png")
        # plt.show()
        # plt.savefig(fig, dpi=300)

        plt.plot(x, y, label='Simulated')
        # plt.show()


if __name__ == '__main__':
    app = Case0()
    app.run()
    app.post_process(app.info_filename)
