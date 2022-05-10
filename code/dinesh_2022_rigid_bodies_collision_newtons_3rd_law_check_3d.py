"""
Amaro2019Improvement

An improvement of rigid bodies contact for particle-based
non-smooth walls modeling

https://doi.org/10.1007/s40571-019-00233-4

An implementation of collision of three rigid bodies.
"""
import numpy as np

from pysph.base.kernels import CubicSpline
# from pysph.base.utils import get_particle_array_rigid_body
from pysph.base.utils import get_particle_array
from pysph.sph.integrator import EPECIntegrator
from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser

from rigid_body_3d import RigidBody3DScheme
from geometry import hydrostatic_tank_2d

from pysph.examples.solid_mech.impact import add_properties
from pysph.examples.rigid_body.sphere_in_vessel_akinci import (create_boundary,
                                                               create_fluid,
                                                               create_sphere)
from pysph.tools.geometry import get_2d_block, get_3d_block, rotate


def get_rotated_3d_block(dx, length, height, depth, angle):
    xb_tmp, yb_tmp = get_2d_block(dx=dx, length=length,
                                  height=height)
    xb, yb, _zs = rotate(xb_tmp, yb_tmp, np.zeros_like(xb_tmp),
                         axis=np.array([0., 0., 1.]),
                         angle=-angle)
    x = np.array([])
    y = np.array([])
    z = np.array([])

    z_linspace = np.arange(0., depth, dx)

    for i in range(len(z_linspace)):
        z1 = np.ones(len(xb), dtype=int) * z_linspace[i]
        x = np.concatenate((x, xb))
        y = np.concatenate((y, yb))
        z = np.concatenate((z, z1))

    return x, y, z


class RigidFluidCoupling(Application):
    def initialize(self):
        spacing = 0.03
        self.hdx = 1.3

        self.fluid_length = 1.0
        self.fluid_height = 1.0
        self.fluid_density = 1000.0
        self.fluid_spacing = spacing

        self.tank_height = 1.5
        self.tank_layers = 3
        self.tank_spacing = spacing

        self.body_height = 0.2 - spacing/2.
        self.body_length = 0.2 - spacing/2.
        self.body_depth = 0.2 - spacing/2.
        self.body_density = 2000
        self.body_spacing = spacing / 2.
        self.body_h = self.hdx * self.body_spacing
        self.angle = 30

        self.h = self.hdx * self.fluid_spacing

        # self.solid_rho = 500
        # self.m = 1000 * self.dx * self.dx
        self.co = 10 * np.sqrt(2 * 9.81 * self.fluid_height)
        self.p0 = self.fluid_density * self.co**2.
        self.c0 = self.co
        self.alpha = 0.1
        self.gy = 0.
        self.dim = 3

    def add_user_options(self, group):
        group.add_argument("--angle",
                           action="store",
                           type=float,
                           dest="angle",
                           default=30.,
                           help="Rotation angle of body in degrees(defaulted to 30)")

    def consume_user_options(self):
        self.angle = self.options.angle

    def create_particles(self):
        xb, yb, zb = get_3d_block(dx=self.body_spacing,
                                  length=self.body_length,
                                  height=self.body_height,
                                  depth=self.body_depth)
        self.body_density = 2000.
        m = self.body_density * self.body_spacing**self.dim
        # left body
        body1 = get_particle_array(name='body1',
                                   x=xb,
                                   y=yb,
                                   z=zb,
                                   h=self.body_h,
                                   m=m,
                                   rho=self.body_density,
                                   m_fluid=0.,
                                   rad_s=self.body_spacing / 2.,
                                   constants={
                                       'E': 69 * 1e9,
                                       'poisson_ratio': 0.3,
                                       'spacing0': self.body_spacing,
                                   })
        body_id = np.zeros(len(xb), dtype=int)
        dem_id = np.zeros(len(xb), dtype=int)
        body1.add_property('body_id', type='int', data=body_id)
        body1.add_property('dem_id', type='int', data=dem_id)
        body1.add_constant('total_no_bodies', [2])

        # right bottom body
        # Body 2 starts
        xb, yb, zb = get_rotated_3d_block(dx=self.body_spacing,
                                          length=self.body_length,
                                          height=self.body_height,
                                          depth=self.body_depth,
                                          angle=self.angle)
        self.body_density = 1500.
        m = self.body_density * self.body_spacing**self.dim
        body2 = get_particle_array(name='body2',
                                   x=xb,
                                   y=yb,
                                   z=zb,
                                   h=self.body_h,
                                   m=m,
                                   rho=self.body_density,
                                   m_fluid=0.,
                                   rad_s=self.body_spacing / 2.,
                                   constants={
                                       'E': 69 * 1e9,
                                       'poisson_ratio': 0.3,
                                       'spacing0': self.body_spacing,
                                   })
        body_id = np.zeros(len(xb), dtype=int)
        dem_id = np.ones(len(xb), dtype=int)
        body2.add_property('body_id', type='int', data=body_id)
        body2.add_property('dem_id', type='int', data=dem_id)
        body2.add_constant('total_no_bodies', [2])

        xc, yc, _zs = rotate(body2.x, body2.y, body2.z,
                             axis=np.array([0., 0., 1.]),
                             angle=-self.angle)
        body2.x[:] = xc[:]
        body2.y[:] = yc[:]

        # xb += max(body1.x) - min(xb) + 0.03
        body2.y += max(body1.y) - min(body2.y) + self.body_spacing * 1.
        # body2.y += self.body_spacing * 1.

        self.scheme.setup_properties([body1, body2])

        body1.add_property('contact_force_is_boundary')
        body1.contact_force_is_boundary[:] = body1.is_boundary[:]
        body2.add_property('contact_force_is_boundary')
        body2.contact_force_is_boundary[:] = body2.is_boundary[:]

        self.scheme.scheme.set_linear_velocity(body2, np.array([0., -1., 0.]))
        # self.scheme.scheme.set_linear_velocity(
        #     body2, np.array([-0.5, 0.0, 0.]))
        # self.scheme.scheme.set_linear_velocity(body3, np.array([0.0, 0.0, 0.]))
        return [body1, body2]

    def create_scheme(self):
        rb3d = RigidBody3DScheme(rigid_bodies=['body1', 'body2'],
                                 boundaries=None, dim=3)
        s = SchemeChooser(default='rb3d', rb3d=rb3d)
        return s

    def configure_scheme(self):
        # dt = 0.125 * self.fluid_spacing * self.hdx / (self.co * 1.1)
        dt = 5e-5
        print("DT: %s" % dt)
        tf = 0.01

        self.scheme.configure_solver(dt=dt, tf=tf, pfreq=100)

    def post_process(self, fname):
        """This function will run once per time step after the time step is
        executed. For some time (self.wall_time), we will keep the wall near
        the cylinders such that they settle down to equilibrium and replicate
        the experiment.
        By running the example it becomes much clear.
        """
        from pysph.solver.utils import iter_output
        import os
        if len(self.output_files) == 0:
            return

        files = self.output_files
        print(len(files))
        t = []
        fx_body_1 = []
        fx_body_2 = []
        fy_body_1 = []
        fy_body_2 = []
        fz_body_1 = []
        fz_body_2 = []

        for sd, body1, body2 in iter_output(files[::1], 'body1', 'body2'):
            _t = sd['t']
            t.append(_t)
            fx_body_1.append(body1.force[0])
            fx_body_2.append(body2.force[0])

            fy_body_1.append(body1.force[1])
            fy_body_2.append(body2.force[1])

            fz_body_1.append(body1.force[2])
            fz_body_2.append(body2.force[2])

        import matplotlib.pyplot as plt
        t = np.asarray(t)
        # t = t - 0.1
        # print(t)

        # data = np.loadtxt('../x_com_zhang.csv', delimiter=',')
        # tx, xcom_zhang = data[:, 0], data[:, 1]

        # plt.plot(tx, xcom_zhang, "s--", label='Simulated PySPH')
        # plt.plot(t, system_x, "s-", label='Experimental')
        # plt.xlabel("time")
        # plt.ylabel("x/L")
        # plt.legend()
        # plt.savefig("xcom", dpi=300)
        # plt.clf()

        # data = np.loadtxt('../y_com_zhang.csv', delimiter=',')
        # ty, ycom_zhang = data[:, 0], data[:, 1]

        # plt.plot(ty, ycom_zhang, "s--", label='Experimental')
        tmp_1 = -np.asarray(fx_body_2)
        tmp_2 = -np.asarray(fy_body_2)
        tmp_3 = -np.asarray(fz_body_2)
        plt.plot(t, fx_body_1, "^-", label='Fx body 1', mfc="none")
        plt.plot(t, tmp_1, "v--", label='Fx body 2', mfc="none")
        plt.plot(t, fy_body_1, "<-", label='Fy body 1', mfc="none")
        plt.plot(t, tmp_2, ">--", label='Fy body 2', mfc="none")
        plt.plot(t, fz_body_1, "<-", label='Fz body 1', mfc="none")
        plt.plot(t, tmp_3, ">--", label='Fz body 2', mfc="none")
        plt.xlabel("time")
        plt.ylabel("Force")
        plt.legend()

        fig = os.path.join(os.path.dirname(fname), "force_vs_time.png")
        plt.savefig(fig, dpi=300)


if __name__ == '__main__':
    app = RigidFluidCoupling()
    app.run()
    app.post_process(app.info_filename)

# Post processing variables
# contact_force_normal_x[0::2], contact_force_normal_y[0::2], contact_force_normal_z[0::2]
# fx, fy, fz
