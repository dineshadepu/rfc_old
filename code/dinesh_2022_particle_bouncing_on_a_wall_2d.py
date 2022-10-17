"""
Single particle bouncing on a wall
"""
import numpy as np

from pysph.base.kernels import QuinticSpline
# from pysph.base.utils import get_particle_array_rigid_body
from pysph.base.utils import get_particle_array
from pysph.sph.integrator import EPECIntegrator
from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser

from dem_2d import DEMScheme
from rigid_body_common import setup_damping_coefficient

from rigid_fluid_coupling import RigidFluidCouplingScheme
from geometry import hydrostatic_tank_2d
from rigid_fluid_coupling import get_files_at_given_times_from_log
import os

from pysph.examples.solid_mech.impact import add_properties
from pysph.examples.rigid_body.sphere_in_vessel_akinci import (create_boundary,
                                                               create_fluid,
                                                               create_sphere)
from pysph.tools.geometry import get_2d_block


class ParticleOnAWall2D(Application):
    def initialize(self):
        spacing = 0.5
        self.hdx = 1.3

        self.fluid_length = 1.0
        self.fluid_height = 1.0
        self.fluid_density = 1000.0
        self.fluid_spacing = spacing

        self.tank_height = 1.5
        self.tank_layers = 5
        self.tank_spacing = spacing

        self.particle_density = 20
        self.particle_spacing = spacing
        self.particle_radius = spacing
        self.particle_h = self.hdx * self.particle_spacing

        self.h = self.hdx * self.fluid_spacing

        # self.solid_rho = 500
        # self.m = 1000 * self.dx * self.dx
        self.co = 10 * np.sqrt(2 * 9.81 * self.fluid_height)
        self.p0 = self.fluid_density * self.co**2.
        self.c0 = self.co
        self.alpha = 0.1
        self.gx = 0.
        self.gy = -9.81
        self.gz = 0.
        self.dim = 2

    # def add_user_options(self, group):
    #     from pysph.sph.scheme import add_bool_argument
    #     add_bool_argument(group, 'two-cubes', dest='use_two_cubes',
    #                       default=False, help='Use two cubes')

    #     add_bool_argument(group, 'three-cubes', dest='use_three_cubes',
    #                       default=False, help='Use three cubes')

    #     add_bool_argument(group, 'pyramid-cubes', dest='use_pyramid_cubes',
    #                       default=False, help='Use pyramid cubes')

    # def consume_user_options(self):
    #     self.use_two_cubes = self.options.use_two_cubes
    #     self.use_three_cubes = self.options.use_three_cubes
    #     self.use_pyramid_cubes = self.options.use_pyramid_cubes

    def create_particles(self):
        xb = np.array([0.])
        yb = np.array([1.5])
        m = self.particle_density * (2 * self.particle_radius)**self.dim
        particle = get_particle_array(name='particle',
                                      x=xb,
                                      y=yb,
                                      h=self.particle_h,
                                      m=m,
                                      rho=self.particle_density,
                                      rad_s=self.particle_radius)
        # ===============================
        # create a tank
        # ===============================
        x = np.array([0.])
        y = np.array([0.])
        m = self.particle_density * self.particle_spacing**2
        h = self.particle_h
        rad_s = self.particle_radius

        xb = np.array([0.])
        yb = np.array([0.])
        wall = get_particle_array(name='wall',
                                  x=x,
                                  y=y,
                                  h=h,
                                  m=m,
                                  rho=self.particle_density,
                                  rad_s=rad_s)
        # ==================================================
        # adjust the rigid body positions on top of the wall
        # ==================================================
        # body.y[:] -= min(body.y) - min(tank.y)
        # body.y[:] += self.tank_layers * self.body_spacing

        self.scheme.setup_properties([particle, wall])

        particle.moi[:] = 1.
        particle.v[:] = -5.

        return [particle, wall]

    def create_scheme(self):
        dem2d = DEMScheme(
            granular_particles=['particle'],
            boundaries=['wall'],
            dim=self.dim,
            gy=self.gy)

        s = SchemeChooser(default='dem2d', dem2d=dem2d)
        return s

    def configure_scheme(self):
        dt = 1e-4
        print("DT: %s" % dt)
        tf = 0.3

        self.scheme.configure_solver(dt=dt, tf=tf, pfreq=100)

    def post_process(self, fname):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from pysph.solver.utils import load, get_files

        info = self.read_info(fname)
        output_files = self.output_files

        from pysph.solver.utils import iter_output

        t, y, v, fy = [], [], [], []

        for sd, pa in iter_output(output_files[::1], 'particle'):
            _t = sd['t']
            t.append(_t)
            _y = (pa.y[0])
            y.append(_y)

            v.append(pa.v[0])
            fy.append(pa.fy[0])

        if 'info' in fname:
            res = os.path.join(os.path.dirname(fname), "results.npz")
        else:
            res = os.path.join(fname, "results.npz")

        np.savez(res,
                 t=t,
                 y=y,
                 v=v)

                 # t_analytical=t_analytical,
                 # v_analytical=v_analytical)

        plt.clf()
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(t, y, "g-")
        ax2.plot(t, v, "b-")
        # plt.plot(t_analytical, v_analytical, "--", label='Analytical')

        plt.title('Y position')
        ax1.set_xlabel('t')
        ax1.set_ylabel('Position', color='g')
        ax2.set_ylabel('Velocity', color='b')
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "y_and_v_vs_time.png")
        plt.savefig(fig, dpi=300)

        plt.clf()
        plt.plot(t, fy, "-", label='force')
        # plt.plot(t_analytical, v_analytical, "--", label='Analytical')

        plt.title('Force')
        plt.xlabel('t')
        plt.ylabel('Force(m)')
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "force_vs_time.png")
        plt.savefig(fig, dpi=300)
        # ========================
        # x amplitude figure
        # ========================


if __name__ == '__main__':
    app = ParticleOnAWall2D()
    app.run()
    app.post_process(app.info_filename)
