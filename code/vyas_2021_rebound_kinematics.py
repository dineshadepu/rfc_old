"""
Test the collision of two rigid bodues made of same particle array
"""
import numpy as np
from math import cos, sin

from pysph.base.kernels import CubicSpline
# from pysph.base.utils import get_particle_array_rigid_body
from pysph.base.utils import get_particle_array
from pysph.sph.integrator import EPECIntegrator
from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser

from rigid_body_3d import RigidBody3DScheme
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


class Simulation1Dinesh2022BouncingCubeOnAWall2D(Application):
    def initialize(self):
        spacing = 0.05
        self.hdx = 1.3

        self.fluid_length = 150 * 1e-3
        self.fluid_height = 10 * 1e-3
        self.fluid_density = 1000.0
        self.fluid_spacing = spacing

        self.tank_height = 10 * 1e-3
        self.tank_layers = 1
        self.tank_spacing = spacing

        self.rigid_body_diameter = 50 * 1e-3
        self.rigid_body_height = 0.2
        self.rigid_body_length = 0.2
        self.rigid_body_density = 2650.
        self.rigid_body_spacing = spacing / 2.
        self.rigid_body_h = self.hdx * self.rigid_body_spacing

        self.h = self.hdx * self.fluid_spacing

        # self.solid_rho = 500
        # self.m = 1000 * self.dx * self.dx
        self.co = 10 * np.sqrt(2 * 9.81 * self.fluid_height)
        self.p0 = self.fluid_density * self.co**2.
        self.c0 = self.co
        self.alpha = 0.1
        self.gx = 0.
        self.gy = 0.
        self.gz = 0.
        self.dim = 2

    def add_user_options(self, group):
        group.add_argument("--spacing",
                           action="store",
                           type=float,
                           dest="spacing",
                           default=1 * 1e-3,
                           help="Spacing (default to 0.05m)")

        group.add_argument("--velocity",
                           action="store",
                           type=float,
                           dest="velocity",
                           default=5.,
                           help="Velocity (default to 5.)")

        group.add_argument("--angle",
                           action="store",
                           type=float,
                           dest="angle",
                           default=10.,
                           help="Angle (default to 10. degrees)")

    def consume_user_options(self):
        self.velocity = self.options.velocity
        self.angle = self.options.angle

        self.dx = self.options.spacing
        self.rigid_body_spacing = self.dx / 2.
        self.hdx = 1.0
        self.h = self.hdx * self.dx
        # self.dt = 0.25 * self.h / ((self.E / self.rho0)**0.5 + 2.85)
        # print("timestep is ", self.dt)

    def create_particles(self):
        xf, yf, xt, yt = hydrostatic_tank_2d(
            self.fluid_length, self.fluid_height, self.tank_height,
            self.tank_layers, self.rigid_body_spacing, self.rigid_body_spacing)

        m_fluid = self.fluid_density * self.fluid_spacing**2.

        xb, yb = create_circle_1(diameter=self.rigid_body_diameter,
                                 spacing=self.rigid_body_spacing, center=None)
        # get_2d_block(dx=self.rigid_body_spacing,
        #                       length=self.rigid_body_length,
        #                       height=self.rigid_body_height)
        m = self.rigid_body_density * self.rigid_body_spacing**self.dim
        rigid_body = get_particle_array(name='rigid_body',
                                  x=xb,
                                  y=yb,
                                  h=self.rigid_body_h,
                                  m=m,
                                  rho=self.rigid_body_density,
                                  m_fluid=m_fluid,
                                  rad_s=self.rigid_body_spacing / 2.,
                                  constants={
                                      'E': 69 * 1e9,
                                      'poisson_ratio': 0.3,
                                      'spacing0': self.rigid_body_spacing,
                                  })
        # rigid_body.y[:] += self.rigid_body_height * 1.

        body_id = np.zeros(len(xb), dtype=int)
        dem_id = np.zeros(len(xb), dtype=int)
        rigid_body.add_property('body_id', type='int', data=body_id)
        rigid_body.add_property('dem_id', type='int', data=dem_id)
        rigid_body.add_constant('total_no_bodies', [2])

        # ===============================
        # create a tank
        # ===============================
        x, y = xt, yt
        dem_id = body_id
        m = self.rigid_body_density * self.rigid_body_spacing**2
        h = self.rigid_body_h
        rad_s = self.rigid_body_spacing / 2.

        tank = get_particle_array(name='tank',
                                  x=x,
                                  y=y,
                                  h=h,
                                  m=m,
                                  rho=self.rigid_body_density,
                                  rad_s=rad_s,
                                  constants={
                                      'E': 69 * 1e9,
                                      'poisson_ratio': 0.3,
                                  })
        max_dem_id = max(dem_id)
        tank.add_property('dem_id', type='int', data=max_dem_id + 1)

        # ==================================================
        # adjust the rigid body positions on top of the wall
        # ==================================================
        rigid_body.y[:] += abs(min(tank.y) - max(rigid_body.y)) + 3. * self.rigid_body_spacing

        self.scheme.setup_properties([rigid_body, tank])

        rigid_body.add_property('contact_force_is_boundary')
        rigid_body.contact_force_is_boundary[:] = rigid_body.is_boundary[:]

        tank.add_property('contact_force_is_boundary')
        tank.contact_force_is_boundary[:] = tank.is_boundary[:]

        vel = self.velocity
        angle = self.angle / 180 * np.pi
        self.scheme.scheme.set_linear_velocity(
            rigid_body, np.array([vel * sin(angle),
                                  -vel * cos(angle), 0.]))

        # remove particles which are not boundary
        indices = []
        for i in range(len(rigid_body.y)):
            if rigid_body.is_boundary[i] == 0:
                indices.append(i)
        rigid_body.remove_particles(indices)

        # set the tank particles as boundayy
        tank.is_boundary[:] = 1.
        tank.add_property('contact_force_is_boundary')
        tank.contact_force_is_boundary[:] = tank.is_boundary[:]
        # # remove particles outside the circle
        # indices = []
        # for i in range(len(tank.x)):
        #     if tank.is_boundary[i] == 0:
        #         indices.append(i)

        coeff_of_rest = np.array([1., 1.0, 1.0, 1.0])
        rigid_body.add_constant('coeff_of_rest', coeff_of_rest)
        setup_damping_coefficient(rigid_body, [rigid_body], boundaries=[tank])

        # print(rigid_body.eta)

        # tank.remove_particles(indices)

        return [rigid_body, tank]

    def create_scheme(self):
        rb3d = RigidBody3DScheme(rigid_bodies=['rigid_body'],
                                 boundaries=['tank'], dim=self.dim,
                                 gy=self.gy, kr=1e7)

        s = SchemeChooser(default='rb3d', rb3d=rb3d)
        return s

    def configure_scheme(self):
        dt = 1 * 1e-5
        print("DT: %s" % dt)
        tf = 0.001

        self.scheme.configure_solver(dt=dt, tf=tf, pfreq=20)

    def post_process(self, fname):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from pysph.solver.utils import load, get_files

        info = self.read_info(fname)
        output_files = self.output_files

        from pysph.solver.utils import iter_output

        t = []
        rad = self.options.angle * np.pi / 180
        non_dim_theta = [np.tan(rad) / self.options.fric_coeff]
        non_dim_omega = []

        for sd, rb in iter_output(output_files[-1:-2:-1], 'rigid_body'):
            _t = sd['t']
            t.append(_t)
            omega = rb.omega[2]
            tmp = 0.5 * self.rigid_body_diameter * omega / self.options.fric_coeff
            non_dim_omega.append(tmp / (5. * cos(rad)))
        print(non_dim_omega)

        path = os.path.abspath(__file__)
        directory = os.path.dirname(path)

        # experimental data (read from file)
        # load the data
        thornton = np.loadtxt(
            os.path.join(directory, 'vyas_2021_rebound_kinematics_Thornton_omega_vs_theta.csv'),
            delimiter=',')
        theta_exp, omega_exp = thornton[:, 0], thornton[:, 1]

        if 'info' in fname:
            res = os.path.join(os.path.dirname(fname), "results.npz")
        else:
            res = os.path.join(fname, "results.npz")

        np.savez(res,
                 non_dim_theta=non_dim_theta,
                 non_dim_omega=non_dim_omega,

                 theta_exp=theta_exp,
                 omega_exp=omega_exp)

        plt.clf()
        plt.scatter(non_dim_theta, non_dim_omega, label='Simulated')
        plt.plot(theta_exp, omega_exp, '^-', label='Thornton')

        plt.title('Theta_vs_Omega')
        plt.xlabel('non dimensional theta')
        plt.ylabel('non dimensional Omega')
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "omega_vs_theta.png")
        plt.savefig(fig, dpi=300)
        # ========================
        # x amplitude figure
        # ========================


if __name__ == '__main__':
    app = Simulation1Dinesh2022BouncingCubeOnAWall2D()
    app.run()
    app.post_process(app.info_filename)
