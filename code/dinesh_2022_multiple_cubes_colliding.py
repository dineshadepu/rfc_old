"""Multiple cubes colliding

"""
from __future__ import print_function
import numpy as np

from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser

from pysph.base.utils import (get_particle_array)

# from rigid_body_3d import RigidBody3DScheme
from rigid_fluid_coupling import RigidFluidCouplingScheme
from pysph.sph.equation import Equation, Group
import os

from pysph.tools.geometry import get_3d_block, get_2d_tank, rotate


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


def create_circle(diameter=1, spacing=0.05, center=None):
    radius = diameter/2.
    xtmp, ytmp = get_2d_block(spacing, diameter+spacing, diameter+spacing)
    x = []
    y = []
    for i in range(len(xtmp)):
        dist = xtmp[i]**2. + ytmp[i]**2.
        if dist < radius**2:
            x.append(xtmp[i])
            y.append(ytmp[i])

    x = np.array(x)
    y = np.array(y)
    x, y = (t.ravel() for t in (x, y))
    if center is None:
        return x, y
    else:
        return x + center[0], y + center[1]


def hydrostatic_tank_2d(fluid_length, fluid_height, tank_height, tank_layers,
                        fluid_spacing, tank_spacing):
    xt, yt = get_2d_tank(dx=tank_spacing,
                         length=fluid_length + 2. * tank_spacing,
                         height=tank_height,
                         num_layers=tank_layers)
    xf, yf = get_2d_block(dx=fluid_spacing,
                          length=fluid_length,
                          height=fluid_height,
                          center=[-1.5, 1])

    xf += (np.min(xt) - np.min(xf))
    yf -= (np.min(yf) - np.min(yt))

    # now adjust inside the tank
    xf += tank_spacing * (tank_layers)
    yf += tank_spacing * (tank_layers)

    return xf, yf, xt, yt


def get_contact_force_is_boundary(x, y, spacing):
    """To understand this snipped please comment few lines and check the
    viewer to see the boundary particles
    """
    max_y = max(y)
    # bottom solid boundary
    indices_1 = (y > max_y - 0.5 * spacing)

    contact_force_is_boundary = np.ones_like(x) * 0.
    contact_force_is_boundary[indices_1] = 1.

    return contact_force_is_boundary


class Dinesh2022SteadyCubesOnAWall3D(Application):
    def initialize(self):
        self.dim = 3
        spacing = 1e-2

        # x dimension
        self.wall_length = 0.5
        # z dimension
        self.wall_height = 0.
        # y dimension
        self.wall_depth = 0.4
        self.wall_spacing = spacing
        self.wall_layers = 0
        self.wall_rho = 2000.

        # x dimension
        self.rigid_body_length = 0.1
        # z dimension
        self.rigid_body_height = 0.1
        # y dimension
        self.rigid_body_depth = 0.1
        self.rigid_body_spacing = spacing
        self.rigid_body_rho = 2700

        self.angle = 30.

        # simulation properties
        self.hdx = 1.3
        self.alpha = 0.1
        self.gx = 0.
        self.gy = 0.
        self.gz = 0.
        self.h = self.hdx * self.rigid_body_spacing

        # solver data
        self.tf = 1.
        self.dt = 1e-4

        # Rigid body collision related data
        self.limit = 6
        self.seval = None

    def add_user_options(self, group):
        from pysph.sph.scheme import add_bool_argument
        add_bool_argument(group, 'two-cubes', dest='use_two_cubes',
                          default=False, help='Use two cubes')

        add_bool_argument(group, 'three-cubes', dest='use_three_cubes',
                          default=False, help='Use three cubes')

        add_bool_argument(group, 'pyramid-cubes', dest='use_pyramid_cubes',
                          default=False, help='Use pyramid cubes')

    def consume_user_options(self):
        self.use_two_cubes = self.options.use_two_cubes
        self.use_three_cubes = self.options.use_three_cubes
        self.use_pyramid_cubes = self.options.use_pyramid_cubes

    def create_two_cubes(self):
        x = np.array([])
        y = np.array([])
        z = np.array([])

        x1, y1, z1 = get_3d_block(dx=self.rigid_body_spacing,
                                  length=self.rigid_body_length,
                                  height=self.rigid_body_height,
                                  depth=self.rigid_body_depth)

        x2, y2, z2 = get_3d_block(dx=self.rigid_body_spacing,
                                  length=self.rigid_body_length,
                                  height=self.rigid_body_height,
                                  depth=self.rigid_body_depth)

        z2[:] += max(z1) - min(z2) + self.rigid_body_spacing

        x = np.concatenate([x1, x2])
        y = np.concatenate([y1, y2])
        z = np.concatenate([z1, z2])

        body_id = np.array([], dtype=int)
        for i in range(2):
            b_id = np.ones(len(x1), dtype=int) * i
            body_id = np.concatenate((body_id, b_id))

        return x, y, z, body_id

    def create_three_cubes(self):
        x = np.array([])
        y = np.array([])
        z = np.array([])

        x1, y1, z1 = get_3d_block(dx=self.rigid_body_spacing,
                                  length=self.rigid_body_length,
                                  height=self.rigid_body_height,
                                  depth=self.rigid_body_depth)

        x2, y2, z2 = get_3d_block(dx=self.rigid_body_spacing,
                                  length=self.rigid_body_length,
                                  height=self.rigid_body_height,
                                  depth=self.rigid_body_depth)

        z2[:] += max(z1) - min(z2) + self.rigid_body_spacing + self.rigid_body_height/2.

        x3, y3, z3 = get_3d_block(dx=self.rigid_body_spacing,
                                  length=self.rigid_body_length,
                                  height=self.rigid_body_height,
                                  depth=self.rigid_body_depth)
        z3[:] += max(z2) - min(z3) + self.rigid_body_spacing + self.rigid_body_height/2.

        x = np.concatenate([x1, x2, x3])
        y = np.concatenate([y1, y2, y3])
        z = np.concatenate([z1, z2, z3])

        body_id = np.array([], dtype=int)
        for i in range(3):
            b_id = np.ones(len(x1), dtype=int) * i
            body_id = np.concatenate((body_id, b_id))

        return x, y, z, body_id

    def create_pyramid_cubes(self):
        x = np.array([])
        y = np.array([])
        z = np.array([])

        x, y, z = get_3d_block(dx=self.rigid_body_spacing,
                               length=self.rigid_body_length,
                               height=self.rigid_body_height,
                               depth=self.rigid_body_depth)

        body_id = np.array([], dtype=int)
        for i in range(1):
            b_id = np.ones(len(x), dtype=int) * i
            body_id = np.concatenate((body_id, b_id))

        return x, y, z, body_id

    def get_boundary_particles(self, no_bodies):
        # create a row of six cylinders
        x, y, z = get_3d_block(dx=self.rigid_body_spacing,
                               length=self.rigid_body_length,
                               height=self.rigid_body_height,
                               depth=self.rigid_body_depth)

        is_boundary = np.zeros_like(x)
        is_boundary[np.where(x == max(x))] = 1
        is_boundary[np.where(x == min(x))] = 1
        is_boundary[np.where(y == max(y))] = 1
        is_boundary[np.where(y == min(y))] = 1
        is_boundary[np.where(z == max(z))] = 1
        is_boundary[np.where(z == min(z))] = 1

        tmp = is_boundary
        is_boundary_tmp = np.tile(tmp, no_bodies)
        is_boundary = is_boundary_tmp.ravel()

        return is_boundary

    def create_particles(self):
        # =============================
        # Create a rigid body
        # =============================
        if self.use_two_cubes:
            xc, yc, zc, body_id = self.create_two_cubes()
        elif self.use_three_cubes:
            xc, yc, zc, body_id = self.create_three_cubes()
        else:
            xc, yc, zc, body_id = self.create_pyramid_cubes()

        dem_id = body_id
        m = self.rigid_body_rho * self.rigid_body_spacing**self.dim
        h = self.h
        rad_s = self.rigid_body_spacing / 2.
        rigid_body = get_particle_array(name='rigid_body',
                                        x=xc,
                                        y=yc,
                                        z=zc,
                                        h=h,
                                        m=m,
                                        rho=self.rigid_body_rho,
                                        rad_s=rad_s,
                                        constants={
                                            'E': 69 * 1e9,
                                            'poisson_ratio': 0.3,
                                            'spacing0': self.rigid_body_spacing,
                                        })
        rigid_body.add_property('dem_id', type='int', data=dem_id)
        rigid_body.add_property('body_id', type='int', data=body_id)
        rigid_body.add_constant('max_tng_contacts_limit', 10)
        rigid_body.add_constant('total_no_bodies', 2)
        # =============================
        # End creation of rigid body
        # =============================

        self.scheme.setup_properties([rigid_body])

        # Add the boundary particle information to the rigid body
        rigid_body.add_property('contact_force_is_boundary')
        is_boundary = self.get_boundary_particles(max(rigid_body.body_id)+1)
        rigid_body.contact_force_is_boundary[:] = is_boundary[:]
        rigid_body.is_boundary[:] = is_boundary[:]

        # set the rigid body initial velocities
        if self.use_two_cubes:
            # self.scheme.scheme.set_linear_velocity(
            #     rigid_body, np.array([0.0, 0.0, 0.1, 0., 0., -0.1]))
            self.scheme.scheme.set_linear_velocity(
                rigid_body, np.array([0.0, 0.0, 0.0, 0., 0., -0.0]))
        elif self.use_three_cubes:
            self.scheme.scheme.set_linear_velocity(
                rigid_body, np.array([0.0, 0.0, 0.1, 0., 0., -0.1, 0., 0., 0.]))
        else:
            print("TODO")

        return [rigid_body]

    def create_scheme(self):
        rfc = RigidFluidCouplingScheme(rigid_bodies=['rigid_body'],
                                       fluids=None,
                                       boundaries=None,
                                       dim=self.dim,
                                       rho0=1000.,
                                       p0=1000 * 100,
                                       c0=10.,
                                       gz=self.gz,
                                       nu=0.,
                                       h=None)

        s = SchemeChooser(default='rfc', rfc=rfc)
        return s

    def configure_scheme(self):
        tf = self.tf

        self.scheme.configure_solver(dt=self.dt, tf=tf, pfreq=100)

    def post_process(self, fname):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from pysph.solver.utils import load, get_files

        output_files = get_files(os.path.dirname(fname))

        from pysph.solver.utils import iter_output

        t, velocity = [], []

        for sd, rb in iter_output(output_files[::1], 'rigid_body'):
            _t = sd['t']
            t.append(_t)
            vel = (rb.vcm[0]**2. + rb.vcm[1]**2. + rb.vcm[2]**2.)**0.5
            velocity.append(vel)

        # analytical data
        theta = np.pi / 6.
        t_analytical = np.linspace(0., max(t), 100)
        v_analytical = (np.sin(theta) - self.options.fric_coeff * np.cos(theta)) * 9.81 * np.asarray(t_analytical)

        if self.options.fric_coeff > np.tan(theta):
            v_analytical = 0. * np.asarray(t_analytical)

        if 'info' in fname:
            res = os.path.join(os.path.dirname(fname), "results.npz")
        else:
            res = os.path.join(fname, "results.npz")

        np.savez(res,
                 t=t,
                 velocity_rbd=velocity,

                 t_analytical=t_analytical,
                 v_analytical=v_analytical)

        plt.clf()
        plt.plot(t, velocity, "-", label='Mohsen')
        plt.plot(t_analytical, v_analytical, "--", label='Analytical')

        plt.title('Velocity')
        plt.xlabel('t')
        plt.ylabel('Velocity (m/s)')
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "velocity_vs_time.png")
        plt.savefig(fig, dpi=300)
        # ========================
        # x amplitude figure
        # ========================
        # generate plots
        i = 0
        output_files = get_files(fname)
        output_times = np.array([0., 5 * 1e-1, 1. * 1e-0,  2. * 1e-0])

        for sd, body, wall in iter_output(output_files, 'rigid_body', 'wall'):
            _t = sd['t']
            # if _t in output_times:
            if _t in output_times:
                s = 0.2
                # print(_t)
                fig, axs = plt.subplots(1, 1)
                axs.scatter(body.x, body.y, s=s, c=body.m)
                # axs.grid()
                axs.set_aspect('equal', 'box')
                # axs.set_title('still a circle, auto-adjusted data limits', fontsize=10)

                # get the maximum and minimum of the geometry
                x_min = min(body.x) - self.rigid_body_height
                x_max = max(body.x) + 3. * self.rigid_body_height
                y_min = min(body.y) - 4. * self.rigid_body_height
                y_max = max(body.y) + 1. * self.rigid_body_height

                filtr_1 = ((wall.x >= x_min) & (wall.x <= x_max)) & (
                    (wall.y >= y_min) & (wall.y <= y_max))
                wall_x = wall.x[filtr_1]
                wall_y = wall.y[filtr_1]
                wall_m = wall.m[filtr_1]

                tmp = axs.scatter(wall_x, wall_y, s=s, c=wall_m)

                # save the figure
                figname = os.path.join(os.path.dirname(fname), "time" + str(i) + ".png")
                fig.savefig(figname, dpi=300)
                # plt.show()
                i = i + 1

        # =======================================
        # =======================================
        # Schematic
        # =======================================
        files = self.output_files
        for sd, body, wall in iter_output(files[0:2], 'rigid_body', 'wall'):
            _t = sd['t']
            if _t == 0.:
                s = 0.3
                # print(_t)
                fig, axs = plt.subplots(1, 1)
                axs.scatter(body.x, body.y, s=s, c=body.m)
                # axs.grid()
                axs.set_aspect('equal', 'box')
                # axs.set_title('still a circle, auto-adjusted data limits', fontsize=10)

                # im_ratio = tmp.shape[0]/tmp.shape[1]
                x_min = min(body.x) - self.rigid_body_height
                x_max = max(body.x) + 3. * self.rigid_body_height
                y_min = min(body.y) - 4. * self.rigid_body_height
                y_max = max(body.y) + 1. * self.rigid_body_height

                filtr_1 = ((wall.x >= x_min) & (wall.x <= x_max)) & (
                    (wall.y >= y_min) & (wall.y <= y_max))
                wall_x = wall.x[filtr_1]
                wall_y = wall.y[filtr_1]
                wall_m = wall.m[filtr_1]
                tmp = axs.scatter(wall_x, wall_y, s=s, c=wall_m)
                axs.axis('off')
                axs.set_xticks([])
                axs.set_yticks([])

                # save the figure
                figname = os.path.join(os.path.dirname(fname), "pre_schematic.png")
                fig.savefig(figname, dpi=300)


if __name__ == '__main__':
    app = Dinesh2022SteadyCubesOnAWall3D()
    app.run()
    app.post_process(app.info_filename)

# ft_x, ft_y, z
# fn_x, fn_y, z
# u, v, w
# delta_lt_x, delta_lt_y, delta_lt_z
