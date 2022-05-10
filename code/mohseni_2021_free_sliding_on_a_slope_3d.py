"""[1] Particle-Based Numerical Simulation Study of Solid Particle
Erosion of Ductile Materials Leading to an Erosion Model,
Including the Particle Shape Effect

https://doi.org/10.3390/ma15010286


3.3.2 Free sliding on a slope

"""
from __future__ import print_function
import numpy as np

from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser

from pysph.base.utils import (get_particle_array)

from rigid_body_3d import RigidBody3DScheme
from rigid_fluid_coupling import get_files_at_given_times_from_log
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


def setup_properties_for_gradual_force(pa):
    pa.add_constant('normal_force_time', 0.)
    pa.add_constant('tmp_normal_force', 0.)
    pa.add_constant('delta_fn', 0.)

    pa.add_constant('tangential_force_time', 0.)
    pa.add_constant('tmp_tangential_force', 0.)
    pa.add_constant('delta_ft', 0.)

    force_idx_fn = np.zeros_like(pa.x)
    force_idx_ft = np.zeros_like(pa.x)
    pa.add_property('force_idx_fn', type='int', data=force_idx_fn)
    pa.add_property('force_idx_ft', type='int', data=force_idx_ft)


def force_index_single(body):
    max_y = np.max(body.y)

    indices_fn = np.where(max_y == body.y)[0]
    body.force_idx_fn[indices_fn] = 1

    min_x = np.min(body.x)
    indices_ft = np.where(min_x == body.x)[0]
    body.force_idx_ft[indices_ft] = 1


class ApplyForceOnRigidBody(Equation):
    def initialize(self, d_idx,
                   d_m,
                   d_fx,
                   d_fy,
                   d_fz,
                   d_normal_force_time,
                   d_tmp_normal_force,
                   d_delta_fn,
                   d_force_idx_fn,
                   d_tangential_force_time,
                   d_tmp_tangential_force,
                   d_delta_ft,
                   d_force_idx_ft,
                   dt, t):
        t_1 = d_normal_force_time[0]

        if t <= t_1:
            if d_idx == 1:
                d_tmp_normal_force[0] += d_delta_fn[0]

        if d_force_idx_fn[d_idx] == 1:
            d_fy[d_idx] += d_tmp_normal_force[0]

        t_2 = d_normal_force_time[0] + d_tangential_force_time[0]
        if t > t_1 and t <= t_2:
            if d_idx == 1:
                d_tmp_tangential_force[0] += d_delta_ft[0]

        if d_force_idx_ft[d_idx] == 1:
            d_fx[d_idx] += d_tmp_tangential_force[0]


class Mohseni2021FreeSlidingOnASlope(Application):
    def initialize(self):
        self.dim = 3
        spacing = 1e-2

        # x dimension
        self.wall_length = 10.
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
        self.rigid_body_depth = 0.2
        self.rigid_body_spacing = spacing
        self.rigid_body_rho = 2700

        self.angle = 30.

        # simulation properties
        self.hdx = 1.5
        self.alpha = 0.1
        self.gx = 0.
        self.gy = -9.81
        self.gz = 0.
        self.h = self.hdx * self.rigid_body_spacing

        # solver data
        self.tf = 1.
        self.dt = 1e-4

        # Rigid body collision related data
        self.limit = 6
        self.seval = None

        # force application parameters
        self.fn = -2. * 1e3
        self.ft = 4. * 1e3

        self.normal_force_time = 0.5
        self.tangential_force_time = 0.5

        timesteps_fn = self.normal_force_time / self.dt
        self.fn_increment = self.fn / timesteps_fn

        timesteps_ft = self.tangential_force_time / self.dt
        self.ft_increment = self.ft / timesteps_ft

    def create_rigid_body(self):
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
        from boundary_particles import (get_boundary_identification_etvf_equations,
                                        add_boundary_identification_properties)
        from pysph.tools.sph_evaluator import SPHEvaluator
        from pysph.base.kernels import (QuinticSpline)
        # create a row of six cylinders
        x, y, z, body_id = self.create_rigid_body()

        m = self.rigid_body_rho * self.rigid_body_spacing**self.dim
        h = self.h
        rad_s = self.rigid_body_spacing / 2.
        pa = get_particle_array(name='pa',
                                x=x,
                                y=y,
                                z=z,
                                h=h,
                                m=m,
                                rho=self.rigid_body_rho,
                                rad_s=rad_s,
                                constants={
                                    'E': 69 * 1e9,
                                    'poisson_ratio': 0.3,
                                    'spacing0': self.rigid_body_spacing,
                                    })

        add_boundary_identification_properties(pa)
        # make sure your rho is not zero
        equations = get_boundary_identification_etvf_equations([pa.name],
                                                               [pa.name])

        sph_eval = SPHEvaluator(arrays=[pa],
                                equations=equations,
                                dim=self.dim,
                                kernel=QuinticSpline(dim=self.dim))

        sph_eval.evaluate(dt=0.1)

        tmp = pa.is_boundary
        is_boundary_tmp = np.tile(tmp, no_bodies)
        is_boundary = is_boundary_tmp.ravel()

        return is_boundary

    def create_particles(self):
        # =============================
        # Create a rigid body
        # =============================
        # get bodyid for each rigid_body
        xc, yc, zc, body_id = self.create_rigid_body()

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

        # =============================================
        # Add additional properties to apply force
        # =============================================
        setup_properties_for_gradual_force(rigid_body)
        force_index_single(rigid_body)

        rigid_body.normal_force_time[0] = self.normal_force_time
        rigid_body.tmp_normal_force[0] = 0.
        rigid_body.delta_fn[0] = 0.

        rigid_body.tangential_force_time[0] = self.tangential_force_time
        rigid_body.tmp_tangential_force[0] = 0.

        # no of particles the tangential force to be applied
        no_par = len(np.where(rigid_body.force_idx_fn == 1.)[0])
        rigid_body.delta_fn[0] = self.fn_increment / no_par

        # no of particles the tangential force to be applied
        no_par = len(np.where(rigid_body.force_idx_ft == 1.)[0])
        rigid_body.delta_ft[0] = self.ft_increment / no_par
        # =============================================
        # Add additional properties to apply force
        # =============================================

        # =============================================
        # Create wall particles
        # =============================================
        length_fac = 1.
        x, y, z = get_3d_block(dx=self.rigid_body_spacing,
                               length=self.wall_length * length_fac,
                               height=self.wall_layers * self.wall_spacing,
                               depth=self.wall_depth)
        contact_force_is_boundary = get_contact_force_is_boundary(x, y, self.rigid_body_spacing)

        # x, y, _z = rotate(x, y, np.zeros(len(x)), axis=np.array([0., 0., 1.]),
        #                   angle=-(90. - self.angle))

        dem_id = body_id
        m = self.rigid_body_rho * self.rigid_body_spacing**self.dim
        h = self.h
        rad_s = self.rigid_body_spacing / 2.

        wall = get_particle_array(name='wall',
                                  x=x,
                                  y=z,
                                  z=y,
                                  h=h,
                                  m=m,
                                  rho=self.rigid_body_rho,
                                  rad_s=rad_s,
                                  contact_force_is_boundary=contact_force_is_boundary,
                                  constants={
                                      'E': 69 * 1e9,
                                      'poisson_ratio': 0.3,
                                  })
        wall.add_property('dem_id', type='int', data=max(body_id) + 1)
        # remove particles outside the circle
        indices = []
        for i in range(len(wall.x)):
            if wall.x[i] < - self.rigid_body_length:
                indices.append(i)

        wall.remove_particles(indices)
        # =============================================
        # End creation wall particles
        # =============================================

        xc, yc, zc = rotate(rigid_body.x, rigid_body.y, rigid_body.z, axis=np.array([0., 0., 1.]),
                            angle=-self.angle)
        x, z, y = rotate(wall.x, wall.y, wall.z, axis=np.array([0., 1., 0.]), angle=self.angle)

        rigid_body.x[:] = xc[:]
        rigid_body.y[:] = yc[:]
        rigid_body.z[:] = zc[:]
        radians = (90. - self.angle) * np.pi / 180.
        rigid_body.x[:] += (self.rigid_body_length / 2. + self.rigid_body_spacing) * np.cos(radians)
        rigid_body.y[:] += (self.rigid_body_length / 2. + self.rigid_body_spacing) * np.sin(radians)

        wall.x[:] = x[:]
        wall.y[:] = y[:]
        wall.z[:] = z[:]

        self.scheme.setup_properties([rigid_body, wall])

        # Add the boundary particle information to the rigid body
        rigid_body.add_property('contact_force_is_boundary')
        is_boundary = self.get_boundary_particles(max(rigid_body.body_id)+1)
        rigid_body.contact_force_is_boundary[:] = is_boundary[:]
        rigid_body.is_boundary[:] = is_boundary[:]

        # self.scheme.scheme.set_linear_velocity(
        #     rigid_body, np.array([0.0, -0.5, 0.]))

        return [rigid_body, wall]

    def create_scheme(self):
        rb3d = RigidBody3DScheme(rigid_bodies=['rigid_body'],
                                 boundaries=['wall'],
                                 gx=0.,
                                 gy=self.gy,
                                 gz=0.,
                                 dim=3,
                                 fric_coeff=0.45)
        s = SchemeChooser(default='rb3d', rb3d=rb3d)
        return s

    def create_equations(self):
        eqns = self.scheme.get_equations()

        # # Apply external force
        # force_eqs = []
        # force_eqs.append(
        #     ApplyForceOnRigidBody(dest="rigid_body", sources=None))

        # eqns.groups[-1].insert(-2, Group(force_eqs))

        return eqns

    def configure_scheme(self):
        tf = self.tf

        output_at_times = np.array([0., 0.5, 1.0])
        self.scheme.configure_solver(dt=self.dt, tf=tf, pfreq=500,
                                     output_at_times=output_at_times)

    def post_process(self, fname):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from pysph.solver.utils import load, get_files
        # from mayavi import mlab
        # mlab.options.offscreen = True

        info = self.read_info(fname)
        output_files = self.output_files

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
        # # generate plots
        # info = self.read_info(fname)
        # output_files = self.output_files
        # print(output_files)
        # # output_times = np.array([0., 5 * 1e-1, 1. * 1e-0,  2. * 1e-0])
        # output_times = np.array([0.])
        # logfile = os.path.join(os.path.dirname(fname), 'mohseni_2021_free_sliding_on_a_slope_3d.log')
        # to_plot = get_files_at_given_times_from_log(output_files, output_times,
        #                                             logfile)

        # mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(800, 800))
        # view = None

        # for i, f in enumerate(to_plot):
        #     print(i, f)
        #     data = load(f)
        #     t = data['solver_data']['t']
        #     body = data['arrays']['rigid_body']
        #     wall = data['arrays']['wall']

        #     bg = mlab.points3d(
        #         body.x, body.y, body.z, body.m, mode='point',
        #         colormap='viridis', vmin=-2, vmax=5
        #     )
        #     bg.actor.property.render_points_as_spheres = True
        #     bg.actor.property.point_size = 10

        #     # get the maximum and minimum of the geometry
        #     x_min = min(body.x) - self.rigid_body_height
        #     x_max = max(body.x) + 3. * self.rigid_body_height
        #     y_min = min(body.y) - 4. * self.rigid_body_height
        #     y_max = max(body.y) + 1. * self.rigid_body_height

        #     filtr_1 = ((wall.x >= x_min) & (wall.x <= x_max)) & (
        #         (wall.y >= y_min) & (wall.y <= y_max))
        #     wall_x = wall.x[filtr_1]
        #     wall_y = wall.y[filtr_1]
        #     wall_z = wall.z[filtr_1]
        #     wall_m = wall.m[filtr_1]

        #     bg = mlab.points3d(
        #         wall_x, wall_y, wall_z, wall_m, mode='point',
        #         colormap='viridis', vmin=-2, vmax=5
        #     )
        #     bg.actor.property.render_points_as_spheres = True
        #     bg.actor.property.point_size = 10

        #     mlab.axes()
        #     cc = mlab.gcf().scene.camera

        #     if i == 0:
        #         cc.position = [0.586053487183876, 0.004303849033652898, 1.473259753499654]
        #         cc.focal_point = [0.06312874772169114, 0.005215998279429021, -0.021841839056437724]
        #         cc.view_angle = 30.0
        #         cc.view_up = [0.001971909716611141, 0.9999980445816752, -0.00014968264907690208]
        #         cc.clipping_range = [0.0020735062077402457, 2.0735062077402455]
        #         cc.compute_view_plane_normal()
        #         mlab.text(0.7, 0.85, f"T = {output_times[i]} sec", width=0.2)

        #     if i == 1:
        #         cc.position = [0.586053487183876, 0.004303849033652898, 1.473259753499654]
        #         cc.focal_point = [0.06312874772169114, 0.005215998279429021, -0.021841839056437724]
        #         cc.view_angle = 30.0
        #         cc.view_up = [0.001971909716611141, 0.9999980445816752, -0.00014968264907690208]
        #         cc.clipping_range = [0.0020735062077402457, 2.0735062077402455]
        #         cc.compute_view_plane_normal()
        #         mlab.text(0.7, 0.85, f"T = {output_times[i]} sec", width=0.2)

        #     # save the figure
        #     figname = os.path.join(os.path.dirname(fname), "time" + str(i) + ".png")
        #     mlab.savefig(figname)
        #     # plt.show()

        # # =======================================
        # # =======================================
        # # Schematic
        # # =======================================
        # files = self.output_files
        # for sd, body, wall in iter_output(files[0:2], 'rigid_body', 'wall'):
        #     _t = sd['t']
        #     if _t == 0.:
        #         s = 0.3
        #         # print(_t)
        #         fig, axs = plt.subplots(1, 1)
        #         axs.scatter(body.x, body.y, s=s, c=body.m)
        #         # axs.grid()
        #         axs.set_aspect('equal', 'box')
        #         # axs.set_title('still a circle, auto-adjusted data limits', fontsize=10)

        #         # im_ratio = tmp.shape[0]/tmp.shape[1]
        #         x_min = min(body.x) - self.rigid_body_height
        #         x_max = max(body.x) + 3. * self.rigid_body_height
        #         y_min = min(body.y) - 4. * self.rigid_body_height
        #         y_max = max(body.y) + 1. * self.rigid_body_height

        #         filtr_1 = ((wall.x >= x_min) & (wall.x <= x_max)) & (
        #             (wall.y >= y_min) & (wall.y <= y_max))
        #         wall_x = wall.x[filtr_1]
        #         wall_y = wall.y[filtr_1]
        #         wall_m = wall.m[filtr_1]
        #         tmp = axs.scatter(wall_x, wall_y, s=s, c=wall_m)
        #         axs.axis('off')
        #         axs.set_xticks([])
        #         axs.set_yticks([])

        #         # save the figure
        #         figname = os.path.join(os.path.dirname(fname), "pre_schematic.png")
        #         fig.savefig(figname, dpi=300)


if __name__ == '__main__':
    app = Mohseni2021FreeSlidingOnASlope()
    app.run()
    app.post_process(app.info_filename)

# ft_x, ft_y, z
# fn_x, fn_y, z
# u, v, w
# delta_lt_x, delta_lt_y, delta_lt_z
