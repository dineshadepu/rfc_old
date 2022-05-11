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

from rigid_body_3d import RigidBody3DScheme, get_files_at_given_times_from_log
from pysph.sph.equation import Equation, Group
import os

from pysph.tools.geometry import get_2d_block, get_2d_tank, rotate


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
        self.dim = 2
        spacing = 25. * 1e-3

        self.wall_length = 5.
        self.wall_height = 0.
        self.wall_spacing = spacing
        self.wall_layers = 0

        self.rigid_body_radius = 0.5
        self.rigid_body_length = 0.1
        self.rigid_body_height = 0.1
        self.rigid_body_spacing = spacing
        self.rigid_body_rho = 2700

        self.angle = 60.

        # simulation properties
        self.hdx = 1.0
        self.alpha = 0.1
        self.gx = 9.81 * np.sin(self.angle * np.pi / 180)
        self.gy = - 9.81 * np.cos(self.angle * np.pi / 180)
        self.gz = 0.
        self.h = self.hdx * self.rigid_body_spacing

        # solver data
        self.tf = 0.6
        self.dt = 1e-4

        # Rigid body collision related data
        self.limit = 6
        self.seval = None

    def get_boundary_particles(self, no_bodies):
        from boundary_particles import (get_boundary_identification_etvf_equations,
                                        add_boundary_identification_properties)
        from pysph.tools.sph_evaluator import SPHEvaluator
        from pysph.base.kernels import (QuinticSpline)
        # create a row of six cylinders
        x, y = create_circle_1(diameter=2. * self.rigid_body_radius,
                               spacing=self.rigid_body_spacing,
                               center=None)

        m = self.rigid_body_rho * self.rigid_body_spacing**2
        h = self.h
        rad_s = self.rigid_body_spacing / 2.
        pa = get_particle_array(name='pa',
                                x=x,
                                y=y,
                                h=h,
                                m=m,
                                rho=self.rigid_body_rho,
                                rad_s=rad_s,
                                E=69 * 1e9,
                                nu=0.3,
                                constants={
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
        xc, yc = create_circle_1(diameter=2. * self.rigid_body_radius,
                                 spacing=self.rigid_body_spacing,
                                 center=None)
        body_id = np.ones(len(xc), dtype=int) * 0

        dem_id = body_id
        m = self.rigid_body_rho * self.rigid_body_spacing**2
        h = self.h
        rad_s = self.rigid_body_spacing / 2.
        rigid_body = get_particle_array(name='rigid_body',
                                        x=xc,
                                        y=yc,
                                        h=h,
                                        m=m,
                                        rho=self.rigid_body_rho,
                                        rad_s=rad_s,
                                        E=69 * 1e9,
                                        nu=0.3,
                                        constants={
                                            'spacing0': self.rigid_body_spacing,
                                        })
        rigid_body.add_property('dem_id', type='int', data=dem_id)
        rigid_body.add_property('body_id', type='int', data=body_id)
        # rigid_body.add_constant('max_tng_contacts_limit', 10)
        rigid_body.add_constant('total_no_bodies', 2)
        # =============================
        # End creation of rigid body
        # =============================
        # =============================================
        # Create wall particles
        # =============================================
        length_fac = 1.
        x, y = get_2d_block(dx=self.rigid_body_spacing,
                            length=self.wall_length * length_fac,
                            height=self.wall_layers * self.wall_spacing)
        contact_force_is_boundary = get_contact_force_is_boundary(x, y, self.rigid_body_spacing)

        m = self.rigid_body_rho * self.rigid_body_spacing**2
        h = self.h
        rad_s = self.rigid_body_spacing / 2.

        wall = get_particle_array(name='wall',
                                  x=x,
                                  y=y,
                                  h=h,
                                  m=m,
                                  rho=self.rigid_body_rho,
                                  rad_s=rad_s,
                                  contact_force_is_boundary=1.,
                                  E=69 * 1e9,
                                  nu=0.3)

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

        rigid_body.x[:] += (self.rigid_body_radius)
        rigid_body.y[:] += (self.rigid_body_radius + 0.45 * self.rigid_body_spacing)

        self.scheme.setup_properties([rigid_body, wall])

        # Add the boundary particle information to the rigid body
        rigid_body.add_property('contact_force_is_boundary')
        is_boundary = self.get_boundary_particles(max(rigid_body.body_id)+1)
        rigid_body.contact_force_is_boundary[:] = is_boundary[:]
        rigid_body.is_boundary[:] = is_boundary[:]

        wall.is_boundary[:] = 1.

        # remove particles which are not boundary
        indices = []
        for i in range(len(rigid_body.y)):
            if rigid_body.is_boundary[i] == 0:
                indices.append(i)
        rigid_body.remove_particles(indices)

        return [rigid_body, wall]

    def create_scheme(self):
        rb3d = RigidBody3DScheme(rigid_bodies=['rigid_body'],
                                 boundaries=['wall'],
                                 gx=self.gx,
                                 gy=self.gy,
                                 gz=0.,
                                 dim=2,
                                 fric_coeff=0.45)
        s = SchemeChooser(default='rb3d', rb3d=rb3d)
        return s

    def configure_scheme(self):
        tf = self.tf

        output_at_times = np.array([0., 0.5, 1.0, 2.0])
        self.scheme.configure_solver(dt=self.dt, tf=tf, pfreq=100,
                                     output_at_times=output_at_times)

    def post_process(self, fname):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from pysph.solver.utils import load, get_files
        from pysph.solver.utils import iter_output

        info = self.read_info(fname)
        output_files = self.output_files

        data = load(output_files[0])
        arrays = data['arrays']
        rb = arrays['rigid_body']
        x0 = rb.xcm[0]

        t, x_com = [], []

        for sd, rb in iter_output(output_files[::1], 'rigid_body'):
            _t = sd['t']
            t.append(_t)
            x_com.append(rb.xcm[0] - x0)

        # analytical data
        theta = np.pi / 3.
        t_analytical = np.linspace(0., max(t), 100)

        if self.options.fric_coeff == 0.3:
            x_analytical = 0.5 * 9.81 * t_analytical**2 * (np.sin(theta) - 0.3 * np.cos(theta))
        elif self.options.fric_coeff == 0.6:
            x_analytical = 1. / 3. * 9.81 * t_analytical**2. * np.sin(theta)

        if 'info' in fname:
            res = os.path.join(os.path.dirname(fname), "results.npz")
        else:
            res = os.path.join(fname, "results.npz")

        np.savez(res,
                 t=t,
                 x_com=x_com,

                 t_analytical=t_analytical,
                 x_analytical=x_analytical)

        plt.clf()
        plt.plot(t, x_com, "^-", label='Simulated')
        plt.plot(t_analytical, x_analytical, "--", label='Analytical')

        plt.title('x-center of mass')
        plt.xlabel('t')
        plt.ylabel('x-center of mass (m)')
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "xcom_vs_time.png")
        plt.savefig(fig, dpi=300)


if __name__ == '__main__':
    app = Mohseni2021FreeSlidingOnASlope()
    app.run()
    app.post_process(app.info_filename)

# ft_x, ft_y, z
# fn_x, fn_y, z
# u, v, w
# delta_lt_x, delta_lt_y, delta_lt_z
