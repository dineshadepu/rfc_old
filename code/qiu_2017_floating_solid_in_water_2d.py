"""[1] A 3D simulation of a moving solid in viscous free-surface flows by
coupling SPH and DEM

[2] Numerical modeling of floating bodies transport for flooding analysis in
nuclear reactor building

[3] A two-way coupled simulation of moving solids in free-surface flows

[1] 3.3 Floating solid in water.

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
from pysph.tools import geometry as G

from pysph.tools.geometry import get_2d_block, get_2d_tank, rotate
from geometry import hydrostatic_tank_2d


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


class Qiu2017FloatingSolidInWater2D(Application):
    def initialize(self):
        self.dim = 2
        spacing = 2. * 1e-3
        self.hdx = 1.0

        # the fluid dimensions are
        # x-dimension (this is in the plane of the paper going right)
        self.fluid_length = 150. * 1e-3
        # y-dimension (this goes into the paper)
        self.fluid_height = 52 * 1e-3

        self.fluid_density = 1000.0
        self.fluid_spacing = spacing

        self.tank_length = 150. * 1e-3
        self.tank_height = 140 * 1e-3
        self.tank_spacing = spacing
        self.tank_layers = 3

        # x dimension
        self.rigid_body_length = 49. * 1e-3
        # y dimension
        self.rigid_body_height = 24. * 1e-3
        self.rigid_body_spacing = spacing
        self.rigid_body_rho = 800.52

        self.h = self.hdx * self.fluid_spacing

        # self.solid_rho = 500
        # self.m = 1000 * self.dx * self.dx
        self.co = 10 * np.sqrt(2 * 9.81 * self.fluid_height)
        self.p0 = self.fluid_density * self.co**2.
        self.c0 = self.co
        # fixme: why is it blowing up with higher alpha
        self.alpha = 0.5
        self.gx = 0.
        self.gy = -9.81
        self.gz = 0.
        self.dim = 2

        # solver data
        self.tf = 1.5
        self.dt = 0.25 * self.fluid_spacing * self.hdx / (self.c0 * 1.1)

        # Rigid body collision related data
        self.limit = 6
        self.seval = None

    def add_user_options(self, group):
        group.add_argument("--dx",
                           action="store",
                           type=float,
                           dest="dx",
                           default=2. * 1e-3,
                           help="Spacing between particles")

    def consume_user_options(self):
        spacing = self.options.dx
        self.fluid_spacing = spacing
        self.tank_spacing = spacing
        self.rigid_body_spacing = spacing

        # update h
        self.h = self.hdx * self.fluid_spacing

    def create_rigid_body(self):
        x = np.array([])
        y = np.array([])

        x, y = get_2d_block(dx=self.rigid_body_spacing,
                            length=self.rigid_body_length,
                            height=self.rigid_body_height)

        body_id = np.array([], dtype=int)
        for i in range(1):
            b_id = np.ones(len(x), dtype=int) * i
            body_id = np.concatenate((body_id, b_id))

        return x, y, body_id

    def get_boundary_particles(self, no_bodies):
        from boundary_particles import (get_boundary_identification_etvf_equations,
                                        add_boundary_identification_properties)
        from pysph.tools.sph_evaluator import SPHEvaluator
        from pysph.base.kernels import (QuinticSpline)
        # create a row of six cylinders
        x, y, body_id = self.create_rigid_body()

        m = self.rigid_body_rho * self.rigid_body_spacing**self.dim
        h = self.h
        rad_s = self.rigid_body_spacing / 2.
        pa = get_particle_array(name='pa',
                                x=x,
                                y=y,
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

        xf, yf, xt, yt = hydrostatic_tank_2d(
            fluid_length=self.fluid_length,
            fluid_height=self.fluid_height,
            tank_height=self.tank_height,
            tank_layers=self.tank_layers,
            fluid_spacing=self.fluid_spacing,
            tank_spacing=self.tank_spacing)

        m_fluid = self.fluid_density * self.fluid_spacing**self.dim

        fluid = get_particle_array(x=xf,
                                   y=yf,
                                   m=m_fluid,
                                   h=self.h,
                                   rho=self.fluid_density,
                                   name="fluid")
        # set the initial pressure
        fluid.p[:] = - self.fluid_density * self.gy * (max(fluid.y) - fluid.y[:])

        # =============================
        # Create a rigid body
        # =============================
        # get bodyid for each rigid_body
        xc, yc, body_id = self.create_rigid_body()

        dem_id = body_id
        m = self.rigid_body_rho * self.rigid_body_spacing**self.dim
        h = self.h
        rad_s = self.rigid_body_spacing / 2.
        rigid_body = get_particle_array(name='rigid_body',
                                        x=xc,
                                        y=yc,
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
        rigid_body.y[:] -= min(rigid_body.y) - min(fluid.y)
        rigid_body.y[:] += self.rigid_body_spacing
        # rigid_body.y[:] += 12. * 1e-3

        # =============================
        # End creation of rigid body
        # =============================
        tank = get_particle_array(x=xt,
                                  y=yt,
                                  m=m_fluid,
                                  m_fluid=m_fluid,
                                  h=self.h,
                                  rho=self.fluid_density,
                                  rad_s=self.fluid_spacing/2.,
                                  contact_force_is_boundary=1.,
                                  name="tank",
                                  constants={
                                      'E': 21 * 1e10,
                                      'poisson_ratio': 0.3,
                                  })
        tank.add_property('dem_id', type='int', data=1)

        # # Translate the tank and fluid so that fluid starts at 0
        # min_xf = abs(np.min(xf))

        # fluid.x += min_xf
        # tank.x += min_xf

        self.scheme.setup_properties([fluid, tank, rigid_body])

        # Add the boundary particle information to the rigid body
        rigid_body.add_property('contact_force_is_boundary')
        is_boundary = self.get_boundary_particles(max(rigid_body.body_id)+1)
        rigid_body.contact_force_is_boundary[:] = is_boundary[:]
        rigid_body.is_boundary[:] = is_boundary[:]

        # Add the rigid fluid coupling properties to the rigid body
        rigid_body.m_fsi[:] = self.fluid_density * self.rigid_body_spacing**self.dim
        rigid_body.rho_fsi[:] = 1000.

        # Remove the fluid particles
        G.remove_overlap_particles(
            fluid, rigid_body, self.rigid_body_spacing, dim=self.dim
        )

        # self.scheme.scheme.set_linear_velocity(
        #     rigid_body, np.array([0.0, -0.5, 0.]))

        return [fluid, tank, rigid_body]

    def create_scheme(self):
        rfc = RigidFluidCouplingScheme(rigid_bodies=['rigid_body'],
                                       fluids=['fluid'],
                                       boundaries=['tank'],
                                       dim=self.dim,
                                       rho0=self.fluid_density,
                                       p0=self.p0,
                                       c0=self.c0,
                                       gx=self.gx,
                                       gy=self.gy,
                                       gz=self.gz,
                                       nu=0.,
                                       alpha=self.alpha,
                                       h=self.h)

        s = SchemeChooser(default='rfc', rfc=rfc)
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

        output_at_times = np.array([0., 0.2, 0.3, 0.4])
        self.scheme.configure_solver(dt=self.dt, tf=tf, pfreq=100,
                                     output_at_times=output_at_times)

    # def post_step(self, solver):
    #     t = solver.t
    #     # dt = solver.dt
    #     # T = self.wall_time
    #     for pa in self.particles:
    #         if pa.name == 'rigid_body':
    #             t_1 = pa.normal_force_time[0]

    #             if t <= t_1:
    #                 pa.tmp_normal_force[0] += pa.delta_fn[0]
    #                 pa.fy[np.where(pa.force_idx_fn == 1)] += pa.tmp_normal_force[0]

    #             t_2 = pa.normal_force_time[0] + pa.tangential_force_time[0]
    #             if t > t_1 and t <= t_2:
    #                 pa.tmp_tangential_force[0] += pa.delta_ft[0]
    #                 pa.fx[np.where(pa.force_idx_ft == 1)] += pa.tmp_tangential_force[0]

    def post_process(self, fname):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from pysph.solver.utils import load, get_files

        output_files = get_files(os.path.dirname(fname))

        from pysph.solver.utils import iter_output

        t, y_cm_simulated = [], []

        for sd, rb in iter_output(output_files[::1], 'rigid_body'):
            _t = sd['t']
            t.append(_t)
            y_cm_simulated.append(rb.xcm[1])

        path = os.path.abspath(__file__)
        directory = os.path.dirname(path)

        # experimental data (read from file)
        # load the data
        data_y_cm_vs_time_exp_qiu_2017_exp = np.loadtxt(
            os.path.join(directory, 'qiu_2017_floating_solid_in_water_vertical_displacement_experimental.csv'),
            delimiter=',')
        t_experimental, y_cm_experimental = data_y_cm_vs_time_exp_qiu_2017_exp[:, 0], data_y_cm_vs_time_exp_qiu_2017_exp[:, 1]

        if 'info' in fname:
            res = os.path.join(os.path.dirname(fname), "results.npz")
        else:
            res = os.path.join(fname, "results.npz")

        np.savez(res,
                 t=t,
                 y_cm_simulated=y_cm_simulated,

                 t_experimental=t_experimental,
                 y_cm_experimental=y_cm_experimental)

        plt.clf()
        plt.plot(t, y_cm_simulated, "-", label='Simulated')
        plt.plot(t_experimental, y_cm_experimental, "--", label='Experimental')

        plt.title('vertical_disp_vs_time')
        plt.xlabel('t')
        plt.ylabel('Vertical displacement (m)')
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "y_cm_vs_time.png")
        plt.savefig(fig, dpi=300)
        # ========================
        # x amplitude figure
        # ========================
        # generate plots
        i = 0
        output_files = get_files(fname)
        output_times = np.array([0., 0.2, 0.3, 0.4])

        for sd, body, fluid, tank in iter_output(output_files, 'rigid_body', 'fluid', 'tank'):
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

                tank_x = tank.x
                tank_y = tank.y
                tank_m = tank.m

                tmp = axs.scatter(tank_x, tank_y, s=s, c=tank_m)

                tmp = axs.scatter(fluid.x, fluid.y, s=s, c=fluid.m)

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
        for sd, body, fluid, tank in iter_output(files[0:2], 'rigid_body', 'fluid', 'tank'):
            _t = sd['t']
            if _t == 0.:
                s = 0.3
                # print(_t)
                fig, axs = plt.subplots(1, 1)
                axs.scatter(body.x, body.y, s=s, c=body.m)
                # axs.grid()
                axs.set_aspect('equal', 'box')
                # axs.set_title('still a circle, auto-adjusted data limits', fontsize=10)

                tank_x = tank.x
                tank_y = tank.y
                tank_m = tank.m

                tmp = axs.scatter(tank_x, tank_y, s=s, c=tank_m)

                tmp = axs.scatter(fluid.x, fluid.y, s=s, c=fluid.m)

                axs.axis('off')
                axs.set_xticks([])
                axs.set_yticks([])

                # save the figure
                figname = os.path.join(os.path.dirname(fname), "pre_schematic.png")
                fig.savefig(figname, dpi=300)


if __name__ == '__main__':
    app = Qiu2017FloatingSolidInWater2D()
    app.run()
    app.post_process(app.info_filename)

# ft_x, ft_y, z
# fn_x, fn_y, z
# u, v, w
# delta_lt_x, delta_lt_y, delta_lt_z
