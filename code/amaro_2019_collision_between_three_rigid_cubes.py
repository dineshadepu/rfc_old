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

from rigid_body_3d import (RigidBody3DScheme, get_files_at_given_times_from_log)
from geometry import hydrostatic_tank_2d

from pysph.examples.solid_mech.impact import add_properties
from pysph.examples.rigid_body.sphere_in_vessel_akinci import (create_boundary,
                                                               create_fluid,
                                                               create_sphere)
from pysph.tools.geometry import get_2d_block


class Amaro2019CollisionBetweenThreeRigidCubes(Application):
    def initialize(self):
        spacing = 0.01
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
        self.body_density = 2000
        self.body_spacing = spacing / 2.
        self.body_h = self.hdx * self.body_spacing

        self.h = self.hdx * self.fluid_spacing

        # self.solid_rho = 500
        # self.m = 1000 * self.dx * self.dx
        self.co = 10 * np.sqrt(2 * 9.81 * self.fluid_height)
        self.p0 = self.fluid_density * self.co**2.
        self.c0 = self.co
        self.alpha = 0.1
        self.gy = 0.
        self.dim = 2

    def create_particles(self):
        xb, yb = get_2d_block(dx=self.body_spacing,
                              length=self.body_length,
                              height=self.body_height)
        self.body1_density = 1.5 / 0.2**2.
        m = self.body1_density * self.body_spacing**self.dim
        # left body
        body1 = get_particle_array(name='body1',
                                   x=xb,
                                   y=yb,
                                   h=self.body_h,
                                   m=m,
                                   rho=self.body1_density,
                                   m_fluid=0.,
                                   rad_s=self.body_spacing / 2.,
                                   E=15 * 1e6,
                                   nu=0.3,
                                   constants={
                                       'spacing0': self.body_spacing,
                                   })
        body_id = np.zeros(len(xb), dtype=int)
        dem_id = np.zeros(len(xb), dtype=int)
        body1.add_property('body_id', type='int', data=body_id)
        body1.add_property('dem_id', type='int', data=dem_id)
        body1.add_constant('total_no_bodies', [3])

        # right bottom body
        # Body 2 starts
        xb, yb = get_2d_block(dx=self.body_spacing,
                              length=self.body_length,
                              height=self.body_height)
        self.body2_density = 3. / 0.2**2.
        m = self.body2_density * self.body_spacing**self.dim
        xb += max(body1.x) - min(xb) + 0.02
        yb -= min(body1.y) - min(yb) + 0.165 - 1. * self.body_spacing
        body2 = get_particle_array(name='body2',
                                   x=xb,
                                   y=yb,
                                   h=self.body_h,
                                   m=m,
                                   rho=self.body2_density,
                                   m_fluid=0.,
                                   rad_s=self.body_spacing / 2.,
                                   E=15 * 1e6,
                                   nu=0.3,
                                   constants={
                                       'spacing0': self.body_spacing,
                                   })
        body_id = np.zeros(len(xb), dtype=int)
        dem_id = np.ones(len(xb), dtype=int)
        body2.add_property('body_id', type='int', data=body_id)
        body2.add_property('dem_id', type='int', data=dem_id)
        body2.add_constant('total_no_bodies', [3])

        # Body 3 starts
        xb, yb = get_2d_block(dx=self.body_spacing,
                              length=self.body_length,
                              height=self.body_height)
        self.body3_density = 1.2 / 0.2**2.
        m = self.body3_density * self.body_spacing**self.dim
        xb += max(body1.x) - min(xb) + 0.02
        yb += min(body1.y) - min(yb) + 0.165 - 1. * self.body_spacing
        # right top body
        body3 = get_particle_array(name='body3',
                                   x=xb,
                                   y=yb,
                                   h=self.body_h,
                                   m=m,
                                   rho=self.body3_density,
                                   m_fluid=0.,
                                   rad_s=self.body_spacing / 2.,
                                   E=15 * 1e6,
                                   nu=0.3,
                                   constants={
                                       'spacing0': self.body_spacing,
                                   })
        body_id = np.zeros(len(xb), dtype=int)
        dem_id = np.ones(len(xb), dtype=int) * 2
        body3.add_property('body_id', type='int', data=body_id)
        body3.add_property('dem_id', type='int', data=dem_id)
        body3.add_constant('total_no_bodies', [3])

        self.scheme.setup_properties([body1, body2, body3])

        body1.add_property('contact_force_is_boundary')
        body1.contact_force_is_boundary[:] = body1.is_boundary[:]
        body2.add_property('contact_force_is_boundary')
        body2.contact_force_is_boundary[:] = body2.is_boundary[:]
        body3.add_property('contact_force_is_boundary')
        body3.contact_force_is_boundary[:] = body2.is_boundary[:]

        self.scheme.scheme.set_linear_velocity(body1, np.array([1., 0., 0.]))
        self.scheme.scheme.set_linear_velocity(
            body2, np.array([-0.5, 0.0, 0.]))
        self.scheme.scheme.set_linear_velocity(body3, np.array([0.0, 0.0, 0.]))
        # print(body1.total_mass)
        # print(body2.total_mass)
        # print(body3.total_mass)
        # print(body1.inertia_tensor_body_frame[8])
        # print(body2.inertia_tensor_body_frame[8])
        # print(body3.inertia_tensor_body_frame[8])


        # print moment of inertia
        return [body1, body2, body3]

    def create_scheme(self):
        rb3d = RigidBody3DScheme(rigid_bodies=['body1', 'body2', 'body3'],
                                 boundaries=None, dim=2)
        s = SchemeChooser(default='rb3d', rb3d=rb3d)
        return s

    def configure_scheme(self):
        # dt = 0.125 * self.fluid_spacing * self.hdx / (self.co * 1.1)
        dt = 1e-4
        print("DT: %s" % dt)
        tf = 0.04

        output_at_times = np.array([0., 0.0135, 0.019, 0.04])

        self.scheme.configure_solver(dt=dt, tf=tf, pfreq=20,
                                     output_at_times=output_at_times)

    def post_process(self, fname):
        """This function will run once per time step after the time step is
        executed. For some time (self.wall_time), we will keep the wall near
        the cylinders such that they settle down to equilibrium and replicate
        the experiment.
        By running the example it becomes much clear.
        """
        from pysph.solver.utils import iter_output, get_files, load
        import os
        # info = self.read_info(fname)
        # files = self.output_files
        # t = []
        # system_x = []
        # system_y = []
        # for sd, array in iter_output(files[::10], 'cylinders'):
        #     _t = sd['t']
        #     t.append(_t)
        #     # get the system center
        #     cm_x = 0
        #     cm_y = 0
        #     for i in range(array.nb[0]):
        #         cm_x += array.xcm[3 * i]
        #         cm_y += array.xcm[3 * i + 1]
        #     cm_x = cm_x / 33
        #     cm_y = cm_y / 33

        #     system_x.append(cm_x / self.dam_length)
        #     system_y.append(cm_y / self.dam_length)

        import matplotlib.pyplot as plt
        # t = np.asarray(t)
        # t = t - self.wall_time

        # # gtvf data
        # path = os.path.abspath(__file__)
        # directory = os.path.dirname(path)

        # data = np.loadtxt(os.path.join(directory, 'x_com_zhang.csv'),
        #                   delimiter=',')
        # tx, xcom_zhang = data[:, 0], data[:, 1]

        # plt.plot(tx, xcom_zhang, "s--", label='Experimental')
        # plt.plot(t, system_x, "s-", label='Simulated PySPH')
        # plt.xlabel("time")
        # plt.ylabel("x/L")
        # plt.legend()
        # fig = os.path.join(os.path.dirname(fname), "xcom.png")
        # plt.savefig(fig, dpi=300)
        # plt.clf()

        # data = np.loadtxt(os.path.join(directory, 'y_com_zhang.csv'),
        #                   delimiter=',')
        # ty, ycom_zhang = data[:, 0], data[:, 1]

        # plt.plot(ty, ycom_zhang, "s--", label='Experimental')
        # plt.plot(t, system_y, "s-", label='Simulated PySPH')
        # plt.xlabel("time")
        # plt.ylabel("y/L")
        # plt.legend()
        # fig = os.path.join(os.path.dirname(fname), "ycom.png")
        # plt.savefig(fig, dpi=300)

        # ============================
        # generate plots
        # ============================
        info = self.read_info(fname)
        output_files = self.output_files
        output_times = np.array([0., 0.0135, 0.019, 0.04])

        logfile = os.path.join(
            os.path.dirname(fname),
            'amaro_2019_collision_between_three_rigid_cubes.log')
        to_plot = get_files_at_given_times_from_log(output_files, output_times,
                                                    logfile)
        for i, f in enumerate(to_plot):
            # print(i, f)
            data = load(f)
            t = data['solver_data']['t']
            body1 = data['arrays']['body1']
            body2 = data['arrays']['body2']
            body3 = data['arrays']['body3']

            s = 0.2
            # print(_t)
            fig, axs = plt.subplots(1, 1)
            axs.scatter(body1.x, body1.y, s=s, c=body1.m)
            axs.scatter(body2.x, body2.y, s=s, c=body2.m)
            axs.grid()
            axs.set_aspect('equal', 'box')
            # axs.set_title('still a circle, auto-adjusted data limits', fontsize=10)

            tmp = axs.scatter(body3.x, body3.y, s=s, c=body3.m)

            # save the figure
            figname = os.path.join(os.path.dirname(fname), "time" + str(i) + ".png")
            fig.savefig(figname, dpi=300)
            # plt.show()
        # ====================================
        # generate plots ends (Snapshots)
        # ====================================

        # ============================
        # generate schematic
        # ============================
        info = self.read_info(fname)
        output_files = self.output_files
        output_times = np.array([0.])
        logfile = os.path.join(
            os.path.dirname(fname),
            'amaro_2019_collision_between_three_rigid_cubes.log')
        to_plot = get_files_at_given_times_from_log(output_files, output_times,
                                                    logfile)
        for i, f in enumerate(to_plot):
            # print(i, f)
            data = load(f)
            t = data['solver_data']['t']
            body1 = data['arrays']['body1']
            body2 = data['arrays']['body2']
            body3 = data['arrays']['body3']

            s = 0.2
            # print(_t)
            fig, axs = plt.subplots(1, 1)
            axs.scatter(body1.x, body1.y, s=s, c=body1.m)
            axs.scatter(body2.x, body2.y, s=s, c=body2.m)
            axs.grid()
            axs.axis('off')
            axs.set_aspect('equal', 'box')
            # axs.set_title('still a circle, auto-adjusted data limits', fontsize=10)

            tmp = axs.scatter(body3.x, body3.y, s=s, c=body3.m)

            # save the figure
            figname = os.path.join(os.path.dirname(fname), "schematic" + ".png")
            fig.savefig(figname, dpi=300)

        # ====================================
        # Schematic
        # ====================================

    # def customize_output(self):
    #     self._mayavi_config('''
    #     for name in ['fluid']:
    #         b = particle_arrays[name]
    #         b.plot.module_manager.scalar_lut_manager.lut_mode = 'seismic'
    #     for name in ['tank', 'wall']:
    #         b = particle_arrays[name]
    #         b.point_size = 0.1
    #     ''')


if __name__ == '__main__':
    app = Amaro2019CollisionBetweenThreeRigidCubes()
    app.run()
    app.post_process(app.info_filename)
