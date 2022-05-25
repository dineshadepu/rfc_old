"""
Test the collision of two rigid bodues made of same particle array
"""
import numpy as np

from pysph.base.kernels import CubicSpline
# from pysph.base.utils import get_particle_array_rigid_body
from pysph.base.utils import get_particle_array
from pysph.sph.integrator import EPECIntegrator
from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser

from rigid_fluid_coupling import RigidFluidCouplingScheme
from geometry import hydrostatic_tank_2d, get_fluid_tank_3d

from pysph.examples.solid_mech.impact import add_properties
from pysph.examples.rigid_body.sphere_in_vessel_akinci import (create_boundary,
                                                               create_fluid,
                                                               create_sphere)
from pysph.tools.geometry import get_3d_block
import os


class Amaro2019DamBreakingFlowHittingOneCube3d(Application):
    def initialize(self):
        spacing = 0.03
        self.hdx = 1.0

        self.fluid_length = 4.5
        self.fluid_height = 0.4
        self.fluid_depth = 0.7
        self.fluid_density = 1000.0
        self.fluid_spacing = spacing

        self.tank_height = 0.7
        self.tank_length = 8.
        self.tank_layers = 3
        self.tank_spacing = spacing

        self.body_height = 0.15
        self.body_length = 0.15
        self.body_depth = 0.15
        self.body_density = 800
        self.body_spacing = spacing

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
        self.dim = 3

    def create_one_cube(self):
        xb1, yb1, zb1 = get_3d_block(dx=self.body_spacing,
                                     length=self.body_length,
                                     height=self.body_height,
                                     depth=self.body_depth)

        xb = np.concatenate([xb1])
        yb = np.concatenate([yb1])
        zb = np.concatenate([zb1])

        body_id1 = np.zeros(len(xb1), dtype=int)
        body_id = np.concatenate([body_id1])

        dem_id = np.concatenate([body_id1])

        return xb, yb, zb, body_id, dem_id

    def create_two_cubes(self):
        xb1, yb1, zb1 = get_3d_block(dx=self.body_spacing,
                                     length=self.body_length,
                                     height=self.body_height,
                                     depth=self.body_depth)

        xb2, yb2, zb2 = get_3d_block(dx=self.body_spacing,
                                     length=self.body_length,
                                     height=self.body_height,
                                     depth=self.body_depth)

        yb2 += max(yb1) - min(yb2) + self.body_spacing * 1.

        xb = np.concatenate([xb1, xb2])
        yb = np.concatenate([yb1, yb2])
        zb = np.concatenate([zb1, zb2])

        body_id1 = np.zeros(len(xb1), dtype=int)
        body_id2 = np.ones(len(xb2), dtype=int)
        body_id = np.concatenate([body_id1, body_id2])

        dem_id = np.concatenate([body_id1, body_id2])

        return xb, yb, zb, body_id, dem_id

    def create_three_cubes(self):
        xb1, yb1, zb1 = get_3d_block(dx=self.body_spacing,
                                     length=self.body_length,
                                     height=self.body_height,
                                     depth=self.body_depth)


        xb2, yb2, zb2 = get_3d_block(dx=self.body_spacing,
                                     length=self.body_length,
                                     height=self.body_height,
                                     depth=self.body_depth)


        xb3, yb3, zb3 = get_3d_block(dx=self.body_spacing,
                                     length=self.body_length,
                                     height=self.body_height,
                                     depth=self.body_depth)


        yb2 += max(yb1) - min(yb2) + self.body_spacing * 1.
        yb3 += max(yb2) - min(yb3) + self.body_spacing * 1.

        xb = np.concatenate([xb1, xb2, xb3])
        yb = np.concatenate([yb1, yb2, yb3])
        zb = np.concatenate([zb1, zb2, zb3])

        body_id1 = np.zeros(len(xb1), dtype=int)
        body_id2 = np.ones(len(xb2), dtype=int)
        body_id3 = np.ones(len(xb3), dtype=int) * 2
        body_id = np.concatenate([body_id1, body_id2, body_id3])

        dem_id = np.concatenate([body_id1, body_id2, body_id3])

        return xb, yb, zb, body_id, dem_id

    def create_pyramid_cubes(self):
        xb1, yb1, zb1 = get_3d_block(dx=self.body_spacing,
                                     length=self.body_length,
                                     height=self.body_height,
                                     depth=self.body_depth)

        xb2, yb2, zb2 = get_3d_block(dx=self.body_spacing,
                                     length=self.body_length,
                                     height=self.body_height,
                                     depth=self.body_depth)

        xb3, yb3, zb3 = get_3d_block(dx=self.body_spacing,
                                     length=self.body_length,
                                     height=self.body_height,
                                     depth=self.body_depth)

        xb4, yb4, zb4 = get_3d_block(dx=self.body_spacing,
                                     length=self.body_length,
                                     height=self.body_height,
                                     depth=self.body_depth)

        xb5, yb5, zb5 = get_3d_block(dx=self.body_spacing,
                                     length=self.body_length,
                                     height=self.body_height,
                                     depth=self.body_depth)

        xb6, yb6, zb6 = get_3d_block(dx=self.body_spacing,
                                     length=self.body_length,
                                     height=self.body_height,
                                     depth=self.body_depth)

        xb1 -= self.body_length
        xb2 += max(xb1) - min(xb2) + self.body_length/3.
        xb3 += max(xb2) - min(xb3) + self.body_length/3.

        xb4 += min(xb1) - min(xb4)
        xb4 += (self.body_length - self.body_length/3.)
        yb4 += max(yb1) - min(yb4) + self.body_spacing * 1.

        yb5 += max(yb4) - max(yb5)
        xb5 += max(xb3) - max(xb5) - (self.body_length - self.body_length/3.)

        yb6 += max(yb4) - min(yb6) + self.body_spacing * 1.
        xb6 += max(xb4) - max(xb6)
        xb6 += (max(xb5) - min(xb4)) / 2. - self.body_length/2.

        xb = np.concatenate([xb1, xb2, xb3, xb4, xb5, xb6])
        yb = np.concatenate([yb1, yb2, yb3, yb4, yb5, yb6])
        zb = np.concatenate([zb1, zb2, zb3, zb4, zb5, zb6])

        body_id1 = np.zeros(len(xb1), dtype=int)
        body_id2 = np.ones(len(xb2), dtype=int)
        body_id3 = np.ones(len(xb3), dtype=int) * 2
        body_id4 = np.ones(len(xb4), dtype=int) * 3
        body_id5 = np.ones(len(xb5), dtype=int) * 4
        body_id6 = np.ones(len(xb5), dtype=int) * 5
        body_id = np.concatenate([body_id1, body_id2, body_id3,
                                  body_id4, body_id5, body_id6])

        dem_id = np.concatenate([body_id1, body_id2, body_id3, body_id4, body_id5,
                                 body_id6])

        return xb, yb, zb, body_id, dem_id

    def get_boundary_particles(self, no_bodies):
        from boundary_particles import (get_boundary_identification_etvf_equations,
                                        add_boundary_identification_properties)
        from pysph.tools.sph_evaluator import SPHEvaluator
        from pysph.base.kernels import (QuinticSpline)
        # create a row of six cylinders
        x, y, z = get_3d_block(dx=self.body_spacing,
                               length=self.body_length,
                               height=self.body_height,
                               depth=self.body_depth)

        m = self.body_density * self.body_spacing**self.dim
        h = self.hdx * self.body_spacing
        rad_s = self.body_spacing / 2.
        pa = get_particle_array(name='foo',
                                x=x,
                                y=y,
                                z=z,
                                rho=self.body_density,
                                h=h,
                                m=m,
                                rad_s=rad_s,
                                constants={
                                    'E': 69 * 1e9,
                                    'poisson_ratio': 0.3,
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
        xf, yf, zf, xt, yt, zt = get_fluid_tank_3d(
            self.fluid_length, self.fluid_height, self.fluid_depth,
            self.tank_length, self.tank_height, self.tank_layers,
            self.body_spacing, self.body_spacing)

        m_fluid = self.fluid_density * self.fluid_spacing**self.dim

        xb, yb, zb, body_id, dem_id = self.create_one_cube()

        m = self.body_density * self.body_spacing**self.dim
        body = get_particle_array(name='body',
                                  x=xb,
                                  y=yb,
                                  z=zb,
                                  h=self.h,
                                  m=m,
                                  rho=self.body_density,
                                  m_fluid=m_fluid,
                                  rad_s=self.body_spacing / 2.,
                                  E=69 * 1e9,
                                  nu=0.3,
                                  constants={
                                      'spacing0': self.body_spacing,
                                  })
        body.y[:] += self.body_height * 2.

        body.add_property('body_id', type='int', data=body_id)
        body.add_property('dem_id', type='int', data=dem_id)
        body.add_constant('total_no_bodies', [max(body_id) + 2])

        # ===============================
        # create a tank
        # ===============================
        x, y, z = xt, yt, zt
        dem_id = body_id
        m = self.fluid_density * self.body_spacing**self.dim
        h = self.h
        rad_s = self.body_spacing / 2.

        tank = get_particle_array(name='tank',
                                  x=x,
                                  y=y,
                                  z=z,
                                  h=h,
                                  m=m,
                                  rho=self.fluid_density,
                                  rad_s=rad_s,
                                  E=69 * 1e9,
                                  nu=0.3)

        max_dem_id = max(dem_id)
        tank.add_property('dem_id', type='int', data=max_dem_id + 1)

        # create the wall which should be lifted for the water to flow
        xw, yw, zw = get_3d_block(self.fluid_spacing, 3. * self.fluid_spacing,
                                  self.fluid_depth, self.tank_height)
        xw += np.max(xf) - np.min(xw) + self.fluid_spacing
        zw += np.min(zt) - np.min(zw) + self.tank_layers * self.fluid_spacing
        wall = get_particle_array(x=xw,
                                  y=yw,
                                  z=zw,
                                  m=m_fluid,
                                  m_fluid=m_fluid,
                                  h=self.h,
                                  rho=self.fluid_density,
                                  rad_s=self.fluid_spacing/2.,
                                  name="wall",
                                  E=69 * 1e9,
                                  nu=0.3)
        wall.add_property('dem_id', type='int', data=2)
        # Translate the tank and fluid so that fluid starts at 0
        wall.y[:] += 5. * self.fluid_spacing
        # ==================================================
        # create fluid
        # ==================================================
        m_fluid = self.fluid_density * self.fluid_spacing**self.dim

        fluid = get_particle_array(x=xf,
                                   y=yf,
                                   z=zf,
                                   m=m_fluid,
                                   h=self.h,
                                   rho=self.fluid_density,
                                   name="fluid")

        # ==================================================
        # adjust the rigid body positions
        # ==================================================
        # put it on top of the wall
        body.y[:] += min(fluid.y) - min(body.y)

        # put in front of the fluid
        body.x[:] += max(fluid.x) - min(body.x) + self.fluid_spacing
        body.x[:] += 1.7

        # adjust the length from left wall
        body.z[:] -= min(body.z) - min(fluid.z)
        body.z[:] += 0.275

        # ==================================================================
        # adjust the particle array positions so that fluid starts at 0.
        # ==================================================================
        # x_scale = abs(min(fluid.x))
        # fluid.x[:] += x_scale
        # wall.x[:] += x_scale
        # tank.x[:] += x_scale
        # body.x[:] += x_scale
        # ==================================================================
        # adjust the particle array positions so that fluid starts at 0.
        # ==================================================================

        self.scheme.setup_properties([body, tank, fluid, wall])

        # reset the boundary particles, this is due to improper boundary
        # particle identification by the setup properties
        is_boundary = self.get_boundary_particles(body.total_no_bodies[0] - 1)
        body.is_boundary[:] = is_boundary[:]

        body.add_property('contact_force_is_boundary')
        body.contact_force_is_boundary[:] = body.is_boundary[:]

        tank.add_property('contact_force_is_boundary')
        tank.contact_force_is_boundary[:] = tank.is_boundary[:]

        wall.add_property('contact_force_is_boundary')
        wall.contact_force_is_boundary[:] = 1.

        # set the rigid-fluid coupling properties for rigid body
        body.rho_fsi[:] = 1000.
        body.m_fsi[:] = 1000. * self.body_spacing**self.dim
        return [body, tank, fluid, wall]

    def create_scheme(self):
        rfc = RigidFluidCouplingScheme(rigid_bodies=['body'],
                                       fluids=['fluid'],
                                       boundaries=['tank', 'wall'],
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

    def configure_scheme(self):
        dt = 1e-4
        print("DT: %s" % dt)
        tf = 1.7

        self.scheme.configure_solver(dt=dt, tf=tf, pfreq=100)

    def post_step(self, solver):
        dt = solver.dt
        for pa in self.particles:
            if pa.name == 'wall':
                pa.y += 1.9 * dt

    def customize_output(self):
        self._mayavi_config('''
        for name in ['fluid']:
            b = particle_arrays[name]
            b.plot.module_manager.scalar_lut_manager.lut_mode = 'seismic'
        for name in ['tank', 'wall']:
            b = particle_arrays[name]
            b.point_size = 0.1
        ''')

    def post_process(self, fname):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from pysph.solver.utils import load, get_files
        from pysph.solver.utils import iter_output

        info = self.read_info(fname)
        output_files = self.output_files

        data = load(output_files[0])
        arrays = data['arrays']
        fluid = arrays['fluid']
        max_x_fluid = max(fluid.x)

        t, x_com = [], []

        for sd, rb in iter_output(output_files[::1], 'body'):
            _t = sd['t']
            t.append(_t)
            x_com.append(rb.xcm[0] - max_x_fluid)

        path = os.path.abspath(__file__)
        directory = os.path.dirname(path)
        # experimental data (read from file)
        # load the data
        data_x_vs_t_exp_canelas_2016 = np.loadtxt(
            os.path.join(directory, 'amaro_2019_dam_breaking_flow_hitting_one_cube_3d_Canelas_exp_x_vs_t.csv'),
            delimiter=',')
        t_exp, x_cm_exp = data_x_vs_t_exp_canelas_2016[:, 0], data_x_vs_t_exp_canelas_2016[:, 1]

        data_x_vs_t_dpdem_li_liu_2022 = np.loadtxt(
            os.path.join(directory, 'amaro_2019_dam_breaking_flow_hitting_one_cube_3d_Lu_Liu_dpdem_eisph_x_vs_t.csv'),
            delimiter=',')
        t_dpdem, x_cm_dpdem = data_x_vs_t_dpdem_li_liu_2022[:, 0], data_x_vs_t_dpdem_li_liu_2022[:, 1]

        if 'info' in fname:
            res = os.path.join(os.path.dirname(fname), "results.npz")
        else:
            res = os.path.join(fname, "results.npz")

        np.savez(res,
                 t=t,
                 x_com=x_com,

                 t_exp=t_exp,
                 x_cm_exp=x_cm_exp,

                 t_dpdem=t_dpdem,
                 x_cm_dpdem=x_cm_dpdem)

        plt.clf()
        plt.plot(t, x_com, "^-", label='Simulated')
        plt.plot(t_exp, x_cm_exp, "--", label='Canelas experiment')
        plt.plot(t_dpdem, x_cm_dpdem, "--", label='DPDEM')

        plt.title('x-center of mass')
        plt.xlabel('t')
        plt.ylabel('x-center of mass (m)')
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "xcom_vs_time.png")
        plt.savefig(fig, dpi=300)


if __name__ == '__main__':
    app = Amaro2019DamBreakingFlowHittingOneCube3d()
    app.run()
    # app.post_process(app.info_filename)
