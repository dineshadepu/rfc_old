import numpy as np

from pysph.sph.scheme import add_bool_argument

from pysph.sph.equation import Equation
from pysph.sph.scheme import Scheme
from pysph.sph.scheme import add_bool_argument
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.sph.equation import Group, MultiStageEquations
from pysph.sph.integrator import EPECIntegrator
from pysph.base.kernels import (QuinticSpline)
from textwrap import dedent
from compyle.api import declare
from math import log, sqrt

from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import IntegratorStep

# from pysph.sph.rigid_body import (BodyForce)

from pysph.base.kernels import CubicSpline

from pysph.sph.wc.gtvf import GTVFIntegrator
from rigid_body_common import (add_properties_stride,
                               set_total_mass, set_center_of_mass,
                               set_body_frame_position_vectors,
                               set_body_frame_normal_vectors,
                               set_moment_of_inertia_and_its_inverse,
                               BodyForce, SumUpExternalForces,
                               normalize_R_orientation,
                               ComputeContactForceNormals,
                               ComputeContactForceDistanceAndClosestPoint,
                               ComputeContactForce)

from contact_force_mohseni_vyas import (
    ComputeContactForceNormalsMV,
    ComputeContactForceDistanceAndClosestPointAndWeightDenominatorMV,
    ComputeContactForceMV,
    TransferContactForceMV)

# compute the boundary particles
from boundary_particles import (get_boundary_identification_etvf_equations,
                                add_boundary_identification_properties)
from numpy import sin, cos


class GTVFDEMStep(IntegratorStep):
    def stage1(self, d_idx, d_m, d_x, d_y, d_z, d_u, d_v, d_w, d_fx, d_fy,
               d_fz, d_wz, d_torz, d_moi, dt):
        dtb2 = 0.5 * dt
        d_u[d_idx] = d_u[d_idx] + dtb2 * d_fx[d_idx] / d_m[d_idx]
        d_v[d_idx] = d_v[d_idx] + dtb2 * d_fy[d_idx] / d_m[d_idx]
        d_w[d_idx] = d_w[d_idx] + dtb2 * d_fz[d_idx] / d_m[d_idx]

        I_inverse = 1. / d_moi[d_idx]
        d_wz[d_idx] += dtb2 * d_torz[d_idx] * I_inverse

    def stage2(self, d_idx, d_m, d_x, d_y, d_z, d_u, d_v, d_w,
               dt):
        d_x[d_idx] = d_x[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y[d_idx] + dt * d_v[d_idx]
        d_z[d_idx] = d_z[d_idx] + dt * d_w[d_idx]

    def stage3(self, d_idx, d_m, d_x, d_y, d_z, d_u, d_v, d_w, d_fx, d_fy,
               d_fz, d_wz, d_torz, d_moi, dt):
        dtb2 = 0.5 * dt
        d_u[d_idx] = d_u[d_idx] + dtb2 * d_fx[d_idx]
        d_v[d_idx] = d_v[d_idx] + dtb2 * d_fy[d_idx]
        d_w[d_idx] = d_w[d_idx] + dtb2 * d_fz[d_idx]

        I_inverse = 1. / d_moi[d_idx]
        d_wz[d_idx] += dtb2 * d_torz[d_idx] * I_inverse


class DEMForce(Equation):
    def __init__(self, dest, sources, kn=1e5, en=1.0):
        self.kn = kn
        self.en = en
        tmp = log(en)
        self.alpha = 2. * sqrt(kn) * abs(tmp) / (sqrt(np.pi**2. + tmp**2.))
        super(DEMForce, self).__init__(dest, sources)

    def loop(self, d_idx, d_m, d_u, d_v, d_w, d_fx, d_fy, d_fz, XIJ, RIJ, s_u,
             s_v, s_w, d_rad_s, s_idx, s_m, s_rad_s, dt, t):
        overlap = -1.

        # check the particles are not on top of each other.
        if RIJ > 0:
            # print("inside")
            overlap = d_rad_s[d_idx] + s_rad_s[s_idx] - RIJ

        # ---------- force computation starts ------------
        # if particles are overlapping
        if overlap > 0:
            # print("inside")
            # normal vector (nij) passing from d_idx to s_idx, i.e., i to j
            rinv = 1.0 / RIJ
            # in PySPH: XIJ[0] = d_x[d_idx] - s_x[s_idx]
            nx = -XIJ[0] * rinv
            ny = -XIJ[1] * rinv
            nz = -XIJ[2] * rinv

            # Now the relative velocity of particle j w.r.t i at the contact
            # point is
            vr_x = s_u[s_idx] - d_u[d_idx]
            vr_y = s_v[s_idx] - d_v[d_idx]
            vr_z = s_w[s_idx] - d_w[d_idx]

            # normal velocity magnitude
            vr_dot_nij = vr_x * nx + vr_y * ny + vr_z * nz
            vn_x = vr_dot_nij * nx
            vn_y = vr_dot_nij * ny
            vn_z = vr_dot_nij * nz

            m_eff = d_m[d_idx] * s_m[s_idx] / (d_m[d_idx] + s_m[s_idx])
            eta_n = self.alpha * sqrt(m_eff)

            ############################
            # normal force computation #
            ############################
            kn_overlap = self.kn * overlap
            fn_x = -kn_overlap * nx + eta_n * vn_x
            fn_y = -kn_overlap * ny + eta_n * vn_y
            fn_z = -kn_overlap * nz + eta_n * vn_z

            d_fx[d_idx] += fn_x
            d_fy[d_idx] += fn_y
            d_fz[d_idx] += fn_z


class DEMScheme(Scheme):
    def __init__(self,
                 granular_particles,
                 boundaries,
                 kn=1e5,
                 en=0.5,
                 dim=2,
                 gx=0.0,
                 gy=0.0,
                 gz=0.0):
        self.granular_particles = granular_particles

        if boundaries is None:
            self.boundaries = []
        else:
            self.boundaries = boundaries

        # assert(dim, 2)

        self.dim = dim

        self.kernel = QuinticSpline

        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.kn = kn
        self.en = en

        self.solver = None

    def add_user_options(self, group):
        group.add_argument("--en", action="store",
                           dest="en", default=1.0,
                           type=float,
                           help="Coefficient of restitution")

        group.add_argument("--gy", action="store",
                           dest="gy", default=-9.81,
                           type=float,
                           help="Gravity")

        group.add_argument("--kn", action="store",
                           dest="kn", default=1e5,
                           type=float,
                           help="Stiffness")

    def consume_user_options(self, options):
        _vars = ['en', 'gy', 'kn']
        data = dict((var, self._smart_getattr(options, var)) for var in _vars)
        self.configure(**data)

    def get_equations(self):
        from pysph.sph.equation import Group, MultiStageEquations
        all = list(set(self.granular_particles + self.boundaries))

        stage1 = []
        g1 = []

        # ------------------------
        # stage 1 equations starts
        # ------------------------
        # There will be no equations for stage 1

        # ------------------------
        # stage 2 equations starts
        # ------------------------
        stage2 = []
        g1 = []

        g2 = []
        for granules in self.granular_particles:
            g2.append(
                BodyForce(dest=granules,
                          sources=None,
                          gx=self.gx,
                          gy=self.gy,
                          gz=self.gz))

        stage2.append(Group(equations=g2, real=False))

        for granules in self.granular_particles:
            g1.append(
                DEMForce(dest=granules, sources=all, kn=self.kn, en=self.en))
        stage2.append(Group(equations=g1, real=False))

        return MultiStageEquations([stage1, stage2])

    def configure_solver(self,
                         kernel=None,
                         integrator_cls=None,
                         extra_steppers=None,
                         **kw):
        from pysph.sph.wc.gtvf import GTVFIntegrator
        from pysph.solver.solver import Solver
        if kernel is None:
            kernel = QuinticSpline(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        for granular_particles in self.granular_particles:
            if granular_particles not in steppers:
                steppers[granular_particles] = GTVFDEMStep()

        cls = integrator_cls if integrator_cls is not None else GTVFIntegrator
        integrator = cls(**steppers)

        self.solver = Solver(dim=self.dim,
                             integrator=integrator,
                             kernel=kernel,
                             **kw)

    def setup_properties(self, particles, clean=True):
        from pysph.examples.solid_mech.impact import add_properties

        pas = dict([(p.name, p) for p in particles])

        for particles in self.granular_particles:
            pa = pas[particles]

            add_properties(pa, 'fx', 'fy', 'fz', 'torx', 'tory', 'torz', 'wx',
                           'wy', 'wz', 'moi')

            pa.set_output_arrays(
                ['x', 'y', 'z', 'u', 'v', 'w', 'fx', 'fy', 'fz', 'm', 'moi'])

    def get_solver(self):
        return self.solver
