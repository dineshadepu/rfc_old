"""
Run it by

python lid_driven_cavity.py --openmp --scheme etvf --integrator pec --internal-flow --pst sun2019 --re 100 --tf 25 --nx 50 --no-edac -d lid_driven_cavity_scheme_etvf_integrator_pec_pst_sun2019_re_100_nx_50_no_edac_output --detailed-output --pfreq 100

"""
import numpy
import numpy as np

from pysph.sph.equation import Equation, Group, MultiStageEquations
from pysph.sph.integrator_step import IntegratorStep
from pysph.sph.scheme import Scheme
from pysph.sph.scheme import add_bool_argument
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.base.kernels import QuinticSpline

from pysph.sph.wc.edac import (SolidWallPressureBC)

from boundary_particles import (add_boundary_identification_properties)

from boundary_particles import (ComputeNormals, SmoothNormals,
                                IdentifyBoundaryParticleCosAngleEDAC)

from common import EDACIntegrator
from pysph.examples.solid_mech.impact import add_properties
from pysph.sph.wc.linalg import mat_vec_mult
from pysph.sph.basic_equations import (ContinuityEquation,
                                       MonaghanArtificialViscosity,
                                       VelocityGradient3D, VelocityGradient2D)
from pysph.sph.solid_mech.basic import (IsothermalEOS,
                                        HookesDeviatoricStressRate)
from pysph.sph.solid_mech.basic import (get_speed_of_sound, get_bulk_mod,
                                        get_shear_modulus)

from solid_mech_common import AddGravityToStructure
from pysph.sph.wc.transport_velocity import (MomentumEquationArtificialViscosity)
from fsi_coupling import (ClampWallPressure, ClampWallPressureFSI)


class EDACEquation(Equation):
    def __init__(self, dest, sources, nu):
        self.nu = nu

        super(EDACEquation, self).__init__(dest, sources)

    def initialize(self, d_idx, d_ap):
        d_ap[d_idx] = 0.0

    def loop(self, d_idx, d_m, d_rho, d_ap, d_p, d_V, s_idx, s_m, s_rho, s_p,
             d_c0_ref, DWIJ, VIJ, XIJ, R2IJ, EPS):
        Vi = d_m[d_idx]/d_rho[d_idx]
        Vj = s_m[s_idx]/s_rho[s_idx]
        Vi2 = Vi * Vi
        Vj2 = Vj * Vj
        cs2 = d_c0_ref[0] * d_c0_ref[0]

        etai = d_rho[d_idx]
        etaj = s_rho[s_idx]
        etaij = 2 * self.nu * (etai * etaj)/(etai + etaj)

        # This is the same as continuity acceleration times cs^2
        rhoi = d_rho[d_idx]
        rhoj = s_rho[s_idx]
        vijdotdwij = DWIJ[0]*VIJ[0] + DWIJ[1]*VIJ[1] + DWIJ[2]*VIJ[2]
        d_ap[d_idx] += rhoi/rhoj*cs2*s_m[s_idx]*vijdotdwij

        # Viscous damping of pressure.
        xijdotdwij = DWIJ[0]*XIJ[0] + DWIJ[1]*XIJ[1] + DWIJ[2]*XIJ[2]
        tmp = 1.0/d_m[d_idx]*(Vi2 + Vj2)*etaij*xijdotdwij/(R2IJ + EPS)
        d_ap[d_idx] += tmp*(d_p[d_idx] - s_p[s_idx])


class EDACEquationFSI(Equation):
    def __init__(self, dest, sources, nu):
        self.nu = nu

        super(EDACEquationFSI, self).__init__(dest, sources)

    def initialize(self, d_idx, d_ap):
        d_ap[d_idx] = 0.0

    def loop(self, d_idx, d_m, d_rho, d_ap, d_p, d_c0_ref, s_idx, s_m_fsi,
             s_rho_fsi, s_p_fsi, DWIJ, VIJ, XIJ, R2IJ, EPS):
        Vi = d_m[d_idx]/d_rho[d_idx]
        Vj = s_m_fsi[s_idx]/s_rho_fsi[s_idx]
        Vi2 = Vi * Vi
        Vj2 = Vj * Vj
        cs2 = d_c0_ref[0] * d_c0_ref[0]

        etai = d_rho[d_idx]
        etaj = s_rho_fsi[s_idx]
        etaij = 2 * self.nu * (etai * etaj)/(etai + etaj)

        # This is the same as continuity acceleration times cs^2
        rhoi = d_rho[d_idx]
        rhoj = s_rho_fsi[s_idx]
        vijdotdwij = DWIJ[0]*VIJ[0] + DWIJ[1]*VIJ[1] + DWIJ[2]*VIJ[2]
        d_ap[d_idx] += rhoi/rhoj*cs2*s_m_fsi[s_idx]*vijdotdwij

        # Viscous damping of pressure.
        xijdotdwij = DWIJ[0]*XIJ[0] + DWIJ[1]*XIJ[1] + DWIJ[2]*XIJ[2]
        tmp = 1.0/d_m[d_idx]*(Vi2 + Vj2)*etaij*xijdotdwij/(R2IJ + EPS)
        d_ap[d_idx] += tmp*(d_p[d_idx] - s_p_fsi[s_idx])


class ContinuityEquationGTVF(Equation):
    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, d_arho, s_idx, s_m,
             d_uhat, d_vhat, d_what,
             s_uhat, s_vhat, s_what, DWIJ):
        uij_x = d_uhat[d_idx] - s_uhat[s_idx]
        uij_y = d_uhat[d_idx] - s_uhat[s_idx]
        uij_z = d_uhat[d_idx] - s_uhat[s_idx]

        vijdotdwij = DWIJ[0]*uij_x + DWIJ[1]*uij_y + DWIJ[2]*uij_z
        d_arho[d_idx] += s_m[s_idx]*vijdotdwij


class MomentumEquationPressureGradient(Equation):
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz
        super(MomentumEquationPressureGradient, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = self.gx
        d_av[d_idx] = self.gy
        d_aw[d_idx] = self.gz

    def loop(self, d_rho, s_rho, d_idx, s_idx, d_p, s_p, s_m, d_au, d_av, d_aw,
             DWIJ, XIJ, RIJ, SPH_KERNEL, HIJ):
        rhoi2 = d_rho[d_idx] * d_rho[d_idx]
        rhoj2 = s_rho[s_idx] * s_rho[s_idx]

        pij = d_p[d_idx] / rhoi2 + s_p[s_idx] / rhoj2

        tmp = -s_m[s_idx] * pij

        d_au[d_idx] += tmp * DWIJ[0]
        d_av[d_idx] += tmp * DWIJ[1]
        d_aw[d_idx] += tmp * DWIJ[2]


class GTVFScheme(Scheme):
    def __init__(self, fluids, boundaries, dim, rho0, p0, c0, h, nu, gamma=7.0,
                 gx=0.0, gy=0.0, gz=0.0, alpha=0.1, beta=0.0, edac_alpha=0.5):
        self.fluids = fluids
        self.boundaries = boundaries

        # fluids parameters
        self.edac = False
        self.edac_alpha = edac_alpha
        self.h = h
        self.art_nu = 0.
        self.nu = nu

        self.dim = dim

        self.kernel = QuinticSpline

        self.rho0 = rho0
        self.p0 = p0
        self.c0 = c0
        self.gamma = gamma

        self.gx = gx
        self.gy = gy
        self.gz = gz

        self.fluid_alpha = alpha
        self.beta = beta

        self.solver = None

    def add_user_options(self, group):
        group.add_argument("--fluid-alpha", action="store",
                           dest="fluid_alpha", default=0.5,
                           type=float,
                           help="Artificial viscosity")

        add_bool_argument(group, 'edac', dest='edac', default=True,
                          help='Use pressure evolution equation EDAC')

    def consume_user_options(self, options):
        _vars = ['fluid_alpha', 'edac']
        data = dict((var, self._smart_getattr(options, var)) for var in _vars)
        self.configure(**data)

    def attributes_changed(self):
        self.edac_nu = self.fluid_alpha * self.h * self.c0 / 8

    def get_equations(self):
        # elastic solid equations
        from pysph.sph.equation import Group
        from pysph.sph.wc.basic import (MomentumEquation, TaitEOS)
        from pysph.sph.wc.transport_velocity import (
            SetWallVelocity,
            MomentumEquationArtificialViscosity)
        from pysph.sph.wc.viscosity import (LaminarViscosity)
        from pysph.sph.basic_equations import (MonaghanArtificialViscosity)
        from pysph.sph.wc.edac import (SolidWallPressureBC)

        all = self.fluids + self.boundaries
        edac_nu = self.edac_nu

        stage1 = []

        eqs = []
        for fluid in self.fluids:
            eqs.append(ContinuityEquationGTVF(
                dest=fluid, sources=self.fluids+self.boundaries), )
            if self.edac is True:
                eqs.append(
                    EDACEquation(
                        dest=fluid, sources=self.fluids+self.boundaries,
                        nu=edac_nu))

        stage1.append(Group(equations=eqs, real=False))

        # ==============================
        # Stage 2 equations
        # ==============================
        stage2 = []
        g2 = []

        if len(self.fluids) > 0:
            tmp = []
            if self.edac is False:
                for fluid in self.fluids:
                    tmp.append(
                        TaitEOS(dest=fluid, sources=None,
                                rho0=self.rho0,
                                c0=self.c0,
                                gamma=self.gamma))

                stage2.append(Group(equations=tmp, real=False))

        if len(self.fluids) > 0:
            tmp = []
            for solid in self.boundaries:
                tmp.append(
                    SetWallVelocity(dest=solid,
                                    sources=self.fluids))
                tmp.append(
                    SolidWallPressureBC(dest=solid,
                                        sources=self.fluids,
                                        gx=self.gx,
                                        gy=self.gy,
                                        gz=self.gz))
                tmp.append(
                    ClampWallPressure(dest=solid, sources=None))

            if len(tmp) > 0:
                stage2.append(Group(equations=tmp, real=False))

        if len(self.fluids) > 0:
            for name in self.fluids:
                alpha = self.fluid_alpha
                g2.append(
                    MomentumEquationPressureGradient(dest=name,
                                                     sources=self.fluids+self.boundaries,
                                                     gx=self.gx,
                                                     gy=self.gy,
                                                     gz=self.gz))

                if abs(alpha) > 1e-14:
                    eq = MomentumEquationArtificialViscosity(dest=name,
                                                             sources=self.fluids,
                                                             c0=self.c0,
                                                             alpha=self.fluid_alpha)
                    g2.insert(-1, eq)

            stage2.append(Group(equations=g2))

        return MultiStageEquations([stage1, stage2])

    def configure_solver(self,
                         kernel=None,
                         integrator_cls=None,
                         extra_steppers=None,
                         **kw):
        from pysph.base.kernels import QuinticSpline
        from pysph.solver.solver import Solver
        if kernel is None:
            kernel = QuinticSpline(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        fluidstep = GTVFFluidStep()
        integrator_cls = GTVFIntegrator

        for fluid in self.fluids:
            if fluid not in steppers:
                steppers[fluid] = fluidstep

        cls = integrator_cls
        integrator = cls(**steppers)

        self.solver = Solver(dim=self.dim,
                             integrator=integrator,
                             kernel=kernel,
                             **kw)

    def setup_properties(self, particles, clean=True):
        from pysph.examples.solid_mech.impact import add_properties

        pas = dict([(p.name, p) for p in particles])

        for boundary in self.boundaries:
            pa = pas[boundary]

            nb = 1
            consts = {
                'total_mass':
                np.zeros(nb, dtype=float),
                'xcm':
                np.zeros(3 * nb, dtype=float),
            }

            for key, elem in consts.items():
                pa.add_constant(key, elem)

            body_id = np.zeros(len(pa.x), dtype=int)
            pa.add_property('body_id', type='int', data=body_id)

            # Adami boundary conditions. SetWallVelocity
            add_properties(pa, 'ug', 'vf', 'vg', 'wg', 'uf', 'wf', 'wij')
            # pa.add_property('m_fluid')

            ####################################################
            # compute the boundary particles of the rigid body #
            ####################################################
            add_boundary_identification_properties(pa)
            # make sure your rho is not zero
            equations = get_boundary_identification_etvf_equations([pa.name],
                                                                   [pa.name])

            sph_eval = SPHEvaluator(arrays=[pa],
                                    equations=equations,
                                    dim=self.dim,
                                    kernel=QuinticSpline(dim=self.dim))

            sph_eval.evaluate(dt=0.1)

        for fluid in self.fluids:
            pa = pas[fluid]

            add_properties(pa, 'rho0', 'u0', 'v0', 'w0', 'x0', 'y0', 'z0',
                           'arho', 'vol', 'cs', 'ap')

            if 'c0_ref' not in pa.constants:
                pa.add_constant('c0_ref', self.c0)

            pa.vol[:] = pa.m[:] / pa.rho[:]

            pa.cs[:] = self.c0
            pa.add_output_arrays(['p'])

    def _get_edac_nu(self):
        if self.art_nu > 0:
            nu = self.art_nu
            print(self.art_nu)
            print("Using artificial viscosity for EDAC with nu = %s" % nu)
        else:
            nu = self.nu
            print("Using real viscosity for EDAC with nu = %s" % self.nu)
        return nu

    def get_solver(self):
        return self.solver
