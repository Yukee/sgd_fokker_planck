from fipy import Variable, FaceVariable, CellVariable, Grid1D, PeriodicGrid1D, ExplicitDiffusionTerm, TransientTerm, DiffusionTerm, Viewer, ExponentialConvectionTerm
from fipy.tools import numerix
import sympy as sym
from sympy.vector import CoordSys3D
from sympy.utilities.lambdify import lambdastr, implemented_function, lambdify
import numpy as np

def get_coeffs(q, print_func=False):
    """
    q: function of w (learned variable) and w0 (fixed, free parameter)
    """
    w, w0, v, ve, l, b = sym.symbols('w w0 v ve l b')
    N = CoordSys3D('N')
    grad_q = sym.diff(q, w)
    h = -l*v*q*grad_q
    g = (l/sym.sqrt(b))*(-v*q*grad_q*N.i + sym.sqrt(v*ve/2)*grad_q*N.j)
    d1 = h + g.dot(sym.diff(g, w))
    d2 = g.dot(g)
    convection = -d1 + sym.diff(d2, w)
    if print_func:
        display("convection term:", sym.factor(convection))
        display("diffusion term:", d2)
    return lambdify((l, v, ve, w0, b, w), convection), lambdify((l, v, ve, w0, b, w), d2)

def run(q, nw, lr, v, ve, w0, b, L, dt, steps, w_left, w_right, zero_prob_left, zero_prob_right, pbc=False):
    """
    Init is 1 inside the interval w_left <= w <= w_right and 0 outside of it

    zero_prob_left: Dirichlet boundary condition on the left end of the interval, with p(left_end) = 0
    """
    
    convection, d2 = get_coeffs(q)

    if pbc:
        mesh = PeriodicGrid1D(nx=nw, dx=1./nw)
    else:
        mesh = Grid1D(nx=nw, Lx=L)
    
    P = CellVariable(mesh=mesh, name=r"$P$")
    P.value = 0.
    w = mesh.cellCenters[0]

    def creneau(w, left, right):
        k = 1./(2.*(right-left))
        return np.where(w < left, -k, k) + np.where(w > right, -k, k)
    
    P.setValue(creneau(w, w_left, w_right))

    if w_left < 0 and w_right > 0 and pbc:
        w_left = w_left % 1.
        P.setValue( .5*(creneau(w, 0., w_right) + creneau(w, w_left, 1.)) )

    if zero_prob_left and not(pbc):
        # set zero probability on the left end of the simulation interval
        P.constrain(0., where=mesh.facesLeft)
    if zero_prob_right and not(pbc):
        # set zero probability on the left end of the simulation interval
        P.constrain(0., where=mesh.facesRight)

    w = mesh.cellCenters[0]
    D2 = d2(lr, v, ve, w0, b, w)
    
    w = mesh.faceCenters[0]
    minusD1plusgradD2 = convection(lr, v, ve, w0, b, w)
    # unit vector
    u = FaceVariable(mesh=mesh, value = 1., rank=1)
    
    # implicit scheme
    eq = TransientTerm() == DiffusionTerm(CellVariable(mesh=mesh, value = D2)) + ExponentialConvectionTerm(u * FaceVariable(mesh=mesh, value = minusD1plusgradD2))
    # explicit scheme: numerically unstable unless dt is small enough compared to spatial discretization
    # eq = TransientTerm() == ExplicitDiffusionTerm(CellVariable(mesh=mesh, value = D2)) + ExponentialConvectionTerm(u * FaceVariable(mesh=mesh, value = minusD1plusgradD2))

    Ps = [P.copy()]
    for step in range(steps):
        eq.solve(var=P, dt=dt)
        Ps.append(P.copy())

    return mesh.x, Ps