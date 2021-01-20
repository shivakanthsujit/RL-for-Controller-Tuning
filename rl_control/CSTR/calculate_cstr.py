import numpy as np
from scipy.integrate import solve_ivp


def dec_ode(u):
    def ode_eqn(t, z):
        qc = u
        q = 100
        Cao = 1
        To = 350
        Tco = 350
        V = 100
        hA = 7e5
        ko = 7.2e10
        AE = 1e4
        delH = 2e5
        rho = 1e3
        rhoc = 1e3
        Cp = 1
        Cpc = 1
        Ca = z[0]
        T = z[1]
        f1 = (q / V * (Cao - Ca)) - (ko * Ca * np.exp(-AE / T))
        f2 = (
            (q / V * (To - T))
            - ((((-delH) * ko * Ca) / (rho * Cp)) * np.exp(-AE / T))
            + (((rhoc * Cpc) / (rho * Cp * V)) * qc * (1 - np.exp(-hA / (qc * rho * Cp))) * (Tco - T))
        )
        return [f1, f2]

    return ode_eqn


Tinit = 438.86881957
yinit = 0.09849321
u = 107.0
t_span = np.linspace(0, 10, 100)


def get_init_states(u):
    sol = solve_ivp(dec_ode(u), t_span=[t_span[0], t_span[-1]], y0=[yinit, Tinit])
    return sol.y[0][-1], sol.y[1][-1]
