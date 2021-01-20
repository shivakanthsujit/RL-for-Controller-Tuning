import control
import gym
import matplotlib.pyplot as plt
import numpy as np
from control.matlab import c2d, ss, ssdata, tf
from dmc_utils import dmccalc, smatgen
from gym import spaces
from scipy.integrate import solve_ivp


class Reactor:
    def __init__(self):
        #  MPC tuning and simulation parameters
        self.n = 50  # model length
        self.p = 10  # prediction horizon
        self.m = 1  # control horizon
        self.ysp = 1  # setpoint change (from 0)
        self.timesp = 0.2  # time of setpoint change
        self.delt = 0.1  # sample time
        self.tfinal = 20  # final simulation time
        self.noise = 0  # noise added to response coefficients

        self.t = np.arange(0, self.tfinal + self.delt, self.delt)  # time vector
        self.kfinal = len(self.t)  # number of time intervals
        self.ksp = int(self.timesp / self.delt)
        self.r = np.append(np.zeros((self.ksp, 1)), np.ones((self.kfinal - self.ksp, 1)) * self.ysp)  # setpoint vector

        """
    ----- insert continuous model here -----------
    model (continuous state space form)
    """
        self.a = np.array([[-2.4048, 0], [0.8333, -2.2381]]).reshape(2, 2)  # a matrix � van de vusse
        self.b = np.array([[7], [-1.117]]).reshape(2, 1)  # b matrix � van de vusse
        self.c = np.array([0, 1]).reshape(1, 2)  # c matrix � van de vusse
        self.d = np.array([0]).reshape(1, 1)  # d matrix � van de vusse
        self.sysc_mod = ss(self.a, self.b, self.c, self.d)  # create LTI "object"

        """
    ----- insert plant here -----------
    perfect model assumption (plant = model)
    """
        self.ap = self.a
        self.bp = self.b
        self.cp = self.c
        self.dp = self.d
        self.sysc_plant = ss(self.ap, self.bp, self.cp, self.dp)

        #  discretize the plant with a sample time, delt
        self.sysd_plant = c2d(self.sysc_plant, self.delt)
        [self.phi, self.gamma, self.cd, self.dd] = ssdata(self.sysd_plant)

        #  evaluate discrete model step response coefficients
        [s, timeresp] = control.matlab.step(
            self.sysc_mod,
            np.arange(self.delt, self.n * self.delt + 2 * self.delt, self.delt),
        )
        self.s = s[-self.n :].reshape(self.n, 1)
        self.timeresp = timeresp[-self.n :].reshape(self.n, 1)

        #  plant initial conditions
        self.xinit = np.zeros((self.a.shape[0], 1))
        self.uinit = 0
        self.yinit = 0

        #  initialize input vector
        self.u = np.zeros((self.kfinal + 1, 1))
        self.u[: min(self.p, self.kfinal)] = np.ones((min(self.p, self.kfinal), 1)) * self.uinit

        self.dup = np.zeros((self.n - 2, 1))
        self.sn = self.s[self.n - 1]  # last step response coefficient

        self.x = np.zeros((self.kfinal + 1, 2, 1))
        self.x[0] = np.array(self.xinit)

        self.y = np.zeros((self.kfinal + 1, 1))
        self.y[0] = np.array([self.yinit])

        self.ymod = np.zeros((self.kfinal + 1, 1))
        self.dist = np.zeros((self.kfinal + 1, 1))
        self.du = np.zeros((self.kfinal, 1))
        self.k = 0

    def step(self, weight):
        # print(self.s, self.p, self.m, self.n, weight)
        Sf, Sp, Kmat = smatgen(self.s, self.p, self.m, self.n, weight)
        self.du[self.k] = dmccalc(
            Sp,
            Kmat,
            self.sn,
            self.dup,
            self.dist[self.k],
            self.r[self.k],
            self.u,
            self.k,
            self.n,
        )
        #  perform control calculation
        if self.k > 0:
            self.u[self.k] = self.u[self.k - 1] + self.du[self.k]  # control input
        else:
            self.u[self.k] = self.uinit + self.du[self.k]

        #  plant equations
        self.x[self.k + 1] = np.matmul(self.phi, self.x[self.k]) + np.matmul(self.gamma, self.u[self.k]).T
        self.y[self.k + 1] = np.matmul(self.cd, self.x[self.k + 1])
        #  model prediction
        if self.k - self.n + 1 >= 0:
            self.ymod[self.k + 1] = (
                self.s[0] * self.du[self.k] + np.matmul(Sp[0, :], self.dup) + self.sn * self.u[self.k - self.n + 1]
            )
        else:
            self.ymod[self.k + 1] = self.s[0] * self.du[self.k] + np.matmul(Sp[0, :], self.dup)

        #  disturbance compensation
        self.dist[self.k + 1] = self.y[self.k + 1] - self.ymod[self.k + 1]
        #  additive disturbance assumption
        #  put input change into vector of past control moves
        self.dup = (np.append(self.du[self.k], self.dup[: self.n - 3])).reshape(self.n - 2, 1)
        self.k = self.k + 1
        return self.r[self.k], self.y[self.k][0]

    def reset(self):
        #  initialize input vector
        self.u = np.zeros((self.kfinal + 1, 1))
        self.u[: min(self.p, self.kfinal)] = np.ones((min(self.p, self.kfinal), 1)) * self.uinit

        self.dup = np.zeros((self.n - 2, 1))
        self.sn = self.s[self.n - 1]  # last step response coefficient

        self.x = np.zeros((self.kfinal + 1, 2, 1))
        self.x[0] = np.array(self.xinit)

        self.y = np.zeros((self.kfinal + 1, 1))
        self.y[0] = np.array([self.yinit])

        self.ymod = np.zeros((self.kfinal + 1, 1))
        self.dist = np.zeros((self.kfinal + 1, 1))
        self.du = np.zeros((self.kfinal, 1))
        self.k = 0
        return self.r[self.k], self.y[self.k][0]

    def plot(self):
        if self.k == 0:
            print("Simulation not started.")
        else:
            plt.figure(figsize=(16, 16))
            plt.subplot(2, 1, 1)
            plt.step(self.t[: self.k], self.r[: self.k], linestyle="dashed", label="Setpoint")
            plt.plot(self.t[: self.k], self.y[: self.k], label="Plant Output")
            plt.ylabel("y")
            plt.xlabel("time")
            plt.title("Plant Output")
            plt.legend()
            plt.subplot(2, 1, 2)
            plt.step(self.t[: self.k], self.u[: self.k], label="Control Input")
            plt.ylabel("u")
            plt.xlabel("time")
            plt.title("Control Action")
            plt.legend()
            plt.show()

    def ise(self):
        return ((self.r[: self.k] - np.squeeze(self.y[: self.k])) ** 2).sum()


class GymReactor(gym.Env):
    def __init__(self):
        super().__init__()

        # Only one action, to choose the weights
        n_actions = 1
        self.action_space = spaces.Box(-1.0, 1.0, (n_actions,))
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)
        self.reactor = Reactor()

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        obs = self.reactor.reset()
        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        return np.array(obs).astype(np.float32)

    def step(self, action):
        reward = 0.0
        if action <= 1.0 and action >= -1.0:
            action = action * 5.0 + 5.0
        else:
            action = 0.0
            reward += -10.0
            # raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        obs = self.reactor.step(action)
        done = bool(self.reactor.k == self.reactor.kfinal - 1)

        e = obs[0] - obs[1]
        error = e ** 2
        reward += -error

        info = {}

        return np.array(obs).astype(np.float32), reward, done, info

    def render(self, mode="plot"):
        if mode != "plot":
            raise NotImplementedError()
        print("ISE: ", self.reactor.ise())
        self.reactor.plot()

    def close(self):
        pass


class NonLinearTank:
    def __init__(self, ubar=16, ybar=4, A=3, ysp=10, timesp=1):
        #  MPC tuning and simulation parameters
        self.n = 60  # model length
        self.p = 3  # prediction horizon
        self.m = 1  # control horizon

        self.ubar = ubar
        self.ybar = ybar
        self.A = A

        self.ysp = ysp  # setpoint change (from 0)
        self.timesp = timesp  # time of setpoint change
        self.delt = 0.1  # sample time
        self.ttfinal = 16  # final simulation time
        self.weight = 0.0

        self.tt = np.arange(0, self.ttfinal + self.delt, self.delt)  # time vector
        self.kfinal = len(self.tt)  # number of time intervals
        self.ksp = int(self.timesp / self.delt)
        self.r = np.append(
            np.ones((self.ksp, 1)) * self.ybar,
            np.ones((self.kfinal - self.ksp, 1)) * self.ysp,
        )  # setpoint vector

        self.linear_sys = tf([0.5], [1.5, 1])

        #  evaluate discrete model step response coefficients
        [s, _] = control.matlab.step(
            self.linear_sys,
            np.arange(self.delt, self.n * self.delt + 2 * self.delt, self.delt),
        )
        self.s = s[-self.n :].reshape(self.n, 1)

        #  plant initial conditions
        self.uinit = self.ubar
        self.yinit = self.ybar

        self.reset()

    def dec_ode(self, u):
        def ode_eqn(t, h):
            beta = self.ubar / np.sqrt(self.ybar)
            hprime = (u - beta * np.sqrt(h)) / self.A
            return np.array(hprime)

        return ode_eqn

    def step(self, weight):
        if self.k == self.kfinal - 1:
            print("Environment is done. Call `reset()` to start again.")
            return
        Sf, Sp, Kmat = smatgen(self.s, self.p, self.m, self.n, weight)
        R = self.r[self.k] - self.ybar
        self.dU[self.k] = dmccalc(Sp, Kmat, self.sn, self.dUp, self.dist[self.k], R, self.U, self.k, self.n)

        #  perform control calculation
        if self.k > 0:
            self.U[self.k] = self.U[self.k - 1] + self.dU[self.k]  # control input
        else:
            self.U[self.k] = self.uinit + self.dU[self.k]

        self.u[self.k] = self.U[self.k] + self.ubar
        sol = solve_ivp(self.dec_ode(self.u[self.k]), [self.tinitial, self.tfinal], [self.ss_s1])
        self.ss_s1 = sol.y[0][-1]
        self.y[self.k + 1] = self.ss_s1
        self.Y[self.k + 1] = self.y[self.k + 1] - self.ybar
        #  model prediction
        if self.k - self.n + 1 >= 0:
            self.Ymod[self.k + 1] = (
                self.s[0] * self.dU[self.k] + np.matmul(Sp[0, :], self.dUp) + self.sn * self.U[self.k - self.n + 1]
            )
        else:
            self.Ymod[self.k + 1] = self.s[0] * self.dU[self.k] + np.matmul(Sp[0, :], self.dUp)

        #  disturbance compensation
        self.dist[self.k + 1] = self.Y[self.k + 1] - self.Ymod[self.k + 1]
        #  additive disturbance assumption
        #  put input change into vector of past control moves
        self.dUp = (np.append(self.dU[self.k], self.dUp[: self.n - 3])).reshape(self.n - 2, 1)
        self.tinitial = self.tfinal
        self.tfinal += self.delt
        self.k = self.k + 1
        return self.r[self.k], self.y[self.k][0]

    def reset_init(self, ubar=16, ybar=4, A=3, ysp=10, timesp=1):
        #  MPC tuning and simulation parameters
        self.n = 60  # model length
        self.p = 3  # prediction horizon
        self.m = 1  # control horizon

        self.ubar = ubar
        self.ybar = ybar
        self.A = A

        self.ysp = ysp  # setpoint change (from 0)
        self.timesp = timesp  # time of setpoint change
        self.delt = 0.1  # sample time
        self.ttfinal = 16  # final simulation time
        self.weight = 0.0

        self.tt = np.arange(0, self.ttfinal + self.delt, self.delt)  # time vector
        self.kfinal = len(self.tt)  # number of time intervals
        self.ksp = int(self.timesp / self.delt)
        self.r = np.append(
            np.ones((self.ksp, 1)) * self.ybar,
            np.ones((self.kfinal - self.ksp, 1)) * self.ysp,
        )  # setpoint vector

        self.linear_sys = tf([0.5], [1.5, 1])

        #  evaluate discrete model step response coefficients
        [s, _] = control.matlab.step(
            self.linear_sys,
            np.arange(self.delt, self.n * self.delt + 2 * self.delt, self.delt),
        )
        self.s = s[-self.n :].reshape(self.n, 1)

        #  plant initial conditions
        self.uinit = self.ubar
        self.yinit = self.ybar

        return self.reset()

    def reset(self):
        #  initialize input vector
        self.u = np.zeros((self.kfinal + 1, 1))
        self.u[: self.ksp] = np.ones((self.ksp, 1)) * self.uinit
        self.U = np.zeros((self.kfinal + 1, 1))
        self.dUp = np.zeros((self.n - 2, 1))
        self.dU = np.zeros((self.kfinal, 1))

        self.y = np.zeros((self.kfinal + 1, 1))
        self.y[: self.ksp] = np.ones((self.ksp, 1)) * self.yinit
        self.Y = np.zeros((self.kfinal + 1, 1))
        self.Ymod = np.zeros((self.kfinal + 1, 1))

        self.sn = self.s[self.n - 1]  # last step response coefficient
        self.dist = np.zeros((self.kfinal + 1, 1))
        self.ss_s1 = self.ybar
        self.tinitial = 0
        self.tfinal = self.tinitial + self.delt
        self.k = self.ksp - 1
        return self.r[self.k], self.y[self.k][0]

    def plot(self):
        if self.k == 0:
            print("Simulation not started.")
        else:
            plt.figure(figsize=(16, 16))
            plt.subplot(2, 1, 1)
            plt.step(
                self.tt[: self.k],
                self.r[: self.k],
                linestyle="dashed",
                label="Setpoint",
            )
            plt.plot(self.tt[: self.k], self.y[: self.k], label="Plant Output")
            plt.ylabel("y")
            plt.xlabel("time")
            plt.title("Plant Output")
            plt.legend()
            plt.subplot(2, 1, 2)
            plt.step(self.tt[: self.k], self.u[: self.k], label="Control Input")
            plt.ylabel("u")
            plt.xlabel("time")
            plt.title("Control Action")
            plt.legend()
            plt.show()

    def ise(self):
        return ((self.r[: self.k] - np.squeeze(self.y[: self.k])) ** 2).sum()


class GymNonLinearTank(gym.Env):
    def __init__(self, deterministic=False, ubar=16, ybar=4, A=3, ysp=10, timesp=1):
        super().__init__()

        self.system = NonLinearTank()

        n_actions = 1
        self.action_space = spaces.Box(-1.0, 1.0, (n_actions,))
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(2,), dtype=np.float32)

        self.deterministic = deterministic
        self.ubar = ubar
        self.ybar = ybar
        self.A = A
        self.ysp = ysp
        self.timesp = timesp

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        if not self.deterministic:
            u = np.random.randint(15, 20)
            y = np.random.randint(3, 8)
            ys = np.random.randint(5, 10)
            ts = np.random.randint(1, 4)
            obs = self.system.reset_init(u, y, 3, ys, ts)
        else:
            obs = self.system.reset()
        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        return np.array(obs).astype(np.float32)

    def step(self, action):
        reward = 0.0
        if action <= 1.0 and action >= -1.0:
            action = action * 10.0 + 10.0
        else:
            action = 0.0
            reward += -100.0
        obs = self.system.step(action)
        done = bool(self.system.k == self.system.kfinal - 1)

        e = obs[0] - obs[1]
        error = e ** 2
        reward += -error

        info = {}

        return np.array(obs).astype(np.float32), reward, done, info

    def render(self, mode="plot"):
        if mode != "plot":
            raise NotImplementedError()
        print("ISE: ", self.system.ise())
        self.system.plot()

    def close(self):
        pass


if __name__ == "__main__":
    sim = Reactor()
    sim.reset()
    for i in range(sim.kfinal - sim.ksp):
        _, _ = sim.step(0.0)
    sim.plot()

    sim = NonLinearTank()
    sim.reset()
    for i in range(sim.kfinal - sim.ksp):
        _, _ = sim.step(0.0)
    sim.plot()
