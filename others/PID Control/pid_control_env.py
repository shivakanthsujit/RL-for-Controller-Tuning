"""
 `control_env.py` contains the latest code. Will remove this file once that is done.
"""
import control
import gym
import matplotlib.pyplot as plt
import numpy as np
from control.matlab import c2d, ssdata, tf, tf2ss
from gym import spaces
from numpy.linalg import inv
from scipy.integrate import solve_ivp


class LinearTankPID:
    def __init__(
        self,
        uinit=16,
        yinit=4,
        A=3,
        R=1.0,
        ysp=10,
        timesp=1,
        delt=0.1,
        ttfinal=20,
        disturbance=False,
    ):
        #  MPC tuning and simulation parameters
        self.reset_init(uinit, yinit, A, R, ysp, timesp, delt, ttfinal, disturbance)

    def reset_init(
        self,
        uinit=16,
        yinit=4,
        A=3,
        R=1.0,
        ysp=10,
        timesp=1,
        delt=0.1,
        ttfinal=20,
        disturbance=False,
    ):
        #  plant initial conditions
        self.uinit = uinit
        self.yinit = yinit

        self.A = A
        self.R = R

        self.ysp = ysp  # setpoint change (from 0)
        self.timesp = timesp  # time of setpoint change
        self.delt = delt  # sample time
        self.ttfinal = ttfinal  # final simulation time

        self.tt = np.arange(0, self.ttfinal + self.delt, self.delt)  # time vector
        self.kfinal = len(self.tt)  # number of time intervals
        self.ksp = int(self.timesp / self.delt)
        self.r = np.append(
            np.ones((self.ksp, 1)) * self.yinit,
            np.ones((self.kfinal - self.ksp, 1)) * self.ysp,
        )  # setpoint vector

        self.disturbance = disturbance

        self.linear_sys = tf([self.R], [self.R * self.A, 1])
        self.linear_ss = tf2ss(self.linear_sys)
        #  discretize the plant with a sample time, delt
        self.sysd_plant = c2d(self.linear_ss, self.delt)
        [self.phi, self.gamma, self.cd, self.dd] = ssdata(self.sysd_plant)
        self.xinit = self.yinit / self.sysd_plant.C
        return self.reset()

    def reset(self):
        #  initialize input vector
        self.u = np.zeros((self.kfinal + 1, 1))
        self.u[: self.ksp] = np.ones((self.ksp, 1)) * self.uinit
        self.du = np.zeros((self.kfinal, 1))

        self.x = np.zeros((self.kfinal + 1, self.sysd_plant.A.shape[0], 1))
        self.x[: self.ksp] = np.repeat(np.array(self.xinit), self.ksp).reshape(self.ksp, self.sysd_plant.A.shape[0], 1)

        self.y = np.zeros((self.kfinal + 1, 1))
        self.y[: self.ksp] = np.ones((self.ksp, 1)) * self.yinit

        self.E = np.zeros((self.kfinal, 1))
        self.tinitial = 0
        self.tfinal = self.tinitial + self.delt
        self.k = self.ksp - 1
        return self.r[self.k], self.y[self.k][0]

    def step(self, Kc, tau_i, tau_d):
        if self.k == self.kfinal - 1:
            print("Environment is done. Call `reset()` to start again.")
            return
        R = self.r[self.k]
        self.E[self.k] = R - self.y[self.k]

        self.du[self.k] = Kc * (
            self.E[self.k]
            - self.E[self.k - 1]
            + (self.delt / (tau_i)) * self.E[self.k]
            + (tau_d / self.delt) * (self.E[self.k] - 2 * self.E[self.k - 1] + self.E[self.k - 2])
        )
        self.du[self.k] = np.clip(self.du[self.k], self.du[self.k - 1] - 400, self.du[self.k - 1] + 400)

        self.u[self.k] = self.u[self.k - 1] + self.du[self.k]

        w = 0.05 * np.random.randn() if self.disturbance else 0.0
        d = 0.05 if self.k >= 80 and self.disturbance else 0.0

        self.u[self.k] = self.u[self.k] + w + d

        #  plant equations
        self.x[self.k + 1] = np.matmul(self.phi, self.x[self.k]) + np.matmul(self.gamma, self.u[self.k]).T
        self.y[self.k + 1] = np.matmul(self.cd, self.x[self.k + 1])

        self.tinitial = self.tfinal
        self.tfinal += self.delt
        self.k = self.k + 1
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
            plt.grid()
            plt.legend()
            plt.subplot(2, 2, 3)
            plt.step(self.tt[: self.k], self.u[: self.k], label="Control Input")
            plt.ylabel("u")
            plt.xlabel("time")
            plt.title("Control Action")
            plt.grid()
            plt.legend()
            plt.subplot(2, 2, 4)
            plt.step(self.tt[: self.k], self.E[: self.k], label="Tracking Error")
            plt.ylabel("e")
            plt.xlabel("time")
            plt.title("Error")
            plt.grid()
            plt.legend()
            plt.show()

    def ise(self):
        return ((self.r[: self.k] - np.squeeze(self.y[: self.k])) ** 2).sum()


class NonLinearTankPID:
    def __init__(
        self,
        ubar=16,
        ybar=4,
        A=3,
        ysp=10,
        timesp=1,
        delt=0.1,
        ttfinal=20,
        disturbance=False,
    ):
        self.reset_init(ubar, ybar, A, ysp, timesp, delt, ttfinal, disturbance)

    def reset_init(
        self,
        ubar=16,
        ybar=4,
        A=3,
        ysp=10,
        timesp=1,
        delt=0.1,
        ttfinal=20,
        disturbance=False,
    ):
        self.ubar = ubar
        self.ybar = ybar
        self.A = A

        self.ysp = ysp  # setpoint change (from 0)
        self.timesp = timesp  # time of setpoint change
        self.delt = delt  # sample time
        self.ttfinal = ttfinal  # final simulation time
        self.weight = 0.0

        self.tt = np.arange(0, self.ttfinal + self.delt, self.delt)  # time vector
        self.kfinal = len(self.tt)  # number of time intervals
        self.ksp = int(self.timesp / self.delt)
        self.r = np.append(
            np.ones((self.ksp, 1)) * self.ybar,
            np.ones((self.kfinal - self.ksp, 1)) * self.ysp,
        )  # setpoint vector

        #  plant initial conditions
        self.uinit = self.ubar
        self.yinit = self.ybar

        self.disturbance = disturbance

        return self.reset()

    def reset(self):
        #  initialize input vector
        self.u = np.zeros((self.kfinal + 1, 1))
        self.u[: self.ksp] = np.ones((self.ksp, 1)) * self.uinit
        self.U = np.zeros((self.kfinal + 1, 1))
        self.dU = np.zeros((self.kfinal, 1))

        self.y = np.zeros((self.kfinal + 1, 1))
        self.y[: self.ksp] = np.ones((self.ksp, 1)) * self.yinit
        self.Y = np.zeros((self.kfinal + 1, 1))

        self.E = np.zeros((self.kfinal, 1))
        self.ss_s1 = self.ybar
        self.tinitial = 0
        self.tfinal = self.tinitial + self.delt
        self.k = self.ksp - 1
        return self.r[self.k], self.y[self.k][0]

    def dec_ode(self, u):
        def ode_eqn(t, h):
            beta = self.ubar / np.sqrt(self.ybar)
            hprime = (u - beta * np.sqrt(np.maximum(0.0, h))) / self.A
            return np.array(hprime)

        return ode_eqn

    def step(self, Kc, tau_i, tau_d):
        if self.k == self.kfinal - 1:
            print("Environment is done. Call `reset()` to start again.")
            return
        R = self.r[self.k]
        self.E[self.k] = R - self.y[self.k]
        self.dU[self.k] = Kc * (
            self.E[self.k]
            - self.E[self.k - 1]
            + (self.delt / tau_i) * self.E[self.k]
            + (tau_d / self.delt) * (self.E[self.k] - 2 * self.E[self.k - 1] + self.E[self.k - 2])
        )
        self.dU[self.k] = np.clip(self.dU[self.k], self.dU[self.k - 1] - 400, self.dU[self.k - 1] + 400)
        self.U[self.k] = self.U[self.k - 1] + self.dU[self.k]
        w = 0.05 * np.random.randn() if self.disturbance else 0.0
        d = 0.05 if self.k >= 80 else 0.0
        self.u[self.k] = self.U[self.k] + self.ubar + w + d
        sol = solve_ivp(self.dec_ode(self.u[self.k]), [self.tinitial, self.tfinal], [self.ss_s1])
        self.ss_s1 = sol.y[0][-1]
        self.y[self.k + 1] = self.ss_s1
        self.Y[self.k + 1] = self.y[self.k + 1] - self.ybar
        self.tinitial = self.tfinal
        self.tfinal += self.delt
        self.k = self.k + 1
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
            plt.grid()
            plt.legend()
            plt.subplot(2, 2, 3)
            plt.step(self.tt[: self.k], self.u[: self.k], label="Control Input")
            plt.ylabel("u")
            plt.xlabel("time")
            plt.title("Control Action")
            plt.grid()
            plt.legend()
            plt.subplot(2, 2, 4)
            plt.step(self.tt[: self.k], self.E[: self.k], label="Tracking Error")
            plt.ylabel("e")
            plt.xlabel("time")
            plt.title("Error")
            plt.grid()
            plt.legend()
            plt.show()

    def ise(self):
        return ((self.r[: self.k] - np.squeeze(self.y[: self.k])) ** 2).sum()


class GymNonLinearTankPID(gym.Env):
    def __init__(
        self,
        deterministic=False,
        ubar=16,
        ybar=4,
        A=3,
        ysp=10,
        timesp=1,
        disturbance=False,
    ):
        super().__init__()
        n_actions = 3
        self.action_space = spaces.Box(-1.0, 1.0, (n_actions,))
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(2,), dtype=np.float32)

        self.deterministic = deterministic
        self.ubar = ubar
        self.ybar = ybar
        self.A = A
        self.ysp = ysp
        self.timesp = timesp
        self.disturbance = disturbance

        self.system = NonLinearTankPID(self.ubar, self.ybar, self.A, self.ysp, self.timesp, self.disturbance)

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
        if (
            action[0] <= 1.0
            and action[0] >= -1.0
            and action[1] <= 1.0
            and action[1] >= -1.0
            and action[2] <= 1.0
            and action[2] >= -1.0
        ):
            action[0] = action[0] * 4.0 + 4.0
            action[1] = action[1] * 0.5 + 0.5 + 1e-3
            action[2] = action[2] * 0.5 + 0.5
        else:
            reward += -5.0
        obs = self.system.step(*action)
        done = bool(self.system.k == self.system.kfinal - 1)

        e = obs[0] - obs[1]
        error = 2.0 * e ** 2
        reward += -np.clip(error, -100.0, 100.0)
        reward += -5.0 if e > 2.0 * obs[0] else 0.0

        info = {}

        return np.array(obs).astype(np.float32), reward, done, info

    def render(self, mode="plot"):
        if mode != "plot":
            raise NotImplementedError()
        print("ISE: ", self.system.ise())
        self.system.plot()

    def close(self):
        pass


class GymLinearTankPID(gym.Env):
    def __init__(
        self,
        deterministic=False,
        ubar=16,
        ybar=4,
        A=3,
        R=1.0,
        ysp=5,
        timesp=1,
        disturbance=False,
    ):
        super().__init__()
        n_actions = 3
        self.action_space = spaces.Box(-1.0, 1.0, (n_actions,))
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(2,), dtype=np.float32)

        self.deterministic = deterministic
        self.ubar = ubar
        self.ybar = ybar
        self.A = A
        self.R = R
        self.ysp = ysp
        self.timesp = timesp
        self.disturbance = disturbance
        self.system = LinearTankPID(
            uinit=self.ubar,
            yinit=self.ybar,
            A=self.A,
            R=self.R,
            ysp=self.ysp,
            timesp=self.timesp,
            disturbance=self.disturbance,
        )

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        if not self.deterministic:
            self.ubar = np.random.randint(15, 20)
            self.ybar = np.random.randint(3, 5)
            self.ysp = np.random.randint(4, 6)
            self.timesp = np.random.randint(1, 4)
            obs = self.system.reset_init(
                uinit=self.ubar,
                yinit=self.ybar,
                A=self.A,
                ysp=self.ysp,
                timesp=self.timesp,
            )
        else:
            obs = self.system.reset()
        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        return np.array(obs).astype(np.float32)

    def step(self, action):
        reward = 0.0
        scaling = 0.5
        if (
            action[0] <= 1.0
            and action[0] >= -1.0
            and action[1] <= 1.0
            and action[1] >= -1.0
            and action[2] <= 1.0
            and action[2] >= -1.0
        ):
            action[0] = action[0] * 4.0 + 4.0
            action[1] = action[1] * scaling + 0.5 + 1e-3
            action[2] = action[2] * scaling + 0.5
        else:
            reward += -5.0
        obs = self.system.step(*action)
        done = bool(self.system.k == self.system.kfinal - 1)

        e = obs[0] - obs[1]
        error = e ** 2 / 1e3
        reward += -np.clip(error, -10.0, 10.0)
        reward += -5.0 if np.abs(e) > 1.5 * (self.ysp - self.ybar) else 0.0

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
    sim = NonLinearTankPID()
    sim.reset()
    for i in range(sim.kfinal - sim.ksp):
        _, _ = sim.step(11.0, 0.05, 0.2)
    print(sim.ise())
    sim.plot()

    sim = LinearTankPID()
    sim.reset()
    for i in range(sim.kfinal - sim.ksp):
        _, _ = sim.step(7, 0.05, 0.2999)
    print(sim.ise())
    sim.plot()
