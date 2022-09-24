import numpy as np
from numpy import linalg
from gym import utils
import os
from gym.envs.mujoco import mujoco_env
import math, random
import matplotlib.pyplot as plt
import argparse
from Common.Utils import *

def hyperparameters():
    parser = argparse.ArgumentParser(description='quadrotor setting')

    # related to initial setting
    parser.add_argument('--mode-cmd', default='velocity', help="velocity, position")
    parser.add_argument('--mode-test', default=False, type=bool, help="If True, It works with PID contoller")
    parser.add_argument('--frame-skip', default=20, type=int, help="time_step = frame skip*initial_time_step")
    parser.add_argument('--pid-skip', default=5, type=int, help="time_step = frame skip*initial_time_step")
    parser.add_argument('--bound-range', default=5, type=int, help="limitation of position that quadrotor can go")

    parser.add_argument('--goal-make', default=False, type=bool, help="If False, goal is origin")
    parser.add_argument('--goal-shape', default='circle', help="circle, helix, custom")
    
    parser.add_argument('--convert-relative', default=False, type=bool, help="convert position to relative position")

    parser.add_argument('--add-disturbance', default=False, type=bool, help="action = action + disturbance")
    parser.add_argument('--form-disturbance', default='sinewave', help="sinewave, impact")
    parser.add_argument('--thrust-noise', default=[0.1, 1000], help="For thrust direction")
    parser.add_argument('--rate-noise', default=[0.0, 2000], help="For rpy direction")


    args = parser.parse_args()

    return args

args = hyperparameters()

class QuadRateEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.goal = np.zeros(6)
        self.local_step = 0
        self.drop_cnt = 0
        self.suc_cnt = 0

        self.error_plot = False

        self.act_min = np.array([3.5, -(1 / 12) * math.pi, -(1 / 12) * math.pi, -0.03])
        self.act_max = np.array([8.5, (1 / 12) * math.pi, (1 / 12) * math.pi, 0.03])

        mujoco_env.MujocoEnv.__init__(self, 'quadrotor.xml', args.frame_skip)
        utils.EzPickle.__init__(self)

        self.timestep = self.model.opt.timestep

    def step(self, input):


        if args.goal_make == True:
            self.goal[:3] = self.make_trajectory(args.goal_shape)

        if args.mode_test == True:
            action = self.PID(self.goal[0:3])
        else:
            if args.mode_cmd == 'velocity':
                action = denomalize(input, self.act_max, self.act_min)
                action = np.clip(action, a_min=self.act_min, a_max=self.act_max)
            elif args.mode_cmd == 'position':
                action = self.PID(0.5 * input[:3])
            else:
                raise Exception(" check your test mode ")
        
        if args.add_disturbance == True:
            action = self.add_disturbance(action)
        
        self.do_simulation(action, self.frame_skip)
        self.local_step += 1

        ob = self._get_obs()
        ob = self._add_error(ob)

        reward, done, info = self._get_reward(ob, action)
    
        if self.local_step == 8000:
            done = True
        if done:
            self.local_step = 0

        return ob, reward, done, info

    def _get_obs(self):
        pos = self.sim.data.qpos * 1e-0
        vel = self.sim.data.qvel * 1e-0
        return np.concatenate([pos.flat, vel.flat])

    def _add_error(self, ob):

        if args.convert_relative == True:
            ob[0:3] = ob[0:3] - self.goal[:3]
            pos = ob[0:3]
            ang = quat2rpy(ob[3:7])
            self.error_pos = linalg.norm(pos)
            self.error_ang = linalg.norm(self.goal[3:] - ang)
        else:
            pos = ob[0:3]
            ang = quat2rpy(ob[3:7])
            self.error_pos = linalg.norm(self.goal[:3] - pos)
            self.error_ang = linalg.norm(self.goal[3:] - ang)

        ob = np.append(ob, [self.error_pos, self.error_ang])

        return ob

    def _get_reward(self, ob, act, bound_range=args.bound_range):

        done = False
        reward = 0

        pos = ob[0:3]
        quat = ob[3:7]
        ang = quat2rpy(quat)
        lin_vel = ob[7:10]
        ang_vel = ob[10:13]
        error_pos = ob[13]
        error_ang = ob[14]

        reward_alive = 2e-1
        penalty_pos = (30 / (error_pos + 0.3) - error_pos) * 1e-1
        penalty_ang = - error_ang * 1e-1
        penalty_lin_vel = -linalg.norm(lin_vel) * 1e-2
        penalty_ang_vel = -linalg.norm(ang_vel) * 1e-2
        penalty_ctrl = - 1e-4 * np.sum(np.square(act))

        reward = reward_alive + \
                 penalty_pos + \
                 penalty_lin_vel + \
                 penalty_ang_vel + \
                 penalty_ctrl

        ############# Goal reach reward ################
        if error_pos < 0.1 and error_ang < 0.25:
            self.suc_cnt += 1
            reward += self.suc_cnt*10
        else:
            self.suc_cnt = 0

        ############# Bound limit penalty ##############
        if pos[2] > bound_range or abs(pos[0]) > bound_range or abs(pos[1]) > bound_range:
            reward -= 100.0
            done = True

        if pos[2] <= -3.95:
            self.drop_cnt += 1
            if self.drop_cnt == 200:
                reward -= 100.0
                self.drop_cnt = 0
                done = True
        else:
            self.drop_cnt = 0

        info = {
            'obx': ob[0],
            'oby': ob[1],
            'obz': ob[2],
            'obvx': ob[7],
            'obvy': ob[8],
            'obvz': ob[9],
        }

        return reward, done, info

    def reset_model(self):
        qpos = self.init_qpos #+ self.np_random.uniform(size=self.model.nq, low=-0.1, high=0.1)
        qvel = self.init_qvel #+ self.np_random.uniform(size=self.model.nv, low=-0.05, high=0.05)
        self.set_state(qpos, qvel)
        ob = self._get_obs()
        ob = self._add_error(ob)

        return ob

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 1
        v.cam.distance = self.model.stat.extent * 4
        v.cam.azimuth = 132.
        # v.cam.lookat[2] += .8
        # v.cam.elevation = 0
        # v.cam.lookat[0] += 1.5
        v.cam.elevation += 0.9

    def get_mass(self):
        mass = np.expand_dims(self.model.body_mass, axis=1)
        return mass

    def make_trajectory(self, shape=args.goal_shape, measure=[3., 2000]):

        if shape == 'circle':
            goal_x = measure[0] * math.cos((math.pi / measure[1]) * self.local_step)
            goal_y = measure[0] * math.sin((math.pi / measure[1]) * self.local_step)
            goal_z = 0

        if shape == 'helix':
            goal_x = measure[0] * math.cos((math.pi / measure[1]) * self.local_step)
            goal_y = measure[0] * math.sin((math.pi / measure[1]) * self.local_step)
            goal_z = 5*(self.local_step/8000)

        goal = np.array([goal_x, goal_y, goal_z])
        return goal

    def PID(self, goal):

        ob = self._get_obs()
        pos = ob[0:3]  # ground height = -3.96504420
        quat = ob[3:7]  # quaternion (cos(a/2), sin(a/2) * (x, y, z))
        lin_vel = ob[7:10]
        ang_vel = ob[10:13]

        euler = self.quat2rpy(quat)

        K_x = np.array([-3.1622776601684, -4.72252397478309, 29.7770194286971])
        K_y = np.array([3.1622776601684, 4.72252397478309, 29.7770194286971])
        K_z = np.array([10.0000000000000, 4.47213595499958])

        error_x = np.array([pos[0] - goal[0], lin_vel[0], euler[1]]).T
        error_y = np.array([pos[1] - goal[1], lin_vel[1], euler[0]]).T
        error_z = np.array([pos[2] - goal[2], lin_vel[2]]).T

        roll_rate = -K_x @ error_y
        pitch_rate = -K_y @ error_x
        yaw_rate = 0
        thrust = -K_z @ error_z + self.get_mass() * 9.81

        return thrust, roll_rate, pitch_rate, yaw_rate

    def add_disturbance(self, action, showme = False, form = args.form_disturbance, T_noise = args.thrust_noise, R_noise = args.rate_noise):

        T_disturb = 0
        R_disturb = 0

        if form == 'sinewave':
            for i in range(len(action)):
                if i == 0:
                    T_disturb = T_noise[0]*math.sin((math.pi / T_noise[1])*self.local_step)
                    action[i] += T_disturb
                else:
                    R_disturb = R_noise[0]*math.sin((math.pi / R_noise[1])*self.local_step)
                    action[i] += R_disturb

        elif form == 'impact':
            for i in range(len(action)):
                if i == 0:
                    if self.local_step % (T_noise[1] / 10) == 0 and self.local_step != 0:
                        T_disturb = T_noise[0]*40*(random.random() - 0.5)*2
                        action[i] += T_disturb
                else:
                    if self.local_step % (R_noise[1] / 10) == 0 and self.local_step != 0:
                        R_disturb = R_noise[0]*40*(random.random() - 0.5)*2
                        action[i] += R_disturb
        if showme is False:
            return action
        else:
            return action, T_disturb, R_disturb 
def main(args):
    pass

if __name__ == '__main__':
    args = hyperparameters()
    main(args)