# -*- coding: utf-8 -*-
"""
@author: OK
"""
import gym
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from PIL import Image


class World:
    width = 10000  # [m]
    height = 4000  # [m]
    dt = 0.2  # [sec]
    time = 0  # [sec]
    score = 0
    reward_city = -15
    reward_open = -1
    reward_fire = -1
    reward_intercept = 4
    g = 9.8  # Gravity [m/sec**2]
    fric = 5e-7  # Air friction [Units of Science]
    rocket_prob = 1  # expected rockets per sec


class Turret:
    x = -2000  # [m]
    y = 0  # [m]
    x_hostile = 4800
    y_hostile = 0
    ang_vel = 30  # Turret angular speed [deg/sec]
    ang = 0  # Turret angle [deg]
    v0 = 800  # Initial speed [m/sec]
    prox_radius = 150  # detonation proximity radius [m]
    reload_time = 1.5  # [sec]
    last_shot_time = -3  # [sec]

    def update(self, action_button, game):
        if action_button == 0:
            self.ang = self.ang - self.ang_vel * game.world.dt
            if self.ang < -90: self.ang = -90

        if action_button == 1:
            pass

        if action_button == 2:
            self.ang = self.ang + self.ang_vel * game.world.dt
            if self.ang > 90: self.ang = 90

        if action_button == 3:
            if game.world.time - self.last_shot_time > self.reload_time:
                Interceptor(game)
                self.last_shot_time = game.world.time  # [sec]


class Interceptor:
    def __init__(self, game):
        self.x = game.turret.x
        self.y = game.turret.y
        self.vx = game.turret.v0 * np.sin(np.deg2rad(game.turret.ang))
        self.vy = game.turret.v0 * np.cos(np.deg2rad(game.turret.ang))
        game.world.score = game.world.score + game.world.reward_fire
        game.interceptor_list.append(self)

    def update(self, game):
        self.v_loss = (self.vx ** 2 + self.vy ** 2) * game.world.fric * game.world.dt
        self.vx = self.vx * (1 - self.v_loss)
        self.vy = self.vy * (1 - self.v_loss) - game.world.g * game.world.dt
        self.x = self.x + self.vx * game.world.dt
        self.y = self.y + self.vy * game.world.dt
        if self.y < 0:
            Explosion(self.x, self.y, game)
            game.interceptor_list.remove(self)
        if np.abs(self.x) > game.world.width / 2:
            game.interceptor_list.remove(self)


class Rocket:
    def __init__(self, game):
        self.x = game.turret.x_hostile  # [m]
        self.y = game.turret.y_hostile  # [m]
        self.v0 = 700 + np.random.rand() * 300  # [m/sec]
        self.ang = -88 + np.random.rand() * 68  # [deg]
        self.vx = self.v0 * np.sin(np.deg2rad(self.ang))
        self.vy = self.v0 * np.cos(np.deg2rad(self.ang))
        game.rocket_list.append(self)

    def update(self, game):
        self.v_loss = (self.vx ** 2 + self.vy ** 2) * game.world.fric * game.world.dt
        self.vx = self.vx * (1 - self.v_loss)
        self.vy = self.vy * (1 - self.v_loss) - game.world.g * game.world.dt
        self.x = self.x + self.vx * game.world.dt
        self.y = self.y + self.vy * game.world.dt


class City:
    def __init__(self, x1, x2, width, game):
        self.x = np.random.randint(x1, x2)  # [m]
        self.width = width  # [m]
        game.city_list.append(self)
        self.img = np.zeros((200, 800))
        for b in range(60):
            h = np.random.randint(30, 180)
            w = np.random.randint(30, 80)
            x = np.random.randint(1, 700)
            self.img[0:h, x:x + w] = np.random.rand()
        self.img = np.flipud(self.img)


class Explosion:
    def __init__(self, x, y, game):
        self.x = x
        self.y = y
        self.size = 500
        self.duration = 0.4  # [sec]
        self.verts1 = (np.random.rand(30, 2) - 0.5) * self.size
        self.verts2 = (np.random.rand(20, 2) - 0.5) * self.size / 2
        self.verts1[:, 0] = self.verts1[:, 0] + x
        self.verts1[:, 1] = self.verts1[:, 1] + y
        self.verts2[:, 0] = self.verts2[:, 0] + x
        self.verts2[:, 1] = self.verts2[:, 1] + y
        self.hit_time = game.world.time
        game.explosion_list.append(self)

    def update(self, game):
        if game.world.time - self.hit_time > self.duration:
            game.explosion_list.remove(self)


class InterceptorGAME:

    def __init__(self):
        self.world = World()
        self.rocket_list = []
        self.interceptor_list = []
        self.turret = Turret()
        self.city_list = []
        self.explosion_list = []
        self.should_show = True
        City(-self.world.width * 0.5 + 400, -self.world.width * 0.25 - 400, 800, self)
        City(-self.world.width * 0.25 + 400, -400, 800, self)
        plt.rcParams['axes.facecolor'] = 'black'

    def check_interception(self):
        for intr in self.interceptor_list:
            for r in self.rocket_list:
                if ((r.x - intr.x) ** 2 + (r.y - intr.y) ** 2) ** 0.5 < self.turret.prox_radius:
                    self.rocket_list.remove(r)
                    Explosion(intr.x, intr.y, self)
                    if intr in self.interceptor_list: self.interceptor_list.remove(intr)
                    self.world.score = self.world.score + self.world.reward_intercept

    def check_ground_hit(self):
        for r in self.rocket_list:
            if r.y < 0:
                city_hit = False
                for c in self.city_list:
                    if np.abs(r.x - c.x) < c.width:
                        city_hit = True
                if city_hit == True:
                    self.world.score = self.world.score + self.world.reward_city
                else:
                    self.world.score = self.world.score + self.world.reward_open
                Explosion(r.x, r.y, self)
                self.rocket_list.remove(r)

    def draw(self):
        plt.cla()
        plt.rcParams['axes.facecolor'] = 'black'
        for r in self.rocket_list:
            plt.plot(r.x, r.y, '.y')
        for intr in self.interceptor_list:
            plt.plot(intr.x, intr.y, 'or')
            C1 = plt.Circle((intr.x, intr.y), radius=self.turret.prox_radius, linestyle='--', color='gray', fill=False)
            ax = plt.gca()
            ax.add_artist(C1)
        for c in self.city_list:
            plt.imshow(c.img, extent=[c.x - c.width / 2, c.x + c.width / 2, 0, c.img.shape[0]])
            plt.set_cmap('bone')
        for e in self.explosion_list:
            P1 = plt.Polygon(e.verts1, True, color='yellow')
            P2 = plt.Polygon(e.verts2, True, color='red')
            ax = plt.gca()
            ax.add_artist(P1)
            ax.add_artist(P2)
        plt.plot(self.turret.x, self.turret.y, 'oc', markersize=12)
        plt.plot([self.turret.x, self.turret.x + 100 * np.sin(np.deg2rad(self.turret.ang))],
                 [self.turret.y, self.turret.y + 100 * np.cos(np.deg2rad(self.turret.ang))], 'c', linewidth=3)
        plt.plot(self.turret.x_hostile, self.turret.y_hostile, 'or', markersize=12)
        plt.axes().set_aspect('equal')
        plt.axis([-self.world.width / 2, self.world.width / 2, 0, self.world.height])
        plt.title('Score: ' + str(self.world.score))
        fig = plt.get_current_fig_manager()
        canvas = fig.canvas
        canvas.draw()
        s, (width, height) = canvas.print_to_buffer()
        buf = np.frombuffer(s, np.uint8).reshape((height, width, 4))
        buf.shape = (width, height, 4)
        return buf

    def game_step(self, action_button):
        self.world.time = self.world.time + self.world.dt

        if np.random.rand() < self.world.rocket_prob * self.world.dt:
            Rocket(game=self)

        for r in self.rocket_list:
            r.update(game=self)

        for intr in self.interceptor_list:
            intr.update(game=self)

        for e in self.explosion_list:
            e.update(game=self)

        self.turret.update(action_button, game=self)
        self.check_interception()
        self.check_ground_hit()

        r_locs = np.zeros(shape=(len(self.rocket_list), 2))
        for ind in range(len(self.rocket_list)):
            r_locs[ind, :] = [self.rocket_list[ind].x, self.rocket_list[ind].y]

        i_locs = np.zeros(shape=(len(self.interceptor_list), 2))
        for ind in range(len(self.interceptor_list)):
            i_locs[ind, :] = [self.interceptor_list[ind].x, self.interceptor_list[ind].y]

        c_locs = np.zeros(shape=(len(self.city_list), 2))
        for ind in range(len(self.city_list)):
            c_locs[ind, :] = [self.city_list[ind].x, self.city_list[ind].width]

        return r_locs, i_locs, c_locs, self.turret.ang, self.world.score


class InterceptorV2Env(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(InterceptorV2Env, self).__init__()
        print("Init()")
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = gym.spaces.Discrete(4)
        # Example for using image as input:
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(640,480, 4))
        self.episode_length = 1000
        self.current_step = 0
        self.interceptor = InterceptorGAME()



    def step(self, action):
        r_locs, i_locs, c_locs, ang, score = self.interceptor.game_step(action_button=action)
        self.current_step += 1
        done = self.current_step >= self.episode_length

        bounded_cloc_1 = np.append([loc[1] for loc in c_locs], np.zeros(80 - len(c_locs)))
        bounded_cloc_0 = np.append([loc[0] for loc in c_locs], np.zeros(80 - len(c_locs)))
        if len(r_locs) > 0:
            bounded_rloc_0 = np.append([loc[0] for loc in r_locs], np.zeros(80 - len(r_locs)))
            bounded_rloc_1 = np.append([loc[1] for loc in r_locs], np.zeros(80 - len(r_locs)))
        else:
            bounded_rloc_0  = np.zeros(80)
            bounded_rloc_1 = np.zeros(80)

        if len(i_locs) > 0:
            bounded_iloc_0 = np.append([loc[0] for loc in i_locs], np.zeros(80 - len(i_locs)))
            bounded_iloc_1 = np.append([loc[1] for loc in i_locs], np.zeros(80 - len(i_locs)))
        else:
            bounded_iloc_0 = np.zeros(80)
            bounded_iloc_1 = np.zeros(80)
        bounded_angle = np.append(np.array(ang), np.zeros(79))
        obs = np.hstack((bounded_rloc_0, bounded_rloc_1, bounded_iloc_0, bounded_iloc_1, bounded_cloc_0, bounded_cloc_1, bounded_angle)).reshape((560, 1))




        return obs, score, done, {}


    def reset(self):
        self.current_step = 0
        self.interceptor = InterceptorGAME()
        obs, score, done, infos = self.step(1)
        return obs

    def render(self, mode='human', close=False):
        self.interceptor.draw()