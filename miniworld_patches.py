from miniworld.miniworld import MiniWorldEnv
from miniworld.entity import Box
from miniworld.params import DEFAULT_PARAMS
import numpy as np
import random
from gymnasium import spaces, utils

# Modified map class (because your map class was adjusted)
from skimage.draw import line_aa

class Map:
    def __init__(self, obs_height, obs_width, fluff, reward_mul=1, decay=1.0):
        self.fluff = fluff
        self.obs_height = obs_height
        self.obs_width = obs_width
        self.decay = decay
        self.reward_mul = reward_mul
        self.map = np.zeros([self.obs_height, self.obs_width])

    def __call__(self):
        return self.map

    def update(self, pos, scale=27, offset=1):
        self.map = self.map * self.decay
        count_pre = np.sum(self.map)
        y, _, x = np.round((pos / scale) * self.obs_height).astype(int)
        y += offset
        x += offset
        self.last_x = x
        self.last_y = y
        self.map[x, y] = 1.0
        if self.fluff:
            self.map[x-self.fluff:x+self.fluff, y-self.fluff:y+self.fluff] = 1.0
        count_post = np.sum(self.map)
        count_delta = count_post - count_pre
        if self.fluff > 1:
            count = count_delta / (self.reward_mul * self.fluff ** 2)
        else:
            count = count_delta
        return count

    def show_all(self, dir, line_len=12):
        self.curr_pos = np.zeros([self.obs_height, self.obs_width])
        self.curr_dir = np.zeros([self.obs_height, self.obs_width])
        self.curr_pos[self.last_x-1:self.last_x+1, self.last_y-1:self.last_y+1] = 1.0
        self.dir_y, _, self.dir_x = np.round(dir * line_len).astype(int)
        rr, cc, val = line_aa(self.last_x, self.last_y, self.last_x + self.dir_x, self.last_y + self.dir_y)
        upper_lim = self.obs_height
        lower_lim = -1
        mask = (lower_lim < rr) & (rr < upper_lim) & (lower_lim < cc) & (cc < upper_lim)
        rr, cc, val = rr[mask], cc[mask], val[mask]
        self.curr_dir[rr, cc] = val
        all_ims = np.stack([self.map, self.curr_pos, self.curr_dir], axis=-1) * 255
        return all_ims.astype(np.uint8)


class MiniWorldEnv_c(MiniWorldEnv):
    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 30,
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        max_episode_steps=10000,
        obs_width=64,
        obs_height=64,
        obs_channels=3,
        window_width=800,
        window_height=600,
        include_maps=True,
        coverage_thresh=0.7,
        patience=1000,
        cam_view=1,
        bw=False,
        decay_param=1.0,
        fluff=2,
        porosity=0.0,
        params=DEFAULT_PARAMS,
        domain_rand=False,
        render_mode=None,
        view="agent",
    ):
        super().__init__(
            max_episode_steps=max_episode_steps,
            obs_width=obs_width,
            obs_height=obs_height,
            obs_channels=obs_channels,
            window_width=window_width,
            window_height=window_height,
            include_maps=include_maps,
            params=params,
            domain_rand=domain_rand,
            render_mode=render_mode,
            view=view,
        )

        self.coverage_thresh = coverage_thresh
        self.patience = patience
        self.cam_view = cam_view
        self.bw = bw
        self.decay_param = decay_param
        self.fluff = fluff
        self.porosity = porosity

        # Override history map
        self.histoire = Map(obs_height=self.obs_height, obs_width=self.obs_width, fluff=self.fluff, decay=self.decay_param)

    def move_agent(self, fwd_dist):
        fwd_dist *= 0.5
        next_pos = self.agent.pos + self.agent.dir_vec * fwd_dist
        if np.any(next_pos[[0, 2]] > 25.2) or np.any(next_pos[[0, 2]] < 0.6):
            return False
        if self.intersect(self.agent, next_pos, self.agent.radius):
            return False
        self.agent.pos = next_pos
        return True

    def strafe_agent(self, fwd_dist):
        fwd_dist *= 0.2
        right_theta = np.arcsin(self.agent.dir_vec[0]) + np.pi / 2
        new_vec = np.array([np.sin(right_theta), 0, -np.cos(right_theta)])
        next_pos = self.agent.pos + new_vec * fwd_dist
        if np.any(next_pos[[0, 2]] > 25.2) or np.any(next_pos[[0, 2]] < 0.6):
            return False
        if self.intersect(self.agent, next_pos, self.agent.radius):
            return False
        self.agent.pos = next_pos
        return True

    def turn_agent(self, turn_angle):
        self.agent.dir += turn_angle * 0.25
        return True

    def step(self, action):
        self.step_count += 1
        self.move_agent(action[0])
        self.strafe_agent(action[1])
        self.turn_agent(action[2])

        obs = self.render_obs()

        expl_reward = 10 * self.histoire.update(self.agent.pos)

        if self.include_maps:
            all_maps = self.histoire.show_all(self.agent.dir_vec)
            if self.bw:
                obs = np.mean(obs, axis=-1, keepdims=True)
                all_maps = np.mean(all_maps, axis=-1, keepdims=True)
            obs = np.concatenate([obs, all_maps], axis=-2)

        reward = -10 + expl_reward
        done = self.step_count >= self.max_episode_steps

        return obs, reward, done, False, {}


class MazeCA(MiniWorldEnv_c, utils.EzPickle):
    def __init__(self, num_rows=8, num_cols=8, room_size=3, max_episode_steps=None, **kwargs):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.room_size = room_size
        self.gap_size = 0.25

        super().__init__(
            max_episode_steps=max_episode_steps or num_rows * num_cols * 24,
            **kwargs,
        )
        utils.EzPickle.__init__(
            self,
            num_rows=num_rows,
            num_cols=num_cols,
            room_size=room_size,
            max_episode_steps=max_episode_steps,
            **kwargs,
        )

        self.action_space = spaces.Box(-1, +1, (3,), dtype=float)

    def _gen_world(self):
        rows = []
        for j in range(self.num_rows):
            row = []
            for i in range(self.num_cols):
                min_x = i * (self.room_size + self.gap_size)
                max_x = min_x + self.room_size
                min_z = j * (self.room_size + self.gap_size)
                max_z = min_z + self.room_size
                room = self.add_rect_room(
                    min_x=min_x,
                    max_x=max_x,
                    min_z=min_z,
                    max_z=max_z,
                    wall_tex="brick_wall",
                )
                row.append(room)
            rows.append(row)

        visited = set()

        def visit(i, j):
            room = rows[j][i]
            visited.add(room)
            neighbors = [(0,1), (0,-1), (-1,0), (1,0)]
            random.shuffle(neighbors)
            for dj, di in neighbors:
                ni, nj = i + di, j + dj
                if 0 <= ni < self.num_cols and 0 <= nj < self.num_rows:
                    neighbor = rows[nj][ni]
                    if neighbor not in visited:
                        if di == 0:
                            self.connect_rooms(room, neighbor, min_x=room.min_x, max_x=room.max_x)
                        else:
                            self.connect_rooms(room, neighbor, min_z=room.min_z, max_z=room.max_z)
                        visit(ni, nj)

        visit(0, 0)

        self.boxes = [
            self.place_entity(Box(color="red")),
            self.place_entity(Box(color="green")),
            self.place_entity(Box(color="blue")),
        ]

        self.place_agent()
