import math
from ctypes import POINTER
from enum import IntEnum
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from scipy.ndimage import binary_dilation
import pyglet
from gymnasium import spaces
from gymnasium.core import ObsType
from pyglet.gl import (
    GL_AMBIENT,
    GL_AMBIENT_AND_DIFFUSE,
    GL_ANY_SAMPLES_PASSED,
    GL_COLOR_BUFFER_BIT,
    GL_COLOR_MATERIAL,
    GL_COMPILE,
    GL_CULL_FACE,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_DIFFUSE,
    GL_FRAMEBUFFER,
    GL_FRONT_AND_BACK,
    GL_LIGHT0,
    GL_LIGHTING,
    GL_MODELVIEW,
    GL_POLYGON,
    GL_POSITION,
    GL_PROJECTION,
    GL_QUADS,
    GL_QUERY_RESULT,
    GL_SMOOTH,
    GL_TEXTURE_2D,
    GLfloat,
    GLubyte,
    GLuint,
    glBegin,
    glBeginQuery,
    glBindFramebuffer,
    glCallList,
    glClear,
    glClearColor,
    glClearDepth,
    glColor3f,
    glColorMaterial,
    glDeleteLists,
    glDeleteQueries,
    glDisable,
    glEnable,
    glEnd,
    glEndList,
    glEndQuery,
    glFlush,
    glGenQueries,
    glGetQueryObjectuiv,
    glLightfv,
    glLoadIdentity,
    glLoadMatrixf,
    glMatrixMode,
    glNewList,
    glNormal3f,
    glOrtho,
    glShadeModel,
    glTexCoord2f,
    gluLookAt,
    gluPerspective,
    glVertex3f,
)

from miniworld.entity import Agent, Entity
from miniworld.math import Y_VEC, intersect_circle_segs
from miniworld.opengl import FrameBuffer, Texture, drawBox
from miniworld.params import DEFAULT_PARAMS

KERNEL = np.ones([3,3])

# Default wall height for room
DEFAULT_WALL_HEIGHT = 2.74

# Texture size/density in texels/meter
TEX_DENSITY = 512

from skimage.draw import line_aa
class map:
    def __init__(self, obs_height, obs_width, fluff, reward_mul=1, decay = 1.0):
        self.fluff = fluff
        self.obs_height = obs_height
        self.obs_width = obs_width
        self.decay = decay
        self.reward_mul = reward_mul
        self.map = np.zeros([self.obs_height, self.obs_width])
    def __call__(self):
        return self.map
    def update(self, pos, scale = 27, offset = 1):
        self.map = self.map * self.decay
        count_pre = np.sum(self.map)
        y,_,x = np.round((pos / scale) * self.obs_height).astype(int)
        y = y + offset
        x = x + offset
        self.last_x = x
        self.last_y = y
        self.map[x, y] = 1.0
        if self.fluff:
            self.map[x-self.fluff:x+self.fluff, y-self.fluff:y+self.fluff] = 1.0
        count_post = np.sum(self.map)
        count_delta = count_post -  count_pre
        if self.fluff > 1:
            count = count_delta / (self.reward_mul * self.fluff ** 2)
        else:
            count = count_delta
        return count
    def show_map(self):
        return self.map
    def show_all(self, dir, line_len = 12):
        self.curr_pos = np.zeros([self.obs_height,self.obs_width])
        self.curr_dir = np.zeros([self.obs_height,self.obs_width])
        self.curr_pos[self.last_x-1:self.last_x+1, self.last_y-1:self.last_y+1] = 1.0
        self.dir_y, _, self.dir_x = np.round(dir * line_len).astype(int)
        rr, cc, val = line_aa(self.last_x, self.last_y, self.last_x + self.dir_x, self.last_y + self.dir_y)
        upper_lim = self.obs_height
        lower_lim = -1
        # Combine all filtering conditions into a single step
        mask = (lower_lim < rr) & (rr < upper_lim) & (lower_lim < cc) & (cc < upper_lim)
        # Apply the mask to filter rr, cc, and val
        rr, cc, val = rr[mask], cc[mask], val[mask]
        self.curr_dir[rr, cc] = val
        all_ims = np.stack([self.map, self.curr_pos, self.curr_dir], axis=-1) * 255
        return all_ims.astype(np.uint8)
    def coverage(self):
        return np.mean(self.map)


def gen_texcs_wall(tex, min_x, min_y, width, height):
    """
    Generate texture coordinates for a wall quad
    """

    xc = TEX_DENSITY / tex.width
    yc = TEX_DENSITY / tex.height

    min_u = min_x * xc
    max_u = (min_x + width) * xc
    min_v = min_y * yc
    max_v = (min_y + height) * yc

    return np.array(
        [
            [min_u, min_v],
            [min_u, max_v],
            [max_u, max_v],
            [max_u, min_v],
        ],
        dtype=np.float32,
    )


def gen_texcs_floor(tex, poss):
    """
    Generate texture coordinates for the floor or ceiling
    This is done by mapping x,z positions directly to texture
    coordinates
    """

    texc_mul = np.array(
        [TEX_DENSITY / tex.width, TEX_DENSITY / tex.height], dtype=float
    )

    coords = np.stack([poss[:, 0], poss[:, 2]], axis=1) * texc_mul

    return coords


class Room:
    """
    Represent an individual room and its contents
    """

    def __init__(
        self,
        outline,
        wall_height=DEFAULT_WALL_HEIGHT,
        floor_tex="floor_tiles_bw",
        wall_tex="concrete",
        ceil_tex="concrete_tiles",
        no_ceiling=False,
    ):
        # The outlien should have shape Nx2
        assert len(outline.shape) == 2
        assert outline.shape[1] == 2
        assert outline.shape[0] >= 3

        # Add a Y coordinate to the outline points
        outline = np.insert(outline, 1, 0, axis=1)

        # Number of outline vertices / walls
        self.num_walls = outline.shape[0]

        # List of 2D points forming the outline of the room
        # Shape is Nx3
        self.outline = outline

        # Compute the min and max x, z extents
        self.min_x = self.outline[:, 0].min()
        self.max_x = self.outline[:, 0].max()
        self.min_z = self.outline[:, 2].min()
        self.max_z = self.outline[:, 2].max()

        # Compute midpoint coordinates
        self.mid_x = (self.max_x + self.min_x) / 2
        self.mid_z = (self.max_z + self.min_z) / 2

        # Compute approximate surface area
        self.area = (self.max_x - self.min_x) * (self.max_z - self.min_z)

        # Compute room edge directions and normals
        # Compute edge vectors (p1 - p0)
        # For the first point, p0 is the last
        # For the last point, p0 is p_n-1
        next_pts = np.concatenate(
            [self.outline[1:], np.expand_dims(self.outline[0], axis=0)], axis=0
        )
        self.edge_dirs = next_pts - self.outline
        self.edge_dirs = (self.edge_dirs.T / np.linalg.norm(self.edge_dirs, axis=1)).T
        self.edge_norms = -np.cross(self.edge_dirs, Y_VEC)
        self.edge_norms = (
            self.edge_norms.T / np.linalg.norm(self.edge_norms, axis=1)
        ).T

        # Height of the room walls
        self.wall_height = wall_height

        # No ceiling flag
        self.no_ceiling = no_ceiling

        # Texture names
        self.wall_tex_name = wall_tex
        self.floor_tex_name = floor_tex
        self.ceil_tex_name = ceil_tex

        # Lists of portals, indexed by wall/edge index
        self.portals = [[] for i in range(self.num_walls)]

        # List of neighbor rooms
        # Same length as list of portals
        self.neighbors = []

    def add_portal(
        self,
        edge,
        start_pos=None,
        end_pos=None,
        min_x=None,
        max_x=None,
        min_z=None,
        max_z=None,
        min_y=0,
        max_y=None,
    ):
        """
        Create a new portal/opening in a wall of this room
        """

        if max_y is None:
            max_y = self.wall_height

        assert edge < self.num_walls, "Edge index out of range"
        assert max_y > min_y, "Invalid vertical extents"

        # Get the edge points, compute the direction vector
        e_p0 = self.outline[edge]
        e_p1 = self.outline[(edge + 1) % self.num_walls]
        e_len = np.linalg.norm(e_p1 - e_p0)
        e_dir = (e_p1 - e_p0) / e_len
        x0, _, z0 = e_p0
        x1, _, z1 = e_p1
        dx, _, dz = e_dir

        # If the portal extents are specified by x coordinates
        if min_x is not None:
            assert min_z is None and max_z is None, "Specify either x or z coordinates for the portal, not both"
            assert start_pos is None and end_pos is None, "Specify either x coordinates or start/end positions, not both"
            assert x0 != x1, "Edge must have a non-zero x dimension"

            m0 = (min_x - x0) / dx
            m1 = (max_x - x0) / dx

            if m1 < m0:
                m0, m1 = m1, m0

            start_pos, end_pos = m0, m1

        # If the portal extents are specified by z coordinates
        elif min_z is not None:
            assert min_x is None and max_x is None, "Specify either x or z coordinates for the portal, not both"
            assert start_pos is None and end_pos is None, "Specify either z coordinates or start/end positions, not both"
            assert z0 != z1, "Edge must have a non-zero z dimension"

            m0 = (min_z - z0) / dz
            m1 = (max_z - z0) / dz

            if m1 < m0:
                m0, m1 = m1, m0

            start_pos, end_pos = m0, m1

        else:
            assert min_x is None and max_x is None, "Specify x coordinates for the portal"
            assert min_z is None and max_z is None, "Specify z coordinates for the portal"

        assert end_pos > start_pos, "Invalid portal dimensions"
        assert start_pos >= 0, "Portal starts outside of wall extents"
        assert end_pos <= e_len, "Portal ends outside of wall extents"

        def portals_overlap(existing_portals, new_start, new_end, new_min_y, new_max_y):
            """
            Check if a new portal overlaps with any existing portals on the same wall.
            """
            for portal in existing_portals:
                # Check vertical overlap
                if not (new_max_y <= portal["min_y"] or new_min_y >= portal["max_y"]):
                    # Check horizontal overlap
                    if new_end > portal["start_pos"] and new_start < portal["end_pos"]:
                        return True
            return False

        # Check for overlaps with existing portals
        if portals_overlap(self.portals[edge], start_pos, end_pos, min_y, max_y):
            # print(f"Portal at edge {edge} overlaps with an existing portal and will not be added.")
            return start_pos, end_pos

        self.portals[edge].append(
            {"start_pos": start_pos, "end_pos": end_pos, "min_y": min_y, "max_y": max_y}
        )

        # Sort the portals by start position
        self.portals[edge].sort(key=lambda e: e["start_pos"])

        return start_pos, end_pos

    def point_inside(self, p):
        """
        Test if a point is inside the room
        """

        # Vector from edge start to test point
        ap = p - self.outline

        # Compute the dot products of normals to AP vectors
        dotNAP = np.sum(self.edge_norms * ap, axis=1)

        # The point is inside if all the dot products are greater than zero
        return np.all(np.greater(dotNAP, 0))

    def _gen_static_data(self, params, rng):
        """
        Generate polygons and static data for this room
        Needed for rendering and collision detection
        Note: the wall polygons are quads, but the floor and
              ceiling can be arbitrary n-gons
        """

        # Load the textures and do texture randomization
        self.wall_tex = Texture.get(self.wall_tex_name, rng)
        self.floor_tex = Texture.get(self.floor_tex_name, rng)
        self.ceil_tex = Texture.get(self.ceil_tex_name, rng)

        # Generate the floor vertices
        self.floor_verts = self.outline
        self.floor_texcs = gen_texcs_floor(self.floor_tex, self.floor_verts)

        # Generate the ceiling vertices
        # Flip the ceiling vertex order because of backface culling
        self.ceil_verts = np.flip(self.outline, axis=0) + self.wall_height * Y_VEC
        self.ceil_texcs = gen_texcs_floor(self.ceil_tex, self.ceil_verts)

        self.wall_verts = []
        self.wall_norms = []
        self.wall_texcs = []
        self.wall_segs = []

        def gen_seg_poly(edge_p0, side_vec, seg_start, seg_end, min_y, max_y):
            if seg_end == seg_start:
                return

            if min_y == max_y:
                return

            s_p0 = edge_p0 + seg_start * side_vec
            s_p1 = edge_p0 + seg_end * side_vec

            # If this polygon starts at ground level, add a collidable segment
            if min_y == 0:
                self.wall_segs.append(np.array([s_p1, s_p0]))

            # Generate the vertices
            # Vertices are listed in counter-clockwise order
            self.wall_verts.append(s_p0 + min_y * Y_VEC)
            self.wall_verts.append(s_p0 + max_y * Y_VEC)
            self.wall_verts.append(s_p1 + max_y * Y_VEC)
            self.wall_verts.append(s_p1 + min_y * Y_VEC)

            # Compute the normal for the polygon
            normal = np.cross(s_p1 - s_p0, Y_VEC)
            normal = -normal / np.linalg.norm(normal)
            for i in range(4):
                self.wall_norms.append(normal)

            # Generate the texture coordinates
            texcs = gen_texcs_wall(
                self.wall_tex, seg_start, min_y, seg_end - seg_start, max_y - min_y
            )
            self.wall_texcs.append(texcs)

        # For each wall
        for wall_idx in range(self.num_walls):
            edge_p0 = self.outline[wall_idx, :]
            edge_p1 = self.outline[(wall_idx + 1) % self.num_walls, :]
            wall_width = np.linalg.norm(edge_p1 - edge_p0)
            side_vec = (edge_p1 - edge_p0) / wall_width

            if len(self.portals[wall_idx]) > 0:
                seg_end = self.portals[wall_idx][0]["start_pos"]
            else:
                seg_end = wall_width

            # Generate the first polygon (going up to the first portal)
            gen_seg_poly(edge_p0, side_vec, 0, seg_end, 0, self.wall_height)

            # For each portal in this wall
            for portal_idx, portal in enumerate(self.portals[wall_idx]):
                portal = self.portals[wall_idx][portal_idx]
                start_pos = portal["start_pos"]
                end_pos = portal["end_pos"]
                min_y = portal["min_y"]
                max_y = portal["max_y"]

                # Generate the bottom polygon
                gen_seg_poly(edge_p0, side_vec, start_pos, end_pos, 0, min_y)

                # Generate the top polygon
                gen_seg_poly(
                    edge_p0, side_vec, start_pos, end_pos, max_y, self.wall_height
                )

                if portal_idx < len(self.portals[wall_idx]) - 1:
                    next_portal = self.portals[wall_idx][portal_idx + 1]
                    next_portal_start = next_portal["start_pos"]
                else:
                    next_portal_start = wall_width

                # Generate the polygon going up to the next portal
                gen_seg_poly(
                    edge_p0, side_vec, end_pos, next_portal_start, 0, self.wall_height
                )

        self.wall_verts = np.array(self.wall_verts)
        self.wall_norms = np.array(self.wall_norms)

        if len(self.wall_segs) > 0:
            self.wall_segs = np.array(self.wall_segs)
        else:
            self.wall_segs = np.array([]).reshape(0, 2, 3)

        if len(self.wall_texcs) > 0:
            self.wall_texcs = np.concatenate(self.wall_texcs)
        else:
            self.wall_texcs = np.array([]).reshape(0, 2)

    def _render(self):
        """
        Render the static elements of the room
        """

        glColor3f(1, 1, 1)

        # Draw the floor
        self.floor_tex.bind()
        glBegin(GL_POLYGON)
        glNormal3f(0, 1, 0)
        for i in range(self.floor_verts.shape[0]):
            glTexCoord2f(*self.floor_texcs[i, :])
            glVertex3f(*self.floor_verts[i, :])
        glEnd()

        # Draw the ceiling
        if not self.no_ceiling:
            self.ceil_tex.bind()
            glBegin(GL_POLYGON)
            glNormal3f(0, -1, 0)
            for i in range(self.ceil_verts.shape[0]):
                glTexCoord2f(*self.ceil_texcs[i, :])
                glVertex3f(*self.ceil_verts[i, :])
            glEnd()

        # Draw the walls
        self.wall_tex.bind()
        glBegin(GL_QUADS)
        for i in range(self.wall_verts.shape[0]):
            glNormal3f(*self.wall_norms[i, :])
            glTexCoord2f(*self.wall_texcs[i, :])
            glVertex3f(*self.wall_verts[i, :])
        glEnd()


class MiniWorldEnv_c(gym.Env):
    """
    Base class for MiniWorld environments. Implements the procedural
    world generation and simulation logic.
    """

    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 30,
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        max_episode_steps: int = 10000,
        obs_width: int = 64,
        obs_height: int = 64,
        obs_channels: int = 3,
        window_width: int = 800,
        window_height: int = 600,
        include_maps: bool = True,
        coverage_thresh: bool = 0.7,
        patience: int = 1000,
        cam_view:int = 1,
        bw:bool = False,
        decay_param: float = 1.0,#0.999999,
        fluff: int = 2,
        porosity: float = 0.0,
        params=DEFAULT_PARAMS,
        domain_rand: bool = False,
        render_mode: Optional[str] = None,
        view: str = "agent",
    ):
        
        self.porosity = porosity

        # Actions are continous values between +1 and -1
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(3,), dtype=float
            )

        self.obs_height = obs_height
        self.obs_width = obs_width

        # Observations are RGB images with pixels in [0, 255]
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.obs_height, self.obs_width*2, obs_channels), dtype=np.uint8
        )

        self.reward_range = (-math.inf, math.inf)

        # Maximum number of steps per episode
        self.max_episode_steps = max_episode_steps
        self.max_patience = patience

        self.cam_view = cam_view

        # Whether to include maps
        self.include_maps = include_maps

        self.coverage_thresh = coverage_thresh
        self.decay_param = decay_param
        self.fluff = fluff

        # Simulation parameters, used for domain randomization
        self.params = params

        self.bw = bw

        # Domain randomization enable/disable flag
        self.domain_rand = domain_rand

        # Window for displaying the environment to humans
        self.window = None

        # Invisible window to render into (shadow OpenGL context)
        self.shadow_window = pyglet.window.Window(width=1, height=1, visible=False)

        # Enable depth testing and backface culling
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)

        # Frame buffer used to render observations
        self.obs_fb = FrameBuffer(self.obs_width, self.obs_height, 8)

        # Frame buffer used for human visualization
        self.vis_fb = FrameBuffer(window_width, window_height, 16)

        # Set rendering mode
        self.render_mode = render_mode

        # Set view type
        assert view in ["agent", "top"]
        self.view = view

        # Compute the observation display size
        self.obs_disp_width = 256
        self.obs_disp_height = self.obs_height * (self.obs_disp_width / self.obs_width)

        # For displaying text
        self.text_label = pyglet.text.Label(
            font_name="Arial",
            font_size=14,
            multiline=True,
            width=400,
            x=window_width + 5,
            y=window_height - (self.obs_disp_height + 19),
        )

        self.histoire = map(obs_height=self.obs_height, obs_width=self.obs_width, fluff = self.fluff, decay = self.decay_param)

        # Initialize the state
        self.reset()

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[ObsType, dict]:
        """
        Reset the simulation at the start of a new episode
        This also randomizes many environment parameters (domain randomization)
        """
        super().reset(seed=seed)

        # Step count since episode start
        self.step_count = 0
        self.patience_count = 0
        self.IC_reward = 0

        # Create the agent
        self.agent = Agent()

        # List of entities contained
        self.entities = []

        # List of rooms in the world
        self.rooms = []

        self.histoire = map(obs_height=self.obs_height, obs_width=self.obs_width, fluff = self.fluff, decay = self.decay_param)

        # Wall segments for collision detection
        # Shape is (N, 2, 3)
        self.wall_segs = []
        # Generate the world
        self._gen_world()

        # Check if domain randomization is enabled or not
        rand = self.np_random if self.domain_rand else None

        # Randomize elements of the world (domain randomization)
        self.params.sample_many(
            rand, self, ["sky_color", "light_pos", "light_color", "light_ambient"]
        )

        # Get the max forward step distance
        self.max_forward_step = self.params.get_max("forward_step")

        # Randomize parameters of the entities
        for ent in self.entities:
            ent.randomize(self.params, rand)

        # Compute the min and max x, z extents of the whole floorplan
        self.min_x = min(r.min_x for r in self.rooms)
        self.max_x = max(r.max_x for r in self.rooms)
        self.min_z = min(r.min_z for r in self.rooms)
        self.max_z = max(r.max_z for r in self.rooms)

        # Generate static data
        if len(self.wall_segs) == 0:
            self._gen_static_data()

        # Pre-compile static parts of the environment into a display list
        self._render_static()

        # Generate the first camera image
        if self.cam_view == 1:
            obs = self.render_obs()
        elif self.cam_view == 4:
            obs = self.render_obs_4()

        if self.include_maps:
            _ = self.histoire.update(self.agent.pos) 
            all_maps = self.histoire.show_all(self.agent.dir_vec)
            # own_map = self.render_top_view()
            # obs = np.concatenate([obs, all_maps, own_map], axis=-1)
            if self.bw:
                obs = np.mean(obs, axis=-1, keepdims=True)
                all_maps = np.mean(all_maps, axis=-1, keepdims=True)
            #axis -1 for older than dreamer models - NOTE
            obs = np.concatenate([obs, all_maps], axis=-2)

        # Return first observation
        return obs, {}

    def _get_carry_pos(self, agent_pos, ent):
        """
        Compute the position at which to place an object being carried
        """

        dist = self.agent.radius + ent.radius + self.max_forward_step
        pos = agent_pos + self.agent.dir_vec * 1.05 * dist

        # Adjust the Y-position so the object is visible while being carried
        y_pos = max(self.agent.cam_height - ent.height - 0.3, 0)
        pos = pos + Y_VEC * y_pos

        return pos

    def move_agent(self, fwd_dist):
        """
        Move the agent forward
        """

        outt = 1

        if fwd_dist < 0.5:
            outt = 2
            if fwd_dist < 0:
                fwd_dist *= 0.1

        fwd_dist = fwd_dist * 0.5

        next_pos = (
            self.agent.pos
            + self.agent.dir_vec * fwd_dist
        )

        if np.any(next_pos[[0,2]] > 25.2) or np.any(next_pos[[0,2]] < 0.6):
            return False

        if self.intersect(self.agent, next_pos, self.agent.radius):
            return False

        self.agent.pos = next_pos

        return outt
    
    def strafe_agent(self, fwd_dist):
        """
        Move the agent forward
        """
        fwd_dist = fwd_dist * 0.2
        right_theta = math.asin(self.agent.dir_vec[0]) + 1.5708
        new_vec = np.array([math.sin(right_theta), 0, -math.cos(right_theta)])

        next_pos = (
            self.agent.pos
            + (new_vec) * fwd_dist
        )

        if np.any(next_pos[[0,2]] > 25.2) or np.any(next_pos[[0,2]] < 0.6):
            return False

        if self.intersect(self.agent, next_pos, self.agent.radius):
            return False

        self.agent.pos = next_pos

        return True

    def turn_agent(self, turn_angle):
        """
        Turn the agent left or right
        """

        self.agent.dir += turn_angle * 0.25 #0.2 #changed on transformer

        return True

    def step(self, action):
        """
        Perform one action and update the simulation

        Multiple rewards here - the lazy reward punishes turning as it returns a 0 reward -0.1 / 0
            - sticky reward gives the agent a reward for repeating the previous action 0.5 / 0
            - IC reward rewards getting a box, increases by 50 each time, resets to 0 upon new episode
              (Sum(+50))/0
            - progress reward rewards moving 9/-4
            - expl reward rewards exploring
        """

        self.step_count += 1
        IC_reward = 0

        mov_stat = self.move_agent(action[0])
        self.strafe_agent(action[1])
        self.turn_agent(action[2]) #turn agent used to be before move, changed on transformer

        # #penalise turning because it annoys me
        # mov_stat -= 0.5 * np.abs(action[2])

        # Generate the current camera image
        if self.cam_view == 1:
            obs = self.render_obs()
        elif self.cam_view == 4:
            obs = self.render_obs_4()

        expl_reward = 10 * self.histoire.update(self.agent.pos)
        # if self.histoire.decay < 1 and reward < 0.0005:
        #     reward = -0.5
        # elif self.histoire.decay == 1 and reward == 0:
        #     reward = -0.5

        if self.include_maps:
            all_maps = self.histoire.show_all(self.agent.dir_vec)
            # own_map = self.render_top_view()
            if self.bw:
                obs = np.mean(obs, axis=-1, keepdims=True)
                all_maps = np.mean(all_maps, axis=-1, keepdims=True)
            #axis -1 for older than dreamer models - NOTE
            obs = np.concatenate([obs, all_maps], axis=-2)

        termination = False
        truncation = False

        to_remove = []
        color = np.zeros(3)
        for i, this_box in enumerate(self.boxes):
            if self.near(this_box):
                IC_reward += 50
                self.entities.remove(this_box)
                to_remove.append(i)
                self.patience_count = 0
            else:
                if this_box.color == 'red':
                    color[0] = self.nearby(this_box)
                elif this_box.color == 'green':
                    color[1] = self.nearby(this_box)
                elif this_box.color == 'blue':
                    color[2] = self.nearby(this_box)

        nearby_reward = sum(color / 76.5)

        # obs[28:36, 28:30] = color    # Left edge (two pixels thick)
        # obs[28:36, 34:36] = color    # Right edge (two pixels thick)
        obs[27:29, 28:36] = color    # Top edge (two pixels thick)
        obs[35:37, 28:36] = color    # Bottom edge (two pixels thick)

        for i in to_remove[::-1]:
            self.boxes.pop(i)

        if len(self.boxes) == 0:
            IC_reward += 150
            termination = True

        # if mov_stat == 1:
        #     progress_reward = 5 #used to be 0.5, changed at free alpha sac
        # elif mov_stat == False:
        #     # termination = True
        #     progress_reward = -5 #used to be -0.2, changed at free alpha sac
        # else:
        #     progress_reward = 0

        # If the maximum time step count is reached
        if self.step_count >= 4096: #self.max_episode_steps:
            truncation = True

        # reward = progress_reward + expl_reward + IC_reward + nearby_reward
        reward = -10 + expl_reward + IC_reward + nearby_reward
        # reward = -10 + IC_reward + nearby_reward
        
        return obs, reward, termination, truncation, {}


    def add_rect_room(self, min_x, max_x, min_z, max_z, **kwargs):
        """
        Create a rectangular room
        """

        # 2D outline coordinates of the room,
        # listed in counter-clockwise order when viewed from the top
        outline = np.array(
            [
                # East wall
                [max_x, max_z],
                # North wall
                [max_x, min_z],
                # West wall
                [min_x, min_z],
                # South wall
                [min_x, max_z],
            ]
        )

        return self.add_room(outline=outline, **kwargs)

    def add_room(self, **kwargs):
        """
        Create a new room
        """

        assert (
            len(self.wall_segs) == 0
        ), "cannot add rooms after static data is generated"

        room = Room(**kwargs)
        self.rooms.append(room)

        return room

    def connect_rooms(
        self, room_a, room_b, min_x=None, max_x=None, min_z=None, max_z=None, max_y=None
    ):
        """
        Connect two rooms along facing edges
        """

        def find_facing_edges():
            for idx_a in range(room_a.num_walls):
                norm_a = room_a.edge_norms[idx_a]

                for idx_b in range(room_b.num_walls):
                    norm_b = room_b.edge_norms[idx_b]

                    # Reject edges that are not facing each other
                    if np.dot(norm_a, norm_b) > -0.9:
                        continue

                    dir = room_b.outline[idx_b] - room_a.outline[idx_a]

                    # Reject edges that are not touching
                    if np.dot(norm_a, dir) > 0.05:
                        continue

                    return idx_a, idx_b

            return None, None

        idx_a, idx_b = find_facing_edges()
        assert idx_a is not None, "matching edges not found in connect_rooms"

        start_a, end_a = room_a.add_portal(
            edge=idx_a, min_x=min_x, max_x=max_x, min_z=min_z, max_z=max_z, max_y=max_y
        )

        start_b, end_b = room_b.add_portal(
            edge=idx_b, min_x=min_x, max_x=max_x, min_z=min_z, max_z=max_z, max_y=max_y
        )

        a = room_a.outline[idx_a] + room_a.edge_dirs[idx_a] * start_a
        b = room_a.outline[idx_a] + room_a.edge_dirs[idx_a] * end_a
        c = room_b.outline[idx_b] + room_b.edge_dirs[idx_b] * start_b
        d = room_b.outline[idx_b] + room_b.edge_dirs[idx_b] * end_b

        # If the portals are directly connected, stop
        if np.linalg.norm(a - d) < 0.001:
            return

        len_a = np.linalg.norm(b - a)
        len_b = np.linalg.norm(d - c)

        # Room outline points must be specified in counter-clockwise order
        outline = np.stack([c, b, a, d])
        outline = np.stack([outline[:, 0], outline[:, 2]], axis=1)

        max_y = max_y if max_y is not None else room_a.wall_height

        room = Room(
            outline,
            wall_height=max_y,
            wall_tex=room_a.wall_tex_name,
            floor_tex=room_a.floor_tex_name,
            ceil_tex=room_a.ceil_tex_name,
            no_ceiling=room_a.no_ceiling,
        )

        self.rooms.append(room)

        room.add_portal(1, start_pos=0, end_pos=len_a)
        room.add_portal(3, start_pos=0, end_pos=len_b)

    def place_entity(
        self,
        ent,
        room=None,
        pos=None,
        dir=None,
        min_x=None,
        max_x=None,
        min_z=None,
        max_z=None,
    ):
        """
        Place an entity/object in the world.
        Find a position that doesn't intersect with any other object.
        """

        assert len(self.rooms) > 0, "create rooms before calling place_entity"
        assert ent.radius is not None, "entity must have physical size defined"

        # Generate collision detection data
        if len(self.wall_segs) == 0:
            self._gen_static_data()

        # If an exact position if specified
        if pos is not None:
            ent.dir = (
                dir if dir is not None else self.np_random.uniform(-math.pi, math.pi)
            )
            ent.pos = pos
            self.entities.append(ent)
            return ent

        # Keep retrying until we find a suitable position
        while True:
            # Pick a room, sample rooms proportionally to floor surface area
            r = (
                room
                if room
                else list(self.rooms)[
                    self.np_random.choice(len(list(self.rooms)), p=self.room_probs)
                ]
            )

            # Choose a random point within the square bounding box of the room
            lx = r.min_x if min_x is None else min_x
            hx = r.max_x if max_x is None else max_x
            lz = r.min_z if min_z is None else min_z
            hz = r.max_z if max_z is None else max_z
            pos = self.np_random.uniform(
                low=[lx - ent.radius, 0, lz - ent.radius],
                high=[hx + ent.radius, 0, hz + ent.radius],
            )

            # Make sure the position is within the room's outline
            if not r.point_inside(pos):
                continue

            # Make sure the position doesn't intersect with any walls
            if self.intersect(ent, pos, ent.radius):
                continue

            # Pick a direction
            d = dir if dir is not None else self.np_random.uniform(-math.pi, math.pi)

            ent.pos = pos
            ent.dir = d
            break

        self.entities.append(ent)

        return ent

    def place_agent(
        self, room=None, dir=None, min_x=None, max_x=None, min_z=None, max_z=None
    ):
        """
        Place the agent in the environment at a random position
        and orientation
        """

        return self.place_entity(
            self.agent,
            room=room,
            dir=dir,
            min_x=min_x,
            max_x=max_x,
            min_z=min_z,
            max_z=max_z,
        )

    def intersect(self, ent, pos, radius):
        """
        Check if an entity intersects with the world
        """

        # Ignore the Y position
        px, _, pz = pos
        pos = np.array([px, 0, pz])

        # Check for intersection with walls
        if intersect_circle_segs(pos, radius, self.wall_segs):
            return True

        # Check for entity intersection
        for ent2 in self.entities:
            # Entities can't intersect with themselves
            if ent2 is ent:
                continue

            px, _, pz = ent2.pos
            pos2 = np.array([px, 0, pz])

            d = np.linalg.norm(pos2 - pos)
            if d < radius + ent2.radius:
                return ent2

        return None

    def near(self, ent0, ent1=None):
        """
        Test if the two entities are near each other.
        Used for "go to" or "put next" type tasks
        """

        if ent1 is None:
            ent1 = self.agent

        dist = np.linalg.norm(ent0.pos - ent1.pos)
        return dist < ent0.radius + ent1.radius + self.max_forward_step

    def nearby(self, ent0, ent1=None):
        """
        Test if the two entities are near each other.
        Used for "go to" or "put next" type tasks
        """

        if ent1 is None:
            ent1 = self.agent

        dist = np.linalg.norm(ent0.pos - ent1.pos)
        new_dist = dist - (ent0.radius + ent1.radius + self.max_forward_step)
        if new_dist < 0:
            new_dist = 0
        elif new_dist > 10:
            new_dist = 0
        else:
            new_dist = int(((10 - new_dist) ** 2) * 2.55) 
        return new_dist


    def _load_tex(self, tex_name):
        """
        Load a texture, with or without domain randomization
        """

        rand = (
            self.np_random if self.params.sample(self.np_random, "tex_rand") else None
        )
        return Texture.get(tex_name, rand)

    def _gen_static_data(self):
        """
        Generate static data needed for rendering and collision detection
        """

        # Generate the static data for each room
        for room in self.rooms:
            room._gen_static_data(
                self.params, self.np_random if self.domain_rand else None
            )

        # Concatenate the wall segments
        self.wall_segs = np.concatenate([r.wall_segs for r in self.rooms])

        # Room selection probabilities
        self.room_probs = np.array([r.area for r in self.rooms], dtype=float)
        self.room_probs /= np.sum(self.room_probs)

    def _gen_world(self):
        """
        Generate the world. Derived classes must implement this method.
        """

        raise NotImplementedError

    def _reward(self):
        """
        Default sparse reward computation
        """

        return 1.0 - 0.2 * (self.step_count / self.max_episode_steps)

    def _render_static(self):
        """
        Render the static elements of the scene into a display list.
        Called once at the beginning of each episode.
        """

        # TODO: manage this automatically
        # glIsList
        glDeleteLists(1, 1)
        glNewList(1, GL_COMPILE)

        # Light position
        glLightfv(GL_LIGHT0, GL_POSITION, (GLfloat * 4)(*self.light_pos + [1]))

        # Background/minimum light level
        glLightfv(GL_LIGHT0, GL_AMBIENT, (GLfloat * 4)(*self.light_ambient))

        # Diffuse light color
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (GLfloat * 4)(*self.light_color))

        # glLightf(GL_LIGHT0, GL_SPOT_CUTOFF, 180)
        # glLightf(GL_LIGHT0, GL_SPOT_EXPONENT, 0)
        # glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 0)
        # glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0)
        # glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, 0)

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)

        glShadeModel(GL_SMOOTH)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        # Render the rooms
        glEnable(GL_TEXTURE_2D)
        for room in self.rooms:
            room._render()

        # Render the static entities
        for ent in self.entities:
            if ent.is_static:
                ent.render()

        glEndList()

    def _render_world(self, frame_buffer, render_agent):
        """
        Render the world from a given camera position into a frame buffer,
        and produce a numpy image array as output.
        """

        # Call the display list for the static parts of the environment
        glCallList(1)

        # TODO: keep the non-static entities in a different list for efficiency?
        # Render the non-static entities
        for ent in self.entities:
            if not ent.is_static and ent is not self.agent:
                ent.render()
                # ent.draw_bound()

        if render_agent:
            self.agent.render()

        # Resolve the rendered image into a numpy array
        img = frame_buffer.resolve()

        return img

    def render_top_view(self, frame_buffer=None):
        """
        Render a top view of the whole map (from above)
        """

        if frame_buffer is None:
            frame_buffer = self.obs_fb

        # Switch to the default OpenGL context
        # This is necessary on Linux Nvidia drivers
        self.shadow_window.switch_to()

        # Bind the frame buffer before rendering into it
        frame_buffer.bind()

        # Clear the color and depth buffers
        glClearColor(*self.sky_color, 1.0)
        glClearDepth(1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Scene extents to render
        min_x = self.min_x - 1
        max_x = self.max_x + 1
        min_z = self.min_z - 1
        max_z = self.max_z + 1

        width = max_x - min_x
        height = max_z - min_z
        aspect = width / height
        fb_aspect = frame_buffer.width / frame_buffer.height

        # Adjust the aspect extents to match the frame buffer aspect
        if aspect > fb_aspect:
            # Want to add to denom, add to height
            new_h = width / fb_aspect
            h_diff = new_h - height
            min_z -= h_diff / 2
            max_z += h_diff / 2
        elif aspect < fb_aspect:
            # Want to add to num, add to width
            new_w = height * fb_aspect
            w_diff = new_w - width
            min_x -= w_diff / 2
            max_x += w_diff / 2

        # Set the projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(min_x, max_x, -max_z, -min_z, -100, 100.0)

        # Setup the camera
        # Y maps to +Z, Z maps to +Y
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        m = [
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            -1,
            0,
            0,
            0,
            0,
            0,
            1,
        ]
        glLoadMatrixf((GLfloat * len(m))(*m))

        return self._render_world(frame_buffer, render_agent=True)

    def render_obs_4(self):
        curr_vec = self.agent.dir_vec
        theta = math.asin(curr_vec[0])
        east_vec = np.array([math.sin(theta + 1.5708), 0, -math.cos(theta + 1.5708)])
        west_vec = np.array([math.sin(theta - 1.5708), 0, -math.cos(theta - 1.5708)])
        back_vec = np.array([math.sin(theta + 3.1416), 0, -math.cos(theta + 3.1416)])

        view_norm = self.render_obs()
        view_east = self.render_obs(look_dir=east_vec)
        view_west = self.render_obs(look_dir=west_vec)
        view_back = self.render_obs(look_dir=back_vec)

        return np.concatenate([view_norm, view_east, view_west, view_back], axis=-1)


    def render_obs(self, frame_buffer=None, look_dir=None):
        """
        Render an observation from the point of view of the agent
        """

        if look_dir is None:
            look_dir = self.agent.cam_dir

        if frame_buffer is None:
            frame_buffer = self.obs_fb

        # Switch to the default OpenGL context
        # This is necessary on Linux Nvidia drivers
        self.shadow_window.switch_to()

        # Bind the frame buffer before rendering into it
        frame_buffer.bind()

        # Clear the color and depth buffers
        glClearColor(*self.sky_color, 1.0)
        glClearDepth(1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set the projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(
            self.agent.cam_fov_y,
            frame_buffer.width / float(frame_buffer.height),
            0.04,
            100.0,
        )

        # Setup the camera
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            # Eye position
            *self.agent.cam_pos,
            # Target
            *(self.agent.cam_pos + look_dir),
            # Up vector
            0,
            1.0,
            0.0,
        )

        return self._render_world(frame_buffer, render_agent=False)

    def render_depth(self, frame_buffer=None):
        """
        Produce a depth map
        Values are floating-point, map shape is (H,W,1)
        Distances are in meters from the observer
        """

        if frame_buffer is None:
            frame_buffer = self.obs_fb

        # Render the world
        self.render_obs(frame_buffer)

        return frame_buffer.get_depth_map(0.04, 100.0)

    def get_visible_ents(self):
        """
        Get a list of visible entities.
        Uses OpenGL occlusion queries to approximate visibility.
        :return: set of objects visible to the agent
        """

        # Allocate the occlusion query ids
        num_ents = len(self.entities)
        query_ids = (GLuint * num_ents)()
        glGenQueries(num_ents, query_ids)

        # Switch to the default OpenGL context
        # This is necessary on Linux Nvidia drivers
        self.shadow_window.switch_to()

        # Use the small observation frame buffer
        frame_buffer = self.obs_fb

        # Bind the frame buffer before rendering into it
        frame_buffer.bind()

        # Clear the color and depth buffers
        glClearColor(*self.sky_color, 1.0)
        glClearDepth(1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set the projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(
            self.agent.cam_fov_y,
            frame_buffer.width / float(frame_buffer.height),
            0.04,
            100.0,
        )

        # Setup the cameravisible objects
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            # Eye position
            *self.agent.cam_pos,
            # Target
            *(self.agent.cam_pos + self.agent.cam_dir),
            # Up vector
            0,
            1.0,
            0.0,
        )

        # Render the rooms, without texturing
        glDisable(GL_TEXTURE_2D)
        for room in self.rooms:
            room._render()

        # For each entity
        for ent_idx, ent in enumerate(self.entities):
            if ent is self.agent:
                continue

            glBeginQuery(GL_ANY_SAMPLES_PASSED, query_ids[ent_idx])
            pos = ent.pos

            # glColor3f(1, 0, 0)
            drawBox(
                x_min=pos[0] - 0.1,
                x_max=pos[0] + 0.1,
                y_min=pos[1],
                y_max=pos[1] + 0.2,
                z_min=pos[2] - 0.1,
                z_max=pos[2] + 0.1,
            )

            glEndQuery(GL_ANY_SAMPLES_PASSED)

        vis_objs = set()

        # Get query results
        for ent_idx, ent in enumerate(self.entities):
            if ent is self.agent:
                continue

            visible = (GLuint * 1)(1)
            glGetQueryObjectuiv(query_ids[ent_idx], GL_QUERY_RESULT, visible)

            if visible[0] != 0:
                vis_objs.add(ent)

        # Free the occlusion query ids
        glDeleteQueries(1, query_ids)

        # img = frame_buffer.resolve()
        # return img

        return vis_objs

    def close(self):
        if self.window:
            self.window.close()
        return

    def render(self):
        """
        Render the environment for human viewing
        """

        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        # Render the human-view image
        if self.view == "agent":
            img = self.render_obs(self.vis_fb)
        else:
            img = self.render_top_view(self.vis_fb)
        img_width = img.shape[1]
        img_height = img.shape[0]

        if self.render_mode == "rgb_array":
            return img

        # Render the agent's view
        obs = self.render_obs()
        obs_width = obs.shape[1]
        obs_height = obs.shape[0]

        window_width = img_width + self.obs_disp_width
        window_height = img_height

        if self.window is None:
            config = pyglet.gl.Config(double_buffer=True)
            self.window = pyglet.window.Window(
                width=window_width, height=window_height, resizable=False, config=config
            )

        self.window.clear()
        self.window.switch_to()

        # Bind the default frame buffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # Clear the color and depth buffers
        glClearColor(0, 0, 0, 1.0)
        glClearDepth(1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Setup orghogonal projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glOrtho(0, window_width, 0, window_height, 0, 10)

        # Draw the human render to the rendering window
        img_flip = np.ascontiguousarray(np.flip(img, axis=0))
        img_data = pyglet.image.ImageData(
            img_width,
            img_height,
            "RGB",
            img_flip.ctypes.data_as(POINTER(GLubyte)),
            pitch=img_width * 3,
        )
        img_data.blit(0, 0, 0, width=img_width, height=img_height)

        # Draw the observation
        obs = np.ascontiguousarray(np.flip(obs, axis=0))
        obs_data = pyglet.image.ImageData(
            obs_width,
            obs_height,
            "RGB",
            obs.ctypes.data_as(POINTER(GLubyte)),
            pitch=obs_width * 3,
        )
        obs_data.blit(
            img_width,
            img_height - self.obs_disp_height,
            0,
            width=self.obs_disp_width,
            height=self.obs_disp_height,
        )

        # Draw the text label in the window
        self.text_label.text = "pos: (%.2f, %.2f, %.2f)\nangle: %d\nsteps: %d" % (
            *self.agent.pos,
            int(self.agent.dir * 180 / math.pi) % 360,
            self.step_count,
        )
        self.text_label.draw()

        # Force execution of queued commands
        glFlush()

        # If we are not running the Pyglet event loop,
        # we have to manually flip the buffers and dispatch events
        if self.render_mode == "human":
            self.window.flip()
            self.window.dispatch_events()

            return

        return img
