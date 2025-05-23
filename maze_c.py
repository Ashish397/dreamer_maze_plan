from gymnasium import spaces, utils

from miniworld.entity import Box
from miniworld.miniworld import MiniWorldEnv
from miniworld.params import DEFAULT_PARAMS

from miniworld_c import MiniWorldEnv_c

class Maze(MiniWorldEnv, utils.EzPickle):
    pass

class MazeCA(MiniWorldEnv_c, utils.EzPickle):

    def __init__(
        self, num_rows=8, num_cols=8, room_size=3, max_episode_steps=None, **kwargs
    ):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.room_size = room_size
        self.gap_size = 0.25

        MiniWorldEnv_c.__init__(
            self,
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

        # Allow only the movement actions
        self.action_space = spaces.Box(-1, +1, (3,), dtype=float)
    
    def _gen_world(self):
        rows = []

        # For each row
        for j in range(self.num_rows):
            row = []

            # For each column
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
                    # floor_tex='asphalt'
                )
                row.append(room)

            rows.append(row)

        visited = set()

        def visit(i, j):
            """
            Recursive backtracking maze construction algorithm
            https://stackoverflow.com/questions/38502
            """

            room = rows[j][i]

            visited.add(room)

            # Reorder the neighbors to visit in a random order
            orders = [(0, 1), (0, -1), (-1, 0), (1, 0)]
            assert 4 <= len(orders)
            neighbors = []

            while len(neighbors) < 4:
                elem = orders[self.np_random.choice(len(orders))]
                orders.remove(elem)
                neighbors.append(elem)

            # For each possible neighbor
            for dj, di in neighbors:
                ni = i + di
                nj = j + dj

                if nj < 0 or nj >= self.num_rows:
                    continue
                if ni < 0 or ni >= self.num_cols:
                    continue

                neighbor = rows[nj][ni]

                if neighbor in visited:
                    continue

                if di == 0:
                    self.connect_rooms(
                        room, neighbor, min_x=room.min_x, max_x=room.max_x
                    )
                elif dj == 0:
                    self.connect_rooms(
                        room, neighbor, min_z=room.min_z, max_z=room.max_z
                    )

                visit(ni, nj)

        # Generate the maze starting from the top-left corner
        visit(0, 0)

        for j in range(self.num_rows):
            for i in range(self.num_cols):
                room = rows[j][i]

                # Look only at right and bottom neighbors to avoid duplicating connections
                if i < self.num_cols - 1:
                    neighbor = rows[j][i + 1]
                    if self.np_random.random() < self.porosity:
                        self.connect_rooms(
                            room, neighbor, min_z=room.min_z, max_z=room.max_z
                        )

                if j < self.num_rows - 1:
                    neighbor = rows[j + 1][i]
                    if self.np_random.random() < self.porosity:
                        self.connect_rooms(
                            room, neighbor, min_x=room.min_x, max_x=room.max_x
                        )
                        
        self.boxes = []
        self.boxes.append(self.place_entity(Box(color="red")))
        self.boxes.append(self.place_entity(Box(color="green")))
        self.boxes.append(self.place_entity(Box(color="blue")))

        self.place_agent()

class MazeS2(Maze):
    pass


class MazeS3(Maze):
    pass


# Parameters for larger movement steps, fast stepping
default_params = DEFAULT_PARAMS.no_random()
default_params.set("forward_step", 0.7)
default_params.set("turn_step", 45)


class MazeS3Fast(Maze):
    pass