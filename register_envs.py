import gymnasium as gym
from gymnasium.envs.registration import register

from maze_c import MazeCA

# Register your custom MazeCA
register(
    id="MiniWorld-MazeCA-v0",
    entry_point="maze_c:MazeCA",
    vector_entry_point="maze_c:MazeCA",
)

print("Custom MiniWorld-MazeCA-v0 registered!")