import pandas as pd
import numpy as np
import os 

class TaskSchedulingEnv(gym.Env):
    def __init__(self, tasks, team, skill_encoder):
        super(TaskSchedulingEnv, self).__init__()
        self.tasks = tasks
        self.team = team
        self.skill_encoder = skill_encoder
        self.action_space = gym.spaces.Discrete(len(team))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4 + len(team),), dtype=np.float32)
        self.current_task_idx = 0
        self.state = self._get_state()
