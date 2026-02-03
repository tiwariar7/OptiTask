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

    def _get_state(self):
        if self.current_task_idx >= len(self.tasks): return np.zeros(self.observation_space.shape, dtype=np.float32)
        task = self.tasks[self.current_task_idx]
        task_vector = np.array([task["Skill"], task["Deadline"], task["Priority"], task["Duration"]], dtype=np.float32)
        team_vector = np.array([member['Workload'] / 20 for member in self.team], dtype=np.float32)
        return np.concatenate([task_vector, team_vector])

    def step(self, action):
        if isinstance(action, (np.ndarray, list)): action = int(action[0])
        member = self.team[action]
        task = self.tasks[self.current_task_idx]
        reward = 0
        member_skill_indices = [self.skill_encoder.transform([s.strip()])[0] for s in member['Skills'] if s.strip() in self.skill_encoder.classes_]
        reward += 20 if task['Skill'] in member_skill_indices else -10
        reward += 10 if member['Workload'] < 10 else -15
        if task['Deadline'] > 0: reward += 10
        elif task['Deadline'] == 0: reward += 5
        else: reward -= 20
        member['Workload'] += task['Duration']
        self.current_task_idx += 1
        done = self.current_task_idx >= len(self.tasks)
        return self._get_state(), float(reward), done, False, {}
