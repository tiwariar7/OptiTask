import pandas as pd
import numpy as np
import os

import streamlit as st
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.preprocessing import LabelEncoder

from pathlib import Path
import pickle

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_task_idx = 0
        for m in self.team: m['Workload'] = 0
        return self._get_state(), {}

# --- Streamlit UI ---
st.set_page_config(page_title="Adaptive Task Scheduler", layout="wide")

st.title(" Adaptive Task Scheduling using RL")
st.markdown("""
This application uses **Reinforcement Learning (PPO)** to optimally assign tasks to team members based on 
skills, deadlines, priorities, and current workloads.
""")

# --- Sidebar: Configuration & Data ---
st.sidebar.header("Data & Model")

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "adaptive_task_scheduling_dataset.csv"

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "adaptive_task_scheduling_dataset.csv"

if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
    all_skills = sorted(df['Skill Requirement'].unique())
    skill_encoder = LabelEncoder().fit(all_skills)
else:
    st.error("Dataset not found. Please ensure 'adaptive_task_scheduling_dataset.csv' is in the project folder.")
    st.stop()
# --- Team Management ---
st.header("Team Configuration")
if 'team' not in st.session_state:
    st.session_state.team = [
        {"Name": "Alice", "Skills": ["Python", "Machine Learning"], "Workload": 0.0},
        {"Name": "Bob", "Skills": ["UI/UX", "JavaScript"], "Workload": 0.0},
        {"Name": "Charlie", "Skills": ["Cloud", "DevOps"], "Workload": 0.0}
    ]

with st.expander("Edit Team Members"):
    new_team = []
    for i, member in enumerate(st.session_state.team):
        col1, col2 = st.columns(2)
        name = col1.text_input(f"Member {i+1} Name", value=member["Name"], key=f"name_{i}")
        skills = col2.multiselect(f"Member {i+1} Skills", options=all_skills, default=[s for s in member["Skills"] if s in all_skills], key=f"skills_{i}")
        new_team.append({"Name": name, "Skills": skills, "Workload": member["Workload"]})
    
    if st.button("Update Team"):
        st.session_state.team = new_team
        st.success("Team updated!")

# --- Training Section ---
st.header("Model Training")
col1, col2 = st.columns(2)
timesteps = col1.slider("Training Timesteps", 1000, 50000, 10000)

if col2.button("Train Model"):
    with st.spinner("Training RL Agent..."):
        # Preprocess some tasks for training
        def preprocess(task):
            days = (pd.to_datetime(task["Task Deadline"]) - pd.Timestamp.now().normalize()).days
            p_map = {"Low": 0, "Medium": 1, "High": 2}
            return {
                "Skill": skill_encoder.transform([task["Skill Requirement"]])[0],
                "Deadline": days,
                "Priority": p_map.get(task["Task Priority"], 0),
                "Duration": task["Estimated Completion Time"] / 8
            }
        
        train_data = df.sample(min(200, len(df)))
        rl_tasks = [preprocess(row) for _, row in train_data.iterrows()]
        
        env = DummyVecEnv([lambda: TaskSchedulingEnv(rl_tasks, st.session_state.team, skill_encoder)])
        model = PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=timesteps)
        
        st.session_state.model = model
        st.success("Model trained and ready!")

# --- Task Assignment ---
st.header("Assign New Task")
if 'model' in st.session_state:
    c1, c2, c3, c4 = st.columns(4)
    task_skill = c1.selectbox("Required Skill", options=all_skills)
    task_deadline = c2.number_input("Days to Deadline", min_value=0, value=7)
    task_priority = c3.selectbox("Priority", options=["Low", "Medium", "High"])
    task_duration = c4.number_input("Duration (Hours)", min_value=1.0, value=4.0)

    if st.button("Assign Task"):
        # Prepare state
        s_idx = skill_encoder.transform([task_skill])[0]
        p_idx = {"Low": 0, "Medium": 1, "High": 2}.get(task_priority, 1)
        dur_norm = task_duration / 8
        
        state = np.concatenate([
            np.array([s_idx, task_deadline, p_idx, dur_norm]),
            [m['Workload'] / 20 for m in st.session_state.team]
        ])
        
        action, _ = st.session_state.model.predict(state.reshape(1, -1), deterministic=True)
        assigned_idx = int(action[0])
        assigned_member = st.session_state.team[assigned_idx]
        
        # Update workload in session state
        st.session_state.team[assigned_idx]['Workload'] += dur_norm
        
        st.balloons()
        st.success(f"### Task assigned to: **{assigned_member['Name']}**")
        
        # Display summary
        st.info(f"**Reasoning:** {assigned_member['Name']} has skills: {', '.join(assigned_member['Skills'])}. Current Workload: {st.session_state.team[assigned_idx]['Workload']*8:.1f} hours.")
else:
    st.warning("Please train the model first to enable task assignment.")

# --- Dashboard ---
st.header("Team Dashboard")
workload_data = pd.DataFrame([
    {"Name": m["Name"], "Workload (Hours)": m["Workload"] * 8} 
    for m in st.session_state.team
])
st.bar_chart(workload_data.set_index("Name"))

if st.button("Reset Workloads"):
    for m in st.session_state.team:
        m['Workload'] = 0.0
    st.rerun()
