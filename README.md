# OptiTask: Adaptive Task Scheduling using RL

OptiTask is an intelligent task scheduling system that leverages Reinforcement Learning (PPO) to optimally assign tasks to team members. It considers multiple factors such as employee skills, task deadlines, priority levels, and current workloads to ensure efficient resource allocation.

## Features

- **Automated Task Assignment**: Uses a Proximal Policy Optimization (PPO) agent to make intelligent scheduling decisions.
- **Dynamic Team Management**: Easily add or edit team members and their associated skill sets.
- **Interactive Dashboard**: Visualize team workloads and task distributions in real-time.
- **On-the-fly Training**: Train the RL model directly through the web interface with customizable timesteps.
- **Data-Driven**: Powered by a comprehensive dataset for realistic simulation and training.

## Project Structure

- `Streamlit File/`: Contains the main web application (`app.py`) and dependencies.
- `Code Files/`: Includes Jupyter notebooks for model exploration and development.
- `data/`: Stores the task scheduling datasets.
- `docx/`: Project documentation and reports.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd OptiTask
   ```

2. **Install dependencies**:
   It is recommended to use a virtual environment.
   ```bash
   pip install -r "Streamlit File/requirements.txt"
   ```

3. **Prepare the dataset**:
   Ensure the `adaptive_task_scheduling_dataset.csv` is present in the `Streamlit File/` directory or update the path in `app.py`.

## Usage

To launch the Streamlit application:

```bash
cd "Streamlit File"
streamlit run app.py
```

### How to use the App:
1. **Configure Team**: Use the sidebar or main panel to set up your team members and their skills.
2. **Train Model**: Click the "Train Model" button to initialize the RL agent.
3. **Assign Tasks**: Enter task details (Skill, Deadline, Priority, Duration) and click "Assign Task".
4. **Monitor**: View the "Team Dashboard" to track the workload distribution.

## Technologies Used

- **Framework**: [Streamlit](https://streamlit.io/)
- **Reinforcement Learning**: [Stable Baselines3](https://stable-baselines3.readthedocs.io/), [Gymnasium](https://gymnasium.farama.org/)
- **Data Analysis**: Pandas, NumPy, Scikit-learn
- **Modeling**: PPO (Proximal Policy Optimization)
