"""
Reinforcement Learning Agent: Deep Q-Network (DQN) for resource allocation.
State  : machine sensor vector + job features (15-dim)
Action : 0 = defer, 1 = assign job to the required machine
Reward : shaping from KPI improvement (tardiness, utilization)
"""

import numpy as np
from collections import deque
import random
from typing import Tuple, List
import warnings
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

STATE_DIM = 15    # 10 machine features + 5 job features
N_ACTIONS = 2     # 0=defer, 1=assign


# ─────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size: int) -> List:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────
# Lightweight Q-Network (pure numpy, no torch dep.)
# ─────────────────────────────────────────────

class QNetwork:
    """Two-layer MLP with ReLU, trained with mini-batch gradient descent."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, lr: float = 1e-3):
        scale1 = np.sqrt(2.0 / in_dim)
        scale2 = np.sqrt(2.0 / hidden)
        self.W1 = np.random.randn(in_dim, hidden).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = np.random.randn(hidden, out_dim).astype(np.float32) * scale2
        self.b2 = np.zeros(out_dim, dtype=np.float32)
        self.lr = lr

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0, x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def update(self, x: np.ndarray, target: np.ndarray):
        """Single gradient step with MSE loss."""
        h = np.maximum(0, x @ self.W1 + self.b1)
        q = h @ self.W2 + self.b2
        delta = q - target                        # (batch, out_dim)

        # Gradients W2, b2
        dW2 = h.T @ delta / len(x)
        db2 = delta.mean(axis=0)

        # Gradients W1, b1
        dh = (delta @ self.W2.T) * (h > 0)
        dW1 = x.T @ dh / len(x)
        db1 = dh.mean(axis=0)

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def copy_weights_from(self, other: "QNetwork"):
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()


# ─────────────────────────────────────────────
# DQN Agent
# ─────────────────────────────────────────────

class DQNAgent:
    def __init__(
        self,
        state_dim: int = STATE_DIM,
        n_actions: int = N_ACTIONS,
        hidden: int = 128,
        lr: float = 5e-4,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        target_update_freq: int = 50,
        buffer_capacity: int = 50_000,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.q_net = QNetwork(state_dim, hidden, n_actions, lr)
        self.target_net = QNetwork(state_dim, hidden, n_actions, lr)
        self.target_net.copy_weights_from(self.q_net)
        self.buffer = ReplayBuffer(buffer_capacity)

        self.steps = 0
        self.losses: List[float] = []
        self.rewards_history: List[float] = []
        self.epsilon_history: List[float] = []

    def select_action(self, state: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        q = self.q_net.forward(state.reshape(1, -1))[0]
        return int(np.argmax(q))

    def store(self, s, a, r, s2, done):
        self.buffer.push(s, a, r, s2, done)

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return 0.0

        batch = self.buffer.sample(self.batch_size)
        S  = np.array([b[0] for b in batch], dtype=np.float32)
        A  = np.array([b[1] for b in batch], dtype=np.int32)
        R  = np.array([b[2] for b in batch], dtype=np.float32)
        S2 = np.array([b[3] for b in batch], dtype=np.float32)
        D  = np.array([b[4] for b in batch], dtype=np.float32)

        # Double DQN target
        q_online_s2 = self.q_net.forward(S2)
        best_a = np.argmax(q_online_s2, axis=1)
        q_target_s2 = self.target_net.forward(S2)
        next_q = q_target_s2[np.arange(len(batch)), best_a]
        td_target = R + self.gamma * next_q * (1 - D)

        # Current Q values
        q_curr = self.q_net.forward(S)
        targets = q_curr.copy()
        targets[np.arange(len(batch)), A] = td_target

        self.q_net.update(S, targets)

        # MSE loss for logging
        loss = float(np.mean((q_curr[np.arange(len(batch)), A] - td_target) ** 2))
        self.losses.append(loss)

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.copy_weights_from(self.q_net)

        return loss

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)


# ─────────────────────────────────────────────
# Training loop helper
# ─────────────────────────────────────────────

def train_dqn(env_factory, n_episodes: int = 30, max_steps: int = 2400) -> DQNAgent:
    """
    Train the DQN agent over multiple simulation episodes.
    env_factory: callable returning a fresh ManufacturingEnvironment.
    """
    from environment import MachineState

    agent = DQNAgent()
    episode_returns = []

    print(f"\nTraining DQN agent over {n_episodes} episodes ...")
    for ep in range(n_episodes):
        env = env_factory()
        obs = env.reset()
        ep_reward = 0.0
        done = False
        t = 0

        while not done and t < max_steps:
            # For each idle machine with available job, decide assign/defer
            for machine in env.machines:
                if machine.state != MachineState.IDLE:
                    continue
                # Find a candidate job for this machine
                candidate = None
                for job in env.job_queue:
                    op = job.current_op
                    if op < len(job.operations) and job.operations[op] == machine.machine_id:
                        candidate = job
                        break
                if candidate is None:
                    continue

                s = _build_state_vec(candidate, machine, env.time, env.sim_duration,
                                     len(env.job_queue))
                a = agent.select_action(s)

                assignments = {}
                if a == 1:
                    assignments[candidate.job_id] = machine.machine_id

                prev_done = len(env.completed_jobs)
                obs, r, done, info = env.step(assignments)
                n_new = len(env.completed_jobs) - prev_done
                r += n_new * 5 - 0.01 * info["avg_tardiness"]

                s2 = _build_state_vec(candidate, machine, env.time, env.sim_duration,
                                      len(env.job_queue))
                agent.store(s, a, r, s2, float(done))
                ep_reward += r

                agent.learn()
                t += 1

        agent.decay_epsilon()
        episode_returns.append(ep_reward)
        if ep % 5 == 0:
            print(f"  Episode {ep+1:3d}/{n_episodes} | Return: {ep_reward:8.1f} "
                  f"| ε={agent.epsilon:.3f} | Buffer={len(agent.buffer)}")

    agent.rewards_history = episode_returns
    print("DQN training complete.")
    return agent


def _build_state_vec(job, machine, t, sim_dur, queue_len) -> np.ndarray:
    op = job.current_op
    pt = job.processing_times[op] if op < len(job.processing_times) else 0
    return np.concatenate([
        machine.sensor_vector(),
        [pt, job.due_date - t, job.priority, queue_len, t / sim_dur]
    ]).astype(np.float32)


if __name__ == "__main__":
    from environment import ManufacturingEnvironment
    agent = train_dqn(
        lambda: ManufacturingEnvironment(n_jobs_total=120, sim_duration=480),
        n_episodes=20,
    )
    print(f"Final epsilon: {agent.epsilon:.3f}")
    print(f"Final avg reward (last 5 ep): {np.mean(agent.rewards_history[-5:]):.1f}")
