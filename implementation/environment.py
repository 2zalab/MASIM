"""
Manufacturing Shop-Floor Simulation Environment
Simulates a smart factory with 6 CNC machines, IoT sensors, and dynamic job scheduling.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import random
import warnings
warnings.filterwarnings("ignore")

random.seed(42)
np.random.seed(42)


class MachineState(Enum):
    IDLE = "idle"
    WORKING = "working"
    MAINTENANCE = "maintenance"
    FAILED = "failed"


@dataclass
class Job:
    job_id: int
    operations: List[int]          # list of machine indices required
    processing_times: List[float]  # processing time per operation (minutes)
    due_date: float                # deadline in simulation minutes
    priority: int = 1              # 1=low, 2=medium, 3=high
    arrival_time: float = 0.0
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    current_op: int = 0

    @property
    def is_complete(self):
        return self.current_op >= len(self.operations)

    @property
    def tardiness(self):
        if self.completion_time is None:
            return 0.0
        return max(0.0, self.completion_time - self.due_date)

    @property
    def lateness(self):
        if self.completion_time is None:
            return 0.0
        return self.completion_time - self.due_date


@dataclass
class Machine:
    machine_id: int
    state: MachineState = MachineState.IDLE
    current_job: Optional[Job] = None
    remaining_time: float = 0.0
    total_working_time: float = 0.0
    total_idle_time: float = 0.0
    total_downtime: float = 0.0
    failure_count: int = 0
    jobs_completed: int = 0

    # Sensor readings
    temperature: float = field(default_factory=lambda: np.random.uniform(20, 30))
    vibration: float = field(default_factory=lambda: np.random.uniform(0.1, 0.5))
    current_load: float = 0.0
    energy_consumed: float = 0.0    # kWh
    wear_level: float = 0.0         # 0-1 cumulative wear
    time_since_maintenance: float = 0.0

    # Failure model parameters (Weibull)
    base_failure_rate: float = field(default_factory=lambda: np.random.uniform(0.001, 0.003))

    def failure_probability(self, dt: float = 1.0) -> float:
        """Weibull-based failure probability increases with wear."""
        wear_factor = 1 + 3.5 * self.wear_level**2
        temp_factor = 1 + 0.02 * max(0, self.temperature - 60)
        vib_factor = 1 + 1.5 * max(0, self.vibration - 1.0)
        rate = self.base_failure_rate * wear_factor * temp_factor * vib_factor
        return 1 - np.exp(-rate * dt)

    def update_sensors(self, dt: float):
        """Update sensor readings with realistic dynamics."""
        if self.state == MachineState.WORKING:
            target_temp = 65 + 20 * self.current_load + np.random.normal(0, 2)
            target_vib = 0.8 + 1.2 * self.wear_level + 0.3 * self.current_load + np.random.normal(0, 0.05)
            self.temperature += 0.3 * (target_temp - self.temperature) * dt + np.random.normal(0, 0.5)
            self.vibration += 0.2 * (target_vib - self.vibration) * dt + np.random.normal(0, 0.02)
            power_kw = (2.5 + 3.5 * self.current_load) * (1 + 0.4 * self.wear_level)
            self.energy_consumed += power_kw * dt / 60.0  # convert min to hours
            self.wear_level = min(1.0, self.wear_level + 0.0001 * dt)
            self.time_since_maintenance += dt
        elif self.state == MachineState.IDLE:
            self.temperature += 0.1 * (25 - self.temperature) * dt
            self.vibration = max(0.05, self.vibration - 0.05 * dt)
            self.energy_consumed += 0.3 * dt / 60.0  # standby power
            self.total_idle_time += dt
        elif self.state == MachineState.MAINTENANCE:
            self.total_downtime += dt
            self.temperature = max(25, self.temperature - 2 * dt)
            self.vibration = max(0.1, self.vibration - 0.1 * dt)

    def apply_maintenance(self):
        """Reset wear after maintenance."""
        self.wear_level = max(0.0, self.wear_level - 0.7)
        self.time_since_maintenance = 0.0
        self.vibration = np.random.uniform(0.1, 0.4)
        self.temperature = np.random.uniform(25, 35)

    def sensor_vector(self) -> np.ndarray:
        return np.array([
            self.temperature,
            self.vibration,
            self.current_load,
            self.wear_level,
            self.time_since_maintenance,
            float(self.state == MachineState.WORKING),
            float(self.state == MachineState.FAILED),
            self.failure_count,
            self.jobs_completed,
            self.energy_consumed,
        ], dtype=np.float32)


class ManufacturingEnvironment:
    """
    Flexible Job-Shop scheduling environment with 6 machines and dynamic disruptions.
    Observation space: machine sensor vectors + job queue status
    Action space: assign job j to machine m (discrete)
    """

    N_MACHINES = 6
    N_JOB_TYPES = 4
    MAINTENANCE_DURATION = 30.0   # minutes
    REPAIR_DURATION = 60.0         # minutes

    def __init__(self, n_jobs_total: int = 200, sim_duration: float = 2400.0):
        self.n_jobs_total = n_jobs_total
        self.sim_duration = sim_duration      # total simulation window (minutes)
        self.machines: List[Machine] = []
        self.job_queue: List[Job] = []
        self.completed_jobs: List[Job] = []
        self.time = 0.0
        self.dt = 1.0                          # simulation step (minutes)
        self._reset_machines()
        self._generate_jobs()

    def _reset_machines(self):
        self.machines = [Machine(machine_id=i) for i in range(self.N_MACHINES)]

    def _generate_jobs(self):
        """Generate heterogeneous jobs with realistic operation sequences."""
        self.all_jobs: List[Job] = []
        job_templates = [
            {"ops": [0, 1, 2], "pt": [15, 20, 10], "base_due": 90,  "priority": 2},
            {"ops": [1, 3, 4], "pt": [25, 15, 30], "base_due": 120, "priority": 1},
            {"ops": [0, 2, 4, 5], "pt": [10, 20, 15, 25], "base_due": 110, "priority": 3},
            {"ops": [2, 3, 5], "pt": [20, 10, 20], "base_due": 80,  "priority": 2},
        ]
        arrival_rate = self.sim_duration / self.n_jobs_total  # avg inter-arrival
        t = 0.0
        for i in range(self.n_jobs_total):
            tmpl = random.choice(job_templates)
            noise = [np.random.uniform(0.8, 1.3) for _ in tmpl["pt"]]
            pt = [tmpl["pt"][k] * noise[k] for k in range(len(tmpl["pt"]))]
            t += np.random.exponential(arrival_rate)
            due = t + tmpl["base_due"] * np.random.uniform(0.9, 1.4)
            self.all_jobs.append(Job(
                job_id=i,
                operations=list(tmpl["ops"]),
                processing_times=pt,
                due_date=due,
                priority=tmpl["priority"],
                arrival_time=t,
            ))
        self.job_pointer = 0

    def reset(self):
        self._reset_machines()
        self._generate_jobs()
        self.job_queue = []
        self.completed_jobs = []
        self.time = 0.0
        return self._get_obs()

    def _get_obs(self) -> Dict:
        return {
            "machines": np.stack([m.sensor_vector() for m in self.machines]),
            "queue_length": len(self.job_queue),
            "time": self.time,
        }

    def _release_arrived_jobs(self):
        while self.job_pointer < len(self.all_jobs):
            j = self.all_jobs[self.job_pointer]
            if j.arrival_time <= self.time:
                self.job_queue.append(j)
                self.job_pointer += 1
            else:
                break

    def step(self, assignments: Dict[int, int]) -> Tuple[Dict, float, bool, Dict]:
        """
        assignments: {job_id: machine_id} for this time step.
        Returns (obs, reward, done, info).
        """
        reward = 0.0

        # Apply assignments
        for job_id, machine_id in assignments.items():
            job = next((j for j in self.job_queue if j.job_id == job_id), None)
            machine = self.machines[machine_id]
            if job is None or machine.state != MachineState.IDLE:
                continue
            op_idx = job.current_op
            required_machine = job.operations[op_idx]
            if required_machine != machine_id:
                continue
            machine.state = MachineState.WORKING
            machine.current_job = job
            machine.remaining_time = job.processing_times[op_idx]
            machine.current_load = np.random.uniform(0.6, 1.0)
            if job.start_time is None:
                job.start_time = self.time

        # Advance simulation
        for machine in self.machines:
            machine.update_sensors(self.dt)

            if machine.state == MachineState.WORKING:
                machine.remaining_time -= self.dt
                machine.total_working_time += self.dt

                # Stochastic failure check
                if np.random.random() < machine.failure_probability(self.dt):
                    machine.state = MachineState.FAILED
                    machine.failure_count += 1
                    if machine.current_job:
                        self.job_queue.insert(0, machine.current_job)
                        machine.current_job = None
                    reward -= 50

                elif machine.remaining_time <= 0:
                    job = machine.current_job
                    job.current_op += 1
                    if job.is_complete:
                        job.completion_time = self.time
                        self.completed_jobs.append(job)
                        self.job_queue.remove(job)
                        machine.jobs_completed += 1
                        reward += 10 - 0.05 * job.tardiness
                    machine.state = MachineState.IDLE
                    machine.current_job = None
                    machine.current_load = 0.0

            elif machine.state == MachineState.FAILED:
                machine.remaining_time -= self.dt
                machine.total_downtime += self.dt
                if machine.remaining_time <= -self.REPAIR_DURATION:
                    machine.state = MachineState.IDLE
                    machine.apply_maintenance()

            elif machine.state == MachineState.MAINTENANCE:
                machine.remaining_time -= self.dt
                machine.total_downtime += self.dt
                if machine.remaining_time <= -self.MAINTENANCE_DURATION:
                    machine.state = MachineState.IDLE
                    machine.apply_maintenance()

        self.time += self.dt
        self._release_arrived_jobs()

        done = self.time >= self.sim_duration
        info = self._compute_kpis()
        return self._get_obs(), reward, done, info

    def _compute_kpis(self) -> Dict:
        n_done = len(self.completed_jobs)
        utilizations = [
            m.total_working_time / max(1, self.time) for m in self.machines
        ]
        tardiness_list = [j.tardiness for j in self.completed_jobs]
        total_energy = sum(m.energy_consumed for m in self.machines)
        failures = sum(m.failure_count for m in self.machines)
        return {
            "time": self.time,
            "jobs_completed": n_done,
            "avg_utilization": float(np.mean(utilizations)),
            "utilizations": utilizations,
            "avg_tardiness": float(np.mean(tardiness_list)) if tardiness_list else 0.0,
            "total_energy_kwh": total_energy,
            "machine_failures": failures,
            "queue_length": len(self.job_queue),
        }

    def trigger_demand_surge(self, n_extra: int = 20):
        """Add urgent jobs to simulate demand surge scenario."""
        t = self.time
        for i in range(n_extra):
            self.job_queue.append(Job(
                job_id=9000 + i,
                operations=[0, 1, 2],
                processing_times=[12, 18, 10],
                due_date=t + 60,
                priority=3,
                arrival_time=t,
            ))

    def trigger_resource_shortage(self):
        """Force machines 0 and 1 into maintenance."""
        for mid in [0, 1]:
            if self.machines[mid].state == MachineState.WORKING:
                if self.machines[mid].current_job:
                    self.job_queue.insert(0, self.machines[mid].current_job)
                    self.machines[mid].current_job = None
            self.machines[mid].state = MachineState.MAINTENANCE
            self.machines[mid].remaining_time = 0.0


if __name__ == "__main__":
    env = ManufacturingEnvironment(n_jobs_total=100, sim_duration=480)
    obs = env.reset()
    print("Environment initialized.")
    print(f"  Machines : {env.N_MACHINES}")
    print(f"  Jobs     : {env.n_jobs_total}")
    print(f"  Duration : {env.sim_duration} min")
