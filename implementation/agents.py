"""
Multi-Agent System for Smart Manufacturing
Implements: MachineAgent, SensorAgent, MaintenanceAgent, ProductionAgent,
            ResourceAgent, GlobalOptimizationAgent with Contract Net Protocol.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from environment import Machine, Job, MachineState, ManufacturingEnvironment
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Message passing infrastructure
# ─────────────────────────────────────────────

@dataclass
class Message:
    sender: str
    receiver: str
    performative: str   # INFORM, REQUEST, PROPOSE, ACCEPT, REJECT, CFP
    content: Dict
    timestamp: float = 0.0


class MessageBus:
    def __init__(self):
        self._inbox: Dict[str, List[Message]] = {}

    def register(self, agent_id: str):
        self._inbox[agent_id] = []

    def send(self, msg: Message):
        if msg.receiver not in self._inbox:
            self._inbox[msg.receiver] = []
        self._inbox[msg.receiver].append(msg)

    def receive(self, agent_id: str) -> List[Message]:
        msgs = self._inbox.get(agent_id, [])
        self._inbox[agent_id] = []
        return msgs

    def broadcast(self, msg: Message, receivers: List[str]):
        for r in receivers:
            self.send(Message(msg.sender, r, msg.performative, msg.content, msg.timestamp))


# ─────────────────────────────────────────────
# Base Agent
# ─────────────────────────────────────────────

class BaseAgent:
    def __init__(self, agent_id: str, bus: MessageBus):
        self.agent_id = agent_id
        self.bus = bus
        bus.register(agent_id)
        self.log: List[str] = []

    def send(self, receiver: str, performative: str, content: Dict, t: float):
        self.bus.send(Message(self.agent_id, receiver, performative, content, t))

    def receive(self) -> List[Message]:
        return self.bus.receive(self.agent_id)

    def step(self, t: float):
        raise NotImplementedError


# ─────────────────────────────────────────────
# Layer 2: Machine Agent
# ─────────────────────────────────────────────

class MachineAgent(BaseAgent):
    """Monitors a physical machine and forwards anomaly reports."""

    FAILURE_ALERT_THRESHOLD = 0.05   # 5% probability triggers alert
    MAINTENANCE_WEAR_THRESHOLD = 0.6

    def __init__(self, machine: Machine, bus: MessageBus, predictor=None):
        super().__init__(f"MachineAgent_{machine.machine_id}", bus)
        self.machine = machine
        self.predictor = predictor   # XGBoost model (injected later)
        self._last_state = machine.state

    def step(self, t: float):
        m = self.machine
        msgs = self.receive()

        # Process incoming assignments
        for msg in msgs:
            if msg.performative == "ASSIGN" and m.state == MachineState.IDLE:
                job: Job = msg.content["job"]
                m.state = MachineState.WORKING
                m.current_job = job
                m.remaining_time = job.processing_times[job.current_op]
                m.current_load = np.random.uniform(0.6, 1.0)
                if job.start_time is None:
                    job.start_time = t

        # Predict failure and alert Maintenance Agent
        p_fail = m.failure_probability()
        if self.predictor is not None:
            feat = m.sensor_vector().reshape(1, -1)
            try:
                p_fail = float(self.predictor.predict_proba(feat)[0, 1])
            except Exception:
                pass

        if p_fail > self.FAILURE_ALERT_THRESHOLD and m.state == MachineState.WORKING:
            self.send("MaintenanceAgent", "INFORM", {
                "machine_id": m.machine_id,
                "p_failure": p_fail,
                "wear": m.wear_level,
                "temperature": m.temperature,
                "vibration": m.vibration,
            }, t)

        if m.wear_level > self.MAINTENANCE_WEAR_THRESHOLD and m.state == MachineState.IDLE:
            self.send("MaintenanceAgent", "REQUEST", {
                "type": "preventive",
                "machine_id": m.machine_id,
            }, t)

        # Report state change to Production Agent
        if m.state != self._last_state:
            self.send("ProductionAgent", "INFORM", {
                "machine_id": m.machine_id,
                "state": m.state.value,
            }, t)
            self._last_state = m.state


# ─────────────────────────────────────────────
# Layer 3: Maintenance Agent
# ─────────────────────────────────────────────

class MaintenanceAgent(BaseAgent):
    """Decides preventive vs. corrective maintenance scheduling."""

    def __init__(self, machines: List[Machine], bus: MessageBus):
        super().__init__("MaintenanceAgent", bus)
        self.machines = machines
        self.maintenance_queue: List[int] = []   # machine ids queued

    def step(self, t: float):
        msgs = self.receive()
        for msg in msgs:
            mid = msg.content.get("machine_id")
            if mid is None:
                continue
            m = self.machines[mid]

            if msg.performative == "INFORM":
                p = msg.content.get("p_failure", 0)
                if p > 0.15 and mid not in self.maintenance_queue:
                    self.maintenance_queue.append(mid)
                    self.log.append(f"t={t:.0f}: Scheduled preventive maint. M{mid} (p={p:.2f})")

            elif msg.performative == "REQUEST":
                if msg.content.get("type") == "corrective":
                    m.state = MachineState.FAILED
                    m.failure_count += 1
                    if m.current_job:
                        m.current_job = None
                    self.log.append(f"t={t:.0f}: Corrective repair M{mid}")

        # Execute scheduled maintenance on idle machines
        still_waiting = []
        for mid in self.maintenance_queue:
            m = self.machines[mid]
            if m.state == MachineState.IDLE:
                m.state = MachineState.MAINTENANCE
                m.remaining_time = 0.0
                self.send("GlobalAgent", "INFORM", {
                    "event": "maintenance_started",
                    "machine_id": mid,
                    "time": t,
                }, t)
            else:
                still_waiting.append(mid)
        self.maintenance_queue = still_waiting


# ─────────────────────────────────────────────
# Layer 3: Resource Agent (Contract Net Protocol)
# ─────────────────────────────────────────────

class ResourceAgent(BaseAgent):
    """
    Implements Contract Net Protocol for machine-job allocation.
    Acts as Manager: issues CFP, collects bids, awards contracts.
    """

    def __init__(self, machines: List[Machine], bus: MessageBus):
        super().__init__("ResourceAgent", bus)
        self.machines = machines
        self._pending_bids: Dict[int, Dict] = {}   # job_id -> best bid info
        self._awarded: Dict[int, int] = {}          # job_id -> machine_id

    def issue_cfp(self, job: Job, t: float):
        """Broadcast Call-For-Proposals to all capable machine agents."""
        required_machine = job.operations[job.current_op]
        self.send(
            f"MachineAgent_{required_machine}", "CFP", {
                "job_id": job.job_id,
                "job": job,
                "required_machine": required_machine,
                "due_date": job.due_date,
                "priority": job.priority,
                "t": t,
            }, t
        )

    def collect_and_award(self, job: Job, t: float) -> Optional[int]:
        """
        Score each candidate machine and award to best one.
        Score = utilization_balance + wear_penalty + urgency_bonus
        """
        required_machine = job.operations[job.current_op]
        m = self.machines[required_machine]

        if m.state != MachineState.IDLE:
            return None

        # Bid score: lower is better
        load_factor = m.total_working_time / max(1, t)
        wear_penalty = 2.0 * m.wear_level
        urgency = max(0, 1 - (job.due_date - t) / 120)  # urgency increases near deadline
        score = load_factor + wear_penalty - urgency * job.priority

        self._awarded[job.job_id] = required_machine
        m.state = MachineState.WORKING
        m.current_job = job
        m.remaining_time = job.processing_times[job.current_op]
        m.current_load = np.random.uniform(0.6, 1.0)
        if job.start_time is None:
            job.start_time = t
        return required_machine

    def step(self, t: float):
        msgs = self.receive()
        # Process any incoming proposal messages (extend for full CNP later)
        _ = msgs


# ─────────────────────────────────────────────
# Layer 3: Production Agent
# ─────────────────────────────────────────────

class ProductionAgent(BaseAgent):
    """
    Manages job sequencing using adaptive dispatching rules
    (ATCS: Apparent Tardiness Cost with Setups).
    """

    def __init__(self, bus: MessageBus):
        super().__init__("ProductionAgent", bus)
        self.dispatching_rule = "ATCS"

    def rank_jobs(self, queue: List[Job], t: float) -> List[Job]:
        if not queue:
            return []

        def atcs_score(job: Job) -> float:
            op_idx = job.current_op
            if op_idx >= len(job.processing_times):
                return -1e9
            pt = job.processing_times[op_idx]
            slack = max(0.001, job.due_date - t - pt)
            urgency = job.priority / pt
            tardiness_risk = 1.0 / slack
            return urgency + 2.0 * tardiness_risk

        return sorted(queue, key=atcs_score, reverse=True)

    def step(self, t: float):
        msgs = self.receive()
        for msg in msgs:
            if msg.performative == "INFORM" and "state" in msg.content:
                pass  # could log state transitions


# ─────────────────────────────────────────────
# Layer 4: Global Optimization Agent
# ─────────────────────────────────────────────

class GlobalOptimizationAgent(BaseAgent):
    """
    Coordinates all sub-agents, applies RL-based resource allocation policy,
    monitors KPIs, and reacts to disruptions.
    """

    def __init__(self, env: ManufacturingEnvironment, bus: MessageBus,
                 rl_policy=None, predictor=None):
        super().__init__("GlobalAgent", bus)
        self.env = env
        self.rl_policy = rl_policy       # RL agent (injected)
        self.predictor = predictor       # XGBoost (injected)

        self.machine_agents = [
            MachineAgent(m, bus, predictor) for m in env.machines
        ]
        self.maintenance_agent = MaintenanceAgent(env.machines, bus)
        self.resource_agent = ResourceAgent(env.machines, bus)
        self.production_agent = ProductionAgent(bus)

        self.kpi_history: List[Dict] = []
        self.event_log: List[str] = []

    def _dispatch(self, t: float) -> Dict[int, int]:
        """Select job-machine assignments for this time step."""
        ranked = self.production_agent.rank_jobs(self.env.job_queue, t)
        assignments = {}

        for job in ranked:
            op_idx = job.current_op
            if op_idx >= len(job.operations):
                continue
            required = job.operations[op_idx]
            machine = self.env.machines[required]
            if machine.state == MachineState.IDLE and job.job_id not in assignments.values():
                # Ask RL or use CNP fallback
                if self.rl_policy is not None:
                    state_vec = self._build_state(job, machine, t)
                    action = self.rl_policy.select_action(state_vec)
                    if action == 1:   # 1 = assign
                        assignments[job.job_id] = required
                else:
                    mid = self.resource_agent.collect_and_award(job, t)
                    if mid is not None:
                        assignments[job.job_id] = mid

        return assignments

    def _build_state(self, job: Job, machine: Machine, t: float) -> np.ndarray:
        op = job.current_op
        pt = job.processing_times[op] if op < len(job.processing_times) else 0
        slack = job.due_date - t
        return np.concatenate([
            machine.sensor_vector(),
            [pt, slack, job.priority, len(self.env.job_queue), t / self.env.sim_duration]
        ]).astype(np.float32)

    def run(self, scenario: str = "normal") -> List[Dict]:
        """Run a complete simulation scenario."""
        obs = self.env.reset()
        done = False
        step_count = 0

        while not done:
            t = self.env.time

            # Scenario injections
            if scenario == "machine_failure" and step_count == 200:
                self.env.machines[2].state = MachineState.FAILED
                self.env.machines[2].failure_count += 1
                self.event_log.append(f"t={t:.0f}: FORCED FAILURE on M2")

            if scenario == "demand_surge" and step_count == 300:
                self.env.trigger_demand_surge(20)
                self.event_log.append(f"t={t:.0f}: DEMAND SURGE +20 jobs")

            if scenario == "resource_shortage" and step_count == 250:
                self.env.trigger_resource_shortage()
                self.event_log.append(f"t={t:.0f}: RESOURCE SHORTAGE M0,M1")

            # Sub-agent steps
            for ma in self.machine_agents:
                ma.step(t)
            self.maintenance_agent.step(t)
            self.production_agent.step(t)
            self.resource_agent.step(t)

            # Dispatch assignments
            assignments = self._dispatch(t)

            # Step environment
            obs, reward, done, info = self.env.step(assignments)
            info["assignments"] = len(assignments)
            info["scenario"] = scenario
            self.kpi_history.append(info)
            step_count += 1

        return self.kpi_history
