"""
Baseline scheduling methods for comparison:
  1. Rule-Based (FIFO / EDD)
  2. Genetic Algorithm (GA)
  3. Centralized Control (greedy makespan minimization)
"""

import numpy as np
import random
from typing import List, Dict, Optional
from environment import ManufacturingEnvironment, Job, MachineState

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────
# 1. Rule-Based Controller (FIFO + EDD)
# ─────────────────────────────────────────────

class RuleBasedController:
    """
    Earliest Due Date (EDD) dispatching on idle machines.
    No predictive maintenance – reacts only after failures.
    """
    name = "Rule-Based (EDD)"

    def dispatch(self, env: ManufacturingEnvironment) -> Dict[int, int]:
        assignments = {}
        queue_sorted = sorted(env.job_queue, key=lambda j: j.due_date)
        for job in queue_sorted:
            op = job.current_op
            if op >= len(job.operations):
                continue
            mid = job.operations[op]
            m = env.machines[mid]
            if m.state == MachineState.IDLE and job.job_id not in assignments:
                assignments[job.job_id] = mid
        return assignments

    def run(self, env: ManufacturingEnvironment, scenario: str = "normal") -> List[Dict]:
        env.reset()
        done = False
        kpis = []
        step = 0
        while not done:
            if scenario == "machine_failure" and step == 200:
                env.machines[2].state = MachineState.FAILED
                env.machines[2].failure_count += 1
            if scenario == "demand_surge" and step == 300:
                env.trigger_demand_surge(20)
            if scenario == "resource_shortage" and step == 250:
                env.trigger_resource_shortage()
            a = self.dispatch(env)
            _, _, done, info = env.step(a)
            kpis.append(info)
            step += 1
        return kpis


# ─────────────────────────────────────────────
# 2. Genetic Algorithm
# ─────────────────────────────────────────────

class GeneticAlgorithmScheduler:
    """
    Offline GA that pre-computes a job-to-machine assignment permutation.
    Each chromosome is an ordered list of (job_id, machine_id) pairs.
    Decoded greedily at simulation time.
    """
    name = "Genetic Algorithm"

    def __init__(self, pop_size: int = 40, generations: int = 50,
                 cx_prob: float = 0.8, mut_prob: float = 0.15):
        self.pop_size = pop_size
        self.generations = generations
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self._schedule: List[int] = []   # job priority order

    def _fitness(self, order: List[int], jobs: List[Job], n_machines: int) -> float:
        """Estimate makespan using list-scheduling on machine time model."""
        machine_end = [0.0] * n_machines
        job_end = {}
        for jid in order:
            job = next((j for j in jobs if j.job_id == jid), None)
            if job is None:
                continue
            t = 0.0
            for op_idx, mid in enumerate(job.operations):
                pt = job.processing_times[op_idx]
                start = max(t, machine_end[mid])
                machine_end[mid] = start + pt
                t = start + pt
            job_end[jid] = t
        makespan = max(machine_end)
        total_tard = sum(max(0, job_end.get(j.job_id, 0) - j.due_date)
                         for j in jobs)
        return makespan + 0.1 * total_tard

    def evolve(self, jobs: List[Job], n_machines: int):
        job_ids = [j.job_id for j in jobs]
        pop = [random.sample(job_ids, len(job_ids)) for _ in range(self.pop_size)]
        best_fit = float("inf")
        best_chrom = pop[0]

        for gen in range(self.generations):
            fits = [self._fitness(c, jobs, n_machines) for c in pop]
            ranked = sorted(zip(fits, pop), key=lambda x: x[0])
            fits_sorted, pop_sorted = zip(*ranked)
            pop_sorted = list(pop_sorted)
            best_fit = fits_sorted[0]
            best_chrom = pop_sorted[0]

            # Elitism: keep top 2
            new_pop = pop_sorted[:2]
            while len(new_pop) < self.pop_size:
                p1, p2 = random.choices(pop_sorted[:20], k=2)
                if random.random() < self.cx_prob:
                    cut = random.randint(1, len(p1) - 1)
                    child = p1[:cut] + [x for x in p2 if x not in p1[:cut]]
                else:
                    child = p1[:]
                if random.random() < self.mut_prob:
                    i, j = random.sample(range(len(child)), 2)
                    child[i], child[j] = child[j], child[i]
                new_pop.append(child)
            pop = new_pop

        self._schedule = best_chrom

    def run(self, env: ManufacturingEnvironment, scenario: str = "normal") -> List[Dict]:
        env.reset()
        # Evolve on initial job set
        self.evolve(env.all_jobs[:env.n_jobs_total], env.N_MACHINES)
        schedule_idx = {jid: idx for idx, jid in enumerate(self._schedule)}

        done = False
        kpis = []
        step = 0
        while not done:
            if scenario == "machine_failure" and step == 200:
                env.machines[2].state = MachineState.FAILED
            if scenario == "demand_surge" and step == 300:
                env.trigger_demand_surge(20)
            if scenario == "resource_shortage" and step == 250:
                env.trigger_resource_shortage()

            queue_sorted = sorted(
                env.job_queue,
                key=lambda j: schedule_idx.get(j.job_id, 9999)
            )
            assignments = {}
            for job in queue_sorted:
                op = job.current_op
                if op >= len(job.operations):
                    continue
                mid = job.operations[op]
                m = env.machines[mid]
                if m.state == MachineState.IDLE:
                    assignments[job.job_id] = mid
            _, _, done, info = env.step(assignments)
            kpis.append(info)
            step += 1
        return kpis


# ─────────────────────────────────────────────
# 3. Centralized Control (greedy / myopic)
# ─────────────────────────────────────────────

class CentralizedController:
    """
    Centralized greedy: assigns available jobs to minimize estimated job finish time.
    No agent autonomy, no predictive maintenance.
    """
    name = "Centralized Control"

    def dispatch(self, env: ManufacturingEnvironment) -> Dict[int, int]:
        assignments = {}
        for job in env.job_queue:
            op = job.current_op
            if op >= len(job.operations):
                continue
            mid = job.operations[op]
            m = env.machines[mid]
            if m.state == MachineState.IDLE:
                assignments[job.job_id] = mid
        return assignments

    def run(self, env: ManufacturingEnvironment, scenario: str = "normal") -> List[Dict]:
        env.reset()
        done = False
        kpis = []
        step = 0
        while not done:
            if scenario == "machine_failure" and step == 200:
                env.machines[2].state = MachineState.FAILED
            if scenario == "demand_surge" and step == 300:
                env.trigger_demand_surge(20)
            if scenario == "resource_shortage" and step == 250:
                env.trigger_resource_shortage()
            a = self.dispatch(env)
            _, _, done, info = env.step(a)
            kpis.append(info)
            step += 1
        return kpis
