from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Generator, Literal
from warnings import warn

import gurobipy as gp

from trigger_arc_tsp.gurobi_tsp_model import GurobiTSPModel
from trigger_arc_tsp.instance import Instance
from trigger_arc_tsp.utils import INSTANCES_DIR, cleanup_instance_name, fisher_yates_shuffle


class TSPPriorEval:
    def __init__(self, instance: Instance) -> None:
        self.instance = instance

    def evaluate_individual(self, tsp_prior: TSPPrior, time_limit_sec: int = 10) -> list[list, float]:
        assert len(tsp_prior.priorities) == self.instance.N

        try:
            best_cost = self.instance.compute_objective(tsp_prior.priorities)
            best_tour = tsp_prior.priorities
        except KeyError:
            best_cost = float("inf")
            best_tour = None

        tsp_edges = self.get_edges_for_tsp_search(tsp_prior=tsp_prior)
        tsp_instance = Instance(
            N=self.instance.N,
            edges=tsp_edges,
            relations={},
            name=self.instance.name,
        )
        self.model = GurobiTSPModel(tsp_instance)
        self.model.formulate()
        self.model.solve_to_optimality(logs=False, time_limit_sec=time_limit_sec)

        if self.model.get_model().Status == gp.GRB.Status.INTERRUPTED:
            raise KeyboardInterrupt

        tours = self.model.get_best_n_tours(15)
        for tour in tours:
            cost = self.instance.compute_objective(tour)
            if cost < best_cost:
                best_tour = tour
                best_cost = cost

        tsp_prior.cost = best_cost
        tsp_prior.best_tour = best_tour
        tsp_prior.rel_gap = self.model.get_model().MIPGap
        return best_tour, best_cost

    def compute_node_dist(self, node_priorities: list) -> dict:
        node_dist = {}
        for i, node_i in enumerate(node_priorities):
            for j, node_j in enumerate(node_priorities):
                if i == j:
                    node_dist[node_i, node_j] = 1
                else:
                    node_dist[node_i, node_j] = min(abs(i - j), self.instance.N - abs(i - j))
        return node_dist

    def get_edges_for_tsp_search(self, tsp_prior: TSPPrior) -> dict:
        # 1. Estimate the probability that an edge is used.
        # This is done with a probability inversely proportional to the
        # distance between the nodes in the node_priorities list

        node_dist = self.compute_node_dist(tsp_prior.priorities)

        edge_used_prob = {edge: 1 / node_dist[edge] for edge in self.instance.edges}

        # 2. Estimate the probability that a relation is active.
        # This is done considering (a) the probability that the trigger is used,
        # (b) the probability that the target is used, and (c) the closeness
        # between the trigger and the target
        relation_active_prob = {
            r: edge_used_prob[r[0], r[1]] * edge_used_prob[r[2], r[3]] * (1 / (node_dist[r[1], r[2]] ** tsp_prior.beta))
            for r in self.instance.relations
        }

        edges_cost = self.instance.edges.copy()
        for a in self.instance.R_a:
            for b in self.instance.R_a[a]:
                val = tsp_prior.alpha * relation_active_prob[*b, *a] * self.instance.relations[*b, *a]
                edges_cost[a] += val
                edges_cost[b] += val
        return edges_cost


@dataclass
class TSPPrior:
    priorities: list
    alpha: float
    beta: float
    cost: float = None
    rel_gap: float = None
    best_tour: list = None


class HeuristicSearch:
    def __init__(self, instance: Instance, search_type: Literal["randomized", "swap_2", "delay_1", "swap_3"]) -> None:
        self.instance = instance
        self.searcher = TSPPriorEval(self.instance)
        self.search_type = search_type

    def generate_n_random_permutations(self, n: int) -> Generator[list]:
        for _ in range(n):
            yield [0, *fisher_yates_shuffle(list(range(1, self.instance.N)))]

    def generate_swap_2_permutations(self, node_priorities: list) -> Generator[list]:
        for i in range(1, self.instance.N - 1):
            for j in range(i + 1, self.instance.N):
                new_prior = node_priorities.copy()
                new_prior[i], new_prior[j] = new_prior[j], new_prior[i]
                yield new_prior

    def generate_delay_1_node_permutations(self, node_priorities: list) -> Generator[list]:
        for i in range(1, self.instance.N - 1):
            for j in range(i + 1, self.instance.N):
                new_prior = (
                    node_priorities[:i]
                    + node_priorities[i + 1 : j + 1]
                    + [node_priorities[i]]
                    + node_priorities[j + 1 :]
                )
                assert len(new_prior) == len(node_priorities)
                assert new_prior[0] == 0
                assert new_prior[j] == node_priorities[i]
                yield new_prior

    def generate_swap_3_permutations(self, node_priorities: list) -> Generator[list]:
        for i in range(1, self.instance.N - 2):
            for j in range(i + 1, self.instance.N - 1):
                for k in range(j + 1, self.instance.N):
                    new_prior = node_priorities.copy()
                    new_prior[i], new_prior[j], new_prior[k] = new_prior[k], new_prior[i], new_prior[j]
                    yield new_prior

    def run(self, n_trials: int | None = None, n_post_trials: int = 10, idx: int = 0) -> None:
        best_tour = self.instance.get_best_known_solution(idx=idx)
        best_cost = self.instance.compute_objective(best_tour)

        if self.search_type == "randomized":
            assert n_trials is not None, "Number of trials must be specified for randomized search"
            tours_to_explore = self.generate_n_random_permutations(n_trials)
        elif self.search_type == "swap_2":
            n_tours_to_explore = (self.instance.N - 2) * (self.instance.N - 1) // 2
            if n_trials is None:
                warn(f"Total number of trials is set to {n_tours_to_explore}", stacklevel=1)
            n_trials = n_tours_to_explore
            tours_to_explore = self.generate_swap_2_permutations(best_tour)
        elif self.search_type == "swap_3":
            n_tours_to_explore = (self.instance.N - 3) * (self.instance.N - 2) * (self.instance.N - 1) // 6
            if n_trials is None:
                warn(f"Total number of trials is set to {n_tours_to_explore}", stacklevel=1)
            n_trials = n_tours_to_explore
            tours_to_explore = self.generate_swap_3_permutations(best_tour)
        elif self.search_type == "delay_1":
            n_tours_to_explore = (self.instance.N - 2) * (self.instance.N - 1) // 2
            if n_trials is None:
                warn(f"Total number of trials is set to {n_tours_to_explore}", stacklevel=1)
            n_trials = n_tours_to_explore
            tours_to_explore = self.generate_delay_1_node_permutations(best_tour)
        else:
            raise ValueError("Invalid search type")

        promising_tsp_priors = []

        it = 0
        try:
            for node_priorities in tours_to_explore:
                tsp_prior = TSPPrior(node_priorities, alpha=random.uniform(0.1, 3), beta=random.uniform(0.1, 3))
                tour, cost = self.searcher.evaluate_individual(tsp_prior=tsp_prior, time_limit_sec=5)
                if cost / best_cost - 1 < 0.05:
                    promising_tsp_priors.append(tsp_prior)
                if improved := cost < best_cost:
                    best_tour = tour
                    best_cost = cost
                    self.instance.save_solution(best_tour, best_cost)
                self.print_log_line(it, cost, best_cost, n_trials=n_trials, improved=improved)
                it += 1

        except KeyboardInterrupt:
            print("--- Interrupted by user ---")

        best_tour, best_cost = self.run_post_trials(promising_tsp_priors, best_cost, best_tour, n_post_trials)
        assert best_cost == self.instance.compute_objective(best_tour)

        print(f"Instnace: {self.instance.name}")
        print(f"Best tour: {best_tour} - Cost: {best_cost}")
        self.instance.save_solution(best_tour, best_cost)

    def run_post_trials(
        self, promising_tsp_priors: list[TSPPrior], best_cost: float, best_tour: list, n_post_trials: int
    ) -> None:
        print("\n--- Running post-trials ---")
        print(f"Promising TSP priors: {len(promising_tsp_priors)}")
        print(f"Number of evaluations: {n_post_trials}")
        print("---------------------------\n")
        evals = 0
        promising_tsp_priors = sorted(promising_tsp_priors, key=lambda x: x.cost)
        try:
            for tsp_prior in promising_tsp_priors:
                if tsp_prior.rel_gap < 0.001:
                    continue
                tour, cost = self.searcher.evaluate_individual(tsp_prior=tsp_prior, time_limit_sec=60)
                evals += 1
                if improved := cost < best_cost:
                    best_tour = tour
                    best_cost = cost
                    self.instance.save_solution(best_tour, best_cost)
                self.print_log_line(evals, cost, best_cost, n_trials=n_post_trials, improved=improved)
                if evals >= n_post_trials:
                    break
        except KeyboardInterrupt:
            print("--- Interrupted by user ---")

        assert best_cost == self.instance.compute_objective(best_tour)
        return best_tour, best_cost

    def print_log_line(self, it: int, cost: float | None, best_cost: float, n_trials: int, *, improved: bool) -> None:
        # Define the width for each column
        it_width = 15  # Width for the iteration number
        cost_width = 10  # Width for the cost
        best_cost_width = 10  # Width for the best cost

        # Format the string with specified widths
        it_string = f"{it} / {n_trials}"
        s = (
            f"Iteration {it_string.ljust(it_width)}"
            f"Cost: {cost:<{cost_width}.2f}"
            f"Best cost: {best_cost:<{best_cost_width}.2f}"
        )

        if improved:
            s += " (*)"

        print(s)


def heuristic_search(
    instance_name: str, search_type: Literal["randomized", "swap_2", "delay_1"], n_trials: int, n_post_trials: int
) -> None:
    instance_name = cleanup_instance_name(instance_name)
    instance = Instance.load_instance_from_file(os.path.join(INSTANCES_DIR, instance_name))
    prior_randomized_search = HeuristicSearch(instance, search_type=search_type)
    prior_randomized_search.run(n_trials=n_trials, n_post_trials=n_post_trials)


if __name__ == "__main__":
    heuristic_search(
        instance_name="instances_release_1/grf11.txt",
        n_trials=None,
        n_post_trials=None,
        search_type="swap_3",
    )
