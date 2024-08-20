from __future__ import annotations

import os

from trigger_arc_tsp.gurobi_tsp_model import GurobiTSPModel
from trigger_arc_tsp.instance import Instance
from trigger_arc_tsp.utils import INSTANCES_DIR


def cleanup_instance_name(instance: str) -> str:
    if not instance.endswith(".txt"):
        error_msg = "Instance file must be a .txt file"
        raise ValueError(error_msg)
    if instance.startswith("instances/"):
        instance = instance[10:]

    if instance.count("/") != 1:
        instance = f"instances_release_1/{instance}"

    # check if file exists
    if not os.path.exists(os.path.join(INSTANCES_DIR, instance)):
        error_msg = f"Instance file {instance} not found"
        raise FileNotFoundError(error_msg)

    return instance


class TSPSearch:
    def __init__(self, instance: Instance) -> None:
        self.instance = instance

    def evaluate_individual(self, node_priorities: list, alpha: float = 1, beta: float = 0.5) -> list[list, float]:
        assert len(node_priorities) == self.instance.N

        tsp_edges = self.get_edges_for_tsp_search(node_priorities, alpha=alpha, beta=beta)
        tsp_instance = Instance(
            N=self.instance.N,
            edges=tsp_edges,
            relations={},
            name=self.instance.name,
        )
        self.model = GurobiTSPModel(tsp_instance)
        self.model.formulate()
        self.model.solve_to_optimality()
        tour = self.model.get_best_tour()
        cost = self.instance.compute_objective(tour)
        return tour, cost

    def compute_node_dist(self, node_priorities: list) -> dict:
        node_dist = {}
        for i, node_i in enumerate(node_priorities):
            for j, node_j in enumerate(node_priorities):
                if i == j:
                    node_dist[node_i, node_j] = 1
                else:
                    node_dist[node_i, node_j] = min(abs(i - j), self.instance.N - abs(i - j))
        return node_dist

    def get_edges_for_tsp_search(self, node_priorities: list, alpha: float = 1, beta: float = 0.5) -> dict:
        # 1. Estimate the probability that an edge is used.
        # This is done with a probability inversely proportional to the
        # distance between the nodes in the node_priorities list

        node_dist = self.compute_node_dist(node_priorities)

        edge_used_prob = {edge: 1 / node_dist[edge] for edge in self.instance.edges}

        # 2. Estimate the probability that a relation is active.
        # This is done considering (a) the probability that the trigger is used,
        # (b) the probability that the target is used, and (c) the closeness
        # between the trigger and the target
        relation_active_prob = {
            r: edge_used_prob[r[0], r[1]] * edge_used_prob[r[2], r[3]] * (1 / (node_dist[r[1], r[2]] ** beta))
            for r in self.instance.relations
        }

        edges_cost = self.instance.edges.copy()
        for a in self.instance.R_a:
            for b in self.instance.R_a[a]:
                val = alpha * relation_active_prob[*b, *a] * self.instance.relations[*b, *a]
                edges_cost[a] += val
                edges_cost[b] += val
        return edges_cost
