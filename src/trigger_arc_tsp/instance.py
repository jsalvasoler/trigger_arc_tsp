from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime

from trigger_arc_tsp.utils import SOLUTIONS_DIR


class Instance:
    def __init__(self, N: int, edges: list, relations: list, name: str) -> None:
        self.name = name

        self.edges = edges
        self.relations = relations

        self.N = N
        self.nodes = set(range(self.N))
        self.A = len(edges)
        self.R = len(self.relations)

        self.R_a = defaultdict(list)
        for b_0, b_1, a_0, a_1 in self.relations:
            self.R_a[(a_0, a_1)].append((b_0, b_1))

        self.delta_in = {node: set() for node in self.nodes}
        self.delta_out = {node: set() for node in self.nodes}
        for i, j in edges:
            self.delta_in[j].add(i)
            self.delta_out[i].add(j)

        self.rescale_costs_of_relations()

    def rescale_costs_of_relations(self) -> None:
        relative_costs = {rel: self.relations[rel] - self.edges[(rel[2], rel[3])] for rel in self.relations}
        max_rel_cost = max(relative_costs.values())
        if max_rel_cost > 0:
            for rel in self.relations:
                self.relations[rel] = relative_costs[rel] - max_rel_cost - 1
        self.offset = -max_rel_cost - 1

    @staticmethod
    def load_instance_from_file(file_path: os.PathLike) -> Instance:
        name = "/".join(file_path.split("/")[-2:])
        with open(file_path) as file:
            lines = file.readlines()
            N, A, R = map(int, lines[0].split())

            edges = {}
            for line in lines[1 : 1 + A]:
                _, i, j, w = map(float, line.split())
                edges[(int(i), int(j))] = w
            assert _ == A - 1

            relations = {}
            for line in lines[1 + A : 1 + A + R]:
                _, _, from_trigger, to_trigger, _, from_arc, to_arc, cost = map(float, line.split())
                relations[(int(from_trigger), int(to_trigger), int(from_arc), int(to_arc))] = cost

            assert int(line.split(" ")[0]) == R - 1

        return Instance(N, edges, relations, name)

    def compute_objective(self, tour: list) -> float:
        path = [(tour[i], tour[i + 1] if i < self.N - 1 else 0) for i in range(self.N)]
        path_set = set(path)
        assert len(path) == self.N
        assert path[0][0] == 0
        assert path[-1][1] == 0

        cost = sum(self.edges[a] for a in path)

        for a in path:
            if self.R_a.get(a):
                # find the relations that could trigger the arc a
                triggering = set(self.R_a[a]).intersection(path_set)
                if not triggering:
                    continue
                # sort the triggering relations by their index in the path (higher index last)
                triggering = sorted(triggering, key=lambda x: path.index(x))
                # remove the triggering arcs that happen after the arc a
                triggering = [rel for rel in triggering if path.index(rel) < path.index(a)]
                if not triggering:
                    continue
                # the last relation in the list is the one that triggers the arc a
                trigger_rel_cost = self.relations[*triggering[-1], *a] - self.offset
                # add the relative cost of the triggering relation to the cost of the arc a
                cost += trigger_rel_cost

        return cost

    def check_solution_correctness(self, tour: list) -> bool:
        if tour[-1] == 0:
            tour = tour[:-1]

        if len(tour) != self.N or len(set(tour)) != self.N or not set(tour).issubset(self.nodes):
            return False

        if tour[0] != 0:
            return False

        return True

    def test_solution(self, tour: list, proposed_objective: float) -> bool:
        if not self.check_solution_correctness(tour):
            return False

        cost = self.compute_objective(tour)

        return abs(cost - proposed_objective) < 1e-6

    def save_solution(self, tour: list, objective: float | None = None) -> None:
        if not self.check_solution_correctness(tour):
            msg = f"Solution {tour} is not a valid solution for instance {self.name}"
            raise ValueError(msg)

        if objective is None:
            objective = self.compute_objective(tour)
        if not self.test_solution(tour, objective):
            msg = f"Solution {tour} does not have the correct objective value for instance {self.name}"
            msg += f" (computed: {self.compute_objective(tour)}, proposed: {objective})"
            raise ValueError(msg)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # noqa: DTZ005
        os.makedirs(os.path.join(SOLUTIONS_DIR, *self.name.split("/")[:-1]), exist_ok=True)

        with open(os.path.join(SOLUTIONS_DIR, *self.name.split("/")), "a") as file:
            if file.tell() != 0:
                file.write("\n")
            assert tour[0] == 0
            assert tour[-1] != 0
            file.write(",".join(map(str, tour)))
            file.write(f" | {objective} | {timestamp}")

    def __str__(self) -> str:
        return f"Instance(N={self.N}, A={self.A}, R={self.R})"
