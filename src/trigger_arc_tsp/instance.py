from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import os


class Instance:
    def __init__(self, N: int, edges: list, relations: list) -> None:
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

        return Instance(N, edges, relations)

    def __str__(self) -> str:
        return f"Instance(N={self.N}, A={self.A}, R={self.R})"
