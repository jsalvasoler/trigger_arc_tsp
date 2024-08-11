from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import os


class Instance:
    def __init__(self, N: int, edges: list, relations: dict) -> None:
        self.edges = edges
        self.relations = relations

        self.N = N
        self.nodes = set(range(self.N))
        self.A = len(edges)
        self.R = len(relations)

        self.delta_in = {node: set() for node in self.nodes}
        self.delta_out = {node: set() for node in self.nodes}
        for i, j in edges:
            self.delta_in[j].add(i)
            self.delta_out[i].add(j)

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
                relations[(int(from_arc), int(to_arc))] = (int(from_trigger), int(to_trigger), cost)

            assert int(line[0]) == R - 1

        return Instance(N, edges, relations)

    def __str__(self) -> str:
        return f"Instance(N={self.N}, A={self.A}, R={self.R})"
