from __future__ import annotations

import os
import pickle
from collections import defaultdict
from datetime import datetime
from warnings import warn

from trigger_arc_tsp.gurobi_tsp_model import GurobiTSPModel
from trigger_arc_tsp.utils import CACHE_DIR, SOLUTIONS_DIR


class Instance:
    def __init__(self, N: int, edges: dict, relations: dict, name: str) -> None:
        self.name = name
        self.model_name = name.removesuffix(".txt") + ".mps"
        self.cache_file = os.path.join(
            CACHE_DIR, name.removesuffix(".txt") + "indices_cache.pkl"
        )

        os.makedirs(CACHE_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

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

        self.relations = {
            rel: self.relations[rel] - self.edges[(rel[2], rel[3])] for rel in self.relations
        }

        if not self._load_indices_from_cache():
            self._generate_z_var_indices()

    def _generate_z_var_indices(self) -> None:
        # print("Instance loading: Generating indices")
        # Optimize z_var_indices creation
        z_var_indices = set()

        for a, b_list in self.R_a.items():
            for b in b_list:
                if b != 0:
                    z_var_indices.add((*a, *b))  # (*a, *b)
                    z_var_indices.add((*b, *a))  # (*b, *a)
                for c in b_list:
                    if b != c and b not in (c, 0):
                        z_var_indices.add((*b, *c))  # (*b, *c)

        self.z_var_indices = list(z_var_indices)
        self._store_indices_in_cache()

    def _store_indices_in_cache(self) -> None:
        """Stores z_var_indices in a cache file using pickle."""
        # print("Instance loading: Storing indices in cache")
        if self.name == "test" or "tsp" in self.name:
            return
        with open(self.cache_file, "wb") as cache:
            pickle.dump(self.z_var_indices, cache)

    def _load_indices_from_cache(self) -> bool:
        """Loads z_var_indices from the cache file if it exists."""
        if self.name == "test" or "tsp" in self.name:
            return False

        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as cache:
                self.z_var_indices = pickle.load(cache)  # noqa: S301
                return True
        return False

    @staticmethod
    def load_instance_from_file(file_path: os.PathLike) -> Instance:
        name = "/".join(file_path.split("/")[-2:])
        print("reading instance file")
        with open(file_path) as file:
            lines = file.readlines()
            N, A, R = map(int, lines[0].split())

            edges = {}
            for line in lines[1 : 1 + A]:
                _, i, j, w = map(float, line.split())
                edges[(int(i), int(j))] = w
            # assert _ == A - 1

            relations = {}
            for line in lines[1 + A : 1 + A + R]:
                _, _, from_trigger, to_trigger, _, from_arc, to_arc, cost = map(
                    float, line.split()
                )
                relations[(int(from_trigger), int(to_trigger), int(from_arc), int(to_arc))] = (
                    cost
                )

            assert int(line.split(" ")[0]) == R - 1

            assert len(edges) == A
            assert len(relations) == R

        print("initialize Instance object")
        return Instance(N, edges, relations, name)

    def compute_objective(self, tour: list) -> float:
        path = [(tour[i], tour[i + 1] if i < self.N - 1 else 0) for i in range(self.N)]
        path_set = set(path)
        assert len(path) == self.N
        assert path[0][0] == 0
        assert path[-1][1] == 0

        cost = sum(self.edges[a] for a in path)

        for a in path:
            if not self.R_a.get(a):
                continue
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
            # add the relative cost of the triggering relation to the cost of the arc a
            cost += self.relations[*triggering[-1], *a]

        return cost

    def check_solution_correctness(self, tour: list) -> bool:
        if tour[-1] == 0:
            tour = tour[:-1]

        if (
            len(tour) != self.N
            or len(set(tour)) != self.N
            or not set(tour).issubset(self.nodes)
        ):
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
            msg = (
                f"Solution {tour} does not have the correct objective value for instance "
                f"{self.name}"
            )
            msg += f" (computed: {self.compute_objective(tour)}, proposed: {objective})"
            warn(msg, stacklevel=1)
            objective = self.compute_objective(tour)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # noqa: DTZ005
        os.makedirs(os.path.join(SOLUTIONS_DIR, *self.name.split("/")[:-1]), exist_ok=True)

        with open(os.path.join(SOLUTIONS_DIR, *self.name.split("/")), "a") as file:
            if file.tell() != 0:
                file.write("\n")
            assert tour[0] == 0, "Tour should start at 0"
            assert tour[-1] != 0, "Tour should not end at 0"
            file.write(",".join(map(str, tour)))
            file.write(f" | {objective} | {timestamp}")

    def __str__(self) -> str:
        return f"Instance(N={self.N}, A={self.A}, R={self.R})"

    def get_best_known_solution(self, idx: int = 0) -> list | None:
        filename = os.path.join(SOLUTIONS_DIR, *self.name.split("/"))
        if not os.path.exists(filename):
            return None

        sols = []
        with open(filename) as file:
            lines = file.readlines()
            for line in lines:
                tour, obj, _ = line.split(" | ")
                if tour not in {s[0] for s in sols}:
                    sols.append((tour, float(obj)))

        sols = sorted(sols, key=lambda x: x[1])
        if idx >= len(sols):
            error = f"Trying to get solution {idx} from {len(sols)} solutions"
            raise ValueError(error)
        tour = list(map(int, sols[idx][0].split(",")))
        print(f"Best known solution for {self.name}: {tour} with objective {sols[idx][1]}")

        return tour

    def get_mip_start(self, *, use_tsp_only: bool = False) -> list[dict, dict, dict, dict]:
        best_known_solution = self.get_best_known_solution()
        tour = (
            best_known_solution
            if best_known_solution and not use_tsp_only
            else self.tsp_solution()
        )
        return self.get_variables_from_tour(tour)

    def get_variables_from_tour(self, tour: list) -> list[dict, dict, dict, dict]:
        tour_edges = [(tour[i], tour[i + 1] if i < self.N - 1 else 0) for i in range(self.N)]
        assert len(tour_edges) == self.N
        x = {edge: 1 if edge in tour_edges else 0 for edge in self.edges}
        u = {node: i for i, node in enumerate(tour)}
        y = dict.fromkeys(self.relations, 0)
        for a in tour_edges:
            assert u[a[0]] < u[a[1]] or a[1] == 0
            if a not in self.R_a:
                continue
            # find the relations that could trigger the arc a
            triggering = set(self.R_a[a]).intersection(tour_edges)
            if not triggering:
                continue
            # sort the triggering relations by their index in the path (higher index last)
            triggering = sorted(triggering, key=lambda x: tour_edges.index(x))
            # remove the triggering arcs that happen after the arc a
            triggering = [
                rel for rel in triggering if tour_edges.index(rel) < tour_edges.index(a)
            ]
            if not triggering:
                continue
            # the last relation in the list is the one that triggers the arc a
            y[*triggering[-1], *a] = 1

        z = {
            (a_1, a_2, b_1, b_2): u[a_1] + 1 <= u[b_1]
            for a_1, a_2, b_1, b_2 in self.z_var_indices
        }

        return x, y, u, z

    def tsp_solution(self) -> list:
        model = GurobiTSPModel(self)
        model.formulate()
        model.solve_to_feasible_solution()
        return model.get_best_tour()
