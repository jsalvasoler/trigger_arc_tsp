import os
from typing import Any, Generator

import pytest

from trigger_arc_tsp.instance import Instance
from trigger_arc_tsp.utils import INSTANCES_DIR, SOLUTIONS_DIR


@pytest.fixture
def clean_test_file() -> Generator[Any, Any, Any]:
    file_path = os.path.join(SOLUTIONS_DIR, "examples/just_a_test.txt")
    # Remove the file if it exists before the test
    if os.path.exists(file_path):
        os.remove(file_path)
    yield file_path
    # Remove the file if it exists after the test
    if os.path.exists(file_path):
        os.remove(file_path)


def test_objective_computation_1() -> None:
    instance = Instance.load_instance_from_file(os.path.join(INSTANCES_DIR, "examples/example_2.txt"))

    tour_1 = [0, 2, 1, 4, 3]
    assert instance.compute_objective(tour_1) == 71.0


def test_objective_computation_2() -> None:
    instance = Instance.load_instance_from_file(os.path.join(INSTANCES_DIR, "examples/example_2.txt"))

    tour_2 = [0, 3, 2, 1, 4]
    assert instance.compute_objective(tour_2) == 62.0


def test_load_instance_from_file() -> None:
    instance = Instance.load_instance_from_file(os.path.join(INSTANCES_DIR, "examples/example_1.txt"))

    assert instance.N == 5
    assert instance.A == 8
    assert instance.R == 4

    assert instance.name == "examples/example_1.txt"


def test_solution_correctness_1() -> None:
    instance = Instance.load_instance_from_file(os.path.join(INSTANCES_DIR, "examples/example_1.txt"))

    # invalid because does not start at 0
    invalid_tour = [2, 1, 4, 3, 0]
    assert not instance.check_solution_correctness(invalid_tour)

    # invalid because of wrong length
    invalid_tour = [0, 2, 1, 4, 3, 2, 1]
    assert not instance.check_solution_correctness(invalid_tour)

    # invalid because of duplicate nodes
    invalid_tour = [0, 2, 1, 4, 3, 3]
    assert not instance.check_solution_correctness(invalid_tour)

    # invalid because of wrong node
    invalid_tour = [0, 2, 1, 4, 18]
    assert not instance.check_solution_correctness(invalid_tour)


def test_write_solution_to_file(clean_test_file: Generator[Any, Any, Any]) -> None:  # noqa: ARG001
    instance = Instance.load_instance_from_file(os.path.join(INSTANCES_DIR, "examples/example_1.txt"))

    invalid_tour = [2, 1, 4, 3]
    with pytest.raises(ValueError, match="Solution "):
        instance.save_solution(invalid_tour, 71.0)

    valid_tour = [0, 2, 1, 4, 3]
    instance.name = "examples/just_a_test.txt"
    instance.save_solution(valid_tour, 71.0)

    with open(os.path.join(SOLUTIONS_DIR, "examples/just_a_test.txt")) as file:
        line = file.readline()
        assert line.startswith("0,2,1,4,3 | 71.0 | ")
