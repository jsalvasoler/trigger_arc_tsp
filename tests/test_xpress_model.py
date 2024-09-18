import xpress as xp

from trigger_arc_tsp.instance import Instance
from trigger_arc_tsp.xpress_model import XpressModel


def test_xpress_model_sample_6() -> None:
    inst = Instance.load_instance_from_file("instances/examples/example_2.txt")
    model = XpressModel(inst)
    model.formulate(read_model=True)
    model.solve_model_with_parameters()

    assert model.get_model().attributes.solstatus == xp.SolStatus.OPTIMAL
    tour, cost = model.get_original_solution()

    assert inst.compute_objective(tour) == cost

    assert cost == 62.0
    assert tour == [0, 3, 2, 1, 4]

    assert inst.compute_objective([0, 3, 2, 1, 4, 0]) == 62.0


def test_xpress_model_sample_7() -> None:
    inst = Instance.load_instance_from_file("instances/examples/example_3.txt")
    model = XpressModel(inst)
    model.formulate(read_model=True)
    model.solve_model_with_parameters()

    assert model.get_model().attributes.solstatus == xp.SolStatus.OPTIMAL
    tour, cost = model.get_original_solution()

    assert inst.compute_objective(tour) == cost

    assert cost == 40.0
    assert tour == [0, 1, 2, 3, 4]

    assert inst.compute_objective([0, 1, 2, 3, 4]) == 40.0
    assert inst.compute_objective([0, 4, 3, 2, 1]) == 50.0


def test_xpress_model_sample_8() -> None:
    inst = Instance.load_instance_from_file("instances/examples/example_4.txt")
    model = XpressModel(inst)
    model.formulate(read_model=True)
    model.solve_model_with_parameters()

    assert model.get_model().attributes.solstatus == xp.SolStatus.OPTIMAL
    tour, cost = model.get_original_solution()

    assert inst.compute_objective(tour) == cost

    assert cost == 50.0
    assert tour == [0, 4, 3, 2, 1]

    assert inst.compute_objective([0, 4, 3, 2, 1]) == 50.0
    assert inst.compute_objective([0, 1, 2, 3, 4]) == 60.0


def test_solve_with_parameters() -> None:
    inst = Instance.load_instance_from_file("instances/examples/example_4.txt")
    model = XpressModel(inst)
    model.formulate(read_model=True)
    model.solve_model_with_parameters(time_limit_sec=1)
    assert model.get_model().getAttrib("time") <= 1.5

    model.solve_model_with_parameters(heuristic_effort=0.658)
    # check that heuristics = 0.658
    assert model.get_model().controls.heursearcheffort == 0.658 * 10


# def test_solve_grf5() -> None:
#     os.environ["XPAUTH_PATH"] = os.path.join(os.getcwd(), "xpauth.xpr")
#     inst = Instance.load_instance_from_file("instances/instances_release_1/grf5.txt")
#     from trigger_arc_tsp.gurobi_model import GurobiModel

#     model = XpressModel(inst)
#     # model = GurobiModel(inst)
#     model.formulate(read_model=True)
#     model.solve_model_with_parameters(1200)
