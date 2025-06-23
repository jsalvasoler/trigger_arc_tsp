import os

from hexaly.optimizer import HexalyOptimizer

from trigger_arc_tsp.instance import Instance

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "instances"
)

M = 1e5  # A large constant to represent "no edge" or "no relation"


def prepare_hexaly_data(instance: Instance, model: HexalyOptimizer):
    edge_exists = [[0] * instance.N for _ in range(instance.N)]
    edge_cost = [[M] * instance.N for _ in range(instance.N)]
    for (i, j), cost in instance.edges.items():
        edge_exists[i][j] = 1
        edge_cost[i][j] = cost
    
    # relations will be accessed as relations[a_1][a_2][b_1][b_2] = instance.R[a_1, a_2, b_1, b_2]
    # this means that a triggers b. If no relation, we set to M, so they will be avoided
    relations = [[[[
        M for _ in range(instance.N)
    ] for _ in range(instance.N)] for _ in range(instance.N)] for _ in range(instance.N)]
    assert relations[0][0][0][0] == M, "relations[0][0][0][0] should be M"

    for (a_1, a_2, b_1, b_2), value in instance.relations.items():
        relations[a_1][a_2][b_1][b_2] = value

    return {
        "edge_exists": model.array(edge_exists),
        "edge_cost": model.array(edge_cost),
        "relations": model.array(relations),
    }

def solve_with_hexaly_v1(instance: Instance, time_limit: int, output_file: str | None):
    """
    Let's try to model as a TSP with valid edges
    """
    with HexalyOptimizer() as optimizer:
        model = optimizer.model

        data = prepare_hexaly_data(instance, model)

        # sequence of nodes
        node_sequence = model.list(instance.N)

        # select N edges
        model.constraint(
            model.count(node_sequence) == instance.N
        )
        model.constraint(
            node_sequence[0] == 0  # start at node 0
        )

        # THIS WORKS FOR TSP
        cost_lambda = model.lambda_function(
            lambda i: data["edge_cost"][node_sequence[i]][node_sequence[i + 1]]
        )
        total_cost = model.sum(
            model.range(instance.N - 1), cost_lambda
        ) + data["edge_cost"][node_sequence[instance.N - 1]][node_sequence[0]]

        def mycost(tour):
            return instance.compute_objective_safe(tour)
        
        func_mycost = model.create_double_external_function(mycost)
        model.minimize(total_cost)

        model.close()

        # Parameterize the optimizer with a time limit.
        optimizer.param.time_limit = time_limit
        optimizer.solve()

        # Print the solution
        print("Total Cost:", total_cost.value)

        tour = "Tour: "
        for x in node_sequence.value:
            tour += str(x) + " -> "
        print(tour + str(node_sequence.value[0]))  # complete the tour by returning to the start


def model_and_solve_with_exaly() -> None:
    instance = Instance.load_instance_from_file(os.path.join(DATA_DIR, "instances_release_1", "grf17.txt"))

    time_limit = 60  # seconds
    output_file = "solution.txt"

    solve_with_hexaly_v1(instance, time_limit, output_file)


    from gurobi_tsp_model import GurobiTSPModel
    gurobi_model = GurobiTSPModel(instance)
    gurobi_model.formulate()
    gurobi_model.solve_to_optimality(time_limit_sec=60)

if __name__ == "__main__":
    model_and_solve_with_exaly()
