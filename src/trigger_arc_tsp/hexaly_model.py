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
    global idx_to_edge
    idx_to_edge = [[u, v] for u, v in instance.edges]
    idx_to_relations = [instance.R_a[idx] for idx in range(instance.A)]
    idx_to_cost = [cost for cost in instance.edges.values()]
    edge_to_idx = [[-1] * instance.N for _ in range(instance.N)]
    edge_to_relations = [[[] for _ in range(instance.N)] for _ in range(instance.N)]
    for idx, ((i, j), cost) in enumerate(instance.edges.items()):
        edge_exists[i][j] = 1
        edge_cost[i][j] = cost
        edge_to_idx[i][j] = idx
        edge_to_relations[i][j] = instance.R_a[i, j]
        idx_to_relations[idx] = instance.R_a[i, j]
    
    idx_to_relations_idx = [[] for _ in range(instance.A)]
    for idx, (i, j) in enumerate(instance.edges):
        idx_to_relations_idx[idx] = [edge_to_idx[r[0]][r[1]] for r in instance.R_a[i, j]]

    relations = [[-1 for _ in range(instance.A)] for _ in range(instance.A)]
    for ((a, b, c, d), cost) in instance.relations.items():
        relations[edge_to_idx[a][b]][edge_to_idx[c][d]] = cost

    return {
        # "edge_exists": model.array(edge_exists),
        # "edge_cost": model.array(edge_cost),
        "idx_to_edge": model.array(idx_to_edge),
        "idx_to_cost": model.array(idx_to_cost),
        "edge_to_idx": model.array(edge_to_idx),
        "relations": model.array(relations),
        # "edge_to_relations": model.array(edge_to_relations),
        # "idx_to_relations": model.array(idx_to_relations),
        "idx_to_relations_idx": model.array(idx_to_relations_idx),
    }

def solve_with_hexaly_v1(instance: Instance, time_limit: int, output_file: str | None):
    """
    Let's try to model as a TSP with valid edges
    """
    with HexalyOptimizer() as optimizer:
        model = optimizer.model

        data = prepare_hexaly_data(instance, model)

        # # sequence of nodes
        node_sequence = model.list(instance.N)
        # select N nodes
        model.constraint(
            model.count(node_sequence) == instance.N
        )
        model.constraint(
            node_sequence[0] == 0  # start at node 0
        )
        
        # # sequence of edges by edge index
        # edge_sequence = model.list(instance.A)
        # # select N edges
        # model.constraint(
        #     model.count(edge_sequence) == instance.N
        # )

        # # make sure that the edge sequence is subordinate to the node sequence
        # model.constraint(
        #     model.and_(model.range(0, model.count(edge_sequence)), 
        #                model.lambda_function(lambda i: edge_sequence[i] == path_ids[i]))
        # )

        path_ids = [data["edge_to_idx"][node_sequence[i]][node_sequence[i + 1] if i < instance.N - 1 else 0] for i in range(instance.N)]
        path_ids = model.array(path_ids)
        # total_cost = instance.compute_objective(path)
        cost = model.sum(model.range(instance.N),
                         model.lambda_function(lambda i: data["idx_to_cost"][path_ids[i]]))

        for i in range(instance.N): # for arcs 0 .. i .. N-1
            # we need to find the last relation that triggers the arc
            trigger = [
                j for j in range(i, 0, -1)
                if data["relations"][path_ids[i]][path_ids[j]] != -1
            ]
            

            # for j in range(i, 0, -1):  # for arcs i down to 1
            #     # add all the relations that could trigger the arc
        
            # if model.count(data["edge_to_relations"][a[0]][a[1]]) == 0:
            #     continue
            # find the relations that could trigger the arc a
            # triggering = model.intersection(data["edge_to_relations"][a[0]][a[1]], path_set)
            # triggering = model.array(triggering)
            # # if model.count(triggering) == 0:
            # #     continue
            # triggering_sorted = model.sort(triggering, sort_lambda)


        # # THIS WORKS FOR TSP with node sequence
        # cost_lambda = model.lambda_function(
        #     lambda i: data["edge_cost"][node_sequence[i]][node_sequence[i + 1]]
        # )
        # total_cost = model.sum(
        #     model.range(instance.N - 1), cost_lambda
        # ) + data["edge_cost"][node_sequence[instance.N - 1]][node_sequence[0]]

        total_cost = cost
        model.minimize(total_cost)

        model.close()

        # Parameterize the optimizer with a time limit.
        optimizer.param.time_limit = 5
        optimizer.solve()

        # Print the solution
        print("Total Cost:", total_cost.value)

        tour = "Tour: "
        for x in node_sequence.value:
            tour += str(x) + " -> "
        print(tour + str(node_sequence.value[0]))  # complete the tour by returning to the start

        # for id_ in edge_sequence.value:
        #     print(idx_to_edge[id_])


def model_and_solve_with_exaly() -> None:
    instance = Instance.load_instance_from_file(os.path.join(DATA_DIR, "instances_release_1", "grf1.txt"))

    time_limit = 60  # seconds
    output_file = "solution.txt"

    solve_with_hexaly_v1(instance, time_limit, output_file)


    # from gurobi_tsp_model import GurobiTSPModel
    # gurobi_model = GurobiTSPModel(instance)
    # gurobi_model.formulate()
    # gurobi_model.solve_to_optimality(time_limit_sec=60)

if __name__ == "__main__":
    model_and_solve_with_exaly()
