from trigger_arc_tsp.instance import Instance
import os
from hexaly.optimizer import HexalyOptimizer


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'instances')

def arc_tsp_objective(instance: Instance, tour: list) -> int:
    pass


def model_and_solve_with_exaly() -> None:
    instance = Instance()
    instance.load_instance_from_file(os.path.join(
        DATA_DIR, 'instances_release_1', 'grf1.txt'))
    
    solver = HexalyOptimizer()
    model = solver.model


    edges = model.list(instance.N)
    model.count(edges) == instance.N

    edge_array = model.array(instance.edges.keys())
    edge_node_lambda_0 = model.lambda_function(lambda i: model.at(edge_array, i)[0])
    edge_node_lambda_1 = model.lambda_function(lambda i: model.at(edge_array, i)[1])

    selected_nodes_1 = model.array(edges, edge_node_lambda_0)
    selected_nodes_2 = model.array(edges, edge_node_lambda_1)

    model.count(selected_nodes_1) + model.count(selected_nodes_2) == instance.N

    visited_nodes = model.array(edges, edge_node_lambda_0)

    model.count(visited_nodes) == instance.N

    import random
    model.minimize(model.sum(random.uniform(0, 1) for _ in range(instance.N)))

    model.close()

    solver.param.time_limit = int(100)
    solver.solve()


if __name__ == '__main__':
    model_and_solve_with_exaly()

    