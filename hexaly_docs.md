
Main example

```python
import hexaly.optimizer

with hexaly.optimizer.HexalyOptimizer() as optimizer:
    weights = [10, 60, 30, 40, 30, 20, 20, 2]
    values = [1, 10, 15, 40, 60, 90, 100, 15]
    knapsack_bound = 102

    #
    # Declare the optimization model
    #
    model = optimizer.model

    # 0-1 decisions
    x = [model.bool() for _ in range(8)]

    # Weight constraint
    knapsack_weight = model.sum(weights[i] * x[i] for i in range(8))
    model.constraint(knapsack_weight <= knapsack_bound)

    # Maximize value
    knapsack_value = model.sum(values[i] * x[i] for i in range(8))
    model.maximize(knapsack_value)

    model.close()

    # Parameterize the optimizer
    optimizer.param.time_limit = 10

    optimizer.solve()

```

Knapsack

```python
import hexaly.optimizer
import sys

if len(sys.argv) < 2:
    print("Usage: python knapsack.py inputFile [outputFile] [timeLimit]")
    sys.exit(1)


def read_integers(filename):
    with open(filename) as f:
        return [int(elem) for elem in f.read().split()]


with hexaly.optimizer.HexalyOptimizer() as optimizer:
    #
    # Read instance data
    #
    file_it = iter(read_integers(sys.argv[1]))

    # Number of items
    nb_items = next(file_it)

    # Items properties
    weights = [next(file_it) for i in range(nb_items)]
    values = [next(file_it) for i in range(nb_items)]

    # Knapsack bound
    knapsack_bound = next(file_it)

    #
    # Declare the optimization model
    #
    model = optimizer.model

    # Decision variables x[i]
    x = [model.bool() for i in range(nb_items)]

    # Weight constraint
    knapsack_weight = model.sum(x[i] * weights[i] for i in range(nb_items))
    model.constraint(knapsack_weight <= knapsack_bound)

    # Maximize value
    knapsack_value = model.sum(x[i] * values[i] for i in range(nb_items))
    model.maximize(knapsack_value)

    model.close()

    # Parameterize the optimizer
    if len(sys.argv) >= 4:
        optimizer.param.time_limit = int(sys.argv[3])
    else:
        optimizer.param.time_limit = 20

    optimizer.solve()

    #
    # Write the solution in a file
    #
    if len(sys.argv) >= 3:
        with open(sys.argv[2], 'w') as f:
            f.write("%d\n" % knapsack_value.value)
            for i in range(nb_items):
                if x[i].value != 1:
                    continue
                f.write("%d " % i)
            f.write("\n")
```

Curve fitting

```python
import hexaly.optimizer
import sys


def read_float(filename):
    with open(filename) as f:
        return [float(elem) for elem in f.read().split()]

#
# Read instance data
#


def read_instance(instance_file):
    file_it = iter(read_float(instance_file))

    # Number of observations
    nb_observations = int(next(file_it))

    # Inputs and outputs
    inputs = []
    outputs = []
    for i in range(nb_observations):
        inputs.append(next(file_it))
        outputs.append(next(file_it))

    return nb_observations, inputs, outputs


def main(instance_file, output_file, time_limit):
    nb_observations, inputs, outputs = read_instance(instance_file)

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Decision variables (parameters of the mapping function)
        a = model.float(-100, 100)
        b = model.float(-100, 100)
        c = model.float(-100, 100)
        d = model.float(-100, 100)

        # Minimize square error between prediction and output
        predictions = [a * model.sin(b - inputs[i]) + c * inputs[i] ** 2 + d
                       for i in range(nb_observations)]
        errors = [predictions[i] - outputs[i] for i in range(nb_observations)]
        square_error = model.sum(model.pow(errors[i], 2) for i in range(nb_observations))
        model.minimize(square_error)

        model.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = time_limit

        optimizer.solve()

        #
        # Write the solution in a file
        #
        if output_file is not None:
            with open(output_file, 'w') as f:
                f.write("Optimal mapping function\n")
                f.write("a = " + str(a.value) + "\n")
                f.write("b = " + str(b.value) + "\n")
                f.write("c = " + str(c.value) + "\n")
                f.write("d = " + str(d.value) + "\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python curve_fitting.py instance_file [output_file] [time_limit]")
        sys.exit(1)

    instance_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 else None
    time_limit = int(sys.argv[3]) if len(sys.argv) >= 4 else 3
    main(instance_file, output_file, time_limit)

```

Facility location

```python
import hexaly.optimizer
import sys


def read_float(filename):
    with open(filename) as f:
        return [float(elem) for elem in f.read().split()]

#
# Read instance data
#


def read_instance(instance_file):
    file_it = iter(read_float(instance_file))

    # Number of observations
    nb_observations = int(next(file_it))

    # Inputs and outputs
    inputs = []
    outputs = []
    for i in range(nb_observations):
        inputs.append(next(file_it))
        outputs.append(next(file_it))

    return nb_observations, inputs, outputs


def main(instance_file, output_file, time_limit):
    nb_observations, inputs, outputs = read_instance(instance_file)

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Decision variables (parameters of the mapping function)
        a = model.float(-100, 100)
        b = model.float(-100, 100)
        c = model.float(-100, 100)
        d = model.float(-100, 100)

        # Minimize square error between prediction and output
        predictions = [a * model.sin(b - inputs[i]) + c * inputs[i] ** 2 + d
                       for i in range(nb_observations)]
        errors = [predictions[i] - outputs[i] for i in range(nb_observations)]
        square_error = model.sum(model.pow(errors[i], 2) for i in range(nb_observations))
        model.minimize(square_error)

        model.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = time_limit

        optimizer.solve()

        #
        # Write the solution in a file
        #
        if output_file is not None:
            with open(output_file, 'w') as f:
                f.write("Optimal mapping function\n")
                f.write("a = " + str(a.value) + "\n")
                f.write("b = " + str(b.value) + "\n")
                f.write("c = " + str(c.value) + "\n")
                f.write("d = " + str(d.value) + "\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python curve_fitting.py instance_file [output_file] [time_limit]")
        sys.exit(1)

    instance_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 else None
    time_limit = int(sys.argv[3]) if len(sys.argv) >= 4 else 3
    main(instance_file, output_file, time_limit)
```

Smalles circle

```python
import hexaly.optimizer
import sys

if len(sys.argv) < 2:
    print("Usage: python smallest_circle.py inputFile [outputFile] [timeLimit]")
    sys.exit(1)


def read_integers(filename):
    with open(filename) as f:
        return [int(elem) for elem in f.read().split()]


with hexaly.optimizer.HexalyOptimizer() as optimizer:
    #
    # Read instance data
    #
    file_it = iter(read_integers(sys.argv[1]))
    # Number of points
    nb_points = next(file_it)

    # Point coordinates
    coord_x = [None] * nb_points
    coord_y = [None] * nb_points

    coord_x[0] = next(file_it)
    coord_y[0] = next(file_it)

    # Minimum and maximum value of the coordinates of the points
    min_x = coord_x[0]
    max_x = coord_x[0]
    min_y = coord_y[0]
    max_y = coord_y[0]

    for i in range(1, nb_points):
        coord_x[i] = next(file_it)
        coord_y[i] = next(file_it)
        if coord_x[i] < min_x:
            min_x = coord_x[i]
        else:
            if coord_x[i] > max_x:
                max_x = coord_x[i]
        if coord_y[i] < min_y:
            min_y = coord_y[i]
        else:
            if coord_y[i] > max_y:
                max_y = coord_y[i]

    #
    # Declare the optimization model
    #
    model = optimizer.model

    # x, y are respectively the abscissa and the ordinate of the origin of the circle
    x = model.float(min_x, max_x)
    y = model.float(min_y, max_y)

    # Distance between the origin and the point i
    radius = [(x - coord_x[i]) ** 2 + (y - coord_y[i]) ** 2 for i in range(nb_points)]

    # Minimize the radius r
    r = model.sqrt(model.max(radius))
    model.minimize(r)

    model.close()

    # Parameterize the optimizer
    if len(sys.argv) >= 4:
        optimizer.param.time_limit = int(sys.argv[3])
    else:
        optimizer.param.time_limit = 6

    optimizer.solve()

    #
    # Write the solution in a file
    #
    if len(sys.argv) >= 3:
        with open(sys.argv[2], 'w') as f:
            f.write("x=%f\n" % x.value)
            f.write("y=%f\n" % y.value)
            f.write("r=%f\n" % r.value)
```


Branin Function

```python
import hexaly.optimizer
import sys

with hexaly.optimizer.HexalyOptimizer() as optimizer:
    # Parameters of the function
    PI = 3.14159265359
    a = 1
    b = 5.1 / (4 * pow(PI, 2))
    c = 5 / PI
    r = 6
    s = 10
    t = 1 / (8 * PI)

    #
    # Declare the optimization model
    #
    model = optimizer.model

    # Numerical decisions
    x1 = model.float(-5.0, 10.0)
    x2 = model.float(0.0, 15.0)

    # f = a(x2 - b*x1^2 + c*x1 - r)^2 + s(1-t)cos(x1) + s
    f = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * model.cos(x1) + s

    # Minimize f
    model.minimize(f)

    model.close()

    # Parameterize the optimizer
    if len(sys.argv) >= 3:
        optimizer.param.time_limit = int(sys.argv[2])
    else:
        optimizer.param.time_limit = 6

    optimizer.solve()

    #
    # Write the solution in a file
    #
    if len(sys.argv) >= 2:
        with open(sys.argv[1], 'w') as f:
            f.write("x1=%f\n" % x1.value)
            f.write("x2=%f\n" % x2.value)

```

Max-Cut

```python
import hexaly.optimizer
import sys

def read_integers(filename):
    with open(filename) as f:
        return [int(elem) for elem in f.read().split()]

#
# Read instance data
#
def read_instance(filename):
    file_it = iter(read_integers(filename))
    # Number of vertices
    n = next(file_it)
    # Number of edges
    m = next(file_it)

    # Origin of each edge
    origin = [None] * m
    # Destination of each edge
    dest = [None] * m
    # Weight of each edge
    w = [None] * m

    for e in range(m):
        origin[e] = next(file_it)
        dest[e] = next(file_it)
        w[e] = next(file_it)
    
    return n, m, origin, dest, w

def main(instance_file, output_file, time_limit):
    n, m, origin, dest, w = read_instance(instance_file)

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Decision variables x[i]
        # True if vertex x[i] is on the right side of the cut
        # and false if it is on the left side of the cut
        x = [model.bool() for i in range(n)]

        # An edge is in the cut-set if it has an extremity in each class of the bipartition
        incut = [None] * m
        for e in range(m):
            incut[e] = model.neq(x[origin[e] - 1], x[dest[e] - 1])

        # Size of the cut
        cut_weight = model.sum(w[e] * incut[e] for e in range(m))
        model.maximize(cut_weight)

        model.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = time_limit

        optimizer.solve()

        #
        # Write the solution in a file with the following format:
        #  - objective value
        #  - each line contains a vertex number and its subset (1 for S, 0 for V-S)
        #
        if output_file != None:
            with open(output_file, 'w') as f:
                f.write("%d\n" % cut_weight.value)
                # Note: in the instances the indices start at 1
                for i in range(n):
                    f.write("%d %d\n" % (i + 1, x[i].value))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python maxcut.py inputFile [outputFile] [timeLimit]")
        sys.exit(1)

    instance_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 else None
    time_limit = int(sys.argv[3]) if len(sys.argv) >= 4 else 10
    main(instance_file, output_file, time_limit)

```

Hosaki Function

```python
import hexaly.optimizer
import sys
import math

#
# External function
#
def hosaki_function(argument_values):
    x1 = argument_values[0]
    x2 = argument_values[1]
    return ((1 - 8 * x1 + 7 * pow(x1, 2) - 7 * pow(x1, 3) / 3 + pow(x1, 4) / 4) 
            * pow(x2, 2) * math.exp(-x2))


def main(evaluation_limit, output_file):
    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Numerical decisions
        x1 = model.float(0, 5)
        x2 = model.float(0, 6)

        # Create and call the function
        f = model.create_double_external_function(hosaki_function)
        func_call = model.call(f, x1, x2)

        # Enable surrogate modeling
        surrogate_params = f.external_context.enable_surrogate_modeling()

        # Minimize function call
        model.minimize(func_call)
        model.close()

        # Parameterize the optimizer
        surrogate_params.evaluation_limit = evaluation_limit

        optimizer.solve()

        # Write the solution in a file
        if output_file is not None:
            with open(output_file, 'w') as f:
                f.write("obj=%f\n" % func_call.value)
                f.write("x1=%f\n" % x1.value)
                f.write("x2=%f\n" % x2.value)


if __name__ == '__main__':
    output_file = sys.argv[1] if len(sys.argv) > 1 else None
    evaluation_limit = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    main(evaluation_limit, output_file)

```

Order Picking

```python

import hexaly.optimizer
import sys

def read_elem(filename) :
    with open(filename) as f :
        return [str(elem) for elem in f.read().split()]

def read_instance(filename) :
    file_iterator = iter(read_elem(filename))
    nb_orders = int(next(file_iterator)) + 1
    distances_data = [None] * nb_orders
    for i in range(nb_orders) :
        distances_data[i] = [None] * nb_orders
        for j in range(nb_orders) :
            distances_data[i][j] = int(next(file_iterator))
    return nb_orders, distances_data

def main(input_file, output_file, time_limit) :
    # Read the instance from input_file
    nb_orders, distances_data = read_instance(input_file)
    
    with hexaly.optimizer.HexalyOptimizer() as optimizer :
        # Declare the model
        model = optimizer.model

        # Declare the list containing the picking order
        picking_list = model.list(nb_orders)

        # All orders must be picked
        model.constraint(model.count(picking_list) == nb_orders)

        # Create an Hexaly array for the distance matrix in order to access it using the "at" operator
        distances_matrix = model.array(distances_data)

        # Lambda expression to compute the distance to the next order
        distance_to_next_order_lambda = model.lambda_function( 
            lambda i : model.at(distances_matrix, picking_list[i], picking_list[i + 1]))

        # The objective is to minimize the total distance required to pick 
        # all the orders and to go back to the initial position
        objective = model.sum(model.range(0, nb_orders - 1), distance_to_next_order_lambda) \
            + model.at(distances_matrix, picking_list[nb_orders - 1], picking_list[0])

        # Store the index of the initial position in the list.
        # It will be used at the end to write the solution starting from the initial point.
        index_initial_position = model.index(picking_list, 0)

        model.minimize(objective)

        # End of the model declaration
        model.close()

        optimizer.param.time_limit = time_limit

        optimizer.solve()

        if output_file != None :
            with open(output_file, 'w') as f:
                f.write("%i\n" % objective.value)
                for i in range(nb_orders):
                    index = (index_initial_position.get_value() + i) % nb_orders
                    f.write("%i " % picking_list.value[index])


if __name__ == '__main__' :
    if len(sys.argv) < 2:
        print("Usage: python order_picking.py input_file [output_file] [time_limit]")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 else None
    time_limit = int(sys.argv[3]) if len(sys.argv) >= 4 else 10
    main(input_file, output_file, time_limit)

```

Car Sequencing

```python

import hexaly.optimizer
import sys


def read_integers(filename):
    with open(filename) as f:
        return [int(elem) for elem in f.read().split()]

#
# Read instance data
#


def read_instance(instance_file):
    file_it = iter(read_integers(instance_file))
    nb_positions = next(file_it)
    nb_options = next(file_it)
    nb_classes = next(file_it)
    max_cars_per_window = [next(file_it) for i in range(nb_options)]
    window_size = [next(file_it) for i in range(nb_options)]
    nb_cars = []
    options = []
    initial_sequence = []

    for c in range(nb_classes):
        next(file_it)  # Note: index of class is read but not used
        nb_cars.append(next(file_it))
        options.append([next(file_it) == 1 for i in range(nb_options)])
        [initial_sequence.append(c) for p in range(nb_cars[c])]

    return nb_positions, nb_options, max_cars_per_window, window_size, options, \
        initial_sequence


def main(instance_file, output_file, time_limit):
    nb_positions, nb_options, max_cars_per_window, window_size, options, \
        initial_sequence = read_instance(instance_file)

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # sequence[i] = j if class initially planned on position j is produced on position i
        sequence = model.list(nb_positions)

        # sequence is a permutation of the initial production plan, all indexes must
        # appear exactly once
        model.constraint(model.partition(sequence))

        # Create Hexaly arrays to be able to access them with "at" operators
        initial_array = model.array(initial_sequence)
        option_array = model.array(options)

        # Number of cars with option o in each window
        nb_cars_windows = [None] * nb_options
        for o in range(nb_options):
            nb_cars_windows[o] = [None] * nb_positions
            for j in range(nb_positions - window_size[o] + 1):
                nb_cars_windows[o][j] = model.sum()
                for k in range(window_size[o]):
                    class_at_position = initial_array[sequence[j + k]]
                    nb_cars_windows[o][j].add_operand(model.at(
                        option_array,
                        class_at_position,
                        o))

        # Number of violations of option o capacity in each window
        nb_violations_windows = [None] * nb_options
        for o in range(nb_options):
            nb_violations_windows[o] = [None] * nb_positions
            for p in range(nb_positions - window_size[o] + 1):
                nb_violations_windows[o][p] = model.max(
                    nb_cars_windows[o][p] - max_cars_per_window[o], 0)

        # Minimize the sum of violations for all options and all windows
        total_violations = model.sum(
            nb_violations_windows[o][p]
            for p in range(nb_positions - window_size[o] + 1) for o in range(nb_options))
        model.minimize(total_violations)

        model.close()

        # Set the initial solution
        sequence.get_value().clear()
        for p in range(nb_positions):
            sequence.get_value().add(p)

        # Parameterize the optimizer
        optimizer.param.time_limit = time_limit

        optimizer.solve()

        #
        # Write the solution in a file with the following format:
        # - 1st line: value of the objective;
        # - 2nd line: for each position p, index of class at positions p.
        #
        if output_file is not None:
            with open(output_file, 'w') as f:
                f.write("%d\n" % total_violations.value)
                for p in range(nb_positions):
                    f.write("%d " % initial_sequence[sequence.value[p]])

                f.write("\n")

```


Car sequencing color

```python
import hexaly.optimizer
import sys

COLOR_HIGH_LOW = 0
HIGH_LOW_COLOR = 1
HIGH_COLOR_LOW = 2
COLOR_HIGH = 3
HIGH_COLOR = 4

def read_integers(filename):
    with open(filename) as f:
        return [int(elem) for elem in f.read().split()]

#
# Read instance data
#
def read_instance(instance_file):
    file_it = iter(read_integers(instance_file))
    nb_positions = next(file_it)
    nb_options = next(file_it)
    nb_classes = next(file_it)
    paint_batch_limit = next(file_it)
    objective_order = next(file_it)
    start_position = next(file_it)

    max_cars_per_window = []
    window_size = []
    is_priority_option = []
    has_low_priority_options = False

    for o in range(nb_options):
        max_cars_per_window.append(next(file_it))
        window_size.append(next(file_it))
        is_prio = next(file_it) == 1
        is_priority_option.append(is_prio)
        if not is_prio:
            has_low_priority_options = True

    if not has_low_priority_options:
        if objective_order == COLOR_HIGH_LOW:
            objective_order = COLOR_HIGH
        elif objective_order == HIGH_COLOR_LOW:
            objective_order = HIGH_COLOR
        elif objective_order == HIGH_LOW_COLOR:
            objective_order = HIGH_COLOR

    color_class = []
    nb_cars = []
    options_data = []

    for c in range(nb_classes):
        color_class.append(next(file_it))
        nb_cars.append(next(file_it))
        options_data.append([next(file_it) == 1 for i in range(nb_options)])

    initial_sequence = [next(file_it) for p in range(nb_positions)]

    return nb_positions, nb_options, paint_batch_limit, objective_order, start_position, \
        max_cars_per_window, window_size, is_priority_option, has_low_priority_options, \
        color_class, options_data, initial_sequence

def main(instance_file, output_file, time_limit):
    nb_positions, nb_options, paint_batch_limit, objective_order, start_position, \
        max_cars_per_window, window_size, is_priority_option, has_low_priority_options, \
        color_class, options_data, initial_sequence = read_instance(instance_file)

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # sequence[i] = j if class initially planned on position j is produced on position i
        sequence = model.list(nb_positions)

        # sequence is a permutation of the initial production plan, all indexes must appear
        # exactly once
        model.constraint(model.partition(sequence))

        # Past classes (before startPosition) can not move
        [model.constraint(sequence[p] == p) for p in range(start_position)]

        # Create Hexaly arrays to be able to access them with "at" operators
        initials = model.array(initial_sequence)
        colors = model.array(color_class)
        options = model.array(options_data)

        # Number of cars with option o in each window
        nb_cars_windows = [None] * nb_options
        for o in range(nb_options):
            nb_cars_windows[o] = [None] * nb_positions
            for j in range(start_position - window_size[o] + 1, nb_positions):
                nb_cars_windows[o][j] = model.sum()
                for k in range(window_size[o]):
                    if j + k >= 0 and j + k < nb_positions:
                        class_at_position = initials[sequence[j + k]]
                        nb_cars_windows[o][j].add_operand(model.at(
                            options,
                            class_at_position,
                            o))

        # Number of violations of option o capacity in each window
        objective_high_priority = model.sum()
        if has_low_priority_options:
            objective_low_priority = model.sum()

        for o in range(nb_options):
            nb_violations_windows = model.sum(
                model.max(
                    nb_cars_windows[o][p] - max_cars_per_window[o], 0)
                    for p in range(start_position - window_size[o] + 1, nb_positions))
            if is_priority_option[o]:
                objective_high_priority.add_operand(nb_violations_windows)
            else:
                objective_low_priority.add_operand(nb_violations_windows)

        # Color change between position p and position p + 1
        color_change = [None] * (nb_positions - 1)
        objective_color = model.sum()
        for p in range(start_position - 1, nb_positions - 1):
            current_class = initials[sequence[p]]
            next_class = initials[sequence[p + 1]]
            color_change[p] = colors[current_class] != colors[next_class]
            objective_color.add_operand(color_change[p])

        # Paint limit constraints: at least one change every paintBatchLimit positions
        for p in range(start_position, nb_positions - paint_batch_limit - 1):
            node_or = model.or_(color_change[p + p2] for p2 in range(paint_batch_limit))
            model.constraint(node_or)

        # Declare the objectives in the correct order
        if objective_order == COLOR_HIGH_LOW:
            model.minimize(objective_color)
            model.minimize(objective_high_priority)
            model.minimize(objective_low_priority)
        elif objective_order == HIGH_COLOR_LOW:
            model.minimize(objective_high_priority)
            model.minimize(objective_color)
            model.minimize(objective_low_priority)
        elif objective_order == HIGH_LOW_COLOR:
            model.minimize(objective_high_priority)
            model.minimize(objective_low_priority)
            model.minimize(objective_color)
        elif objective_order == COLOR_HIGH:
            model.minimize(objective_color)
            model.minimize(objective_high_priority)
        elif objective_order == HIGH_COLOR:
            model.minimize(objective_high_priority)
            model.minimize(objective_color)

        model.close()


        # Set the initial solution
        sequence.get_value().clear()
        for p in range(nb_positions):
            sequence.get_value().add(p)

        # Parameterize the optimizer
        optimizer.param.time_limit = time_limit

        optimizer.solve()

        #
        # Write the solution in a file with the following format:
        # - 1st line: value of the objectives;
        # - 2nd line: for each position p, index of class at positions p.
        #
        if output_file is not None:
            with open(output_file, 'w') as f:
                f.write("%d " % objective_color.value)
                f.write("%d " % objective_high_priority.value)
                f.write("%d\n" % objective_low_priority.value)
                for p in range(nb_positions):
                    f.write("%d " % sequence.value[p])

                f.write("\n")
```


Social Golfer

```python

import hexaly.optimizer
import sys

if len(sys.argv) < 2:
    print("Usage: python social_golfer.py inputFile [outputFile] [timeLimit]")
    sys.exit(1)


def read_integers(filename):
    with open(filename) as f:
        return [int(elem) for elem in f.read().split()]


with hexaly.optimizer.HexalyOptimizer() as optimizer:
    #
    # Read instance data
    #
    file_it = iter(read_integers(sys.argv[1]))
    nb_groups = next(file_it)
    group_size = next(file_it)
    nb_weeks = next(file_it)
    nb_golfers = nb_groups * group_size

    #
    # Declare the optimization model
    #
    model = optimizer.model

    # Decision variables
    # 0-1 decisions variables: x[w][gr][gf]=1 if golfer gf is in group gr on week w
    x = [[[model.bool() for gf in range(nb_golfers)]
          for gr in range(nb_groups)] for w in range(nb_weeks)]

    # Each week, each golfer is assigned to exactly one group
    for w in range(nb_weeks):
        for gf in range(nb_golfers):
            model.constraint(
                model.eq(model.sum(x[w][gr][gf] for gr in range(nb_groups)), 1))

    # Each week, each group contains exactly group_size golfers
    for w in range(nb_weeks):
        for gr in range(nb_groups):
            model.constraint(
                model.eq(model.sum(x[w][gr][gf] for gf in range(nb_golfers)), group_size))

    # Golfers gf0 and gf1 meet in group gr on week w if both are
    # assigned to this group for week w
    meetings = [None] * nb_weeks
    for w in range(nb_weeks):
        meetings[w] = [None] * nb_groups
        for gr in range(nb_groups):
            meetings[w][gr] = [None] * nb_golfers
            for gf0 in range(nb_golfers):
                meetings[w][gr][gf0] = [None] * nb_golfers
                for gf1 in range(gf0 + 1, nb_golfers):
                    meetings[w][gr][gf0][gf1] = model.and_(x[w][gr][gf0], x[w][gr][gf1])

    # The number of meetings of golfers gf0 and gf1 is the sum
    # of their meeting variables over all weeks and groups
    redundant_meetings = [None] * nb_golfers
    for gf0 in range(nb_golfers):
        redundant_meetings[gf0] = [None] * nb_golfers
        for gf1 in range(gf0 + 1, nb_golfers):
            nb_meetings = model.sum(meetings[w][gr][gf0][gf1] for w in range(nb_weeks)
                                    for gr in range(nb_groups))
            redundant_meetings[gf0][gf1] = model.max(nb_meetings - 1, 0)

    # the goal is to minimize the number of redundant meetings
    obj = model.sum(redundant_meetings[gf0][gf1] for gf0 in range(nb_golfers)
                    for gf1 in range(gf0 + 1, nb_golfers))
    model.minimize(obj)

    model.close()

    # Parameterize the optimizer
    optimizer.param.nb_threads = 1
    if len(sys.argv) >= 4:
        optimizer.param.time_limit = int(sys.argv[3])
    else:
        optimizer.param.time_limit = 10

    optimizer.solve()

    #
    # Write the solution in a file with the following format:
    # - the objective value
    # - for each week and each group, write the golfers of the group
    # (nb_weeks x nbGroupes lines of group_size numbers).
    #
    if len(sys.argv) >= 3:
        with open(sys.argv[2], 'w') as f:
            f.write("%d\n" % obj.value)
            for w in range(nb_weeks):
                for gr in range(nb_groups):
                    for gf in range(nb_golfers):
                        if x[w][gr][gf].value:
                            f.write("%d " % (gf))
                    f.write("\n")
                f.write("\n")
```


Steel Mill Slab Design

```python

import hexaly.optimizer
import sys

if len(sys.argv) < 2:
    print("Usage: python steel_mill_slab_design.py inputFile [outputFile] [timeLimit]")
    sys.exit(1)

def read_integers(filename):
    with open(filename) as f:
        return [int(elem) for elem in f.read().split()]


# Compute the vector waste_for_content
def pre_compute_waste_for_content(slab_sizes, sum_size_orders):
    # No waste when a slab is empty
    waste_for_content = [0] * sum_size_orders

    prev_size = 0
    for size in slab_sizes:
        if size < prev_size:
            print("Slab sizes should be sorted in ascending order")
            sys.exit(1)
        for content in range(prev_size + 1, size):
            waste_for_content[content] = size - content
        prev_size = size
    return waste_for_content


with hexaly.optimizer.HexalyOptimizer() as optimizer:
    #
    # Read instance data
    #
    nb_colors_max_slab = 2

    file_it = iter(read_integers(sys.argv[1]))
    nb_slab_sizes = next(file_it)
    slab_sizes = [next(file_it) for i in range(nb_slab_sizes)]
    max_size = slab_sizes[nb_slab_sizes - 1]

    nb_colors = next(file_it)
    nb_orders = next(file_it)
    nb_slabs = nb_orders

    sum_size_orders = 0

    # List of quantities and colors for each order
    quantities_data = []
    colors_data = []
    for o in range(nb_orders):
        quantities_data.append(next(file_it))
        colors_data.append(next(file_it))
        sum_size_orders += quantities_data[o]

    waste_for_content = pre_compute_waste_for_content(slab_sizes, sum_size_orders)

    #
    # Declare the optimization model
    #
    model = optimizer.model

    # Create array and function to retrieve the orders's colors and quantities
    colors = model.array(colors_data)
    color_lambda = model.lambda_function(lambda l: colors[l])
    quantities = model.array(quantities_data)
    quantity_lambda = model.lambda_function(lambda o: quantities[o])

    # Set decisions: slab[k] represents the orders in slab k
    slabs = [model.set(nb_orders) for s in range(nb_slabs)]

    # Each order must be in one slab and one slab only
    model.constraint(model.partition(slabs))

    slabContent = []
    
    for s in range(nb_slabs):

        # The number of colors per slab must not exceed a specified value
        model.constraint(model.count(model.distinct(slabs[s], color_lambda)) <= nb_colors_max_slab)

        # The content of each slab must not exceed the maximum size of the slab
        slabContent.append(model.sum(slabs[s], quantity_lambda))
        model.constraint(slabContent[s] <= max_size)

    waste_for_content_array = model.array(waste_for_content)

    # Wasted steel is computed according to the content of the slab
    wasted_steel = [waste_for_content_array[slabContent[s]] for s in range(nb_slabs)]

    # Minimize the total wasted steel
    total_wasted_steel = model.sum(wasted_steel)
    model.minimize(total_wasted_steel)

    model.close()

    # Parameterize the optimizer
    if len(sys.argv) >= 4:
        optimizer.param.time_limit = int(sys.argv[3])
    else:
        optimizer.param.time_limit = 60
    optimizer.solve()

    #
    # Write the solution in a file with the following format:
    #  - total wasted steel
    #  - number of slabs used
    #  - for each slab used, the number of orders in the slab and the list of orders
    #
    if len(sys.argv) >= 3:
        with open(sys.argv[2], 'w') as f:
            f.write("%d\n" % total_wasted_steel.value)
            actual_nb_slabs = 0
            for s in range(nb_slabs):
                if slabs[s].value.count() > 0:
                    actual_nb_slabs += 1
            f.write("%d\n" % actual_nb_slabs)

            for s in range(nb_slabs):
                nb_orders_in_slab = slabs[s].value.count()
                if nb_orders_in_slab == 0:
                    continue
                f.write("%d" % nb_orders_in_slab)
                for o in slabs[s].value:
                    f.write(" %d" % (o + 1))
                f.write("\n")
```


Bin Packing

```python

import hexaly.optimizer
import sys
import math

if len(sys.argv) < 2:
    print("Usage: python bin_packing.py inputFile [outputFile] [timeLimit]")
    sys.exit(1)


def read_integers(filename):
    with open(filename) as f:
        return [int(elem) for elem in f.read().split()]


with hexaly.optimizer.HexalyOptimizer() as optimizer:
    # Read instance data
    file_it = iter(read_integers(sys.argv[1]))

    nb_items = int(next(file_it))
    bin_capacity = int(next(file_it))

    weights_data = [int(next(file_it)) for i in range(nb_items)]
    nb_min_bins = int(math.ceil(sum(weights_data) / float(bin_capacity)))
    nb_max_bins = min(nb_items, 2 * nb_min_bins)

    #
    # Declare the optimization model
    #
    model = optimizer.model

    # Set decisions: bins[k] represents the items in bin k
    bins = [model.set(nb_items) for _ in range(nb_max_bins)]

    # Each item must be in one bin and one bin only
    model.constraint(model.partition(bins))

    # Create an array and a function to retrieve the item's weight
    weights = model.array(weights_data)
    weight_lambda = model.lambda_function(lambda i: weights[i])

    # Weight constraint for each bin
    bin_weights = [model.sum(b, weight_lambda) for b in bins]
    for w in bin_weights:
        model.constraint(w <= bin_capacity)

    # Bin k is used if at least one item is in it
    bins_used = [model.count(b) > 0 for b in bins]

    # Count the used bins
    total_bins_used = model.sum(bins_used)

    # Minimize the number of used bins
    model.minimize(total_bins_used)
    model.close()

    # Parameterize the optimizer
    if len(sys.argv) >= 4:
        optimizer.param.time_limit = int(sys.argv[3])
    else:
        optimizer.param.time_limit = 5

    # Stop the search if the lower threshold is reached
    optimizer.param.set_objective_threshold(0, nb_min_bins)

    optimizer.solve()

    # Write the solution in a file
    if len(sys.argv) >= 3:
        with open(sys.argv[2], 'w') as f:
            for k in range(nb_max_bins):
                if bins_used[k].value:
                    f.write("Bin weight: %d | Items: " % bin_weights[k].value)
                    for e in bins[k].value:
                        f.write("%d " % e)
                    f.write("\n")
```

Bin Packing with Conflicts 

```python
import hexaly.optimizer
import sys
import math

if len(sys.argv) < 2:
    print("Usage: python bin_packing_conflicts.py inputFile [outputFile] [timeLimit]")
    sys.exit(1)


def read_integers(filename):
    with open(filename) as f:
        return [int(elem) for elem in f.read().split()]


with hexaly.optimizer.HexalyOptimizer() as optimizer:
    # Read instance data
    filename = sys.argv[1]
    count = 0
    weights_data = []
    forbidden_items = []
    with open(filename) as f:
        for line in f:
            line = line.split()
            if count == 0:
                nb_items = int(line[0])
                bin_capacity = int(line[1])
            else:
                weights_data.append(int(line[1]))
                forbidden_items.append([])
                for i in range(2, len(line)):
                    forbidden_items[count-1].append(int(line[i]))
            count += 1
                    
    nb_min_bins = int(math.ceil(sum(weights_data) / float(bin_capacity)))
    nb_max_bins = min(nb_items, 2 * nb_min_bins)

    #
    # Declare the optimization model
    #
    model = optimizer.model

    # Set decisions: bins[k] represents the items in bin k
    bins = [model.set(nb_items) for _ in range(nb_max_bins)]

    # Transform bins and itemFordbidden list into hx expression
    bins_array = model.array(bins)
    forbidden_items_array = model.array(forbidden_items)

    # Find the bin where an item is packed
    bin_for_item = [model.find(bins_array, i) for i in range(nb_items)]

    # Each item must be in one bin and one bin only
    model.constraint(model.partition(bins))

    # Create an array and a function to retrieve the item's weight
    weights = model.array(weights_data)
    weight_lambda = model.lambda_function(lambda i: weights[i])

    # Forbidden constraint for each items
    for i in range(nb_items):
        items_intersection = model.intersection(forbidden_items_array[i], bins_array[bin_for_item[i]])
        model.constraint(model.count(items_intersection) == 0)

    # Weight constraint for each bin
    bin_weights = [model.sum(b, weight_lambda) for b in bins]
    for w in bin_weights:
        model.constraint(w <= bin_capacity)

    # Bin k is used if at least one item is in it
    bins_used = [model.count(b) > 0 for b in bins]

    # Count the used bins
    total_bins_used = model.sum(bins_used)

    # Minimize the number of used bins
    model.minimize(total_bins_used)
    model.close()

    # Parameterize the optimizer
    if len(sys.argv) >= 4:
        optimizer.param.time_limit = int(sys.argv[3])
    else:
        optimizer.param.time_limit = 5

    # Stop the search if the lower threshold is reached
    optimizer.param.set_objective_threshold(0, nb_min_bins)

    optimizer.solve()

    # Write the solution in a file
    if len(sys.argv) >= 3:
        with open(sys.argv[2], 'w') as f:
            for k in range(nb_items):
                f.write("item:%d Weight:%d" % (k, weights_data[k]))
                f.write("\n")
            for k in range(nb_max_bins):
                if bins_used[k].value:
                    f.write("Bin weight: %d | Items: " % bin_weights[k].value)
                    for e in bins[k].value:
                        f.write("%d " % e)
                    f.write("\n")
```

Capacitated Facility Location

```python
import hexaly.optimizer
import sys


def main(instanceFile, strTimeLimit, solFile):

    #
    # Read instance data
    #
    nb_max_facilities, nb_sites, capacity_data, opening_price_data, \
        demand_data, allocation_price_data = read_data(instanceFile)

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Facilities are represented by the set of sites they provide
        facility_assignments = [model.set(nb_sites) for _ in range(nb_max_facilities)]

        # Each site is covered by exactly one facility
        model.constraint(model.partition(facility_assignments))

        # Converting demand and allocationPrice into Hexaly array
        demand = model.array(demand_data)
        allocation_price = model.array(allocation_price_data)

        cost = [None] * nb_max_facilities
        for f in range(nb_max_facilities):
            facility = facility_assignments[f]
            size = model.count(facility)

            # Capacity constraint
            demand_lambda = model.lambda_function(lambda i: demand[i])
            model.constraint(model.sum(facility, demand_lambda) <= capacity_data[f])

            # Cost (allocation price + opening price)
            costSelector = model.lambda_function(lambda i: model.at(allocation_price, f, i))
            cost[f] = model.sum(facility, costSelector) + opening_price_data[f] * (size > 0)

        # Objective : minimize total cost
        totalCost = model.sum(cost)
        model.minimize(totalCost)

        model.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = int(strTimeLimit)

        optimizer.solve()

        #
        # Write the solution in a file with the following format:
        # - value of the objective
        # - indices of the open facilities followed by all the sites they provide
        #
        if solFile:
            with open(solFile, 'w') as outputFile:
                outputFile.write("%d" % totalCost.value)
                for f in range(nb_max_facilities):
                    if cost[f].value > 0:
                        outputFile.write("%d\n" % f)
                        for site in facility_assignments[f].value:
                            outputFile.write("%d " % site)
                        outputFile.write("\n")


def read_elem(filename):
    with open(filename) as f:
        return [str(elem) for elem in f.read().split()]


def read_data(filename):
    file_it = iter(read_elem(filename))

    nb_max_facilities = int(next(file_it))
    nb_sites = int(next(file_it))

    capacity_data = []
    opening_price_data = []
    demand_data = []
    allocation_price_data = []

    for f in range(nb_max_facilities):
        # List of facilities capacities
        capacity_data.append(float(next(file_it)))
        # List of fixed costs induced by the facilities opening
        opening_price_data.append(float(next(file_it)))
        allocation_price_data.append([])

    # Demand of each site
    for s in range(nb_sites):
        demand_data.append(float(next(file_it)))

    # Allocation price between sites and facilities
    for f in range(nb_max_facilities):
        for s in range(nb_sites):
            allocation_price_data[f].append(float(next(file_it)))

    return nb_max_facilities, nb_sites, capacity_data, opening_price_data, \
        demand_data, allocation_price_data


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python capacitated_facility_location.py input_file \
            [output_file] [time_limit]")
        sys.exit(1)

    instanceFile = sys.argv[1]
    solFile = sys.argv[2] if len(sys.argv) > 2 else None
    strTimeLimit = sys.argv[3] if len(sys.argv) > 3 else "20"

    main(instanceFile, strTimeLimit, solFile)
```

Optimal Bucket

```python
import hexaly.optimizer
import sys

with hexaly.optimizer.HexalyOptimizer() as optimizer:
    PI = 3.14159265359

    #
    # Declare the optimization model
    #
    m = optimizer.model

    # Numerical decisions
    R = m.float(0, 1)
    r = m.float(0, 1)
    h = m.float(0, 1)

    # Surface must not exceed the surface of the plain disc
    surface = PI * r ** 2 + PI * (R + r) * m.sqrt((R - r) ** 2 + h ** 2)
    m.constraint(surface <= PI)

    # Maximize the volume
    volume = PI * h / 3 * (R ** 2 + R * r + r ** 2)
    m.maximize(volume)

    m.close()

    #
    # Parametrize the optimizer
    #
    if len(sys.argv) >= 3:
        optimizer.param.time_limit = int(sys.argv[2])
    else:
        optimizer.param.time_limit = 2

    optimizer.solve()

    #
    # Write the solution in a file with the following format:
    #  - surface and volume of the bucket
    #  - values of R, r and h
    #
    if len(sys.argv) >= 2:
        with open(sys.argv[1], 'w') as f:
            f.write("%f %f\n" % (surface.value, volume.value))
            f.write("%f %f %f\n" % (R.value, r.value, h.value))
```

Portfolio Selection Optimization Problem

```python
def read_instance(filename):
    with open(filename) as f:
        lines = f.readlines()

    first_line = lines[0].split()

    # Expected profit, in percentage of the portfolio
    expected_profit = float(first_line[0])

    second_line = lines[2].split()

    # Number of stocks
    nb_stocks = int(second_line[0])

    # Covariance among the stocks
    sigma_stocks = [[0 for i in range(nb_stocks)] for j in range(nb_stocks)]
    for s in range(nb_stocks):
        line = lines[s+4].split()
        for t in range(nb_stocks):
            sigma_stocks[s][t] = float(line[t])

    # Variation of the price of each stock
    delta_stock = [0 for i in range(nb_stocks)]
    line = lines[nb_stocks+5].split()
    for s in range(nb_stocks):
        delta_stock[s] = float(line[s])
        print(delta_stock[s])

    return expected_profit, nb_stocks, sigma_stocks, delta_stock


def main(instance_file, output_file, time_limit):
    expected_profit, nb_stocks, sigma_stocks, delta_stock = read_instance(
        instance_file)

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Proportion of the portfolio invested in each stock
        portfolio_stock = [model.float(0, 1) for s in range(nb_stocks)]

        # Risk of the portfolio
        risk = model.sum(portfolio_stock[s] * portfolio_stock[t] * sigma_stocks[s][t]
                         for t in range(nb_stocks) for s in range(nb_stocks))

        # Return of the portfolio in percentage
        profit = model.sum(portfolio_stock[s] * delta_stock[s]
                           for s in range(nb_stocks))

        # All the portfolio is used
        model.constraint(
            model.sum(portfolio_stock[s] for s in range(nb_stocks)) == 1.0)

        # The profit is at least the expected profit
        model.constraint(profit >= expected_profit)

        # Minimize the risk
        model.minimize(risk)

        model.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = time_limit

        optimizer.solve()

        # Write the solution in a file with the following format:
        # - for each stock, the proportion of the porfolio invested
        # - the final profit in percentage of the portfolio
        if output_file != None:
            with open(output_file, "w") as f:
                print("Solution written in file", output_file)
                for s in range(nb_stocks):
                    proportion = portfolio_stock[s].value
                    f.write("Stock " + str(s+1) + ": " + str(round(proportion * 100, 1))
                            + "%" + "\n")
                f.write("Profit: " + str(round(profit.value, 4)) + "%")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(
            "Usage: python portfolio.py instance_file [output_file] [time_limit]")
        sys.exit(1)

    instance_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 else None
    time_limit = int(sys.argv[3]) if len(sys.argv) >= 4 else 60
    main(instance_file, output_file, time_limit)
```

K-Means Clustering (MSSC)

```python

import hexaly.optimizer
import sys

def read_elem(filename):
    with open(filename) as f:
        return [str(elem) for elem in f.read().split()]

#
# Read instance data
#
def read_instance(filename):
    file_it = iter(read_elem(filename))

    # Data properties
    nb_observations = int(next(file_it))
    nb_dimensions = int(next(file_it))

    coordinates_data = [None] * nb_observations
    for o in range(nb_observations):
        coordinates_data[o] = [None] * (nb_dimensions)
        for d in range(nb_dimensions):
            coordinates_data[o][d] = float(next(file_it))
        next(file_it) # skip initial clusters

    return nb_observations, nb_dimensions, coordinates_data

def main(instance_file, output_file, time_limit, k):
    nb_observations, nb_dimensions, coordinates_data = read_instance(instance_file)

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # clusters[c] represents the points in cluster c
        clusters = [model.set(nb_observations) for c in range(k)]

        # Each point must be in one cluster and one cluster only
        model.constraint(model.partition(clusters))

        # Coordinates of points
        coordinates = model.array(coordinates_data)

        # Compute variances
        variances = []
        for cluster in clusters:
            size = model.count(cluster)

            # Compute centroid of cluster
            centroid = [0 for d in range(nb_dimensions)]
            for d in range(nb_dimensions):
                coordinate_lambda = model.lambda_function(
                    lambda i: model.at(coordinates, i, d))
                centroid[d] = model.iif(
                    size == 0,
                    0,
                    model.sum(cluster, coordinate_lambda) / size)

            # Compute variance of cluster
            variance = model.sum()
            for d in range(nb_dimensions):
                dimension_variance_lambda = model.lambda_function(lambda i: model.sum(
                    model.pow(model.at(coordinates, i, d) - centroid[d], 2)))
                dimension_variance = model.sum(cluster, dimension_variance_lambda)
                variance.add_operand(dimension_variance)
            variances.append(variance)

        # Minimize the total variance
        obj = model.sum(variances)
        model.minimize(obj)

        model.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = time_limit

        optimizer.solve()

        #
        # Write the solution in a file in the following format:
        #  - objective value
        #  - k
        #  - for each cluster, a line with the elements in the cluster
        #    (separated by spaces)
        #
        if output_file != None:
            with open(output_file, 'w') as f:
                f.write("%f\n" % obj.value)
                f.write("%d\n" % k)
                for c in range(k):
                    for o in clusters[c].value:
                        f.write("%d " % o)
                    f.write("\n")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python kmeans.py inputFile [outputFile] [timeLimit] [k value]")
        sys.exit(1)
```


Quadratic Assignment (QAP)

```python
import hexaly.optimizer
import sys

if len(sys.argv) < 2:
    print("Usage: python qap.py inputFile [outputFile] [timeLimit]")
    sys.exit(1)


def read_integers(filename):
    with open(filename) as f:
        return [int(elem) for elem in f.read().split()]


with hexaly.optimizer.HexalyOptimizer() as optimizer:
    #
    # Read instance data
    #
    file_it = iter(read_integers(sys.argv[1]))

    # Number of points
    n = next(file_it)

    # Distance between locations
    A = [[next(file_it) for j in range(n)] for i in range(n)]
    # Flow between factories
    B = [[next(file_it) for j in range(n)] for i in range(n)]

    #
    # Declare the optimization model
    #
    model = optimizer.model

    # Permutation such that p[i] is the facility on the location i
    p = model.list(n)

    # The list must be complete
    model.constraint(model.eq(model.count(p), n))

    # Create B as an array to be accessed by an at operator
    array_B = model.array(B)

    # Minimize the sum of product distance*flow
    obj = model.sum(A[i][j] * model.at(array_B, p[i], p[j])
                    for j in range(n) for i in range(n))
    model.minimize(obj)

    model.close()

    # Parameterize the optimizer
    if len(sys.argv) >= 4:
        optimizer.param.time_limit = int(sys.argv[3])
    else:
        optimizer.param.time_limit = 30
    optimizer.solve()

    #
    # Write the solution in a file with the following format:
    #  - n objValue
    #  - permutation p
    #
    if len(sys.argv) >= 3:
        with open(sys.argv[2], 'w') as outfile:
            outfile.write("%d %d\n" % (n, obj.value))
            for i in range(n):
                outfile.write("%d " % p.value[i])
            outfile.write("\n")
```



Assembly Line Balancing

```python
import hexaly.optimizer
import sys


#
# Functions to read the instances
#
def read_elem(filename):
    with open(filename) as f:
        return [str(elem) for elem in f.read().split()]


def read_instance(instance_file):
    file_it = iter(read_elem(instance_file))

    for _ in range(3):
        next(file_it)

    # Read number of tasks
    nb_tasks = int(next(file_it))
    max_nb_stations = nb_tasks
    for _ in range(2):
        next(file_it)

    # Read the cycle time limit
    cycle_time = int(next(file_it))
    for _ in range(5):
        next(file_it)

    # Read the processing times
    processing_time_dict = {}
    for _ in range(nb_tasks):
        task = int(next(file_it)) - 1
        processing_time_dict[task] = int(next(file_it))
    for _ in range(2):
        next(file_it)
    processing_time = [elem[1] for elem in sorted(processing_time_dict.items(),
                                                  key=lambda x: x[0])]

    # Read the successors' relations
    successors = {}
    while True:
        try:
            pred, succ = next(file_it).split(',')
            pred = int(pred) - 1
            succ = int(succ) - 1
            if pred in successors:
                successors[pred].append(succ)
            else:
                successors[pred] = [succ]
        except:
            break
    return nb_tasks, max_nb_stations, cycle_time, processing_time, successors


def main(instance_file, output_file, time_limit):
    nb_tasks, max_nb_stations, cycle_time, processing_time_data, \
        successors_data = read_instance(instance_file)

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Decision variables: station_vars[s] is the set of tasks assigned to station s
        station_vars = [model.set(nb_tasks) for s in range(max_nb_stations)]
        stations = model.array(station_vars)
        model.constraint(model.partition(stations))

        # Objective: nb_used_stations is the total number of used stations
        nb_used_stations = model.sum(
            (model.count(station_vars[s]) > 0) for s in range(max_nb_stations))

        # All stations must respect the cycleTime constraint
        processing_time = model.array(processing_time_data)
        time_lambda = model.lambda_function(lambda i: processing_time[i])
        time_in_station = [model.sum(station_vars[s], time_lambda)
                           for s in range(max_nb_stations)]
        for s in range(max_nb_stations):
            model.constraint(time_in_station[s] <= cycle_time)

        # The stations must respect the succession's order of the tasks
        task_station = [model.find(stations, i) for i in range(nb_tasks)]
        for i in range(nb_tasks):
            if i in successors_data.keys():
                for j in successors_data[i]:
                    model.constraint(task_station[i] <= task_station[j])

        # Minimization of the number of active stations
        model.minimize(nb_used_stations)

        model.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = time_limit

        optimizer.solve()

        # Write the solution in a file following the format:
        # - 1st line: value of the objective
        # - 2nd line: number of tasks
        # - following lines: task's number, station's number
        if output_file is not None:
            with open(output_file, 'w') as f:
                f.write("%d\n" % nb_used_stations.value)
                f.write("%d\n" % nb_tasks)
                for i in range(nb_tasks):
                    f.write("{},{}\n".format(i + 1, task_station[i].value + 1))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python assembly_line_balancing.py instance_file \
            [output_file] [time_limit]")
        sys.exit(1)

    instance_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 else None
    time_limit = int(sys.argv[3]) if len(sys.argv) >= 4 else 20
    main(instance_file, output_file, time_limit)
```


Flow Shop

```python

import hexaly.optimizer
import sys

def read_integers(filename):
    with open(filename) as f:
        return [int(elem) for elem in f.read().split()]

#
# Read instance data
#
def read_instance(instance_file):
    file_it = iter(read_integers(instance_file))

    nb_jobs = int(next(file_it))
    nb_machines = int(next(file_it))
    next(file_it)
    next(file_it)
    next(file_it)

    processing_time_data = [[int(next(file_it)) for j in range(nb_jobs)]
                            for j in range(nb_machines)]

    return nb_jobs, nb_machines, processing_time_data

def main(instance_file, output_file, time_limit):
    nb_jobs, nb_machines, processing_time_data = read_instance(instance_file)

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Permutation of jobs
        jobs = model.list(nb_jobs)

        # All jobs have to be assigned
        model.constraint(model.eq(model.count(jobs), nb_jobs))

        # For each machine create proccessingTime[m] as an array to be able
        # to access it with an 'at' operator
        processing_time = [model.array(processing_time_data[m])
                           for m in range(nb_machines)]

        # On machine 0, the jth job ends on the time it took to be processed
        # after the end of the previous job
        job_end = [None] * nb_machines

        first_end_lambda = model.lambda_function(lambda i, prev:
                                                 prev + processing_time[0][jobs[i]])

        job_end[0] = model.array(model.range(0, nb_jobs), first_end_lambda, 0)

        # The jth job on machine m starts when it has been processed by machine n-1
        # AND when job j-1 has been processed on machine m.
        # It ends after it has been processed.
        for m in range(1, nb_machines):
            mL = m
            end_lambda = model.lambda_function(lambda i, prev:
                model.max(prev, job_end[mL - 1][i]) + processing_time[mL][jobs[i]])
            job_end[m] = model.array(model.range(0, nb_jobs), end_lambda, 0)

        # Minimize the makespan: end of the last job on the last machine
        makespan = job_end[nb_machines - 1][nb_jobs - 1]
        model.minimize(makespan)

        model.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = time_limit

        optimizer.solve()

        #
        # Write the solution in a file
        #
        if output_file is not None:
            with open(output_file, 'w') as f:
                f.write("%d\n" % makespan.value)
                for j in jobs.value:
                    f.write("%d " % j)
                f.write("\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python flowshop.py instance_file [output_file] [time_limit]")
        sys.exit(1)

    instance_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 else None
    time_limit = int(sys.argv[3]) if len(sys.argv) >= 4 else 5
    main(instance_file, output_file, time_limit)
```


Job Shop 


```python
import hexaly.optimizer
import sys


# The input files follow the "Taillard" format
def read_instance(filename):
    with open(filename) as f:
        lines = f.readlines()

    first_line = lines[1].split()
    # Number of jobs
    nb_jobs = int(first_line[0])
    # Number of machines
    nb_machines = int(first_line[1])

    # Processing times for each job on each machine (given in the processing order)
    processing_times_in_processing_order = [[int(lines[i].split()[j])
                                             for j in range(nb_machines)]
                                            for i in range(3, 3 + nb_jobs)]

    # Processing order of machines for each job
    machine_order = [[int(lines[i].split()[j]) - 1 for j in range(nb_machines)]
                     for i in range(4 + nb_jobs, 4 + 2 * nb_jobs)]

    # Reorder processing times: processing_time[j][m] is the processing time of the
    # task of job j that is processed on machine m
    processing_time = [[processing_times_in_processing_order[j][machine_order[j].index(m)]
                        for m in range(nb_machines)]
                       for j in range(nb_jobs)]

    # Trivial upper bound for the end times of the tasks
    max_end = sum(sum(processing_time[j]) for j in range(nb_jobs))

    return nb_jobs, nb_machines, processing_time, machine_order, max_end


def main(instance_file, output_file, time_limit):
    nb_jobs, nb_machines, processing_time, machine_order, max_end = read_instance(instance_file)

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Interval decisions: time range of each task
        # tasks[j][m] is the interval of time of the task of job j which is processed
        # on machine m
        tasks = [[model.interval(0, max_end) for m in range(nb_machines)]
                 for j in range(nb_jobs)]

        # Task duration constraints
        for j in range(nb_jobs):
            for m in range(0, nb_machines):
                model.constraint(model.length(tasks[j][m]) == processing_time[j][m])

        # Create an Hexaly array in order to be able to access it with "at" operators
        task_array = model.array(tasks)

        # Precedence constraints between the tasks of a job
        for j in range(nb_jobs):
            for k in range(nb_machines - 1):
                model.constraint(
                    tasks[j][machine_order[j][k]] < tasks[j][machine_order[j][k + 1]])

        # Sequence of tasks on each machine
        jobs_order = [model.list(nb_jobs) for m in range(nb_machines)]

        for m in range(nb_machines):
            # Each job has a task scheduled on each machine
            sequence = jobs_order[m]
            model.constraint(model.eq(model.count(sequence), nb_jobs))

            # Disjunctive resource constraints between the tasks on a machine
            sequence_lambda = model.lambda_function(
                lambda i: model.lt(model.at(task_array, sequence[i], m),
                                   model.at(task_array, sequence[i + 1], m)))
            model.constraint(model.and_(model.range(0, nb_jobs - 1), sequence_lambda))

        # Minimize the makespan: end of the last task of the last job
        makespan = model.max([model.end(tasks[j][machine_order[j][nb_machines - 1]])
                             for j in range(nb_jobs)])
        model.minimize(makespan)

        model.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = time_limit

        optimizer.solve()

        #
        # Write the solution in a file with the following format:
        # - for each machine, the job sequence
        #
        if output_file != None:
            final_jobs_order = [list(jobs_order[m].value) for m in range(nb_machines)]
            with open(output_file, "w") as f:
                print("Solution written in file ", output_file)
                for m in range(nb_machines):
                    for j in range(nb_jobs):
                        f.write(str(final_jobs_order[m][j]) + " ")
                    f.write("\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python jobshop.py instance_file [output_file] [time_limit]")
        sys.exit(1)

    instance_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 else None
    time_limit = int(sys.argv[3]) if len(sys.argv) >= 4 else 60
    main(instance_file, output_file, time_limit)
```


Job Shop Scheduling Problem with Intensity

```python

import hexaly.optimizer
import sys


def read_instance(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    first_line = lines[1].split()
    # Number of jobs
    nb_jobs = int(first_line[0])
    # Number of machines
    nb_machines = int(first_line[1])
    # Time horizon: number of time steps
    time_horizon = int(first_line[2])

    # Processing times for each job on each machine (given in the processing order)
    processing_times_in_processing_order = [[int(lines[i].split()[j])
                                             for j in range(nb_machines)]
                                            for i in range(3, 3 + nb_jobs)]

    # Processing order of machines for each job
    machine_order = [[int(lines[i].split()[j]) - 1 for j in range(nb_machines)]
                     for i in range(4 + nb_jobs, 4 + 2 * nb_jobs)]

    # Reorder processing times: processing_time[j][i] is the processing time
    # of the task of job j that is processed on machine m
    processing_time = [[processing_times_in_processing_order[j][machine_order[j].index(m)]
                        for m in range(nb_machines)] for j in range(nb_jobs)]

    # Intensity for each machine for each time step
    intensity = [[int(lines[i].split()[j]) for j in range(time_horizon)]
                 for i in range(5 + 2 * nb_jobs, 5 + 2 * nb_jobs + nb_machines)]

    return nb_jobs, nb_machines, time_horizon, processing_time, machine_order, intensity


def main(instance_file, output_file, time_limit):
    nb_jobs, nb_machines, time_horizon, processing_time, \
        machine_order, intensity = read_instance(instance_file)

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Interval decisions: time range of each task
        # tasks[j][m] is the interval of time of the task of job j which is processed
        # on machine m
        tasks = [[model.interval(0, time_horizon) for m in range(nb_machines)]
                 for j in range(nb_jobs)]

        # Create Hexaly arrays to be able to access them with "at" operators
        task_array = model.array(tasks)
        intensity_array = model.array(intensity)

        # The sum of the machine's intensity over the duration of the task must be
        # greater than its processing time
        for m in range(nb_machines):
            intensity_lambda = model.lambda_function(lambda t:
                                                     model.at(intensity_array, m, t))
            for j in range(nb_jobs):
                model.constraint(
                    model.sum(tasks[j][m], intensity_lambda)
                    >= processing_time[j][m])

        # Precedence constraints between the tasks of a job
        for j in range(nb_jobs):
            for k in range(nb_machines - 1):
                model.constraint(
                    tasks[j][machine_order[j][k]] < tasks[j][machine_order[j][k + 1]])

        # Sequence of tasks on each machine
        jobs_order = [model.list(nb_jobs) for m in range(nb_machines)]

        for m in range(nb_machines):
            # Each job has a task scheduled on each machine
            sequence = jobs_order[m]
            model.constraint(model.eq(model.count(sequence), nb_jobs))

            # Disjunctive resource constraints between the tasks on a machine
            sequence_lambda = model.lambda_function(
                lambda i: model.at(task_array, sequence[i], m) < model.at(task_array, sequence[i + 1], m))
            model.constraint(model.and_(model.range(0, nb_jobs - 1), sequence_lambda))

        # Minimize the makespan: end of the last task of the last job
        makespan = model.max([model.end(tasks[j][machine_order[j][nb_machines - 1]])
                              for j in range(nb_jobs)])
        model.minimize(makespan)

        model.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = time_limit

        optimizer.solve()

        #
        # Write the solution in a file with the following format:
        # - for each machine, the job sequence
        #
        if output_file != None:
            final_jobs_order = [list(jobs_order[m].value) for m in range(nb_machines)]
            with open(output_file, "w") as f:
                print("Solution written in file ", output_file)
                for m in range(nb_machines):
                    for j in range(nb_jobs):
                        f.write(str(final_jobs_order[m][j]) + " ")
                    f.write("\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python jobshop_intensity.py instance_file [output_file] [time_limit]")
        sys.exit(1)

    instance_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 else None
    time_limit = int(sys.argv[3]) if len(sys.argv) >= 4 else 10
    main(instance_file, output_file, time_limit)
```

Flexible Resource Constrained Project Scheduling Problem

```python
import hexaly.optimizer
import sys


def read_instance(filename):
    with open(filename) as f:
        lines = f.readlines()

    first_line = lines[0].split()

    # Number of tasks
    nb_tasks = int(first_line[0])

    # Number of resources
    nb_resources = int(first_line[1])

    # Maximum capacity of each resource
    capacity = [int(lines[1].split()[r]) for r in range(nb_resources)]

    # Duration of task i if task i is done by resource r
    task_processing_time_data = [[] for i in range(nb_tasks)]

    # Resource weight of resource r required for task i
    weight = [[] for r in range(nb_resources)]

    # Number of successors
    nb_successors = [0 for i in range(nb_tasks)]

    # Successors of each task i
    successors = [[] for i in range(nb_tasks)]

    for i in range(nb_tasks):
        line_d_w = lines[i + 2].split()
        for r in range(nb_resources):
            task_processing_time_data[i].append(int(line_d_w[2 * r]))
            weight[r].append(int(line_d_w[2 * r + 1]))

        line_succ = lines[i + 2 + nb_tasks].split()
        nb_successors[i] = int(line_succ[0])
        successors[i] = [int(elm) for elm in line_succ[1::]]

    # Trivial upper bound for the end times of the tasks
    horizon = sum(max(task_processing_time_data[i][r] for r in range(nb_resources)) for i in range(nb_tasks))

    return (nb_tasks, nb_resources, capacity, task_processing_time_data, weight, nb_successors, successors, horizon)


def main(instance_file, output_file, time_limit):
    nb_tasks, nb_resources, capacity, task_processing_time_data, weight,\
        nb_successors, successors, horizon = read_instance(instance_file)

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Set of tasks done by each resource
        resources_tasks = [model.set(nb_tasks) for r in range(nb_resources)]
        resources = model.array(resources_tasks)

        # Only compatible resources can be selected for a task
        for i in range(nb_tasks):
            for r in range(nb_resources):
                if task_processing_time_data[i][r] == 0 and weight[r][i] == 0:
                    model.constraint(model.contains(resources_tasks[r], i) == 0)

        # For each task, the selected resource
        task_resource = [model.find(resources, t) for t in range(nb_tasks)]

        # All tasks are scheduled on the resources
        model.constraint(model.partition(resources))

        # Interval decisions: time range of each task
        tasks = [model.interval(0, horizon) for i in range(nb_tasks)]

        # Create Hexaly arrays to be able to access them with an "at" operator
        tasks_array = model.array(tasks)
        task_processing_time = model.array(task_processing_time_data)
        weight_array = model.array(weight)

        # Task duration constraints
        for i in range(nb_tasks):
            model.constraint(model.length(tasks[i]) == task_processing_time[i][task_resource[i]])

        # Precedence constraints between the tasks
        for i in range(nb_tasks):
            for s in range(nb_successors[i]):
                model.constraint(tasks[i] < tasks[successors[i][s]])

        # Makespan: end of the last task
        makespan = model.max([model.end(tasks[i]) for i in range(nb_tasks)])

        # Cumulative resource constraints
        for r in range(nb_resources):
            capacity_respected = model.lambda_function(
                lambda t: model.sum(resources_tasks[r], model.lambda_function(
                    lambda i: model.at(weight_array, r, i) * model.contains(tasks_array[i], t)))
                <= capacity[r])
            model.constraint(model.and_(model.range(makespan), capacity_respected))

        # Minimize the makespan
        model.minimize(makespan)

        model.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = time_limit

        optimizer.solve()

        #
        # Write the solution in a file with the following format:
        # - total makespan
        # - for each task, the task id, the selected resource, the start and end times
        #
        if output_file != None:
            with open(output_file, "w") as f:
                print("Solution written in file", output_file)
                f.write(str(makespan.value) + "\n")
                for i in range(nb_tasks):
                    f.write(
                        str(i) + " " + str(task_resource[i].value) + " " + str(tasks[i].value.start()) + " " +
                        str(tasks[i].value.end()))
                    f.write("\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python flexible_cumulative.py instance_file [output_file] [time_limit]")
        sys.exit(1)

    instance_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 else None
    time_limit = int(sys.argv[3]) if len(sys.argv) >= 4 else 60
    main(instance_file, output_file, time_limit)
```

Flexible Job Shop (FJSP)

```python
import hexaly.optimizer
import sys

# Constant for incompatible machines
INFINITE = 1000000


def read_instance(filename):
    with open(filename) as f:
        lines = f.readlines()

    first_line = lines[0].split()
    # Number of jobs
    nb_jobs = int(first_line[0])
    # Number of machines
    nb_machines = int(first_line[1])

    # Number of operations for each job
    nb_operations = [int(lines[j + 1].split()[0]) for j in range(nb_jobs)]

    # Number of tasks
    nb_tasks = sum(nb_operations[j] for j in range(nb_jobs))

    # Processing time for each task, for each machine
    task_processing_time = [[INFINITE for m in range(nb_machines)] for i in range(nb_tasks)]

    # For each job, for each operation, the corresponding task id
    job_operation_task = [[0 for o in range(nb_operations[j])] for j in range(nb_jobs)]

    id = 0
    for j in range(nb_jobs):
        line = lines[j + 1].split()
        tmp = 0
        for o in range(nb_operations[j]):
            nb_machines_operation = int(line[tmp + o + 1])
            for i in range(nb_machines_operation):
                machine = int(line[tmp + o + 2 * i + 2]) - 1
                time = int(line[tmp + o + 2 * i + 3])
                task_processing_time[id][machine] = time
            job_operation_task[j][o] = id
            id = id + 1
            tmp = tmp + 2 * nb_machines_operation

    # Trivial upper bound for the end times of the tasks
    max_end = sum(
        max(task_processing_time[i][m] for m in range(nb_machines) if task_processing_time[i][m] != INFINITE)
        for i in range(nb_tasks))

    return nb_jobs, nb_machines, nb_tasks, task_processing_time, job_operation_task, nb_operations, max_end


def main(instance_file, output_file, time_limit):
    nb_jobs, nb_machines, nb_tasks, task_processing_time_data, job_operation_task, \
        nb_operations, max_end = read_instance(instance_file)

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Sequence of tasks on each machine
        jobs_order = [model.list(nb_tasks) for _ in range(nb_machines)]
        machines = model.array(jobs_order)

        # Each task is scheduled on a machine
        model.constraint(model.partition(machines))

        # Only compatible machines can be selected for a task
        for i in range(nb_tasks):
            for m in range(nb_machines):
                if task_processing_time_data[i][m] == INFINITE:
                    model.constraint(model.not_(model.contains(jobs_order[m], i)))

        # For each task, the selected machine
        task_machine = [model.find(machines, i) for i in range(nb_tasks)]

        task_processing_time = model.array(task_processing_time_data)

        # Interval decisions: time range of each task
        tasks = [model.interval(0, max_end) for _ in range(nb_tasks)]

        # The task duration depends on the selected machine
        duration = [model.at(task_processing_time, i, task_machine[i]) for i in range(nb_tasks)]
        for i in range(nb_tasks):
            model.constraint(model.length(tasks[i]) == duration[i])

        task_array = model.array(tasks)

        # Precedence constraints between the operations of a job
        for j in range(nb_jobs):
            for o in range(nb_operations[j] - 1):
                i1 = job_operation_task[j][o]
                i2 = job_operation_task[j][o + 1]
                model.constraint(tasks[i1] < tasks[i2])

        # Disjunctive resource constraints between the tasks on a machine
        for m in range(nb_machines):
            sequence = jobs_order[m]
            sequence_lambda = model.lambda_function(
                lambda i: task_array[sequence[i]] < task_array[sequence[i + 1]])
            model.constraint(model.and_(model.range(0, model.count(sequence) - 1), sequence_lambda))

        # Minimize the makespan: end of the last task
        makespan = model.max([model.end(tasks[i]) for i in range(nb_tasks)])
        model.minimize(makespan)

        model.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = time_limit

        optimizer.solve()

        # Write the solution in a file with the following format:
        # - for each operation of each job, the selected machine, the start and end dates
        if output_file != None:
            with open(output_file, "w") as f:
                print("Solution written in file", output_file)
                for j in range(nb_jobs):
                    for o in range(0, nb_operations[j]):
                        taskIndex = job_operation_task[j][o]
                        f.write(str(j + 1) + "\t" + str(o + 1)
                                + "\t" + str(task_machine[taskIndex].value + 1)
                                + "\t" + str(tasks[taskIndex].value.start())
                                + "\t" + str(tasks[taskIndex].value.end()) + "\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python flexible_jobshop.py instance_file [output_file] [time_limit]")
        sys.exit(1)

    instance_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 else None
    time_limit = int(sys.argv[3]) if len(sys.argv) >= 4 else 60
    main(instance_file, output_file, time_limit)
```

Flexible Job Shop Scheduling Problem with Sequence-Dependent Setup Times (FJSP-SDST)

```python
import hexaly.optimizer
import sys

# Constant for incompatible machines
INFINITE = 1000000


def read_instance(filename):
    with open(filename) as f:
        lines = f.readlines()

    first_line = lines[0].split()
    # Number of jobs
    nb_jobs = int(first_line[0])
    # Number of machines
    nb_machines = int(first_line[1])

    # Number of operations for each job
    nb_operations = [int(lines[j + 1].split()[0]) for j in range(nb_jobs)]

    # Number of tasks
    nb_tasks = sum(nb_operations[j] for j in range(nb_jobs))

    # Processing time for each task, for each machine
    task_processing_time = [[INFINITE for m in range(nb_machines)] for i in range(nb_tasks)]

    # For each job, for each operation, the corresponding task id
    job_operation_task = [[0 for o in range(nb_operations[j])] for j in range(nb_jobs)]

    # Setup time between every two consecutive tasks, for each machine
    task_setup_time = [[[-1 for r in range(nb_tasks)] for i in range(nb_tasks)] for m in range(nb_machines)]

    id = 0
    for j in range(nb_jobs):
        line = lines[j + 1].split()
        tmp = 0
        for o in range(nb_operations[j]):
            nb_machines_operation = int(line[tmp + o + 1])
            for i in range(nb_machines_operation):
                machine = int(line[tmp + o + 2 * i + 2]) - 1
                time = int(line[tmp + o + 2 * i + 3])
                task_processing_time[id][machine] = time
            job_operation_task[j][o] = id
            id = id + 1
            tmp = tmp + 2 * nb_machines_operation

    id_line = nb_jobs + 2
    max_setup = 0
    for m in range(nb_machines):
        for i1 in range(nb_tasks):
            task_setup_time[m][i1] = list(map(int, lines[id_line].split()))
            max_setup = max(max_setup, max(s if s != INFINITE else 0 for s in task_setup_time[m][i1]))
            id_line += 1

    # Trivial upper bound for the end times of the tasks
    max_sum_processing_times = sum(
        max(task_processing_time[i][m] for m in range(nb_machines) if task_processing_time[i][m] != INFINITE)
        for i in range(nb_tasks))
    max_end = max_sum_processing_times + nb_tasks * max_setup

    return nb_jobs, nb_machines, nb_tasks, task_processing_time, job_operation_task, \
        nb_operations, task_setup_time, max_end


def main(instance_file, output_file, time_limit):
    nb_jobs, nb_machines, nb_tasks, task_processing_time_data, job_operation_task, \
        nb_operations, task_setup_time_data, max_end = read_instance(instance_file)

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Sequence of tasks on each machine
        jobs_order = [model.list(nb_tasks) for _ in range(nb_machines)]
        machines = model.array(jobs_order)

        # Each task is scheduled on a machine
        model.constraint(model.partition(machines))

        # Only compatible machines can be selected for a task
        for i in range(nb_tasks):
            for m in range(nb_machines):
                if task_processing_time_data[i][m] == INFINITE:
                    model.constraint(model.not_(model.contains(jobs_order[m], i)))

        # For each task, the selected machine
        task_machine = [model.find(machines, i) for i in range(nb_tasks)]

        task_processing_time = model.array(task_processing_time_data)
        task_setup_time = model.array(task_setup_time_data)

        # Interval decisions: time range of each task
        tasks = [model.interval(0, max_end) for _ in range(nb_tasks)]

        # The task duration depends on the selected machine
        duration = [model.at(task_processing_time, i, task_machine[i]) for i in range(nb_tasks)]
        for i in range(nb_tasks):
            model.constraint(model.length(tasks[i]) == duration[i])

        task_array = model.array(tasks)

        # Precedence constraints between the operations of a job
        for j in range(nb_jobs):
            for o in range(nb_operations[j] - 1):
                i1 = job_operation_task[j][o]
                i2 = job_operation_task[j][o + 1]
                model.constraint(tasks[i1] < tasks[i2])

        # Disjunctive resource constraints between the tasks on a machine
        for m in range(nb_machines):
            sequence = jobs_order[m]
            sequence_lambda = model.lambda_function(
                lambda i: model.start(task_array[sequence[i + 1]]) >= model.end(task_array[sequence[i]])
                + model.at(task_setup_time, m, sequence[i], sequence[i + 1]))
            model.constraint(model.and_(model.range(0, model.count(sequence) - 1), sequence_lambda))

        # Minimize the makespan: end of the last task
        makespan = model.max([model.end(tasks[i]) for i in range(nb_tasks)])
        model.minimize(makespan)

        model.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = time_limit

        optimizer.solve()

        # Write the solution in a file with the following format:
        # - for each operation of each job, the selected machine, the start and end dates
        if output_file != None:
            with open(output_file, "w") as f:
                print("Solution written in file", output_file)
                for j in range(nb_jobs):
                    for o in range(0, nb_operations[j]):
                        taskIndex = job_operation_task[j][o]
                        f.write(str(j + 1) + "\t" + str(o + 1) + "\t" + str(task_machine[taskIndex].value + 1)
                                + "\t" + str(tasks[taskIndex].value.start())
                                + "\t" + str(tasks[taskIndex].value.end()) + "\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python flexible_jobshop_setup.py instance_file [output_file] [time_limit]")
        sys.exit(1)

    instance_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 else None
    time_limit = int(sys.argv[3]) if len(sys.argv) >= 4 else 60
    main(instance_file, output_file, time_limit)
```


Flexible Job Shop Scheduling Problem with Machine-Dependent Changeover Times

```python
import hexaly.optimizer
import sys

# Constant for incompatible machines
INFINITE = 1000000


def read_instance(filename):
    with open(filename) as f:
        lines = f.readlines()

    first_line = lines[0].split()
    # Number of jobs
    nb_jobs = int(first_line[0])
    # Number of machines
    nb_machines = int(first_line[1])

    # Number of operations for each job
    nb_operations = [int(lines[j + 1].split()[0]) for j in range(nb_jobs)]

    # Number of tasks
    nb_tasks = sum(nb_operations[j] for j in range(nb_jobs))

    # Processing time for each task, for each machine
    task_processing_time = [[INFINITE for m in range(nb_machines)] for i in range(nb_tasks)]

    # For each job, for each operation, the corresponding task id
    job_operation_task = [[0 for o in range(nb_operations[j])] for j in range(nb_jobs)]

    id = 0
    for j in range(nb_jobs):
        line = lines[j + 1].split()
        tmp = 0
        for o in range(nb_operations[j]):
            nb_machines_operation = int(line[tmp + o + 1])
            for i in range(nb_machines_operation):
                machine = int(line[tmp + o + 2 * i + 2]) - 1
                time = int(line[tmp + o + 2 * i + 3])
                task_processing_time[id][machine] = time
            job_operation_task[j][o] = id
            id = id + 1
            tmp = tmp + 2 * nb_machines_operation

    # Changeover time between two machines
    machine_changeover_time = [[0 for m2 in range(nb_machines)] for m1 in range(nb_machines)]

    for m1 in range(nb_machines):
        line = lines[nb_jobs + 1 + m1].split()
        for m2 in range(nb_machines):
            machine_changeover_time[m1][m2] = int(line[m2])


    # Trivial upper bound for the end times of the tasks
    max_end = sum(
        max(task_processing_time[i][m] for m in range(nb_machines) if task_processing_time[i][m] != INFINITE)
        for i in range(nb_tasks))
    
    return nb_jobs, nb_machines, nb_tasks, task_processing_time, job_operation_task, nb_operations, max_end, machine_changeover_time


def main(instance_file, output_file, time_limit):
    nb_jobs, nb_machines, nb_tasks, task_processing_time_data, job_operation_task, \
        nb_operations, max_end, machine_changeover_time_data = read_instance(instance_file)

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Sequence of tasks on each machine
        jobs_order = [model.list(nb_tasks) for _ in range(nb_machines)]
        machines = model.array(jobs_order)

        # Each task is scheduled on a machine
        model.constraint(model.partition(machines))

        # Only compatible machines can be selected for a task
        for i in range(nb_tasks):
            for m in range(nb_machines):
                if task_processing_time_data[i][m] == INFINITE:
                    model.constraint(model.not_(model.contains(jobs_order[m], i)))

        # For each task, the selected machine
        task_machine = [model.find(machines, i) for i in range(nb_tasks)]

        task_processing_time = model.array(task_processing_time_data)

        # Interval decisions: time range of each task
        tasks = [model.interval(0, max_end) for _ in range(nb_tasks)]

        # The task duration depends on the selected machine
        duration = [model.at(task_processing_time, i, task_machine[i]) for i in range(nb_tasks)]
        for i in range(nb_tasks):
            model.constraint(model.length(tasks[i]) == duration[i])

        task_array = model.array(tasks)

        machine_changeover_time = model.array(machine_changeover_time_data)
        # Precedence constraints between the operations of a job with machine-dependent changeover times
        for j in range(nb_jobs):
            for o in range(nb_operations[j] - 1):
                i1 = job_operation_task[j][o]
                i2 = job_operation_task[j][o + 1]
                model.constraint(model.start(tasks[i2]) >= model.end(tasks[i1]) 
                        + machine_changeover_time[task_machine[i1]][task_machine[i2]])

        # Disjunctive resource constraints between the tasks on a machine
        for m in range(nb_machines):
            sequence = jobs_order[m]
            sequence_lambda = model.lambda_function(
                lambda i: task_array[sequence[i]] < task_array[sequence[i + 1]])
            model.constraint(model.and_(model.range(0, model.count(sequence) - 1), sequence_lambda))

        # Minimize the makespan: end of the last task
        makespan = model.max([model.end(tasks[i]) for i in range(nb_tasks)])
        model.minimize(makespan)

        model.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = time_limit

        optimizer.solve()

        # Write the solution in a file with the following format:
        # - for each operation of each job, the selected machine, the start and end dates
        if output_file != None:
            with open(output_file, "w") as f:
                print("Solution written in file", output_file)
                for j in range(nb_jobs):
                    for o in range(0, nb_operations[j]):
                        taskIndex = job_operation_task[j][o]
                        f.write(str(j + 1) + "\t" + str(o + 1)
                                + "\t" + str(task_machine[taskIndex].value + 1)
                                + "\t" + str(tasks[taskIndex].value.start())
                                + "\t" + str(tasks[taskIndex].value.end()) + "\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python flexiblejobshop_changeover.py instance_file [output_file] [time_limit]")
        sys.exit(1)

    instance_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 else None
    time_limit = int(sys.argv[3]) if len(sys.argv) >= 4 else 60
    main(instance_file, output_file, time_limit)
```


Open Shop Scheduling Problem

```python
import hexaly.optimizer
import sys


def read_instance(filename):
    # The input files follow the "Taillard" format
    with open(filename, 'r') as f:
        lines = f.readlines()

    first_line = lines[1].split()
    nb_jobs = int(first_line[0])
    nb_machines = int(first_line[1])

    # Processing times for each job on each machine
    # (given in the task order, the processing order is a decision variable)
    processing_times_task_order = [[int(proc_time) for proc_time in line.split()]
                                   for line in lines[3:3 + nb_jobs]]

    # Index of machines for each task
    machine_index = [[int(machine_i) - 1 for machine_i in line.split()]
                     for line in lines[4 + nb_jobs:4 + 2 * nb_jobs]]

    # Reorder processing times: processingTime[j][m] is the processing time of the
    # task of job j that is processed on machine m
    processing_times = [[processing_times_task_order[j][machine_index[j].index(m)]
                         for m in range(nb_machines)] for j in range(nb_jobs)]

    # Trivial upper bound for the end time of tasks
    max_end = sum(map(lambda processing_times_job: sum(processing_times_job), processing_times))

    return nb_jobs, nb_machines, processing_times, max_end


def main(instance_file, output_file, time_limit):
    nb_jobs, nb_machines, processing_times, max_end = read_instance(instance_file)

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Interval decisions: time range of each task
        # tasks[j][m] is the interval of time of the task of job j
        # which is processed on machine m
        tasks = [[model.interval(0, max_end) for _ in range(nb_machines)] for _ in range(nb_jobs)]

        # Task duration constraints
        for j in range(nb_jobs):
            for m in range(0, nb_machines):
                model.constraint(model.length(tasks[j][m]) == processing_times[j][m])

        # Create an Hexaly array in order to be able to access it with "at" operators
        task_array = model.array(tasks)

        # List of the jobs on each machine
        jobs_order = [model.list(nb_jobs) for _ in range(nb_machines)]
        for m in range(nb_machines):
            # Each job is scheduled on every machine
            model.constraint(model.eq(model.count(jobs_order[m]), nb_jobs))

            # Every machine executes a single task at a time
            sequence_lambda = model.lambda_function(lambda i:
                model.at(task_array, jobs_order[m][i], m) < model.at(task_array, jobs_order[m][i + 1], m))
            model.constraint(model.and_(model.range(0, nb_jobs - 1), sequence_lambda))

        # List of the machines for each job
        machines_order = [model.list(nb_machines) for _ in range(nb_jobs)]
        for j in range(nb_jobs):
            # Every task is scheduled on its corresponding machine
            model.constraint(model.eq(model.count(machines_order[j]), nb_machines))

            # A job has a single task at a time
            sequence_lambda = model.lambda_function(lambda k:
                    model.at(task_array, j, machines_order[j][k]) < model.at(task_array, j, machines_order[j][k + 1]))
            model.constraint(model.and_(model.range(0, nb_machines - 1), sequence_lambda))

        # Minimize the makespan: the end of the last task
        makespan = model.max([model.end(model.at(task_array, j, m))
                             for j in range(nb_jobs) for m in range(nb_machines)])
        model.minimize(makespan)

        model.close()

        # Parametrize the optimizer
        optimizer.param.time_limit = time_limit
        optimizer.solve()

        #
        # Write the solution in a file with the following format:
        #  - for each machine, the job sequence
        #
        if output_file is not None:
            with open(output_file, 'w') as f:
                for m in range(nb_machines):
                    line = ""
                    for j in range(nb_jobs):
                        line += str(jobs_order[m].value[j]) + " "
                    f.write(line + "\n")
            print("Solution written in file ", output_file)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(
            "Usage: python openshop.py instance_file [output_file] [time_limit]")
        sys.exit(1)

    instance_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 else None
    time_limit = int(sys.argv[3]) if len(sys.argv) >= 4 else 60
    main(instance_file, output_file, time_limit)
```


Stochastic Packing

```python
from __future__ import print_function
import random
import math
import hexaly.optimizer


def generate_scenarios(nb_items, nb_scenarios, rng_seed):
    random.seed(rng_seed)

    # Pick random parameters for each item distribution
    items_dist = []
    for _ in range(nb_items):
        item_min = random.randint(10, 100)
        item_max = item_min + random.randint(0, 50)
        items_dist.append((item_min, item_max))

    # Sample the distributions to generate the scenarios
    scenario_item_weights = [[random.randint(*dist) for dist in items_dist]
                               for _ in range(nb_scenarios)]
    return scenario_item_weights


def main(nb_items, nb_bins, nb_scenarios, seed, time_limit):
    # Generate instance data
    scenario_item_weights_data = generate_scenarios(nb_items, nb_scenarios, seed)

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Set decisions: bins[k] represents the items in bin k
        bins = [model.set(nb_items) for _ in range(nb_bins)]

        # Each item must be in one bin and one bin only
        model.constraint(model.partition(bins))

        scenarios_item_weights = model.array(scenario_item_weights_data)

        # Compute max weight for each scenario
        scenarios_max_weights = model.array(
            model.max(
                model.sum(bin,
                          model.lambda_function(
                              lambda i:
                                  model.at(scenarios_item_weights, k, i)))
                for bin in bins) for k in range(nb_scenarios))

        # Compute the 9th decile of scenario max weights
        stochastic_max_weight = \
            model.sort(scenarios_max_weights)[int(math.ceil(0.9 * (nb_scenarios - 1)))]

        model.minimize(stochastic_max_weight)
        model.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = time_limit

        optimizer.solve()

        #
        # Write the solution
        #
        print()
        print("Scenario item weights:")
        for i, scenario in enumerate(scenario_item_weights_data):
            print(i, ': ', scenario, sep='')

        print()
        print("Bins:")
        for k, bin in enumerate(bins):
            print(k, ': ', bin.value, sep='')


if __name__ == '__main__':
    nb_items = 10
    nb_bins = 2
    nb_scenarios = 3
    rng_seed = 42
    time_limit = 2

    main(
        nb_items,
        nb_bins,
        nb_scenarios,
        rng_seed,
        time_limit
    )


```

Stochastic Job Shop Scheduling Problem

```python
import hexaly.optimizer
import sys


def read_instance(filename):
    with open(filename) as f:
        lines = f.readlines()

    first_line = lines[1].split()
    # Number of jobs
    nb_jobs = int(first_line[0])
    # Number of machines
    nb_machines = int(first_line[1])
    # Number of scenarios
    nb_scenarios = int(first_line[2])

    # Processing times for each job on each machine (given in the processing order)
    processing_times_in_processing_order_per_scenario = [[[int(lines[s*(nb_jobs+1)+i].split()[j])
                                                           for j in range(nb_machines)]
                                                          for i in range(3, 3 + nb_jobs)]
                                                         for s in range(nb_scenarios)]

    # Processing order of machines for each job
    machine_order = [[int(lines[i].split()[j]) - 1 for j in range(nb_machines)]
                     for i in range(4 + nb_scenarios*(nb_jobs+1), 4 + nb_scenarios*(nb_jobs+1) + nb_jobs)]

    # Reorder processing times: processing_time[s][j][m] is the processing time of the
    # task of job j that is processed on machine m in the scenario s
    processing_time_per_scenario = [[[processing_times_in_processing_order_per_scenario[s][j][machine_order[j].index(m)]
                                      for m in range(nb_machines)]
                                     for j in range(nb_jobs)]
                                    for s in range(nb_scenarios)]

    # Trivial upper bound for the end times of the tasks
    max_end = max([sum(sum(processing_time_per_scenario[s][j])
                    for j in range(nb_jobs)) for s in range(nb_scenarios)])

    return nb_jobs, nb_machines, nb_scenarios, processing_time_per_scenario, machine_order, max_end


def main(instance_file, output_file, time_limit):
    nb_jobs, nb_machines, nb_scenarios, processing_time_per_scenario, machine_order, max_end = read_instance(
        instance_file)

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Interval decisions: time range of each task
        # tasks[s][j][m] is the interval of time of the task of job j which is processed
        # on machine m in the scenario s
        tasks = [[[model.interval(0, max_end) for m in range(nb_machines)]
                  for j in range(nb_jobs)]
                 for s in range(nb_scenarios)]

        # Task duration constraints
        for s in range(nb_scenarios):
            for j in range(nb_jobs):
                for m in range(0, nb_machines):
                    model.constraint(model.length(tasks[s][j][m]) == processing_time_per_scenario[s][j][m])

        # Create an Hexaly array in order to be able to access it with "at" operators
        task_array = model.array(tasks)

        # Precedence constraints between the tasks of a job
        for s in range(nb_scenarios):
            for j in range(nb_jobs):
                for k in range(nb_machines - 1):
                    model.constraint(
                        tasks[s][j][machine_order[j][k]] < tasks[s][j][machine_order[j][k + 1]])

        # Sequence of tasks on each machine
        jobs_order = [model.list(nb_jobs) for m in range(nb_machines)]

        for m in range(nb_machines):
            # Each job has a task scheduled on each machine
            sequence = jobs_order[m]
            model.constraint(model.eq(model.count(sequence), nb_jobs))

            # Disjunctive resource constraints between the tasks on a machine
            for s in range(nb_scenarios):
                sequence_lambda = model.lambda_function(
                    lambda i: model.lt(model.at(task_array, s, sequence[i], m),
                                       model.at(task_array, s, sequence[i + 1], m)))
                model.constraint(model.and_(model.range(0, nb_jobs - 1), sequence_lambda))

        # Minimize the maximum makespan: end of the last task of the last job
        # over all scenarios
        makespans = [model.max([model.end(tasks[s][j][machine_order[j][nb_machines - 1]]) for j in range(nb_jobs)])
                     for s in range(nb_scenarios)]
        max_makespan = model.max(makespans)
        model.minimize(max_makespan)

        model.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = time_limit

        optimizer.solve()

        #
        # Write the solution in a file with the following format:
        # - for each machine, the job sequence
        #
        if output_file != None:
            final_jobs_order = [list(jobs_order[m].value) for m in range(nb_machines)]
            with open(output_file, "w") as f:
                print("Solution written in file ", output_file)
                for m in range(nb_machines):
                    for j in range(nb_jobs):
                        f.write(str(final_jobs_order[m][j]) + " ")
                    f.write("\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python stochastic_jobshop.py instance_file [output_file] [time_limit]")
        sys.exit(1)

    instance_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 else None
    time_limit = int(sys.argv[3]) if len(sys.argv) >= 4 else 60
    main(instance_file, output_file, time_limit)
```


Resource Constrained Project Scheduling Problem (RCPSP)

```python
import hexaly.optimizer
import sys


# The input files follow the "Patterson" format
def read_instance(filename):
    with open(filename) as f:
        lines = f.readlines()

    first_line = lines[0].split()

    # Number of tasks
    nb_tasks = int(first_line[0])

    # Number of resources
    nb_resources = int(first_line[1])

    # Maximum capacity of each resource
    capacity = [int(lines[1].split()[r]) for r in range(nb_resources)]

    # Duration of each task
    duration = [0 for i in range(nb_tasks)]

    # Resource weight of resource r required for task i
    weight = [[] for i in range(nb_tasks)]

    # Number of successors
    nb_successors = [0 for i in range(nb_tasks)]

    # Successors of each task i
    successors = [[] for i in range(nb_tasks)]

    for i in range(nb_tasks):
        line = lines[i + 2].split()
        duration[i] = int(line[0])
        weight[i] = [int(line[r + 1]) for r in range(nb_resources)]
        nb_successors[i] = int(line[nb_resources + 1])
        successors[i] = [int(line[nb_resources + 2 + s]) - 1 for s in range(nb_successors[i])]

    # Trivial upper bound for the end times of the tasks
    horizon = sum(duration[i] for i in range(nb_tasks))

    return (nb_tasks, nb_resources, capacity, duration, weight, nb_successors, successors, horizon)


def main(instance_file, output_file, time_limit):
    nb_tasks, nb_resources, capacity, duration, weight, nb_successors, successors, horizon = read_instance(
        instance_file)

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Interval decisions: time range of each task
        tasks = [model.interval(0, horizon) for i in range(nb_tasks)]

        # Task duration constraints
        for i in range(nb_tasks):
            model.constraint(model.length(tasks[i]) == duration[i])

        # Precedence constraints between the tasks
        for i in range(nb_tasks):
            for s in range(nb_successors[i]):
                model.constraint(tasks[i] < tasks[successors[i][s]])

        # Makespan: end of the last task
        makespan = model.max([model.end(tasks[i]) for i in range(nb_tasks)])

        # Cumulative resource constraints
        for r in range(nb_resources):
            capacity_respected = model.lambda_function(
                lambda t: model.sum(weight[i][r] * model.contains(tasks[i], t)
                                    for i in range(nb_tasks))
                <= capacity[r])
            model.constraint(model.and_(model.range(makespan), capacity_respected))

        # Minimize the makespan
        model.minimize(makespan)

        model.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = time_limit

        optimizer.solve()

        #
        # Write the solution in a file with the following format:
        # - total makespan
        # - for each task, the task id, the start and end times
        #
        if output_file != None:
            with open(output_file, "w") as f:
                print("Solution written in file", output_file)
                f.write(str(makespan.value) + "\n")
                for i in range(nb_tasks):
                    f.write(str(i + 1) + " " + str(tasks[i].value.start()) + " " + str(tasks[i].value.end()))
                    f.write("\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python rcpsp.py instance_file [output_file] [time_limit]")
        sys.exit(1)

    instance_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 else None
    time_limit = int(sys.argv[3]) if len(sys.argv) >= 4 else 60
    main(instance_file, output_file, time_limit)
```



Preemptive Resource Constrained Project Scheduling Problem

```python
import hexaly.optimizer
import sys


# The input files follow the "Patterson" format
def read_instance(filename):
    with open(filename) as f:
        lines = f.readlines()

    first_line = lines[0].split()

    # Number of tasks
    nb_tasks = int(first_line[0])

    # Number of resources
    nb_resources = int(first_line[1])

    # Maximum capacity of each resource
    capacity = [int(lines[1].split()[r]) for r in range(nb_resources)]

    # Duration of each task
    duration = [0 for i in range(nb_tasks)]

    # Resource weight of resource r required for task i
    weight = [[] for i in range(nb_tasks)]

    # Number of successors
    nb_successors = [0 for i in range(nb_tasks)]

    # Successors of each task i
    successors = [[] for i in range(nb_tasks)]

    for i in range(nb_tasks):
        line = lines[i + 2].split()
        duration[i] = int(line[0])
        weight[i] = [int(line[r + 1]) for r in range(nb_resources)]
        nb_successors[i] = int(line[nb_resources + 1])
        successors[i] = [int(line[nb_resources + 2 + s]) - 1 for s in range(nb_successors[i])]

    # Trivial upper bound for the end times of the tasks
    horizon = sum(duration[i] for i in range(nb_tasks))

    # Number of intervals authorized for each task (pseudo preemption)
    max_nb_preemptions = 4 

    return (nb_tasks, nb_resources, capacity, duration, weight, nb_successors, successors, horizon, max_nb_preemptions)


def main(instance_file, output_file, time_limit):
    nb_tasks, nb_resources, capacity, duration, weight, nb_successors, successors, horizon, max_nb_preemptions = read_instance(
        instance_file)

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Interval decisions: time range of each task
        # Each task can be split into max_nb_preemtptions subtasks
        tasks = [[model.interval(0, horizon) for t in range(max_nb_preemptions)] 
                 for i in range(nb_tasks)]

        for i in range(nb_tasks):
            # Task duration constraints
            model.constraint(
                    model.sum(model.length(tasks[i][t]) for t in range(max_nb_preemptions)) == duration[i])
            
            # Precedence constraints between each task's subtasks
            for t in range(max_nb_preemptions - 1):
                model.constraint(tasks[i][t] < tasks[i][t + 1])
           
        # Precedence constraints between the tasks
        for i in range(nb_tasks):
            for s in range(nb_successors[i]):
                model.constraint(tasks[i][max_nb_preemptions - 1] < tasks[successors[i][s]][0])

        # Makespan: end of the last task
        makespan = model.max([model.end(tasks[i][max_nb_preemptions - 1]) for i in range(nb_tasks)])

        # Cumulative resource constraints
        for r in range(nb_resources):
            capacity_respected = model.lambda_function(
                lambda t: model.sum(weight[i][r] * model.contains(tasks[i][k], t)
                                    for i in range(nb_tasks) for k in range(max_nb_preemptions))
                <= capacity[r])
            model.constraint(model.and_(model.range(makespan), capacity_respected))

        # Minimize the makespan
        model.minimize(makespan)

        model.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = time_limit

        optimizer.solve()

        #
        # Write the solution in a file with the following format:
        # - total makespan
        # - for each task, the task id, the start and end times of each subtask
        #
        if output_file != None:
            with open(output_file, "w") as f:
                print("Solution written in file", output_file)
                f.write(str(makespan.value) + "\n")
                for i in range(nb_tasks):
                    start_time = tasks[i][0].value.start()
                    f.write(str(i + 1))
                    for k in range(max_nb_preemptions):
                        if tasks[i][k].value.end() != start_time :
                            f.write(" (" + str(start_time) + " " + str(tasks[i][k].value.end()) + ")")
                            if k < max_nb_preemptions - 1 :
                                start_time = tasks[i][k + 1].value.start()
                    f.write("\n")
                     
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python rcpsp.py instance_file [output_file] [time_limit]")
        sys.exit(1)

    instance_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 else None
    time_limit = int(sys.argv[3]) if len(sys.argv) >= 4 else 60
    main(instance_file, output_file, time_limit)
```


Batch Scheduling Problem

```python
import hexaly.optimizer
import sys

def read_instance(filename):
    # The import files follow the "Taillard" format
    with open(filename, 'r') as f:
        lines = f.readlines()

    first_line = lines[0].split()
    # Number of tasks
    nb_tasks = int(first_line[0])
    # Number of resources
    nb_resources = int(first_line[1])

    second_line = lines[1].split()
    # Capacity of each resource
    capacity = [int(l) for l in second_line]

    # Remaining lines contain task information at each row
    nb_tasks_per_resource = [0 for _ in range(nb_resources)]
    types_in_resource = [[] for _ in range(nb_resources)]
    tasks_in_resource = [[] for _ in range(nb_resources)]
    task_index_in_resource = []
    types, resources, duration, nb_successors = [], [], [], []
    successors = [[] for _ in range(nb_tasks)]

    for i in range(nb_tasks):

        # Extract dataset line related to this task
        task_line = i + 2
        task_information = lines[task_line].split()
        
        # Type of task i
        types.append(int(task_information[0]))
        # Resource required for task i
        resources.append(int(task_information[1]))
        
        # Index of task i on resource[i]
        task_index_in_resource.append(nb_tasks_per_resource[resources[i]])
        # Map from name of task i on resource[i] to task i type
        types_in_resource[resources[i]].append(types[i])
        # Map from name of task i on resource[i] to task i
        tasks_in_resource[resources[i]].append(i)
        # Increment number of tasks required by this resource
        nb_tasks_per_resource[resources[i]] += 1
        
        # Task duration
        duration.append(int(task_information[2]))

        # Number of successors of this task
        nb_successors.append(int(task_information[3]))
        # Tasks that must succeed current task
        for succeeding_task in task_information[4:]:
            successors[i].append(int(succeeding_task))

    # Trivial time horizon
    time_horizon = sum(duration[t] for t in range(nb_tasks))

    return nb_tasks, nb_resources, capacity, types, resources, duration, \
            nb_successors, successors, nb_tasks_per_resource, \
              task_index_in_resource, types_in_resource, tasks_in_resource, \
                nb_tasks_per_resource, time_horizon


def main(instance_file, output_file, time_limit):

    nb_tasks, nb_resources, capacity, types, resources, duration, \
            nb_successors, successors, nb_tasks_per_resource, \
              task_index_in_resource, types_in_resource, tasks_in_resource, \
                nb_tasks_per_resource, time_horizon \
                 = read_instance(instance_file)
    
    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # For each resource, the contents of each batch of tasks performed
        batch_content = [[model.set(nb_tasks_per_resource[r]) 
                          for b in range(nb_tasks_per_resource[r])]
                            for r in range(nb_resources)]
        
        # Create HexalyOptimizer arrays in order to be able to access them with "at" operators
        batch_content_arrays = [model.array(batch_content[r]) for r in range(nb_resources)]

        # All tasks are assigned to a batch
        for r in range(nb_resources):
            model.constraint(model.partition(batch_content_arrays[r]))

        # Each batch must consist of tasks with the same type
        types_in_resource_array = model.array(types_in_resource)
        for r in range(nb_resources):
            resource_type_lambda = model.lambda_function(lambda i: types_in_resource_array[r][i])
            for batch in batch_content[r]:
                model.constraint(model.count( model.distinct( batch, resource_type_lambda ) ) <= 1)

        # Each batch cannot exceed the maximum capacity of the resource
        for r in range(nb_resources):
            for batch in batch_content[r]:
                model.constraint(model.count(batch) <= capacity[r])

        # Interval decisions: time range of each batch of tasks
        batch_interval = [[model.interval(0, time_horizon) 
                           for _ in range(nb_tasks_per_resource[r])]
                            for r in range(nb_resources)]
        batch_interval_arrays = [model.array(batch_interval[r]) for r in range(nb_resources)]
    
        # Non-overlap of batch intervals on the same resource
        for r in range(nb_resources):
            for b in range(1, nb_tasks_per_resource[r]):
                model.constraint(batch_interval[r][b-1] < batch_interval[r][b])
        
        # Interval decisions: time range of each task
        task_interval = [None for _ in range(nb_tasks)]
        for t in range(nb_tasks):
            # Retrieve the batch index and resource for this task
            r = resources[t]
            b = model.find( batch_content_arrays[r], task_index_in_resource[t] )
            # Task interval interval associated with task t
            task_interval[t] = batch_interval_arrays[r][b]

        # Task durations
        for t in range(nb_tasks):
            model.constraint(model.length(task_interval[t]) == duration[t])

        # Precedence constraints between tasks
        for t in range(nb_tasks):
            for s in successors[t]:
                model.constraint( task_interval[t] < task_interval[s])

        # Makespan: end of the last task
        makespan = model.max([model.end(model.at(batch_interval_arrays[r], i))
                                for i in range(nb_tasks_per_resource[r]) 
                                    for r in range(nb_resources)])
        model.minimize(makespan)
        model.close()

        # Parametrize the optimizer
        optimizer.param.time_limit = time_limit
        optimizer.solve()

        #
        # Write the solution in a file with the following format:
        #  - makespan
        #  - machine number
        #  - preceeding lines are the ordered intervals of the tasks 
        #    for the corresponding machine number
        if output_file is not None:
            with open(output_file, 'w') as f:
                f.write(str(makespan.value) + "\n")
                for r in range(nb_resources):
                    f.write(str(r) + "\n")
                    for b in range(nb_tasks_per_resource[r]):
                        t = tasks_in_resource[r][b]
                        line = str(t) + " " + str(task_interval[t].value.start()) + " " + str(task_interval[t].value.end())
                        f.write(line + "\n")
            print("Solution written in file ", output_file)


if __name__=="__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python batch_scheduling.py instance_file [output_file] [time_limit]")
        sys.exit(1)
    
    instance_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 else None
    time_limit = int(sys.argv[3]) if len(sys.argv) >= 4 else 60

    main(instance_file, output_file, time_limit)
```

Project Scheduling Problem with Production and Consumption of Resources

```python
import hexaly.optimizer
import sys


# The input files follow the "Patterson" format
def read_instance(filename):
    with open(filename) as f:
        lines = f.readlines()

    first_line = lines[0].split()

    # Number of tasks
    nb_tasks = int(first_line[0])

    # Number of resources
    nb_resources = int(first_line[1])

    # Number of inventories
    nb_inventories = int(first_line[2])
    
    second_line = lines[1].split()

    # Maximum capacity of each resource
    capacity = [int(second_line[r]) for r in range(nb_resources)]

    # Initial level of each inventory
    init_level = [int(second_line[r + nb_resources]) for r in range(nb_inventories)]

    # Duration of each task
    duration = [0 for i in range(nb_tasks)]

    # Resource weight of resource r required for task i
    weight = [[] for i in range(nb_tasks)]

    # Inventory consumed at beginning of task i
    start_cons = [[] for i in range(nb_tasks)]

    # Inventory produced at end of task i
    end_prod = [[] for i in range(nb_tasks)]

    # Number of successors
    nb_successors = [0 for i in range(nb_tasks)]

    # Successors of each task i
    successors = [[] for i in range(nb_tasks)]

    for i in range(nb_tasks):
        line = lines[i + 2].split()
        duration[i] = int(line[0])
        weight[i] = [int(line[r + 1]) for r in range(nb_resources)]
        start_cons[i] = [int(line[2*r + nb_resources + 1]) for r in range(nb_inventories)]
        end_prod[i] = [int(line[2*r + nb_resources + 2]) for r in range(nb_inventories)]
        nb_successors[i] = int(line[2*nb_inventories + nb_resources + 1])
        successors[i] = [int(line[2*nb_inventories + nb_resources + 2 + s]) - 1 for s in range(nb_successors[i])]

    # Trivial upper bound for the end times of the tasks
    horizon = sum(duration[i] for i in range(nb_tasks))

    return (nb_tasks, nb_resources, nb_inventories, capacity, init_level, duration, weight, start_cons, end_prod, nb_successors, successors, horizon)


def main(instance_file, output_file, time_limit):
    nb_tasks, nb_resources, nb_inventories, capacity, init_level, duration, weight, start_cons, end_prod, nb_successors, successors, horizon = read_instance(
        instance_file)

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Interval decisions: time range of each task
        tasks = [model.interval(0, horizon) for _ in range(nb_tasks)]

        # Task duration constraints
        for i in range(nb_tasks):
            model.constraint(model.length(tasks[i]) == duration[i])

        # Precedence constraints between the tasks
        for i in range(nb_tasks):
            for s in range(nb_successors[i]):
                model.constraint(tasks[i] < tasks[successors[i][s]])

        # Makespan: end of the last task
        makespan = model.max([model.end(tasks[i]) for i in range(nb_tasks)])

        # Cumulative resource constraints
        for r in range(nb_resources):
            capacity_respected = model.lambda_function(
                lambda t: model.sum(weight[i][r] * model.contains(tasks[i], t)
                                    for i in range(nb_tasks))
                <= capacity[r])
            model.constraint(model.and_(model.range(makespan), capacity_respected))

        # Non-negative inventory constraints
        for r in range(nb_resources):
            inventory_value = model.lambda_function(
                lambda t: model.sum(end_prod[i][r] * (model.end(tasks[i]) <= t)
                                        - start_cons[i][r] * (model.start(tasks[i]) <= t)
                                    for i in range(nb_tasks)) 
                                    + init_level[r]
                >= 0)
            model.constraint(model.and_(model.range(makespan), inventory_value))

        # Minimize the makespan
        model.minimize(makespan)

        model.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = time_limit

        optimizer.solve()

        #
        # Write the solution in a file with the following format:
        # - total makespan
        # - for each task, the task id, the start and end times
        #
        if output_file != None:
            with open(output_file, "w") as f:
                print("Solution written in file", output_file)
                f.write(str(makespan.value) + "\n")
                for i in range(nb_tasks):
                    f.write(str(i + 1) + " " + str(tasks[i].value.start()) + " " + str(tasks[i].value.end()))
                    f.write("\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python rcpsp_producer_consumer.py instance_file [output_file] [time_limit]")
        sys.exit(1)

    instance_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 else None
    time_limit = int(sys.argv[3]) if len(sys.argv) >= 4 else 60
    main(instance_file, output_file, time_limit)
```

Travelling Salesman Problem (TSP)

```python
import hexaly.optimizer
import sys

if len(sys.argv) < 2:
    print("Usage: python tsp.py inputFile [outputFile] [timeLimit]")
    sys.exit(1)


def read_elem(filename):
    with open(filename) as f:
        return [str(elem) for elem in f.read().split()]


with hexaly.optimizer.HexalyOptimizer() as optimizer:
    #
    # Read instance data
    #
    file_it = iter(read_elem(sys.argv[1]))

    # The input files follow the TSPLib "explicit" format
    for pch in file_it:
        if pch == "DIMENSION:":
            nb_cities = int(next(file_it))
        if pch == "EDGE_WEIGHT_SECTION":
            break

    # Distance from i to j
    dist_matrix_data = [[int(next(file_it)) for i in range(nb_cities)]
                        for j in range(nb_cities)]

    #
    # Declare the optimization model
    #
    model = optimizer.model

    # A list variable: cities[i] is the index of the ith city in the tour
    cities = model.list(nb_cities)

    # All cities must be visited
    model.constraint(model.count(cities) == nb_cities)

    # Create an Hexaly array for the distance matrix in order to be able
    # to access it with "at" operators
    dist_matrix = model.array(dist_matrix_data)

    # Minimize the total distance
    dist_lambda = model.lambda_function(lambda i:
                                        model.at(dist_matrix, cities[i - 1], cities[i]))
    obj = model.sum(model.range(1, nb_cities), dist_lambda) \
        + model.at(dist_matrix, cities[nb_cities - 1], cities[0])
    model.minimize(obj)

    model.close()

    # Parameterize the optimizer
    if len(sys.argv) >= 4:
        optimizer.param.time_limit = int(sys.argv[3])
    else:
        optimizer.param.time_limit = 5

    optimizer.solve()

    #
    # Write the solution in a file
    #
    if len(sys.argv) >= 3:
        # Write the solution in a file
        with open(sys.argv[2], 'w') as f:
            f.write("%d\n" % obj.value)
            for c in cities.value:
                f.write("%d " % c)
            f.write("\n")
```


Capacitated Vehicle Routing (CVRP)

```python
import hexaly.optimizer
import sys
import math


def read_elem(filename):
    with open(filename) as f:
        return [str(elem) for elem in f.read().split()]


def main(instance_file, str_time_limit, output_file, str_nb_trucks):
    nb_trucks = int(str_nb_trucks)

    #
    # Read instance data
    #
    nb_customers, truck_capacity, dist_matrix_data, dist_depot_data, \
        demands_data = read_input_cvrp(instance_file)

    # The number of trucks is usually given in the name of the file
    # nb_trucks can also be given in command line
    if nb_trucks == 0:
        nb_trucks = get_nb_trucks(instance_file)

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Sequence of customers visited by each truck
        customers_sequences = [model.list(nb_customers) for _ in range(nb_trucks)]

        # All customers must be visited by exactly one truck
        model.constraint(model.partition(customers_sequences))

        # Create Hexaly arrays to be able to access them with an "at" operator
        demands = model.array(demands_data)
        dist_matrix = model.array(dist_matrix_data)
        dist_depot = model.array(dist_depot_data)

        # A truck is used if it visits at least one customer
        trucks_used = [(model.count(customers_sequences[k]) > 0) for k in range(nb_trucks)]

        dist_routes = [None] * nb_trucks
        for k in range(nb_trucks):
            sequence = customers_sequences[k]
            c = model.count(sequence)

            # The quantity needed in each route must not exceed the truck capacity
            demand_lambda = model.lambda_function(lambda j: demands[j])
            route_quantity = model.sum(sequence, demand_lambda)
            model.constraint(route_quantity <= truck_capacity)

            # Distance traveled by each truck
            dist_lambda = model.lambda_function(lambda i:
                                                model.at(dist_matrix,
                                                         sequence[i - 1],
                                                         sequence[i]))
            dist_routes[k] = model.sum(model.range(1, c), dist_lambda) \
                + model.iif(c > 0,
                            dist_depot[sequence[0]] + dist_depot[sequence[c - 1]],
                            0)

        # Total number of trucks used
        nb_trucks_used = model.sum(trucks_used)

        # Total distance traveled
        total_distance = model.sum(dist_routes)

        # Objective: minimize the number of trucks used, then minimize the distance traveled
        model.minimize(nb_trucks_used)
        model.minimize(total_distance)

        model.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = int(str_time_limit)

        optimizer.solve()

        #
        # Write the solution in a file with the following format:
        #  - number of trucks used and total distance
        #  - for each truck the customers visited (omitting the start/end at the depot)
        #
        if output_file is not None:
            with open(output_file, 'w') as f:
                f.write("%d %d\n" % (nb_trucks_used.value, total_distance.value))
                for k in range(nb_trucks):
                    if trucks_used[k].value != 1:
                        continue
                    # Values in sequence are in 0...nbCustomers. +2 is to put it back
                    # in 2...nbCustomers+2 as in the data files (1 being the depot)
                    for customer in customers_sequences[k].value:
                        f.write("%d " % (customer + 2))
                    f.write("\n")


# The input files follow the "Augerat" format
def read_input_cvrp(filename):
    file_it = iter(read_elem(filename))

    nb_nodes = 0
    while True:
        token = next(file_it)
        if token == "DIMENSION":
            next(file_it)  # Removes the ":"
            nb_nodes = int(next(file_it))
            nb_customers = nb_nodes - 1
        elif token == "CAPACITY":
            next(file_it)  # Removes the ":"
            truck_capacity = int(next(file_it))
        elif token == "EDGE_WEIGHT_TYPE":
            next(file_it)  # Removes the ":"
            token = next(file_it)
            if token != "EUC_2D":
                print("Edge Weight Type " + token + " is not supported (only EUD_2D)")
                sys.exit(1)
        elif token == "NODE_COORD_SECTION":
            break

    customers_x = [None] * nb_customers
    customers_y = [None] * nb_customers
    depot_x = 0
    depot_y = 0
    for n in range(nb_nodes):
        node_id = int(next(file_it))
        if node_id != n + 1:
            print("Unexpected index")
            sys.exit(1)
        if node_id == 1:
            depot_x = int(next(file_it))
            depot_y = int(next(file_it))
        else:
            # -2 because original customer indices are in 2..nbNodes
            customers_x[node_id - 2] = int(next(file_it))
            customers_y[node_id - 2] = int(next(file_it))

    distance_matrix = compute_distance_matrix(customers_x, customers_y)
    distance_depots = compute_distance_depots(depot_x, depot_y, customers_x, customers_y)

    token = next(file_it)
    if token != "DEMAND_SECTION":
        print("Expected token DEMAND_SECTION")
        sys.exit(1)

    demands = [None] * nb_customers
    for n in range(nb_nodes):
        node_id = int(next(file_it))
        if node_id != n + 1:
            print("Unexpected index")
            sys.exit(1)
        if node_id == 1:
            if int(next(file_it)) != 0:
                print("Demand for depot should be 0")
                sys.exit(1)
        else:
            # -2 because original customer indices are in 2..nbNodes
            demands[node_id - 2] = int(next(file_it))

    token = next(file_it)
    if token != "DEPOT_SECTION":
        print("Expected token DEPOT_SECTION")
        sys.exit(1)

    depot_id = int(next(file_it))
    if depot_id != 1:
        print("Depot id is supposed to be 1")
        sys.exit(1)

    end_of_depot_section = int(next(file_it))
    if end_of_depot_section != -1:
        print("Expecting only one depot, more than one found")
        sys.exit(1)

    return nb_customers, truck_capacity, distance_matrix, distance_depots, demands


# Compute the distance matrix
def compute_distance_matrix(customers_x, customers_y):
    nb_customers = len(customers_x)
    distance_matrix = [[None for i in range(nb_customers)] for j in range(nb_customers)]
    for i in range(nb_customers):
        distance_matrix[i][i] = 0
        for j in range(nb_customers):
            dist = compute_dist(customers_x[i], customers_x[j], customers_y[i], customers_y[j])
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist
    return distance_matrix


# Compute the distances to depot
def compute_distance_depots(depot_x, depot_y, customers_x, customers_y):
    nb_customers = len(customers_x)
    distance_depots = [None] * nb_customers
    for i in range(nb_customers):
        dist = compute_dist(depot_x, customers_x[i], depot_y, customers_y[i])
        distance_depots[i] = dist
    return distance_depots


def compute_dist(xi, xj, yi, yj):
    exact_dist = math.sqrt(math.pow(xi - xj, 2) + math.pow(yi - yj, 2))
    return int(math.floor(exact_dist + 0.5))


def get_nb_trucks(filename):
    begin = filename.rfind("-k")
    if begin != -1:
        begin += 2
        end = filename.find(".", begin)
        return int(filename[begin:end])
    print("Error: nb_trucks could not be read from the file name. Enter it from the command line")
    sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python cvrp.py input_file [output_file] [time_limit] [nb_trucks]")
        sys.exit(1)

    instance_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    str_time_limit = sys.argv[3] if len(sys.argv) > 3 else "20"
    str_nb_trucks = sys.argv[4] if len(sys.argv) > 4 else "0"

    main(instance_file, str_time_limit, output_file, str_nb_trucks)
```

Vehicle Routing with Time Windows (CVRPTW)

```python
import hexaly.optimizer
import sys
import math


def read_elem(filename):
    with open(filename) as f:
        return [str(elem) for elem in f.read().split()]


def main(instance_file, str_time_limit, output_file):
    #
    # Read instance data
    #
    nb_customers, nb_trucks, truck_capacity, dist_matrix_data, dist_depot_data, \
        demands_data, service_time_data, earliest_start_data, latest_end_data, \
        max_horizon = read_input_cvrptw(instance_file)

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Sequence of customers visited by each truck
        customers_sequences = [model.list(nb_customers) for k in range(nb_trucks)]

        # All customers must be visited by exactly one truck
        model.constraint(model.partition(customers_sequences))

        # Create Hexaly arrays to be able to access them with an "at" operator
        demands = model.array(demands_data)
        earliest = model.array(earliest_start_data)
        latest = model.array(latest_end_data)
        service_time = model.array(service_time_data)
        dist_matrix = model.array(dist_matrix_data)
        dist_depot = model.array(dist_depot_data)

        dist_routes = [None] * nb_trucks
        end_time = [None] * nb_trucks
        home_lateness = [None] * nb_trucks
        lateness = [None] * nb_trucks

        # A truck is used if it visits at least one customer
        trucks_used = [(model.count(customers_sequences[k]) > 0) for k in range(nb_trucks)]
        nb_trucks_used = model.sum(trucks_used)

        for k in range(nb_trucks):
            sequence = customers_sequences[k]
            c = model.count(sequence)

            # The quantity needed in each route must not exceed the truck capacity
            demand_lambda = model.lambda_function(lambda j: demands[j])
            route_quantity = model.sum(sequence, demand_lambda)
            model.constraint(route_quantity <= truck_capacity)

            # Distance traveled by each truck
            dist_lambda = model.lambda_function(
                lambda i: model.at(dist_matrix, sequence[i - 1], sequence[i]))
            dist_routes[k] = model.sum(model.range(1, c), dist_lambda) \
                + model.iif(c > 0, dist_depot[sequence[0]] + dist_depot[sequence[c - 1]], 0)

            # End of each visit
            end_time_lambda = model.lambda_function(
                lambda i, prev:
                    model.max(
                        earliest[sequence[i]],
                        model.iif(
                            i == 0, dist_depot[sequence[0]],
                            prev + model.at(dist_matrix, sequence[i - 1], sequence[i])))
                    + service_time[sequence[i]])

            end_time[k] = model.array(model.range(0, c), end_time_lambda, 0)

            # Arriving home after max horizon
            home_lateness[k] = model.iif(
                trucks_used[k],
                model.max(
                    0,
                    end_time[k][c - 1] + dist_depot[sequence[c - 1]] - max_horizon),
                0)

            # Completing visit after latest end
            late_lambda = model.lambda_function(
                lambda i: model.max(0, end_time[k][i] - latest[sequence[i]]))
            lateness[k] = home_lateness[k] + model.sum(model.range(0, c), late_lambda)

        # Total lateness
        total_lateness = model.sum(lateness)

        # Total distance traveled
        total_distance = model.div(model.round(100 * model.sum(dist_routes)), 100)

        # Objective: minimize the number of trucks used, then minimize the distance traveled
        model.minimize(total_lateness)
        model.minimize(nb_trucks_used)
        model.minimize(total_distance)

        model.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = int(str_time_limit)

        optimizer.solve()

        #
        # Write the solution in a file with the following format:
        #  - number of trucks used and total distance
        #  - for each truck the customers visited (omitting the start/end at the depot)
        #
        if output_file is not None:
            with open(output_file, 'w') as f:
                f.write("%d %d\n" % (nb_trucks_used.value, total_distance.value))
                for k in range(nb_trucks):
                    if trucks_used[k].value != 1:
                        continue
                    # Values in sequence are in 0...nbCustomers. +1 is to put it back in
                    # 1...nbCustomers+1 as in the data files (0 being the depot)
                    for customer in customers_sequences[k].value:
                        f.write("%d " % (customer + 1))
                    f.write("\n")


# The input files follow the "Solomon" format
def read_input_cvrptw(filename):
    file_it = iter(read_elem(filename))

    for i in range(4):
        next(file_it)

    nb_trucks = int(next(file_it))
    truck_capacity = int(next(file_it))

    for i in range(13):
        next(file_it)

    depot_x = int(next(file_it))
    depot_y = int(next(file_it))

    for i in range(2):
        next(file_it)

    max_horizon = int(next(file_it))

    next(file_it)

    customers_x = []
    customers_y = []
    demands = []
    earliest_start = []
    latest_end = []
    service_time = []

    while True:
        val = next(file_it, None)
        if val is None:
            break
        i = int(val) - 1
        customers_x.append(int(next(file_it)))
        customers_y.append(int(next(file_it)))
        demands.append(int(next(file_it)))
        ready = int(next(file_it))
        due = int(next(file_it))
        stime = int(next(file_it))
        earliest_start.append(ready)
        # in input files due date is meant as latest start time
        latest_end.append(due + stime)
        service_time.append(stime)

    nb_customers = i + 1

    # Compute distance matrix
    distance_matrix = compute_distance_matrix(customers_x, customers_y)
    distance_depots = compute_distance_depots(depot_x, depot_y, customers_x, customers_y)

    return nb_customers, nb_trucks, truck_capacity, distance_matrix, distance_depots, \
        demands, service_time, earliest_start, latest_end, max_horizon


# Computes the distance matrix
def compute_distance_matrix(customers_x, customers_y):
    nb_customers = len(customers_x)
    distance_matrix = [[None for i in range(nb_customers)] for j in range(nb_customers)]
    for i in range(nb_customers):
        distance_matrix[i][i] = 0
        for j in range(nb_customers):
            dist = compute_dist(customers_x[i], customers_x[j],
                                customers_y[i], customers_y[j])
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist
    return distance_matrix


# Computes the distances to depot
def compute_distance_depots(depot_x, depot_y, customers_x, customers_y):
    nb_customers = len(customers_x)
    distance_depots = [None] * nb_customers
    for i in range(nb_customers):
        dist = compute_dist(depot_x, customers_x[i], depot_y, customers_y[i])
        distance_depots[i] = dist
    return distance_depots


def compute_dist(xi, xj, yi, yj):
    return math.sqrt(math.pow(xi - xj, 2) + math.pow(yi - yj, 2))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python cvrptw.py input_file [output_file] [time_limit]")
        sys.exit(1)

    instance_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    str_time_limit = sys.argv[3] if len(sys.argv) > 3 else "20"

    main(instance_file, str_time_limit, output_file)
```


Pickup and Delivery with Time Windows (PDPTW)

```python
import hexaly.optimizer
import sys
import math


def read_elem(filename):
    with open(filename) as f:
        return [str(elem) for elem in f.read().split()]


def main(instance_file, str_time_limit, sol_file):
    #
    # Read instance data
    #
    nb_customers, nb_trucks, truck_capacity, dist_matrix_data, dist_depot_data, \
        demands_data, service_time_data, earliest_start_data, latest_end_data, \
        pick_up_index, delivery_index, max_horizon = read_input_pdptw(instance_file)

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Sequence of customers visited by each truck
        customers_sequences = [model.list(nb_customers) for k in range(nb_trucks)]

        # All customers must be visited by exactly one truck
        model.constraint(model.partition(customers_sequences))

        # /Create Hexaly arrays to be able to access them with "at" operators
        demands = model.array(demands_data)
        earliest = model.array(earliest_start_data)
        latest = model.array(latest_end_data)
        service_time = model.array(service_time_data)
        dist_matrix = model.array(dist_matrix_data)
        dist_depot = model.array(dist_depot_data)

        dist_routes = [None] * nb_trucks
        end_time = [None] * nb_trucks
        home_lateness = [None] * nb_trucks
        lateness = [None] * nb_trucks

        # A truck is used if it visits at least one customer
        trucks_used = [(model.count(customers_sequences[k]) > 0) for k in range(nb_trucks)]
        nb_trucks_used = model.sum(trucks_used)

        # Pickups and deliveries
        customers_sequences_array = model.array(customers_sequences)
        for i in range(nb_customers):
            if pick_up_index[i] == -1:
                pick_up_list_index = model.find(customers_sequences_array, i)
                delivery_list_index = model.find(customers_sequences_array, delivery_index[i])
                model.constraint(pick_up_list_index == delivery_list_index)
                pick_up_list = model.at(customers_sequences_array, pick_up_list_index)
                delivery_list = model.at(customers_sequences_array, delivery_list_index)
                model.constraint(model.index(pick_up_list, i) < model.index(delivery_list, delivery_index[i]))

        for k in range(nb_trucks):
            sequence = customers_sequences[k]
            c = model.count(sequence)

            # The quantity needed in each route must not exceed the truck capacity at any
            # point in the sequence
            demand_lambda = model.lambda_function(
                lambda i, prev: prev + demands[sequence[i]])
            route_quantity = model.array(model.range(0, c), demand_lambda, 0)

            quantity_lambda = model.lambda_function(
                lambda i: route_quantity[i] <= truck_capacity)
            model.constraint(model.and_(model.range(0, c), quantity_lambda))

            # Distance traveled by each truck
            dist_lambda = model.lambda_function(
                lambda i: model.at(dist_matrix, sequence[i - 1], sequence[i]))
            dist_routes[k] = model.sum(model.range(1, c), dist_lambda) \
                + model.iif(c > 0, dist_depot[sequence[0]] + dist_depot[sequence[c - 1]], 0)

            # End of each visit
            end_lambda = model.lambda_function(
                lambda i, prev:
                    model.max(
                        earliest[sequence[i]],
                        model.iif(
                            i == 0,
                            dist_depot[sequence[0]],
                            prev + model.at(dist_matrix, sequence[i - 1], sequence[i])))
                    + service_time[sequence[i]])

            end_time[k] = model.array(model.range(0, c), end_lambda, 0)

            # Arriving home after max_horizon
            home_lateness[k] = model.iif(
                trucks_used[k],
                model.max(
                    0,
                    end_time[k][c - 1] + dist_depot[sequence[c - 1]] - max_horizon),
                0)

            # Completing visit after latest_end
            late_selector = model.lambda_function(
                lambda i: model.max(0, end_time[k][i] - latest[sequence[i]]))
            lateness[k] = home_lateness[k] + model.sum(model.range(0, c), late_selector)

        # Total lateness (must be 0 for the solution to be valid)
        total_lateness = model.sum(lateness)

        # Total distance traveled
        total_distance = model.div(model.round(100 * model.sum(dist_routes)), 100)

        # Objective: minimize the number of trucks used, then minimize the distance traveled
        model.minimize(total_lateness)
        model.minimize(nb_trucks_used)
        model.minimize(total_distance)

        model.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = int(str_time_limit)

        optimizer.solve()

        #
        # Write the solution in a file with the following format:
        #  - number of trucks used and total distance
        #  - for each truck the customers visited (omitting the start/end at the depot)
        #
        if sol_file is not None:
            with open(sol_file, 'w') as f:
                f.write("%d %.2f\n" % (nb_trucks_used.value, total_distance.value))
                for k in range(nb_trucks):
                    if trucks_used[k].value != 1:
                        continue
                    # Values in sequence are in 0...nbCustomers. +2 is to put it back in
                    # 2...nbCustomers+2 as in the data files (1 being the depot)
                    for customer in customers_sequences[k].value:
                        f.write("%d " % (customer + 1))
                    f.write("\n")


# The input files follow the "Li & Lim" format
def read_input_pdptw(filename):
    file_it = iter(read_elem(filename))

    nb_trucks = int(next(file_it))
    truck_capacity = int(next(file_it))
    next(file_it)

    next(file_it)

    depot_x = int(next(file_it))
    depot_y = int(next(file_it))

    for i in range(2):
        next(file_it)

    max_horizon = int(next(file_it))

    for i in range(3):
        next(file_it)

    customers_x = []
    customers_y = []
    demands = []
    earliest_start = []
    latest_end = []
    service_time = []
    pick_up_index = []
    delivery_index = []

    while True:
        val = next(file_it, None)
        if val is None:
            break
        i = int(val) - 1
        customers_x.append(int(next(file_it)))
        customers_y.append(int(next(file_it)))
        demands.append(int(next(file_it)))
        ready = int(next(file_it))
        due = int(next(file_it))
        stime = int(next(file_it))
        pick = int(next(file_it))
        delivery = int(next(file_it))
        earliest_start.append(ready)
        # in input files due date is meant as latest start time
        latest_end.append(due + stime)
        service_time.append(stime)
        pick_up_index.append(pick - 1)
        delivery_index.append(delivery - 1)

    nb_customers = i + 1

    distance_matrix = compute_distance_matrix(customers_x, customers_y)
    distance_depots = compute_distance_depots(depot_x, depot_y, customers_x, customers_y)

    return nb_customers, nb_trucks, truck_capacity, distance_matrix, distance_depots, \
            demands, service_time, earliest_start, latest_end, pick_up_index, \
            delivery_index, max_horizon


# Compute the distance matrix
def compute_distance_matrix(customers_x, customers_y):
    nb_customers = len(customers_x)
    distance_matrix = [[None for i in range(nb_customers)] for j in range(nb_customers)]
    for i in range(nb_customers):
        distance_matrix[i][i] = 0
        for j in range(nb_customers):
            dist = compute_dist(customers_x[i], customers_x[j],
                                customers_y[i], customers_y[j])
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist
    return distance_matrix


# Compute the distances to the depot
def compute_distance_depots(depot_x, depot_y, customers_x, customers_y):
    nb_customers = len(customers_x)
    distance_depots = [None] * nb_customers
    for i in range(nb_customers):
        dist = compute_dist(depot_x, customers_x[i], depot_y, customers_y[i])
        distance_depots[i] = dist
    return distance_depots


def compute_dist(xi, xj, yi, yj):
    return math.sqrt(math.pow(xi - xj, 2) + math.pow(yi - yj, 2))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python pdptw.py input_file [output_file] [time_limit]")
        sys.exit(1)

    instance_file = sys.argv[1]
    sol_file = sys.argv[2] if len(sys.argv) > 2 else None
    str_time_limit = sys.argv[3] if len(sys.argv) > 3 else "20"

    main(instance_file, str_time_limit, sol_file)
```

Dial-A-Ride Problem (DARP)

```python
import hexaly.optimizer
import sys
import json

def read_data(filename):
    with open(filename) as f:
        return json.load(f)
    
def read_input_darp(instance_file):
    instance = read_data(instance_file)

    nb_clients = instance["nbClients"]
    nb_nodes = instance["nbNodes"]
    nb_vehicles = instance["nbVehicles"]
    depot_tw_end = instance["depot"]["twEnd"]
    capacity = instance["capacity"]
    scale = instance["scale"]

    quantities = [-1 for i in range(2 * nb_clients)]
    distances = instance["distanceMatrix"]
    starts = [-1.0 for i in range(2 * nb_clients)]
    ends = [-1.0 for i in range(2 * nb_clients)]
    loading_times = [-1.0 for i in range(2 * nb_clients)]
    max_travel_times = [-1.0 for i in range(2 * nb_clients)]
    for k in range(nb_clients):
        quantities[k] = instance["clients"][k]["nbClients"]
        quantities[k+nb_clients] = -instance["clients"][k]["nbClients"]

        starts[k] = instance["clients"][k]["pickup"]["start"]
        ends[k] = instance["clients"][k]["pickup"]["end"]
        
        starts[k+nb_clients] = instance["clients"][k]["delivery"]["start"]
        ends[k+nb_clients] = instance["clients"][k]["delivery"]["end"]

        loading_times[k] = instance["clients"][k]["pickup"]["loadingTime"]
        loading_times[k+nb_clients] = instance["clients"][k]["delivery"]["loadingTime"]

        max_travel_times[k] = instance["clients"][k]["pickup"]["maxTravelTime"]
        max_travel_times[k+nb_clients] = instance["clients"][k]["delivery"]["maxTravelTime"]
        
    factor = 1.0 / (scale * instance["speed"])

    distance_warehouse = [-1.0 for i in range(nb_nodes)]
    time_warehouse = [-1.0 for i in range(nb_nodes)]
    distance_matrix = [[-1.0 for i in range(nb_nodes)] for j in range(nb_nodes)]
    time_matrix = [[-1.0 for i in range(nb_nodes)] for j in range(nb_nodes)]
    for i in range(nb_nodes):
        distance_warehouse[i] = distances[0][i+1]
        time_warehouse[i] = distance_warehouse[i] * factor
        for j in range(nb_nodes):
            distance_matrix[i][j] = distances[i+1][j+1]
            time_matrix[i][j] = distance_matrix[i][j] * factor

    return nb_clients, nb_nodes, nb_vehicles, depot_tw_end, capacity, scale, quantities, \
        starts, ends, loading_times, max_travel_times, distance_warehouse, time_warehouse, \
        distance_matrix, time_matrix

def main(instance_file, str_time_limit, sol_file):

    nb_clients, nb_nodes, nb_vehicles, depot_tw_end, capacity, scale, quantities_data, \
        starts_data, ends_data, loading_times_data, max_travel_times, distance_warehouse_data, \
        time_warehouse_data, distance_matrix_data, time_matrix_data = read_input_darp(instance_file)

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        model = optimizer.model

        # routes[k] represents the nodes visited by vehicle k
        routes = [model.list(nb_nodes) for k in range(nb_vehicles)]
        depot_starts = [model.float(0, depot_tw_end) for k in range(nb_vehicles)]
        # Each node is taken by one vehicle
        model.constraint(model.partition(routes))

        quantities = model.array(quantities_data)
        time_warehouse = model.array(time_warehouse_data)
        time_matrix = model.array(time_matrix_data)
        loading_times = model.array(loading_times_data)
        starts = model.array(starts_data)
        ends = model.array(ends_data)
        # waiting[k] is the waiting time at node k
        waiting = [model.float(0, depot_tw_end) for k in range(nb_nodes)]
        waiting_array = model.array(waiting)
        distance_matrix = model.array(distance_matrix_data)
        distance_warehouse = model.array(distance_warehouse_data)

        times = [None] * nb_vehicles
        lateness = [None] * nb_vehicles
        home_lateness = [None] * nb_vehicles
        route_distances = [None] * nb_vehicles

        for k in range(nb_vehicles):
            route = routes[k]
            c = model.count(route)

            demand_lambda = model.lambda_function(lambda i, prev: prev + quantities[route[i]])
            # route_quantities[k][i] indicates the number of clients in vehicle k
            # at its i-th taken node
            route_quantities = model.array(model.range(0, c), demand_lambda)
            quantity_lambda = model.lambda_function(lambda i: route_quantities[i] <= capacity)
            # Vehicles have a maximum capacity
            model.constraint(model.and_(model.range(0, c), quantity_lambda))

            times_lambda = model.lambda_function(
                lambda i, prev: model.max(
                    starts[route[i]],
                    model.iif(
                        i == 0,
                        depot_starts[k] + time_warehouse[route[0]],
                        prev + time_matrix[route[i-1]][route[i]]
                    )
                ) + waiting_array[route[i]] + loading_times[route[i]]
            )
            # times[k][i] is the time at which vehicle k leaves the i-th node
            # (after waiting and loading time at node i)
            times[k] = model.array(model.range(0, c), times_lambda)

            lateness_lambda = model.lambda_function(
                lambda i: model.max(
                    0,
                    times[k][i] - loading_times[route[i]] - ends[route[i]]
                )
            )
            # Total lateness of the k-th route
            lateness[k] = model.sum(model.range(0, c), lateness_lambda)

            home_lateness[k] = model.iif(
                c > 0,
                model.max(0, times[k][c-1] + time_warehouse[route[c-1]] - depot_tw_end),
                0
            )

            route_dist_lambda = model.lambda_function(
                lambda i: distance_matrix[route[i-1]][route[i]]
            )
            route_distances[k] = model.sum(
                model.range(1, c),
                route_dist_lambda
            ) + model.iif(
                c > 0,
                distance_warehouse[route[0]] + distance_warehouse[route[c-1]],
                0
            )

        routes_array = model.array(routes)
        times_array = model.array(times)
        client_lateness = [None] * nb_clients
        for k in range(nb_clients):
            # For each pickup node k, its associated delivery node is k + nb_clients
            pickup_list_index = model.find(routes_array, k)
            delivery_list_index = model.find(routes_array, k + nb_clients)
            # A client picked up in route i is delivered in route i
            model.constraint(pickup_list_index == delivery_list_index)

            client_list = routes_array[pickup_list_index]
            pickup_index = model.index(client_list, k)
            delivery_index = model.index(client_list, k + nb_clients)
            # Pickup before delivery
            model.constraint(pickup_index < delivery_index)

            pickup_time = times_array[pickup_list_index][pickup_index]
            delivery_time = times_array[delivery_list_index][delivery_index] \
                - loading_times[k + nb_clients]
            travel_time = delivery_time - pickup_time
            client_lateness[k] = model.max(travel_time - max_travel_times[k], 0)

        total_lateness = model.sum(lateness + home_lateness)
        total_client_lateness = model.sum(client_lateness)
        total_distance = model.sum(route_distances)

        model.minimize(total_lateness)
        model.minimize(total_client_lateness)
        model.minimize(total_distance / scale)

        model.close()
        optimizer.param.time_limit = int(str_time_limit)
        optimizer.solve()

        #
        # Write the solution in a file with the following format:
        #  - total lateness on the routes, total client lateness, and total distance
        #  - for each vehicle, the depot start time, the nodes visited (omitting the start/end at the
        # depot), and the waiting time at each node
        #
        if sol_file is not None:
            with open(sol_file, 'w') as f:
                f.write("%d %d %.2f\n" % (
                    total_lateness.value,
                    total_client_lateness.value,
                    total_distance.value
                ))
                for k in range(nb_vehicles):
                    f.write("Vehicle %d (%.2f): " %(k + 1, depot_starts[k].value))
                    for node in routes[k].value:
                        f.write("%d (%.2f), " % (node, waiting[node].value))
                    f.write("\n")
    return 0

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python darp.py input_file [output_file] [time_limit]")
        sys.exit(1)

    instance_file = sys.argv[1]
    sol_file = sys.argv[2] if len(sys.argv) > 2 else None
    str_time_limit = sys.argv[3] if len(sys.argv) > 3 else "20"

    main(instance_file, str_time_limit, sol_file)
```


Clustered Vehicle Routing

```python
import hexaly.optimizer
import sys
import math

def read_elem(filename):
    with open(filename) as f:
        return [str(elem) for elem in f.read().split()]
    

def main(instance_file, str_time_limit, output_file):
    # Read instance data
    nb_customers, nb_trucks, nb_clusters, truck_capacity, dist_matrix_data, dist_depot_data, \
        demands_data, clusters_data = read_input_cvrp(instance_file)


    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        # Declare the optimization model
        model = optimizer.model

        # Create Hexaly arrays to be able to access them with an "at" operator
        demands = model.array(demands_data)
        dist_matrix = model.array(dist_matrix_data)
        clusters = model.array(clusters_data)
        dist_depot = model.array(dist_depot_data)
        
        # A list is created for each cluster, to determine the order within the cluster
        clusters_sequences = []
        for k in range(nb_clusters):
            clusters_sequences.append(model.list(len(clusters_data[k])))
            # All customers in the cluster must be visited 
            model.constraint(model.count(clusters_sequences[k]) == len(clusters_data[k]))

        clustersDistances = model.array()
        initialNodes = model.array()
        endNodes = model.array()
        for k in range(nb_clusters):
            sequence = clusters_sequences[k]
            c = model.count(sequence)
            # Distance traveled within cluster k
            clustersDistances_lambda = model.lambda_function(lambda i:
                    model.at(dist_matrix, clusters[k][sequence[i - 1]],
                    clusters[k][sequence[i]]))
            clustersDistances.add_operand(model.sum(model.range(1,c), 
                    clustersDistances_lambda))
            
            # First and last point when visiting cluster k
            initialNodes.add_operand(clusters[k][sequence[0]]) 
            endNodes.add_operand(clusters[k][sequence[c - 1]])
        
        # Sequences of clusters visited by each truck
        truckSequences = [model.list(nb_clusters) for _ in range(nb_trucks)]

        # Each cluster must be visited by exactly one truck
        model.constraint(model.partition(truckSequences))

        routeDistances = [None] * nb_trucks
        for k in range(nb_trucks):
            sequence = truckSequences[k]
            c = model.count(sequence)

            # The quantity needed in each route must not exceed the truck capacity
            demand_lambda = model.lambda_function(lambda j: demands[j])
            route_quantity = model.sum(sequence, demand_lambda)
            model.constraint(route_quantity <= truck_capacity)

            # Distance traveled by each truck
            # = distance in each cluster + distance between clusters + distance with depot at 
            # the beginning end at the end of a route
            routeDistances_lambda = model.lambda_function(lambda i:
                    model.at(clustersDistances, sequence[i]) + model.at(dist_matrix,
                    endNodes[sequence[i - 1]], initialNodes[sequence[i]]))
            routeDistances[k] = model.sum(model.range(1, c), routeDistances_lambda) \
                    + model.iif(c > 0, model.at(clustersDistances, sequence[0]) 
                    + dist_depot[initialNodes[sequence[0]]]
                    + dist_depot[endNodes[sequence[c - 1]]], 0)

        # Total distance traveled
        total_distance = model.sum(routeDistances)

        # Objective:  minimize the distance traveled
        model.minimize(total_distance)

        model.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = int(str_time_limit)

        optimizer.solve() 
        # Write the solution in a file with the following format:
        # - total distance
        # - for each truck the customers visited (omitting the start/end at the depot)
        if output_file is not None:
            with open(output_file, 'w') as f:
                f.write("%d\n" % (total_distance.value))
                for k in range(nb_trucks):
                    # Values in sequence are in [0..nbCustomers - 1]. +2 is to put it back
                    # in [2..nbCustomers+1] as in the data files (1 being the depot)
                    for cluster in truckSequences[k].value:
                        for customer in clusters_sequences[cluster].value:
                            f.write("%d " % (clusters_data[cluster][customer] + 2))
                    f.write("\n")

# The input files follow the "Augerat" format
def read_input_cvrp(filename):
    file_it = iter(read_elem(filename))

    nb_nodes = 0
    while True:
        token = next(file_it)
        if token == "DIMENSION:":
            nb_nodes = int(next(file_it))
            nb_customers = nb_nodes - 1
        if token == "VEHICLES:":
            nb_trucks = int(next(file_it))
        elif token == "GVRP_SETS:":
            nb_clusters = int(next(file_it))
        elif token == "CAPACITY:":
            truck_capacity = int(next(file_it))
        elif token == "NODE_COORD_SECTION":
            break

    customers_x = [None] * nb_customers
    customers_y = [None] * nb_customers
    depot_x = 0
    depot_y = 0
    for n in range(nb_nodes):
        node_id = int(next(file_it))
        if node_id != n + 1:
            print("Unexpected index")
            sys.exit(1)
        if node_id == 1:
            depot_x = int(float(next(file_it)))
            depot_y = int(float(next(file_it)))
        else:
            # -2 because original customer indices are in 2..nbNodes
            customers_x[node_id - 2] = int(float(next(file_it)))
            customers_y[node_id - 2] = int(float(next(file_it)))

    distance_matrix = compute_distance_matrix(customers_x, customers_y)
    distance_depots = compute_distance_depots(depot_x, depot_y, customers_x, customers_y)
    
    token = next(file_it)
    if token != "GVRP_SET_SECTION":
        print("Expected token GVRP_SET_SECTION")
        sys.exit(1)
    clusters_data = [None]*nb_clusters
    for n in range(nb_clusters):
        node_id = int(next(file_it))
        if node_id != n + 1:
            print("Unexpected index")
            sys.exit(1)
        cluster = []
        value = int(next(file_it))
        while value != -1:
            # -2 because original customer indices are in 2..nbNodes
            cluster.append(value-2)
            value = int(next(file_it))
        clusters_data[n] = cluster
    token = next(file_it)
    if token != "DEMAND_SECTION":
        print("Expected token DEMAND_SECTION")
        sys.exit(1)

    demands = [None] * nb_clusters
    for n in range(nb_clusters):
        node_id = int(next(file_it))
        if node_id != n + 1:
            print("Unexpected index")
            sys.exit(1)
        demands[n] = int(next(file_it))
    return nb_customers, nb_trucks, nb_clusters, truck_capacity, distance_matrix, \
        distance_depots, demands, clusters_data

# Compute the distance matrix
def compute_distance_matrix(customers_x, customers_y):
    nb_customers = len(customers_x)
    distance_matrix = [[None for i in range(nb_customers)] for j in range(nb_customers)]
    for i in range(nb_customers):
        distance_matrix[i][i] = 0
        for j in range(nb_customers):
            dist = compute_dist(customers_x[i], customers_x[j], customers_y[i], customers_y[j])
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist
    return distance_matrix

# Compute the distances to depot
def compute_distance_depots(depot_x, depot_y, customers_x, customers_y):
    nb_customers = len(customers_x)
    distance_depots = [None] * nb_customers
    for i in range(nb_customers):
        dist = compute_dist(depot_x, customers_x[i], depot_y, customers_y[i])
        distance_depots[i] = dist
    return distance_depots


def compute_dist(xi, xj, yi, yj):
    exact_dist = math.sqrt(math.pow(xi - xj, 2) + math.pow(yi - yj, 2))
    return int(math.floor(exact_dist + 0.5))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python clustered-vehicle-routing.py input_file [output_file] [time_limit]")
        sys.exit(1)

    instance_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    str_time_limit = sys.argv[3] if len(sys.argv) > 3 else "20"

    main(instance_file, str_time_limit, output_file)

```

Inventory routing

```python
import hexaly.optimizer
import sys
import math


def read_elem(filename):
    with open(filename) as f:
        return [str(elem) for elem in f.read().split()]


def main(instance_file, str_time_limit, sol_file):
    #
    # Read instance data
    #
    nb_customers, horizon_length, capacity, start_level_supplier, production_rate_supplier, \
        holding_cost_supplier, start_level, max_level, demand_rate, holding_cost, \
        dist_matrix_data, dist_supplier_data = read_input_irp(instance_file)

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Quantity of product delivered at each discrete time instant of
        # the planning time horizon to each customer
        delivery = [[model.float(0, capacity) for i in range(nb_customers)]
                    for _ in range(horizon_length)]

        # Sequence of customers visited at each discrete time instant of
        # the planning time horizon
        route = [model.list(nb_customers) for t in range(horizon_length)]

        # Customers receive products only if they are visited
        is_delivered = [[model.contains(route[t], i) for i in range(nb_customers)]
                        for t in range(horizon_length)]

        # Create Hexaly arrays to be able to access them with an "at" operator
        dist_matrix = model.array(dist_matrix_data)
        dist_supplier = model.array(dist_supplier_data)

        dist_routes = [None for _ in range(horizon_length)]

        for t in range(horizon_length):
            sequence = route[t]
            c = model.count(sequence)

            # Distance traveled at instant t
            dist_lambda = model.lambda_function(
                lambda i:
                    model.at(
                        dist_matrix,
                        sequence[i - 1],
                        sequence[i]))
            dist_routes[t] = model.iif(
                c > 0,
                dist_supplier[sequence[0]]
                + model.sum(model.range(1, c), dist_lambda)
                + dist_supplier[sequence[c - 1]],
                0)

        # Stockout constraints at the supplier
        inventory_supplier = [None for _ in range(horizon_length + 1)]
        inventory_supplier[0] = start_level_supplier
        for t in range(1, horizon_length + 1):
            inventory_supplier[t] = inventory_supplier[t - 1] - model.sum(
                delivery[t - 1][i] for i in range(nb_customers)) + production_rate_supplier
            if t != horizon_length:
                model.constraint(
                    inventory_supplier[t] >= model.sum(delivery[t][i]
                                                       for i in range(nb_customers)))

        # Stockout constraints at the customers
        inventory = [[None for _ in range(horizon_length + 1)] for _ in range(nb_customers)]
        for i in range(nb_customers):
            inventory[i][0] = start_level[i]
            for t in range(1, horizon_length + 1):
                inventory[i][t] = inventory[i][t - 1] + delivery[t - 1][i] - demand_rate[i]
                model.constraint(inventory[i][t] >= 0)

        for t in range(horizon_length):
            # Capacity constraints
            model.constraint(
                model.sum((delivery[t][i]) for i in range(nb_customers)) <= capacity)

            # Maximum level constraints
            for i in range(nb_customers):
                model.constraint(delivery[t][i] <= max_level[i] - inventory[i][t])
                model.constraint(delivery[t][i] <= max_level[i] * is_delivered[t][i])

        # Total inventory cost at the supplier
        total_cost_inventory_supplier = holding_cost_supplier * model.sum(
            inventory_supplier[t] for t in range(horizon_length + 1))

        # Total inventory cost at customers
        total_cost_inventory = model.sum(model.sum(
            holding_cost[i] * inventory[i][t] for t in range(horizon_length + 1))
            for i in range(nb_customers))

        # Total transportation cost
        total_cost_route = model.sum(dist_routes[t] for t in range(horizon_length))

        # Objective: minimize the sum of all costs
        objective = total_cost_inventory_supplier + total_cost_inventory + total_cost_route
        model.minimize(objective)

        model.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = int(str_time_limit)

        optimizer.solve()

        #
        # Write the solution in a file with the following format :
        # - total distance run by the vehicle
        # - the nodes visited at each time step (omitting the start/end at the supplier)
        #
        if len(sys.argv) >= 3:
            with open(sol_file, 'w') as f:
                f.write("%d\n" % (total_cost_route.value))
                for t in range(horizon_length):
                    for customer in route[t].value:
                        f.write("%d " % (customer + 1))
                    f.write("\n")


# The input files follow the "Archetti" format
def read_input_irp(filename):
    file_it = iter(read_elem(filename))

    nb_customers = int(next(file_it)) - 1
    horizon_length = int(next(file_it))
    capacity = int(next(file_it))

    x_coord = [None] * nb_customers
    y_coord = [None] * nb_customers
    start_level = [None] * nb_customers
    max_level = [None] * nb_customers
    min_level = [None] * nb_customers
    demand_rate = [None] * nb_customers
    holding_cost = [None] * nb_customers

    next(file_it)
    x_coord_supplier = float(next(file_it))
    y_coord_supplier = float(next(file_it))
    start_level_supplier = int(next(file_it))
    production_rate_supplier = int(next(file_it))
    holding_cost_supplier = float(next(file_it))
    for i in range(nb_customers):
        next(file_it)
        x_coord[i] = float(next(file_it))
        y_coord[i] = float(next(file_it))
        start_level[i] = int(next(file_it))
        max_level[i] = int(next(file_it))
        min_level[i] = int(next(file_it))
        demand_rate[i] = int(next(file_it))
        holding_cost[i] = float(next(file_it))

    distance_matrix = compute_distance_matrix(x_coord, y_coord)
    distance_supplier = compute_distance_supplier(x_coord_supplier, y_coord_supplier, x_coord, y_coord)

    return nb_customers, horizon_length, capacity, start_level_supplier, \
        production_rate_supplier, holding_cost_supplier, start_level, max_level, \
        demand_rate, holding_cost, distance_matrix, distance_supplier


# Compute the distance matrix
def compute_distance_matrix(x_coord, y_coord):
    nb_customers = len(x_coord)
    distance_matrix = [[None for i in range(nb_customers)] for j in range(nb_customers)]
    for i in range(nb_customers):
        distance_matrix[i][i] = 0
        for j in range(nb_customers):
            dist = compute_dist(x_coord[i], x_coord[j], y_coord[i], y_coord[j])
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist
    return distance_matrix


# Compute the distances to the supplier
def compute_distance_supplier(x_coord_supplier, y_coord_supplier, x_coord, y_coord):
    nb_customers = len(x_coord)
    distance_supplier = [None] * nb_customers
    for i in range(nb_customers):
        dist = compute_dist(x_coord_supplier, x_coord[i], y_coord_supplier, y_coord[i])
        distance_supplier[i] = dist
    return distance_supplier


def compute_dist(xi, xj, yi, yj):
    return round(math.sqrt(math.pow(xi - xj, 2) + math.pow(yi - yj, 2)))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python irp.py input_file [output_file] [time_limit]")
        sys.exit(1)

    instance_file = sys.argv[1]
    sol_file = sys.argv[2] if len(sys.argv) > 2 else None
    str_time_limit = sys.argv[3] if len(sys.argv) > 3 else "20"

    main(instance_file, str_time_limit, sol_file)
```

Split Delivery Vehicle Routing (SDVRP)

```python
import hexaly.optimizer
import sys
import math


def read_elem(filename):
    with open(filename) as f:
        return [str(elem) for elem in f.read().split()]


def main(instance_file, str_time_limit, sol_file):
    #
    # Read instance data
    #
    nb_customers, truck_capacity, dist_matrix_data, dist_depot_data, demands_data = \
        read_input_sdvrp(instance_file)
    nb_trucks = nb_customers

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Sequence of customers visited by each truck
        customers_sequences = [model.list(nb_customers) for _ in range(nb_trucks)]

        # Quantity carried by each truck for each customer
        quantity = [None] * nb_trucks
        for k in range(nb_trucks):
            quantity[k] = [model.float(0, demands_data[i]) for i in range(nb_customers)]

        # All customers must be visited at least by one truck
        model.constraint(model.cover(customers_sequences))

        # Create Hexaly arrays to be able to access them with "at" operators
        dist_matrix = model.array(dist_matrix_data)
        dist_depot = model.array(dist_depot_data)

        route_distances = [None] * nb_trucks
        trucks_used = [None] * nb_trucks

        for k in range(nb_trucks):
            sequence = customers_sequences[k]
            c = model.count(sequence)

            # A truck is used if it visits at least one customer
            trucks_used[k] = model.count(sequence) > 0

            # The quantity carried in each route must not exceed the truck capacity
            quantity_array = model.array(quantity[k])
            quantity_lambda = model.lambda_function(lambda j: quantity_array[j])
            route_quantity = model.sum(sequence, quantity_lambda)
            model.constraint(route_quantity <= truck_capacity)

            # Distance traveled by each truck
            dist_lambda = model.lambda_function(
                lambda i: model.at(dist_matrix, sequence[i - 1], sequence[i]))
            route_distances[k] = model.sum(model.range(1, c), dist_lambda) \
                + model.iif(
                    trucks_used[k],
                    dist_depot[sequence[0]] + dist_depot[sequence[c - 1]],
                    0)

        for i in range(nb_customers):
            # Each customer must receive at least its demand
            quantity_served = model.sum(
                quantity[k][i] * model.contains(customers_sequences[k], i)
                for k in range(nb_trucks))
            model.constraint(quantity_served >= demands_data[i])

        total_distance = model.sum(route_distances)

        # Objective: minimize the distance traveled
        model.minimize(total_distance)

        model.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = int(str_time_limit)

        optimizer.solve()

        #
        # Write the solution in a file with the following format:
        # each line k contain the customers visited by the truck k
        #
        if len(sys.argv) >= 3:
            with open(sol_file, 'w') as f:
                f.write("%d \n" % total_distance.value)
                for k in range(nb_trucks):
                    if trucks_used[k].value != 1:
                        continue
                    # Values in sequence are in 0...nbCustomers. +1 is to put it back in 1...nbCustomers+1
                    for customer in customers_sequences[k].value:
                        f.write("%d " % (customer + 1))
                    f.write("\n")


def read_input_sdvrp(filename):
    file_it = iter(read_elem(filename))

    nb_customers = int(next(file_it))
    capacity = int(next(file_it))

    demands = [None] * nb_customers
    for i in range(nb_customers):
        demands[i] = int(next(file_it))

    # Extracting the coordinates of the depot and the customers
    customers_x = [None] * nb_customers
    customers_y = [None] * nb_customers
    depot_x = float(next(file_it))
    depot_y = float(next(file_it))
    for i in range(nb_customers):
        customers_x[i] = float(next(file_it))
        customers_y[i] = float(next(file_it))

    distance_matrix = compute_distance_matrix(customers_x, customers_y)
    distance_depots = compute_distance_depots(depot_x, depot_y, customers_x, customers_y)
    return nb_customers, capacity, distance_matrix, distance_depots, demands


# Compute the distance between two customers
def compute_distance_matrix(customers_x, customers_y):
    nb_customers = len(customers_x)
    distance_matrix = [[None for _ in range(nb_customers)] for _ in range(nb_customers)]
    for i in range(nb_customers):
        distance_matrix[i][i] = 0
        for j in range(nb_customers):
            dist = compute_dist(customers_x[i], customers_x[j], customers_y[i], customers_y[j])
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist
    return distance_matrix


# Compute the distances to depot
def compute_distance_depots(depot_x, depot_y, customers_x, customers_y):
    nb_customers = len(customers_x)
    distance_depots = [None] * nb_customers
    for i in range(nb_customers):
        dist = compute_dist(depot_x, customers_x[i], depot_y, customers_y[i])
        distance_depots[i] = dist
    return distance_depots


def compute_dist(xi, xj, yi, yj):
    exact_dist = math.sqrt(math.pow(xi - xj, 2) + math.pow(yi - yj, 2))
    return int(math.floor(exact_dist + 0.5))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python sdvrp.py input_file [output_file] [time_limit]")
        sys.exit(1)

    instance_file = sys.argv[1]
    sol_file = sys.argv[2] if len(sys.argv) > 2 else None
    str_time_limit = sys.argv[3] if len(sys.argv) > 3 else "20"

    main(instance_file, str_time_limit, sol_file)
```

Time Dependent Capacitated Vehicle Routing Problem with Time Windows (TDCVRPTW)

```python
import hexaly.optimizer
import sys
import math


def read_elem(filename):
    with open(filename) as f:
        return [str(elem) for elem in f.read().split()]


def main(instance_file, str_time_limit, output_file):
    #
    # Read instance data
    #
    nb_customers, nb_trucks, truck_capacity, dist_matrix_data, travel_time_data, \
        time_to_matrix_idx_data, dist_depot_data, travel_time_warehouse_data,\
        demands_data, service_time_data, earliest_start_data, latest_end_data, \
        max_horizon = read_input_cvrptw(instance_file)

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Sequence of customers visited by each truck
        customers_sequences = [model.list(nb_customers) for k in range(nb_trucks)]

        # All customers must be visited by exactly one truck
        model.constraint(model.partition(customers_sequences))

        # Create Hexaly arrays to be able to access them with an "at" operator
        demands = model.array(demands_data)
        earliest = model.array(earliest_start_data)
        latest = model.array(latest_end_data)
        service_time = model.array(service_time_data)
        dist_matrix = model.array(dist_matrix_data)
        travel_time = model.array(travel_time_data)
        time_to_matrix_idx = model.array(time_to_matrix_idx_data)
        dist_depot = model.array(dist_depot_data)
        travel_time_warehouse = model.array(travel_time_warehouse_data)

        dist_routes = [None] * nb_trucks
        end_time = [None] * nb_trucks
        home_lateness = [None] * nb_trucks
        lateness = [None] * nb_trucks

        # A truck is used if it visits at least one customer
        trucks_used = [(model.count(customers_sequences[k]) > 0) for k in range(nb_trucks)]
        nb_trucks_used = model.sum(trucks_used)

        for k in range(nb_trucks):
            sequence = customers_sequences[k]
            c = model.count(sequence)

            # The quantity needed in each route must not exceed the truck capacity
            demand_lambda = model.lambda_function(lambda j: demands[j])
            route_quantity = model.sum(sequence, demand_lambda)
            model.constraint(route_quantity <= truck_capacity)

            # Distance traveled by each truck
            dist_lambda = model.lambda_function(
                lambda i: model.at(dist_matrix, sequence[i - 1], sequence[i]))
            dist_routes[k] = model.sum(model.range(1, c), dist_lambda) \
                + model.iif(c > 0, dist_depot[sequence[0]] + dist_depot[sequence[c - 1]], 0)

            # End of each visit according to the traffic
            end_time_lambda = model.lambda_function(
                lambda i, prev:
                    model.max(
                        earliest[sequence[i]],
                        model.iif(
                            i == 0, model.at(travel_time_warehouse, sequence[0], time_to_matrix_idx[0]),
                            prev + model.at(travel_time, sequence[i - 1], sequence[i],
                                            time_to_matrix_idx[model.round(prev)])))
                    + service_time[sequence[i]])

            end_time[k] = model.array(model.range(0, c), end_time_lambda)

            # Arriving home after max horizon
            home_lateness[k] = model.iif(
                trucks_used[k],
                model.max(
                    0,
                    end_time[k][c - 1] + model.at(travel_time_warehouse, sequence[c - 1],
                                                  time_to_matrix_idx[model.round(end_time[k][c - 1])]) - max_horizon),
                0)

            # Completing visit after latest end
            late_lambda = model.lambda_function(
                lambda i: model.max(0, end_time[k][i] - latest[sequence[i]]))
            lateness[k] = home_lateness[k] + model.sum(model.range(0, c), late_lambda)

        # Total lateness
        total_lateness = model.sum(lateness)

        # Total distance traveled
        total_distance = model.div(model.round(100 * model.sum(dist_routes)), 100)

        # Objective: minimize the number of trucks used, then minimize the distance traveled
        model.minimize(total_lateness)
        model.minimize(nb_trucks_used)
        model.minimize(total_distance)

        model.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = int(str_time_limit)

        optimizer.solve()

        #
        # Write the solution in a file with the following format:
        #  - number of trucks used and total distance
        #  - for each truck the customers visited (omitting the start/end at the depot)
        #
        if output_file is not None:
            with open(output_file, 'w') as f:
                f.write("%d %d\n" % (nb_trucks_used.value, total_distance.value))
                for k in range(nb_trucks):
                    if trucks_used[k].value != 1:
                        continue
                    # Values in sequence are in 0...nbCustomers. +1 is to put it back in
                    # 1...nbCustomers+1 as in the data files (0 being the depot)
                    for customer in customers_sequences[k].value:
                        f.write("%d " % (customer + 1))
                    f.write("\n")


# The input files follow the "Solomon" format
def read_input_cvrptw(filename):
    file_it = iter(read_elem(filename))

    for i in range(4):
        next(file_it)

    nb_trucks = int(next(file_it))
    truck_capacity = int(next(file_it))

    for i in range(13):
        next(file_it)

    depot_x = int(next(file_it))
    depot_y = int(next(file_it))

    for i in range(2):
        next(file_it)

    max_horizon = int(next(file_it))

    next(file_it)

    customers_x = []
    customers_y = []
    demands = []
    earliest_start = []
    latest_end = []
    service_time = []

    while True:
        val = next(file_it, None)
        if val is None:
            break
        i = int(val) - 1
        customers_x.append(int(next(file_it)))
        customers_y.append(int(next(file_it)))
        demands.append(int(next(file_it)))
        ready = int(next(file_it))
        due = int(next(file_it))
        stime = int(next(file_it))
        earliest_start.append(ready)
        # in input files due date is meant as latest start time
        latest_end.append(due + stime)
        service_time.append(stime)

    nb_customers = i + 1

    short_distance_travel_time_profile = [1.00, 2.50, 1.75, 2.50, 1.00]
    medium_distance_travel_time_profile = [1.00, 2.00, 1.50, 2.00, 1.00]
    long_distance_travel_time_profile = [1.00, 1.60, 1.10, 1.60, 1.00]
    travel_time_profile_matrix = [
        short_distance_travel_time_profile,
        medium_distance_travel_time_profile,
        long_distance_travel_time_profile
    ]
    distance_levels = [10, 25]
    time_interval_steps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    nb_time_intervals = len(time_interval_steps) - 1
    nb_distance_levels = len(distance_levels)

    # Compute distance matrices
    distance_matrix, travel_time, time_to_matrix_idx = compute_distance_matrices(
        customers_x, customers_y, max_horizon, travel_time_profile_matrix, time_interval_steps, nb_time_intervals,
        distance_levels, nb_distance_levels)
    distance_depots, travel_time_warehouse = compute_distance_depots(
        depot_x, depot_y, customers_x, customers_y, travel_time_profile_matrix, nb_time_intervals, distance_levels,
        nb_distance_levels)

    return nb_customers, nb_trucks, truck_capacity, distance_matrix, travel_time, time_to_matrix_idx, distance_depots, \
        travel_time_warehouse, demands, service_time, earliest_start, latest_end, max_horizon


# Computes the distance matrices
def compute_distance_matrices(customers_x, customers_y, max_horizon, travel_time_profile_matrix,
                              time_interval_steps, nb_time_intervals, distance_levels, nb_distance_levels):

    nb_customers = len(customers_x)
    distance_matrix = [[None for _ in range(nb_customers)] for _ in range(nb_customers)]
    travel_time = [[[None for _ in range(nb_time_intervals)] for _ in range(nb_customers)] for _ in range(nb_customers)]
    time_to_matrix_idx = [None for _ in range(max_horizon)]
    for i in range(nb_customers):
        distance_matrix[i][i] = 0

        for k in range(nb_time_intervals):
            travel_time[i][i][k] = 0

        for j in range(nb_customers):
            dist = compute_dist(customers_x[i], customers_x[j],
                                customers_y[i], customers_y[j])
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist

            profile_idx = get_profile(dist, distance_levels, nb_distance_levels)
            for k in range(nb_time_intervals):
                local_travel_time = travel_time_profile_matrix[profile_idx][k] * dist
                travel_time[i][j][k] = local_travel_time
                travel_time[j][i][k] = local_travel_time

    for i in range(nb_time_intervals):
        time_step_start = int(round(time_interval_steps[i] * max_horizon))
        time_step_end = int(round(time_interval_steps[i+1] * max_horizon))
        for j in range(time_step_start, time_step_end):
            time_to_matrix_idx[j] = i
    return distance_matrix, travel_time, time_to_matrix_idx


# Computes the distances to depot
def compute_distance_depots(depot_x, depot_y, customers_x, customers_y, travel_time_profile_matrix,
                            nb_time_intervals, distance_levels, nb_distance_levels):
    nb_customers = len(customers_x)
    distance_depots = [None] * nb_customers
    travel_time_warehouse = [[None for _ in range(nb_time_intervals)] for _ in range(nb_customers)]
    for i in range(nb_customers):
        dist = compute_dist(depot_x, customers_x[i], depot_y, customers_y[i])
        distance_depots[i] = dist

        profile_idx = get_profile(dist, distance_levels, nb_distance_levels)
        for j in range(nb_time_intervals):
            local_travel_time_warehouse = travel_time_profile_matrix[profile_idx][j] * dist
            travel_time_warehouse[i][j] = local_travel_time_warehouse
    return distance_depots, travel_time_warehouse


def compute_dist(xi, xj, yi, yj):
    return math.sqrt(math.pow(xi - xj, 2) + math.pow(yi - yj, 2))


def get_profile(dist, distance_levels, nb_distance_levels):
    idx = 0
    while idx < nb_distance_levels and dist > distance_levels[idx]:
        idx += 1
    return idx


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python tdcvrptw.py input_file [output_file] [time_limit]")
        sys.exit(1)

    instance_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    str_time_limit = sys.argv[3] if len(sys.argv) > 3 else "20"

    main(instance_file, str_time_limit, output_file)
```



Capacitated Arc Routing Problem (CARP)

```python
import sys
import hexaly.optimizer


def main(instance_file, output_file, str_time_limit):
    #
    # Read instance data
    #
    instance = CarpInstance(instance_file)

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Sequence of edges visited and "serviced" by each truck
        edges_sequences_vars = [model.list(2 * instance.nb_required_edges)
                                for _ in range(instance.nb_trucks)]
        edges_sequences = model.array(edges_sequences_vars)

        # Create distance and cost arrays to be able to access it with an "at" operator
        demands = model.array(instance.demands_data)
        costs = model.array(instance.costs_data)
        dist_from_depot = model.array(instance.dist_from_depot_data)
        dist_to_depot = model.array(instance.dist_to_depot_data)
        edges_dist = model.array()
        for n in range(2 * instance.nb_required_edges):
            edges_dist.add_operand(model.array(instance.edges_dist_data[n]))

        # An edge must be serviced by at most one truck
        model.constraint(model.disjoint(edges_sequences))

        # An edge can be travelled in both directions but its demand must be 
        # satisfied only once
        for i in range(instance.nb_required_edges):
            model.constraint(
                model.contains(edges_sequences, 2 * i)
                + model.contains(edges_sequences, 2 * i + 1) 
                == 1)

        route_distances = [None] * instance.nb_trucks
        for k in range(instance.nb_trucks):
            sequence = edges_sequences_vars[k]
            c = model.count(sequence)

            # Quantity in each truck
            demand_lambda = model.lambda_function(lambda j: demands[j])
            route_quantity = model.sum(sequence, demand_lambda)
            # Capacity constraint : a truck must not exceed its capacity
            model.constraint(route_quantity <= instance.truck_capacity)

            # Distance travelled by each truck
            dist_lambda = model.lambda_function(
                lambda i:
                    costs[sequence[i]]
                    + model.at(
                        edges_dist,
                        sequence[i - 1],
                        sequence[i]))
            route_distances[k] = model.sum(model.range(1, c), dist_lambda) \
                + model.iif(
                    c > 0,
                    costs[sequence[0]] + dist_from_depot[sequence[0]] \
                        + dist_to_depot[sequence[c - 1]],
                    0)

        # Total distance travelled
        total_distance = model.sum(route_distances)

        # Objective: minimize the distance travelled
        model.minimize(total_distance)

        model.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = int(str_time_limit)

        optimizer.solve()

        #
        # Write the solution in a file with the following format:
        #  - total distance
        #  - number of routes
        #  - for each truck, the edges visited
        #
        if output_file:
            with open(output_file, 'w') as f:
                f.write("Objective function value : {}\nNumber of routes : {}\n".format(
                    total_distance.value, instance.nb_trucks))
                for k in range(instance.nb_trucks):
                    f.write("Sequence of truck {}: ".format(k + 1))
                    sequence = edges_sequences_vars[k].value
                    c = len(sequence)
                    for i in range(c):
                        f.write("({}, {})  ".format(instance.origins_data[sequence[i]],
                                instance.destinations_data[sequence[i]]))
                    f.write("\n")


class CarpInstance:

    def read_elem(self, filename):
        with open(filename) as f:
            return [str(elem) for elem in f.read().strip().split("\n")]

    # The input files follow the format of the DIMACS challenge
    def __init__(self, filename):
        file_it = iter(self.read_elem(filename))
        for _ in range(2):
            next(file_it)
        nb_nodes = int(next(file_it).strip().split(":")[1])
        self.nb_required_edges = int(next(file_it).strip().split(":")[1])
        nb_not_required_edges = int(next(file_it).strip().split(":")[1])
        self.nb_trucks = int(next(file_it).strip().split(":")[1])
        self.truck_capacity = int(next(file_it).strip().split(":")[1])
        for _ in range(3):
            next(file_it)
        self.demands_data = list()
        self.costs_data = list()
        self.origins_data = list()
        self.destinations_data = list()
        required_nodes = list()
        node_neighbors = [([0] * nb_nodes) for _ in range(nb_nodes)]
        for _ in range(self.nb_required_edges):
            elements = next(file_it)
            edge = tuple(map(int, elements.strip().split("   ")[0][2:-1].strip().split(",")))
            cost = int(elements.strip().split("   ")[1].strip().split()[1])
            demand = int(elements.strip().split("   ")[2].strip().split()[1])
            for _ in range(2):
                self.costs_data.append(cost)
                self.demands_data.append(demand)
            # even indices store direct edges, and odd indices store reverse edges
            self.origins_data.append(edge[0])
            self.destinations_data.append(edge[1])
            self.origins_data.append(edge[1])
            self.destinations_data.append(edge[0])
            if edge[0] not in required_nodes:
                required_nodes.append(edge[0])
            if edge[1] not in required_nodes:
                required_nodes.append(edge[1])
            node_neighbors[edge[0] - 1][edge[1] - 1] = cost
            node_neighbors[edge[1] - 1][edge[0] - 1] = cost
        if nb_not_required_edges > 0:
            next(file_it)
            for _ in range(nb_not_required_edges):
                elements = next(file_it)
                edge = tuple(map(int, elements.strip().split("   ")[0][2:-1].strip().split(",")))
                cost = int(elements.strip().split("   ")[1].strip().split()[1])
                node_neighbors[edge[0] - 1][edge[1] - 1] = cost
                node_neighbors[edge[1] - 1][edge[0] - 1] = cost
        depot_node = int(next(file_it).strip().split(":")[1])
        # Finds the shortest path from one "required node" to another
        nb_required_nodes = len(required_nodes)
        required_distances = list()
        for node in required_nodes:
            paths = self.shortest_path_finder(node, nb_nodes, node_neighbors)
            required_distances.append(paths)
        # Since we can explore the edges in both directions, we will represent all possible
        # edges with an index
        self.edges_dist_data = None
        self.find_distance_between_edges(nb_required_nodes, required_nodes, required_distances)
        self.dist_to_depot_data = None
        self.find_distance_to_depot(nb_required_nodes, depot_node,
                                    required_nodes, required_distances)
        self.dist_from_depot_data = None
        self.find_distance_from_depot(nb_required_nodes, nb_nodes, depot_node,
                                      required_nodes, required_distances, node_neighbors)

    # Finds the shortest path from one node "origin" to all the other nodes of the graph
    # thanks to the Dijkstra's algorithm
    def min_distance(self, nb_nodes, shortest_path, sptSet):
        min = sys.maxsize
        for i in range(nb_nodes):
            if shortest_path[i] < min and sptSet[i] == False:
                min = shortest_path[i]
                min_index = i
        return min_index

    def shortest_path_finder(self, origin, nb_nodes, node_neighbors):
        shortest_path = [sys.maxsize] * nb_nodes
        shortest_path[origin - 1] = 0
        sptSet = [False] * nb_nodes
        for _ in range(nb_nodes):
            current_node = self.min_distance(nb_nodes, shortest_path, sptSet)
            sptSet[current_node] = True
            current_neighbors = node_neighbors[current_node]
            for neighbor in range(nb_nodes):
                if current_neighbors[neighbor] != 0:
                    distance = current_neighbors[neighbor]
                    if ((sptSet[neighbor] == False) and
                            (shortest_path[current_node] + distance < shortest_path[neighbor])):
                        shortest_path[neighbor] = distance + shortest_path[current_node]
        return shortest_path

    def find_distance_between_edges(self, nb_required_nodes, required_nodes, required_distances):
        self.edges_dist_data = [[None] * (2 * self.nb_required_edges)
                                for _ in range(2 * self.nb_required_edges)]
        for i in range(2 * self.nb_required_edges):
            for j in range(2 * self.nb_required_edges):
                if self.destinations_data[i] == self.origins_data[j]:
                    self.edges_dist_data[i][j] = 0
                else:
                    for k in range(nb_required_nodes):
                        if required_nodes[k] == self.destinations_data[i]:
                            self.edges_dist_data[i][j] = required_distances[k][
                                self.origins_data[j] - 1]

    def find_distance_to_depot(
            self, nb_required_nodes, depot_node, required_nodes, required_distances):
        self.dist_to_depot_data = [None] * (2 * self.nb_required_edges)
        for i in range(2 * self.nb_required_edges):
            if self.destinations_data[i] == depot_node:
                self.dist_to_depot_data[i] = 0
            else:
                for k in range(nb_required_nodes):
                    if required_nodes[k] == self.destinations_data[i]:
                        self.dist_to_depot_data[i] = required_distances[k][depot_node-1]

    def find_distance_from_depot(self, nb_required_nodes, nb_nodes, depot_node,
                                 required_nodes, required_distances, node_neighbors):
        self.dist_from_depot_data = [None] * (2 * self.nb_required_edges)
        for i in range(2 * self.nb_required_edges):
            if depot_node == self.origins_data[i]:
                self.dist_from_depot_data[i] = 0
            else:
                depot_is_required_node = False
                for k in range(nb_required_nodes):
                    if required_nodes[k] == depot_node:
                        depot_is_required_node = True
                        self.dist_from_depot_data[i] = required_distances[k][
                            self.origins_data[i] - 1]
                if not depot_is_required_node:
                    shortest_paths_from_depot = self.shortest_path_finder(
                        depot_node, nb_nodes, node_neighbors)
                    self.dist_from_depot_data[i] = shortest_paths_from_depot[self.origins_data[i] - 1]


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python capacitated_arc_routing.py input_file [output_file] [time_limit]")
        sys.exit(1)

    instance_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    str_time_limit = sys.argv[3] if len(sys.argv) > 3 else "20"

    main(instance_file, output_file, str_time_limit)
```

Multi Trip Capacitated Vehicle Routing Problem (MTCVRP)

```python
import hexaly.optimizer
import sys
import math


def read_elem(filename):
    with open(filename) as f:
        return [str(elem) for elem in f.read().split()]


def main(instance_file, str_time_limit, output_file):
    nb_customers, nb_trucks, truck_capacity, dist_matrix_data, nb_depots, \
    nb_depot_copies, nb_total_locations, demands_data, max_dist = read_input_multi_trip_vrp(instance_file)
    
    with hexaly.optimizer.HexalyOptimizer() as optimizer:

        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Locations visited by each truck (customer or depot)
        # Add copies of the depots (so that they can be visited multiple times)
        # Add an extra fictive truck (who will visit every depot that will not be visited by real trucks)
        visit_orders = [model.list(nb_total_locations) for k in range(nb_trucks+1)]

        # The fictive truck cannot visit customers
        for i in range(nb_customers):
            model.constraint((model.contains(visit_orders[nb_trucks],i)) == False)

        # All customers must be visited by exactly one truck
        model.constraint(model.partition(visit_orders))

        # Create Hexaly arrays to be able to access them with an "at" operator
        demands = model.array(demands_data)
        dist_matrix = model.array(dist_matrix_data)
        
         # A truck is used if it visits at least one customer
        trucks_used = [(model.count(visit_orders[k]) > 0) for k in range(nb_trucks)]

        dist_routes = [None] * nb_trucks
        for k in range(nb_trucks):
            sequence = visit_orders[k]
            c = model.count(sequence)

            # Compute the quantity in the truck at each step
            route_quantity_lambda = model.lambda_function(lambda i,prev: \
                model.iif(sequence[i] < nb_customers, prev+demands[sequence[i]],0))
            route_quantity = model.array(model.range(0, c), route_quantity_lambda, 0)

            # Trucks cannot carry more than their capacity
            quantity_lambda = model.lambda_function(
                lambda i: route_quantity[i] <= truck_capacity)
            model.constraint(model.and_(model.range(0, c), quantity_lambda))

            # Distance traveled by each truck
            dist_lambda = model.lambda_function(lambda i:
                                                model.at(dist_matrix,
                                                         sequence[i - 1],
                                                         sequence[i]))
            dist_routes[k] = model.sum(model.range(1, c), dist_lambda) \
                + model.iif(c > 0,
                            model.at(dist_matrix,nb_customers,sequence[0]) +\
                            model.at(dist_matrix,sequence[c-1],nb_customers),\
                            0)
            
            # Limit distance traveled
            model.constraint( dist_routes[k] <= max_dist)

        # Total distance traveled
        total_distance = model.sum(dist_routes)

        # Objective: minimize the distance traveled
        model.minimize(total_distance)

        model.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = int(str_time_limit)

        optimizer.solve()

        # Write solution output
        if output_file != None:
            with open(output_file, 'w') as file:
                file.write("File name: %s; totalDistance = %d \n" % (instance_file,total_distance.value))
                for k in range(nb_trucks):
                    if trucks_used[k].value:
                        file.write("Truck %d : " % (k))
                        for customer in visit_orders[k].value:
                            file.write( "%d" % (customer) if customer<nb_customers else "%d" % (-(math.floor((customer-nb_customers)/nb_depot_copies) + 1)))
                            file.write(" ")
                        file.write("\n")


def read_input_multi_trip_vrp(filename):
    if filename.endswith(".dat"):
        return read_input_multi_trip_vrp_dat(filename)
    else:
        raise Exception("Unknown file format")

def read_input_multi_trip_vrp_dat(filename):
    file_it = iter(read_elem(filename))

    nb_customers = int(next(file_it))
    nb_depots = int(next(file_it))

    depots_x = [None] * nb_depots
    depots_y = [None] * nb_depots
    for i in range(nb_depots):
        depots_x[i] = int(next(file_it))
        depots_y[i] = int(next(file_it))

    customers_x = [None] * nb_customers
    customers_y = [None] * nb_customers
    for i in range(nb_customers):
        customers_x[i] = int(next(file_it))
        customers_y[i] = int(next(file_it))

    truck_capacity = int(next(file_it))//2

    # Skip depots capacity infos (not related to the problem)
    for i in range(nb_depots):
        next(file_it)

    demands_data = [None] * nb_customers
    for i in range(nb_customers):
        demands_data[i] = int(next(file_it))

    nb_depot_copies = 20

    nb_total_locations  = nb_customers + nb_depots*nb_depot_copies

    max_dist = 400

    nb_trucks = 3

    dist_matrix_data = compute_distance_matrix(depots_x, depots_y, customers_x, customers_y, nb_depot_copies)

    return  nb_customers, nb_trucks, truck_capacity, dist_matrix_data, nb_depots, \
        nb_depot_copies, nb_total_locations, demands_data, max_dist


# Compute the distance matrix
def compute_distance_matrix(depots_x, depots_y, customers_x, customers_y, nb_depot_copies):
    nb_customers = len(customers_x)
    nb_depots = len(depots_x)
    nb_total_locations = nb_customers + nb_depots*nb_depot_copies
    dist_matrix = [[0 for _ in range(nb_total_locations)] for _ in range(nb_total_locations)]
    for i in range(nb_customers):
        dist_matrix[i][i] = 0
        for j in range(i,nb_customers):
            dist = compute_dist(customers_x[i], customers_x[j], customers_y[i], customers_y[j])
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist
        for d in range(nb_depots):
            dist = compute_dist(customers_x[i], depots_x[d], customers_y[i], depots_y[d])
            for c in range(nb_depot_copies):
                j = nb_customers+d*nb_depot_copies + c
                dist_matrix[i][j] = dist
                dist_matrix[j][i] = dist
    
    for i in range(nb_customers, nb_total_locations):
        for j in range(nb_customers, nb_total_locations):
            # Going from one depot to an other is never worth it
            dist_matrix[i][j] = 100000

    return dist_matrix


def compute_dist(xi, xj, yi, yj):
    exact_dist = math.sqrt(math.pow(xi - xj, 2) + math.pow(yi - yj, 2))
    return int(math.floor(exact_dist + 0.5))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python cvrp.py input_file [output_file] [time_limit]")
        sys.exit(1)

    instance_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    str_time_limit = sys.argv[3] if len(sys.argv) > 3 else "20"

    main(instance_file, str_time_limit, output_file)
```


Multi Depot Vehicle Routing Problem (MDVRP)

```python
import hexaly.optimizer
import sys
import math


def main(instance_file, str_time_limit, output_file):
    #
    # Read instance data
    #
    nb_trucks_per_depot, nb_customers, nb_depots, route_duration_capacity_data, \
        truck_capacity_data, demands_data, service_time_data, \
        distance_matrix_customers_data, distance_warehouse_data = read_input_mdvrp(instance_file)

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Sequence of customers visited by each truck
        customer_sequences = [[model.list(nb_customers) for _ in range(nb_trucks_per_depot)] for _ in range(nb_depots)]

        # Vectorization for partition constraint
        customer_sequences_constraint = [customer_sequences[d][k]
                                         for d in range(nb_depots) for k in range(nb_trucks_per_depot)]

        # All customers must be visited by exactly one truck
        model.constraint(model.partition(customer_sequences_constraint))

        # Create Hexaly arrays to be able to access them with an "at" operator
        demands = model.array(demands_data)
        service_time = model.array(service_time_data)
        dist_customers = model.array(distance_matrix_customers_data)

        # Distances traveled by each truck from each depot
        route_distances = [[None for _ in range(nb_trucks_per_depot)] for _ in range(nb_depots)]

        # Total distance traveled
        total_distance = model.sum()

        for d in range(nb_depots):
            dist_depot = model.array(distance_warehouse_data[d])
            for k in range(nb_trucks_per_depot):
                sequence = customer_sequences[d][k]
                c = model.count(sequence)

                # The quantity needed in each route must not exceed the truck capacity
                demand_lambda = model.lambda_function(lambda j: demands[j])
                route_quantity = model.sum(sequence, demand_lambda)
                model.constraint(route_quantity <= truck_capacity_data[d])

                # Distance traveled by truck k of depot d
                dist_lambda = model.lambda_function(lambda i: model.at(dist_customers, sequence[i - 1], sequence[i]))
                route_distances[d][k] = model.sum(model.range(1, c), dist_lambda) \
                    + model.iif(c > 0, dist_depot[sequence[0]] + dist_depot[sequence[c - 1]], 0)

                # We add service Time
                service_lambda = model.lambda_function(lambda j: service_time[j])
                route_service_time = model.sum(sequence, service_lambda)

                total_distance.add_operand(route_distances[d][k])

                # The total distance should not exceed the duration capacity of the truck
                # (only if we define such a capacity)
                if (route_duration_capacity_data[d] > 0):
                    model.constraint(route_distances[d][k] + route_service_time <= route_duration_capacity_data[d])

        # Objective: minimize the total distance traveled
        model.minimize(total_distance)

        model.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = int(str_time_limit)

        optimizer.solve()

        #
        # Write the solution in a file with the following format:
        #  - instance, time_limit, total distance
        #  - for each depot and for each truck in this depot, the customers visited
        #
        if output_file is not None:
            with open(output_file, 'w') as f:
                f.write("Instance: " + instance_file + " ; " + "time_limit: " + str_time_limit + " ; " +
                        "Objective value: " + str(total_distance.value))
                f.write("\n")
                for d in range(nb_depots):
                    trucks_used = []
                    for k in range(nb_trucks_per_depot):
                        if (len(customer_sequences[d][k].value) > 0):
                            trucks_used.append(k)
                    if len(trucks_used) > 0:
                        f.write("Depot " + str(d + 1) + "\n")
                        for k in range(len(trucks_used)):
                            f.write("Truck " + str(k + 1) + " : ")
                            customers_collection = customer_sequences[d][trucks_used[k]].value
                            for p in range(len(customers_collection)):
                                f.write(str(customers_collection[p] + 1) + " ")
                            f.write("\n")
                        f.write("\n")


# Input files following "Cordeau"'s format
def read_input_mdvrp(filename):
    with open(filename) as f:
        instance = f.readlines()

    nb_line = 0
    datas = instance[nb_line].split()

    # Numbers of trucks per depot, customers and depots
    nb_trucks_per_depot = int(datas[1])
    nb_customers = int(datas[2])
    nb_depots = int(datas[3])

    route_duration_capacity = [None]*nb_depots  # Time capacity for every type of truck from every depot
    truck_capacity = [None]*nb_depots  # Capacity for every type of truck from every depot

    for d in range(nb_depots):
        nb_line += 1
        capacities = instance[nb_line].split()

        route_duration_capacity[d] = int(capacities[0])
        truck_capacity[d] = int(capacities[1])

    # Coordinates X and Y, service time and demand for customers
    nodes_xy = [[None, None]] * nb_customers
    service_time = [None] * nb_customers
    demands = [None] * nb_customers

    for n in range(nb_customers):
        nb_line += 1
        customer = instance[nb_line].split()

        nodes_xy[n] = [float(customer[1]), float(customer[2])]

        service_time[n] = int(customer[3])
        demands[n] = int(customer[4])

    # Coordinates X and Y of every depot
    depot_xy = [None] * nb_depots

    for d in range(nb_depots):
        nb_line += 1
        depot = instance[nb_line].split()

        depot_xy[d] = [float(depot[1]), float(depot[2])]

    # Compute the distance matrices
    distance_matrix_customers = compute_distance_matrix_customers(nodes_xy)
    distance_warehouse = compute_distance_warehouse(depot_xy, nodes_xy)

    return nb_trucks_per_depot, nb_customers, nb_depots, route_duration_capacity, \
        truck_capacity, demands, service_time, distance_matrix_customers, distance_warehouse


# Compute the distance matrix for customers
def compute_distance_matrix_customers(nodes_xy):
    nb_customers = len(nodes_xy)
    distance_matrix = [[0 for _ in range(nb_customers)] for _ in range(nb_customers)]
    for i in range(nb_customers):
        for j in range(i+1, nb_customers):
            distij = compute_dist(nodes_xy[i], nodes_xy[j])
            distance_matrix[i][j] = distij
            distance_matrix[j][i] = distij
    return distance_matrix


# Compute the distance matrix for warehouses/depots
def compute_distance_warehouse(depot_xy, nodes_xy):
    nb_customers = len(nodes_xy)
    nb_depots = len(depot_xy)
    distance_warehouse = [[0 for _ in range(nb_customers)] for _ in range(nb_depots)]

    for i in range(nb_customers):
        for d in range(nb_depots):
            distance_warehouse[d][i] = compute_dist(depot_xy[d], nodes_xy[i])

    return distance_warehouse


# Compute the distance between two points
def compute_dist(p, q):
    return math.sqrt(math.pow(p[0] - q[0], 2) + math.pow(p[1] - q[1], 2))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python mdvrp.py input_file [output_file] [time_limit]")
        sys.exit(1)

    instance_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    str_time_limit = sys.argv[3] if len(sys.argv) > 3 else "20"

    main(instance_file, str_time_limit, output_file)
```


Location Routing Problem

```python
import hexaly.optimizer
import sys
import math


def read_elem(filename):
    with open(filename) as f:
        return [str(elem) for elem in f.read().split()]


def main(instance_file, str_time_limit, sol_file):
    #
    # Read instance data
    #
    nb_customers, nb_depots, vehicle_capacity, opening_route_cost, demands_data, \
        capacity_depots, opening_depots_cost, dist_matrix_data, dist_depots_data = \
        read_input_lrp(instance_file)

    min_nb_trucks = int(math.ceil(sum(demands_data) / vehicle_capacity))
    nb_trucks = int(math.ceil(1.5 * min_nb_trucks))

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        m = optimizer.model

        # A route is represented as a list containing the customers in the order they are
        # visited
        customers_sequences = [m.list(nb_customers) for _ in range(nb_trucks)]
        # All customers should be assigned to a route
        m.constraint(m.partition(customers_sequences))

        # A depot is represented as a set containing the associated sequences
        depots = [m.set(nb_trucks) for _ in range(nb_depots)]
        # All the sequences should be assigned to a depot
        m.constraint(m.partition(depots))

        route_costs = [None] * nb_trucks
        sequence_used = [None] * nb_trucks
        dist_routes = [None] * nb_trucks
        associated_depot = [None] * nb_trucks

        # Create Hexaly arrays to be able to access them with "at" operators
        demands = m.array(demands_data)
        dist_matrix = m.array()
        dist_depots = m.array()
        quantity_served = m.array()
        for i in range(nb_customers):
            dist_matrix.add_operand(m.array(dist_matrix_data[i]))
            dist_depots.add_operand(m.array(dist_depots_data[i]))

        for r in range(nb_trucks):
            sequence = customers_sequences[r]
            c = m.count(sequence)

            # A sequence is used if it serves at least one customer
            sequence_used[r] = c > 0
            # The "find" function gets the depot that is assigned to the sequence
            associated_depot[r] = m.find(m.array(depots), r)

            # The quantity needed in each sequence must not exceed the vehicle capacity
            demand_lambda = m.lambda_function(lambda j: demands[j])
            quantity_served.add_operand(m.sum(sequence, demand_lambda))
            m.constraint(quantity_served[r] <= vehicle_capacity)

            # Distance traveled by each truck
            dist_lambda = m.lambda_function(
                lambda i: m.at(dist_matrix, sequence[i], sequence[i + 1]))
            depot = associated_depot[r]
            dist_routes[r] = m.sum(m.range(0, c - 1), dist_lambda) + m.iif(
                sequence_used[r],
                m.at(dist_depots, sequence[0], depot)
                + m.at(dist_depots, sequence[c - 1], depot),
                0)

            # The sequence cost is the sum of the opening cost and the sequence length
            route_costs[r] = sequence_used[r] * opening_route_cost + dist_routes[r]

        depot_cost = [None] * nb_depots
        for d in range(nb_depots):
            # A depot is open if at least one sequence starts from there
            depot_cost[d] = (m.count(depots[d]) > 0) * opening_depots_cost[d]

            # The total demand served by a depot must not exceed its capacity
            depot_lambda = m.lambda_function(lambda r: quantity_served[r])
            depot_quantity = m.sum(depots[d], depot_lambda)
            m.constraint(depot_quantity <= capacity_depots[d])

        depots_cost = m.sum(depot_cost)
        routing_cost = m.sum(route_costs)
        totalCost = routing_cost + depots_cost

        m.minimize(totalCost)

        m.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = int(str_time_limit)

        optimizer.solve()

        if sol_file != None:
            with open(sol_file, 'w') as file:
                file.write("File name: %s; totalCost = %d \n" % (instance_file, totalCost.value))
                for r in range(nb_trucks):
                    if sequence_used[r].value:
                        file.write("Route %d, assigned to depot %d: " % (r, associated_depot[r].value))
                        for customer in customers_sequences[r].value:
                            file.write("%d " % customer)
                        file.write("\n")


def read_input_lrp_dat(filename):
    file_it = iter(read_elem(filename))

    nb_customers = int(next(file_it))
    nb_depots = int(next(file_it))

    x_depot = [None] * nb_depots
    y_depot = [None] * nb_depots
    for i in range(nb_depots):
        x_depot[i] = int(next(file_it))
        y_depot[i] = int(next(file_it))

    x_customer = [None] * nb_customers
    y_customer = [None] * nb_customers
    for i in range(nb_customers):
        x_customer[i] = int(next(file_it))
        y_customer[i] = int(next(file_it))

    vehicle_capacity = int(next(file_it))
    capacity_depots = [None] * nb_depots
    for i in range(nb_depots):
        capacity_depots[i] = int(next(file_it))

    demands = [None] * nb_customers
    for i in range(nb_customers):
        demands[i] = int(next(file_it))

    temp_opening_cost_depot = [None] * nb_depots
    for i in range(nb_depots):
        temp_opening_cost_depot[i] = float(next(file_it))
    temp_opening_route_cost = int(next(file_it))
    are_cost_double = int(next(file_it))

    opening_depots_cost = [None] * nb_depots
    if are_cost_double == 1:
        opening_depots_cost = temp_opening_cost_depot
        opening_route_cost = temp_opening_route_cost
    else:
        opening_route_cost = round(temp_opening_route_cost)
        for i in range(nb_depots):
            opening_depots_cost[i] = round(temp_opening_cost_depot[i])

    distance_customers = compute_distance_matrix(x_customer, y_customer, are_cost_double)
    distance_customers_depots = compute_distance_depot(x_customer, y_customer,
                                                       x_depot, y_depot, are_cost_double)

    return nb_customers, nb_depots, vehicle_capacity, opening_route_cost, demands, \
        capacity_depots, opening_depots_cost, distance_customers, distance_customers_depots

# Compute the distance matrix
def compute_distance_matrix(customers_x, customers_y, are_cost_double):
    nb_customers = len(customers_x)
    dist_customers = [[None for _ in range(nb_customers)] for _ in range(nb_customers)]
    for i in range(nb_customers):
        dist_customers[i][i] = 0
        for j in range(nb_customers):
            dist = compute_dist(customers_x[i], customers_x[j],
                                customers_y[i], customers_y[j], are_cost_double)
            dist_customers[i][j] = dist
            dist_customers[j][i] = dist
    return dist_customers

# Compute the distance depot matrix
def compute_distance_depot(customers_x, customers_y, depot_x, depot_y, are_cost_double):
    nb_customers = len(customers_x)
    nb_depots = len(depot_x)
    distance_customers_depots = [[None for _ in range(nb_depots)] for _ in range(nb_customers)]
    for i in range(nb_customers):
        for d in range(nb_depots):
            dist = compute_dist(customers_x[i], depot_x[d],
                                customers_y[i], depot_y[d], are_cost_double)
            distance_customers_depots[i][d] = dist
    return distance_customers_depots


def compute_dist(xi, xj, yi, yj, are_cost_double):
    dist = math.sqrt(math.pow(xi - xj, 2) + math.pow(yi - yj, 2))
    if are_cost_double == 0:
        dist = math.ceil(100 * dist)
    return dist


def read_input_lrp(filename):
    if filename.endswith(".dat"):
        return read_input_lrp_dat(filename)
    else:
        raise Exception("Unknown file format")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python location_routing_problem.py input_file \
            [output_file] [time_limit]")
        sys.exit(1)

    instance_file = sys.argv[1]
    sol_file = sys.argv[2] if len(sys.argv) > 2 else None
    str_time_limit = sys.argv[3] if len(sys.argv) > 3 else "20"

    main(instance_file, str_time_limit, sol_file)
```

Vehicle Routing Problem with Transhipment Facilities

```python
import hexaly.optimizer
import sys
import math


def read_elem(filename):
    with open(filename) as f:
        return [str(elem) for elem in f.read().split()]


def main(instance_file, str_time_limit, sol_file):
    #
    # Read instance data
    #
    nb_customers, nb_facilities, capacity, customers_demands, \
        depot_distances_data, distance_matrix_data, assignement_costs_data = read_input(instance_file)

    # A point is either a customer or a facility
    # Facilities are duplicated for each customer
    nb_points = nb_customers + nb_customers * nb_facilities

    demands_data = [None] * nb_points
    for c in range(nb_customers):
        demands_data[c] = customers_demands[c]
        for f in range(nb_facilities):
            demands_data[nb_customers + c * nb_facilities + f] = customers_demands[c]

    min_nb_trucks = int(math.ceil(sum(customers_demands) / capacity))
    nb_trucks = int(math.ceil(1.5 * min_nb_trucks))

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        m = optimizer.model

        # Each route is represented as a list containing the points in the order they are visited
        routes_sequences = [m.list(nb_points) for _ in range(nb_trucks)]
        routes = m.array(routes_sequences)

        # Each point must be visited at most once
        m.constraint(m.disjoint(routes_sequences))

        dist_routes = [None] * nb_trucks
        assignement_cost_routes = [None] * nb_trucks

        # Create Hexaly arrays to be able to access them with "at" operators
        demands = m.array(demands_data)
        dist_matrix = m.array()
        dist_depots = m.array(depot_distances_data)
        assignement_costs = m.array(assignement_costs_data)
        for i in range(nb_points):
            dist_matrix.add_operand(m.array(distance_matrix_data[i]))

        for c in range(nb_customers):
            start_facilities = nb_customers + c * nb_facilities
            end_facilities = start_facilities + nb_facilities

            # Each customer is either contained in a route or assigned to a facility
            facility_used = [m.contains(routes, f) for f in range(start_facilities, end_facilities)]
            delivery_count = m.contains(routes, c) + m.sum(facility_used)
            m.constraint(delivery_count == 1)

        for r in range(nb_trucks):
            route = routes_sequences[r]
            c = m.count(route)

            # Each truck cannot carry more than its capacity
            demand_lambda = m.lambda_function(lambda j: demands[j])
            quantity_served = m.sum(route, demand_lambda)
            m.constraint(quantity_served <= capacity)

            # Distance traveled by each truck
            dist_lambda = m.lambda_function(
                lambda i: m.at(dist_matrix, route[i], route[i + 1]))
            dist_routes[r] = m.sum(m.range(0, c - 1), dist_lambda) + m.iif(
                c > 0,
                m.at(dist_depots, route[0])
                + m.at(dist_depots, route[c - 1]),
                0)
            
            # Cost to assign customers to their facility
            assignment_cost_lambda = m.lambda_function(
                lambda i: assignement_costs[i]
            )
            assignement_cost_routes[r] = m.sum(route, assignment_cost_lambda)

        # The total distance travelled
        total_distance_cost = m.sum(dist_routes)
        # The total assignement cost
        total_assignement_cost = m.sum(assignement_cost_routes)

        # Objective: minimize the sum of the total distance travelled and the total assignement cost
        total_cost = total_distance_cost + total_assignement_cost

        m.minimize(total_cost)

        m.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = int(str_time_limit)

        optimizer.solve()

        if sol_file != None:
            with open(sol_file, 'w') as file:
                file.write("File name: {}; totalCost = {}; totalDistance = {}; totalAssignementCost = {}\n"
                           .format(instance_file, total_cost.value, total_distance_cost.value, total_assignement_cost.value))
                for r in range(nb_trucks):
                    route = routes_sequences[r].value
                    if len(route) == 0:
                        continue
                    file.write("Route {} [".format(r))
                    for i, point in enumerate(route):
                        if point < nb_customers:
                            file.write("Customer {}".format(point))
                        else:
                            file.write("Facility {} assigned to Customer {}"
                                       .format(point % nb_customers, (point - nb_customers) // nb_facilities))
                        if i < len(route) - 1:
                            file.write(", ")
                    file.write("]\n")


def read_input_dat(filename):
    file_it = iter(read_elem(filename))

    nb_customers = int(next(file_it))
    nb_facilities = int(next(file_it))

    facilities_x = [None] * nb_facilities
    facilities_y = [None] * nb_facilities
    for i in range(nb_facilities):
        facilities_x[i] = int(next(file_it))
        facilities_y[i] = int(next(file_it))

    customers_x = [None] * nb_customers
    customers_y = [None] * nb_customers
    for i in range(nb_customers):
        customers_x[i] = int(next(file_it))
        customers_y[i] = int(next(file_it))

    truck_capacity = int(next(file_it))

    # Facility capacities : skip
    for f in range(nb_facilities):
        next(file_it)

    customer_demands = [None] * nb_customers
    for i in range(nb_customers):
        customer_demands[i] = int(next(file_it))

    depot_x, depot_y = compute_depot_coordinates(customers_x, customers_y,
                                                 facilities_x, facilities_y)
    depot_distances, distance_matrix = compute_distances(customers_x, customers_y,
                                                         facilities_x, facilities_y,
                                                         depot_x, depot_y)
    assignement_costs = compute_assignment_costs(nb_customers, nb_facilities, distance_matrix)

    return nb_customers, nb_facilities, truck_capacity, customer_demands, \
        depot_distances, distance_matrix, assignement_costs


def compute_depot_coordinates(customers_x, customers_y, facilities_x, facilities_y):
    # Compute the coordinates of the bounding box containing all of the points
    x_min = min(min(customers_x), min(facilities_x))
    x_max = max(max(customers_x), max(facilities_x))
    y_min = min(min(customers_y), min(facilities_y))
    y_max = max(max(customers_y), max(facilities_y))

    # We assume that the depot is at the center of the bounding box
    return x_min + (x_max - x_min) // 2, y_min + (y_max - y_min) // 2


def compute_distances(customers_x, customers_y, facilities_x, facilities_y, depot_x, depot_y):
    nb_customers = len(customers_x)
    nb_facilities = len(facilities_x)
    nb_points = nb_customers + nb_customers * nb_facilities


    # Distance to depot
    depot_distances = [None] * nb_points

    # Customer to depot
    for c in range(nb_customers):
        depot_distances[c] = compute_dist(customers_x[c], depot_x, customers_y[c], depot_y)

    # Facility to depot
    for c in range(nb_customers):
        for f in range(nb_facilities):
            depot_distances[nb_customers + c * nb_facilities + f] = \
                compute_dist(facilities_x[f], depot_x, facilities_y[f], depot_y)

    # Distance between points
    distance_matrix = [[None for _ in range(nb_points)] for _ in range(nb_points)]

    # Distances between customers
    for c_1 in range(nb_customers):
        for c_2 in range(nb_customers):
            distance_matrix[c_1][c_2] = \
                compute_dist(customers_x[c_1], customers_x[c_2],
                             customers_y[c_1], customers_y[c_2])

    # Distances between customers and facilities
    for c_1 in range(nb_customers):
        for f in range(nb_facilities):
            distance = compute_dist(facilities_x[f], customers_x[c_1],
                                    facilities_y[f], customers_y[c_1])
            for c_2 in range(nb_customers):
                # Index representing serving c_2 through facility f
                facility_index = nb_customers + c_2 * nb_facilities + f
                distance_matrix[facility_index][c_1] = distance
                distance_matrix[c_1][facility_index] = distance

    # Distances between facilities
    for f_1 in range(nb_facilities):
        for f_2 in range(nb_facilities):
            dist = compute_dist(facilities_x[f_1], facilities_x[f_2], facilities_y[f_1], facilities_y[f_2])
            for c_1 in range(nb_customers):
                for c_2 in range(nb_customers):
                    distance_matrix[nb_customers + c_1 * nb_facilities + f_1]\
                        [nb_customers + c_2 * nb_facilities + f_2] = dist

    return depot_distances, distance_matrix


def compute_assignment_costs(nb_customers, nb_facilities, distance_matrix):
    # Compute assignment cost for each point
    nb_points = nb_customers + nb_customers * nb_facilities
    assignment_costs = [0] * nb_points
    for c in range(nb_customers):
        for f in range(nb_facilities):
            #  Cost of serving customer c through facility f
            assignment_costs[nb_customers + c * nb_facilities + f] = \
                distance_matrix[c][nb_customers + c * nb_facilities + f]
    return assignment_costs


def compute_dist(xi, xj, yi, yj):
    exact_dist = math.sqrt(math.pow(xi - xj, 2) + math.pow(yi - yj, 2))
    return round(exact_dist)


def read_input(filename):
    if filename.endswith(".dat"):
        return read_input_dat(filename)
    else:
        raise Exception("Unknown file format")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python vrptf.py input_file \
            [output_file] [time_limit]")
        sys.exit(1)

    instance_file = sys.argv[1]
    sol_file = sys.argv[2] if len(sys.argv) > 2 else None
    str_time_limit = sys.argv[3] if len(sys.argv) > 3 else "20"

    main(instance_file, str_time_limit, sol_file)
```

Prize-Collecting Vehicle Routing (PCVRP)

```python
import hexaly.optimizer
import sys
import math


def read_elem(filename):
    with open(filename) as f:
        return [str(elem) for elem in f.read().split()]


def main(instance_file, str_time_limit, output_file):
    #
    # Read instance data
    #
    nb_customers, nb_trucks, truck_capacity, dist_matrix_data, dist_depot_data, \
        demands_data, demands_to_satisfy, prizes_data = read_input_pcvrp(instance_file)

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Sequence of customers visited by each truck
        customers_sequences = [model.list(nb_customers) for _ in range(nb_trucks)]

        # A customer might be visited by only one truck
        model.constraint(model.disjoint(customers_sequences))

        # Create Hexaly arrays to be able to access them with an "at" operator
        demands = model.array(demands_data)
        prizes = model.array(prizes_data)
        dist_matrix = model.array(dist_matrix_data)
        dist_depot = model.array(dist_depot_data)

        # A truck is used if it visits at least one customer
        trucks_used = [(model.count(customers_sequences[k]) > 0) for k in range(nb_trucks)]

        dist_routes = [None] * nb_trucks
        route_prizes = [None] * nb_trucks
        route_quantities = [None] * nb_trucks

        for k in range(nb_trucks):
            sequence = customers_sequences[k]
            c = model.count(sequence)

            # The quantity needed in each route must not exceed the truck capacity
            demand_lambda = model.lambda_function(lambda j: demands[j])
            route_quantities[k] = model.sum(sequence, demand_lambda)
            model.constraint(route_quantities[k] <= truck_capacity)

            # Distance traveled by each truck
            dist_lambda = model.lambda_function(lambda i:
                                                model.at(dist_matrix,
                                                         sequence[i - 1],
                                                         sequence[i]))
            dist_routes[k] = model.sum(model.range(1, c), dist_lambda) \
                + model.iif(c > 0,
                            dist_depot[sequence[0]] + dist_depot[sequence[c - 1]],
                            0)
            
            # Route prize of truck k
            prize_lambda = model.lambda_function(lambda j: prizes[j])
            route_prizes[k] = model.sum(sequence, prize_lambda)

        # Total nb demands satisfied
        total_quantity = model.sum(route_quantities)
    
        # Minimal number of demands to satisfy
        model.constraint(total_quantity >= demands_to_satisfy)

        # Total nb trucks used
        nb_trucks_used = model.sum(trucks_used)

        # Total distance traveled
        total_distance = model.sum(dist_routes)

        # Total prize
        total_prize = model.sum(route_prizes)

        # Objective: minimize the number of trucks used, then maximize the total prize and minimize the distance traveled
        model.minimize(nb_trucks_used)
        model.maximize(total_prize)
        model.minimize(total_distance)

        model.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = int(str_time_limit)

        optimizer.solve()

        #
        # Write the solution in a file with the following format:
        #  - total prize, number of trucks used and total distance
        #  - for each truck the customers visited (omitting the start/end at the depot)
        #  - number of unvisited customers, demands satisfied
        if output_file is not None:
            with open(output_file, 'w') as f:
                f.write("%d %d %d\n" % (total_prize.value, nb_trucks_used.value, total_distance.value))
                nb_unvisited_customers = nb_customers
                for k in range(nb_trucks):
                    if trucks_used[k].value != 1:
                        continue
                    # Values in sequence are in 0...nbCustomers. +1 is to put it back in 1...nbCustomers+1
                    # as in the data files (0 being the depot)
                    for customer in customers_sequences[k].value:
                        f.write("%d " % (customer + 1))
                        nb_unvisited_customers -= 1
                    f.write("\n")
                f.write("%d %d\n" % (nb_unvisited_customers, total_quantity.value))


# The input files follow the "longjianyu" format
def read_input_pcvrp(filename):
    file_it = iter(read_elem(filename))

    nb_trucks = int(next(file_it))
    truck_capacity = int(next(file_it))
    demands_to_satisfy = int(next(file_it))

    n = 0
    customers_x = []
    customers_y = []
    depot_x = 0
    depot_y = 0
    demands = []
    prizes = []
    
    it = next(file_it, None)
    while (it != None):
        node_id = int(it)
        if node_id != n:
            print("Unexpected index")
            sys.exit(1)

        if n == 0:
            depot_x = int(next(file_it))
            depot_y = int(next(file_it))
        else:
            customers_x.append(int(next(file_it)))
            customers_y.append(int(next(file_it)))
            demands.append(int(next(file_it)))
            prizes.append(int(next(file_it)))
        it = next(file_it, None)
        n += 1

    distance_matrix = compute_distance_matrix(customers_x, customers_y)
    distance_depots = compute_distance_depots(depot_x, depot_y, customers_x, customers_y)

    nb_customers = n - 1

    return nb_customers, nb_trucks, truck_capacity, distance_matrix, distance_depots, \
        demands, demands_to_satisfy, prizes


# Compute the distance matrix
def compute_distance_matrix(customers_x, customers_y):
    nb_customers = len(customers_x)
    distance_matrix = [[None for _ in range(nb_customers)] for _ in range(nb_customers)]
    for i in range(nb_customers):
        distance_matrix[i][i] = 0
        for j in range(nb_customers):
            dist = compute_dist(customers_x[i], customers_x[j], customers_y[i], customers_y[j])
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist
    return distance_matrix


# Compute the distances to depot
def compute_distance_depots(depot_x, depot_y, customers_x, customers_y):
    nb_customers = len(customers_x)
    distance_depots = [None] * nb_customers
    for i in range(nb_customers):
        dist = compute_dist(depot_x, customers_x[i], depot_y, customers_y[i])
        distance_depots[i] = dist
    return distance_depots


def compute_dist(xi, xj, yi, yj):
    exact_dist = math.sqrt(math.pow(xi - xj, 2) + math.pow(yi - yj, 2))
    return int(math.floor(exact_dist + 0.5))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python pcvrp.py input_file [output_file] [time_limit]")
        sys.exit(1)

    instance_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    str_time_limit = sys.argv[3] if len(sys.argv) > 3 else "20"

    main(instance_file, str_time_limit, output_file)
```

Aircraft Landing

```python

import hexaly.optimizer
import sys


def read_elem(filename):
    with open(filename) as f:
        return [str(elem) for elem in f.read().split()]


#
# Read instance data
#
def read_instance(instance_file):
    file_it = iter(read_elem(instance_file))
    nb_planes = int(next(file_it))
    next(file_it)  # Skip freezeTime value
    earliest_time_data = []
    target_time_data = []
    latest_time_data = []
    earliness_cost_data = []
    tardiness_cost_data = []
    separation_time_data = []

    for p in range(nb_planes):
        next(file_it)  # Skip appearanceTime values
        earliest_time_data.append(int(next(file_it)))
        target_time_data.append(int(next(file_it)))
        latest_time_data.append(int(next(file_it)))
        earliness_cost_data.append(float(next(file_it)))
        tardiness_cost_data.append(float(next(file_it)))
        separation_time_data.append([None] * nb_planes)

        for pp in range(nb_planes):
            separation_time_data[p][pp] = int(next(file_it))

    return nb_planes, earliest_time_data, target_time_data, latest_time_data, \
        earliness_cost_data, tardiness_cost_data, separation_time_data


def get_min_landing_time(p, prev, model, separation_time, landing_order):
    return model.iif(
        p > 0,
        prev + model.at(separation_time, landing_order[p - 1], landing_order[p]),
        0)


def main(instance_file, output_file, time_limit):
    nb_planes, earliest_time_data, target_time_data, latest_time_data, \
        earliness_cost_data, tardiness_cost_data, separation_time_data = \
        read_instance(instance_file)

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # A list variable: landingOrder[i] is the index of the ith plane to land
        landing_order = model.list(nb_planes)

        # All planes must be scheduled
        model.constraint(model.count(landing_order) == nb_planes)

        # Create Hexaly arrays to be able to access them with an "at" operator
        target_time = model.array(target_time_data)
        latest_time = model.array(latest_time_data)
        earliness_cost = model.array(earliness_cost_data)
        tardiness_cost = model.array(tardiness_cost_data)
        separation_time = model.array(separation_time_data)

        # Int variable: preferred landing time for each plane
        preferred_time_vars = [model.int(earliest_time_data[p], target_time_data[p])
                               for p in range(nb_planes)]
        preferred_time = model.array(preferred_time_vars)

        # Landing time for each plane
        landing_time_lambda = model.lambda_function(
            lambda p, prev:
                model.max(
                    preferred_time[landing_order[p]],
                    get_min_landing_time(p, prev, model, separation_time, landing_order)))
        landing_time = model.array(model.range(0, nb_planes), landing_time_lambda)

        # Landing times must respect the separation time with every previous plane
        for p in range(1, nb_planes):
            last_separation_end = [
                landing_time[previous_plane]
                + model.at(
                    separation_time,
                    landing_order[previous_plane],
                    landing_order[p])
                for previous_plane in range(p)]
            model.constraint(landing_time[p] >= model.max(last_separation_end))

        total_cost = model.sum()
        for p in range(nb_planes):
            plane_index = landing_order[p]

            # Constraint on latest landing time
            model.constraint(landing_time[p] <= latest_time[plane_index])

            # Cost for each plane
            difference_to_target_time = abs(landing_time[p] - target_time[plane_index])
            unit_cost = model.iif(
                landing_time[p] < target_time[plane_index],
                earliness_cost[plane_index],
                tardiness_cost[plane_index])
            total_cost.add_operand(unit_cost * difference_to_target_time)

        # Minimize the total cost
        model.minimize(total_cost)

        model.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = time_limit

        optimizer.solve()

        #
        # Write the solution in a file with the following format:
        # - 1st line: value of the objective;
        # - 2nd line: for each position p, index of plane at position p.
        #
        if output_file is not None:
            with open(output_file, 'w') as f:
                f.write("%d\n" % total_cost.value)
                for p in landing_order.value:
                    f.write("%d " % p)
                f.write("\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python aircraft_landing.py instance_file [output_file] [time_limit]")
        sys.exit(1)

    instance_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 else None
    time_limit = int(sys.argv[3]) if len(sys.argv) >= 4 else 20
    main(instance_file, output_file, time_limit)
```

Movie Shoot Scheduling

```python
import hexaly.optimizer
import sys


def read_integers(filename):
    with open(filename) as f:
        return [int(elem) for elem in f.read().split()]


class MssInstance:

    #
    # Read instance data
    #
    def __init__(self, filename):
        file_it = iter(read_integers(filename))
        self.nb_actors = next(file_it)
        self.nb_scenes = next(file_it)
        self.nb_locations = next(file_it)
        self.nb_precedences = next(file_it)
        self.actor_cost = [next(file_it) for i in range(self.nb_actors)]
        self.location_cost = [next(file_it) for i in range(self.nb_locations)]
        self.scene_duration = [next(file_it) for i in range(self.nb_scenes)]
        self.scene_location = [next(file_it) for i in range(self.nb_scenes)]
        self.is_actor_in_scene = [[next(file_it) for i in range(self.nb_scenes)]
                                  for i in range(self.nb_actors)]
        self.precedences = [[next(file_it) for i in range(2)]
                            for i in range(self.nb_precedences)]

        self.actor_nb_worked_days = self._compute_nb_worked_days()

    def _compute_nb_worked_days(self):
        actor_nb_worked_days = [0] * self.nb_actors
        for a in range(self.nb_actors):
            for s in range(self.nb_scenes):
                if self.is_actor_in_scene[a][s]:
                    actor_nb_worked_days[a] += self.scene_duration[s]
        return actor_nb_worked_days


def main(instance_file, output_file, time_limit):
    data = MssInstance(instance_file)

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Decision variable: A list, shoot_order[i] is the index of the ith scene to be shot
        shoot_order = model.list(data.nb_scenes)

        # All scenes must be scheduled
        model.constraint(model.count(shoot_order) == data.nb_scenes)

        # Constraint of precedence between scenes
        for i in range(data.nb_precedences):
            model.constraint(model.index(shoot_order, data.precedences[i][0])
                             < model.index(shoot_order, data.precedences[i][1]))

        # Minimize external function
        cost_function = CostFunction(data)
        func = model.create_int_external_function(cost_function.compute_cost)
        func.external_context.lower_bound = 0
        cost = func(shoot_order)
        model.minimize(cost)
        model.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = time_limit if len(sys.argv) >= 4 else 20
        optimizer.solve()

        # Write the solution in a file in the following format:
        # - 1st line: value of the objective;
        # - 2nd line: for each i, the index of the ith scene to be shot.
        if len(sys.argv) >= 3:
            with open(output_file, 'w') as f:
                f.write("%d\n" % cost.value)
                for i in shoot_order.value:
                    f.write("%d " % i)
                f.write("\n")


class CostFunction:

    def __init__(self, data):
        self.data = data

    def compute_cost(self, context):
        shoot_order = context[0]
        if len(shoot_order) < self.data.nb_scenes:
            # Infeasible solution if some scenes are missing
            return sys.maxsize

        location_extra_cost = self._compute_location_cost(shoot_order)
        actor_extra_cost = self._compute_actor_cost(shoot_order)
        return location_extra_cost + actor_extra_cost

    def _compute_location_cost(self, shoot_order):
        nb_location_visits = [0] * self.data.nb_locations
        previous_location = -1
        for i in range(self.data.nb_scenes):
            current_location = self.data.scene_location[shoot_order[i]]
            # When we change location, we increment the number of scenes of the new location
            if previous_location != current_location:
                nb_location_visits[current_location] += 1
                previous_location = current_location
        location_extra_cost = sum(cost * (nb_visits - 1)
            for cost, nb_visits in zip(self.data.location_cost, nb_location_visits))
        return location_extra_cost

    def _compute_actor_cost(self, shoot_order):
        # Compute first and last days of work for each actor
        actor_first_day = [0] * self.data.nb_actors
        actor_last_day = [0] * self.data.nb_actors
        for j in range(self.data.nb_actors):
            has_actor_started_working = False
            start_day_of_scene = 0
            for i in range(self.data.nb_scenes):
                current_scene = shoot_order[i]
                end_day_of_scene = start_day_of_scene \
                    + self.data.scene_duration[current_scene] - 1
                if self.data.is_actor_in_scene[j][current_scene]:
                    actor_last_day[j] = end_day_of_scene
                    if not has_actor_started_working:
                        has_actor_started_working = True
                        actor_first_day[j] = start_day_of_scene
                # The next scene begins the day after the end of the current one
                start_day_of_scene = end_day_of_scene + 1

        # Compute actor extra cost due to days paid but not worked
        actor_extra_cost = 0
        for j in range(self.data.nb_actors):
            nb_paid_days = actor_last_day[j] - actor_first_day[j] + 1
            actor_extra_cost += (nb_paid_days - self.data.actor_nb_worked_days[j]) \
                * self.data.actor_cost[j]
        return actor_extra_cost


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python movie_shoot_scheduling.py instance_file \
            [output_file] [time_limit]")
        sys.exit(1)

    instance_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 else None
    time_limit = int(sys.argv[3]) if len(sys.argv) >= 4 else 60
    main(instance_file, output_file, time_limit)
```



Revenue Management

```python
import hexaly.optimizer
import sys
import math
import random


class RevenueManagementFunction:

    def __init__(self, seed):
        self.nb_periods = 3
        self.prices = [100, 300, 400]
        self.mean_demands = [50, 20, 30]
        self.purchase_price = 80
        self.evaluated_points = [{
            "point": [100, 50, 30],
            "value": 4740.99
        }]
        self.nb_simulations = int(1e6)
        self.seed = seed

    # External function
    def evaluate(self, argument_values):
        variables = [argument_values.get(i) for i in range(argument_values.count())]
        # Initial quantity purchased
        nb_units_purchased = variables[0]
        # Number of units that should be left for future periods
        nb_units_reserved = variables[1:] + [0]

        # Set seed for reproducibility
        random.seed(self.seed)
        # Create distribution
        X = [gamma_sample() for i in range(self.nb_simulations)]
        Y = [[exponential_sample() for i in range(self.nb_periods)]
             for j in range(self.nb_simulations)]

        # Run simulations
        sum_profit = 0.0
        for i in range(self.nb_simulations):
            remaining_capacity = nb_units_purchased
            for j in range(self.nb_periods):
                # Generate demand for period j
                demand_j = int(self.mean_demands[j] * X[i] * Y[i][j])
                nb_units_sold = min(
                    max(remaining_capacity - nb_units_reserved[j], 0),
                    demand_j)
                remaining_capacity = remaining_capacity - nb_units_sold
                sum_profit += self.prices[j] * nb_units_sold

        # Calculate mean revenue
        mean_profit = sum_profit / self.nb_simulations
        mean_revenue = mean_profit - self.purchase_price * nb_units_purchased

        return mean_revenue


def exponential_sample(rate_param=1.0):
    u = random.random()
    return math.log(1 - u) / (-rate_param)


def gamma_sample(scale_param=1.0):
    return exponential_sample(scale_param)


def solve(evaluation_limit, time_limit, output_file):
    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Generate data
        revenue_management = RevenueManagementFunction(1)
        nb_periods = revenue_management.nb_periods
        # Declare decision variables
        variables = [model.int(0, 100) for _ in range(nb_periods)]

        # Create the function
        func_expr = model.create_double_external_function(revenue_management.evaluate)
        # Call function
        func_call = model.call(func_expr)
        func_call.add_operands(variables)

        # Declare constraints
        for i in range(1, nb_periods):
            model.constraint(variables[i] <= variables[i - 1])

        # Maximize function call
        model.maximize(func_call)

        # Enable surrogate modeling
        context = func_expr.external_context
        surrogate_params = context.enable_surrogate_modeling()

        # Set lower bound
        context.lower_bound = 0.0

        model.close()

        # Parametrize the optimizer
        if time_limit is not None:
            optimizer.param.time_limit = time_limit

        # Set the maximum number of evaluations
        surrogate_params.evaluation_limit = evaluation_limit

        # Add evaluation points
        for evaluated_point in revenue_management.evaluated_points:
            evaluation_point = surrogate_params.create_evaluation_point()
            for i in range(nb_periods):
                evaluation_point.add_argument(evaluated_point["point"][i])
            evaluation_point.set_return_value(evaluated_point["value"])

        optimizer.solve()

        # Write the solution in a file
        if output_file is not None:
            with open(output_file, 'w') as f:
                f.write("obj=%f\n" % func_call.value)
                f.write("b=%f\n" % variables[0].value)
                for i in range(1, nb_periods):
                    f.write("r%f=%f\n" % (i + 1, variables[i].value))


if __name__ == '__main__':
    output_file = sys.argv[1] if len(sys.argv) > 1 else None
    time_limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    evaluation_limit = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    solve(evaluation_limit, time_limit, output_file)
```

Pooling

```python
import hexaly.optimizer
import json
import sys


class PoolingInstance:

    #
    # Read instance data
    #
    def __init__(self, instance_file):
        with open(instance_file) as problem:
            problem = json.load(problem)

            self.nbComponents = len(problem["components"])
            self.nbProducts = len(problem["products"])
            self.nbAttributes = len(problem["components"][0]["quality"])
            self.nbPools = len(problem["pool_size"])

            # Components
            self.componentPrices = [problem["components"][c]["price"]
                                    for c in range(self.nbComponents)]
            self.componentSupplies = [problem["components"][c]["upper"]
                                      for c in range(self.nbComponents)]
            self.componentQuality = [list(problem["components"][c]["quality"].values())
                                     for c in range(self.nbComponents)]
            self.componentNames = [problem["components"][c]["name"]
                                   for c in range(self.nbComponents)]

            componentsIdx = {}
            for c in range(self.nbComponents):
                componentsIdx[problem["components"][c]["name"]] = c

            # Final products (blendings)
            self.productPrices = [problem["products"][p]["price"]
                                  for p in range(self.nbProducts)]
            self.productCapacities = [problem["products"][p]["upper"]
                                      for p in range(self.nbProducts)]
            self.demand = [problem["products"][p]["lower"]
                           for p in range(self.nbProducts)]
            self.productNames = [problem["products"][p]["name"]
                                 for p in range(self.nbProducts)]

            productIdx = {}
            for p in range(self.nbProducts):
                productIdx[problem["products"][p]["name"]] = p

            self.minTolerance = [[0 for _ in range(self.nbAttributes)]
                if (problem["products"][p]["quality_lower"] == None)
                else list(problem["products"][p]["quality_lower"].values())
                for p in range(self.nbProducts)]
            self.maxTolerance = [list(problem["products"][p]["quality_upper"].values())
                                 for p in range(self.nbProducts)]

            # Intermediate pools
            self.poolNames = list(problem["pool_size"].keys())
            self.poolCapacities = [problem["pool_size"][o] for o in self.poolNames]
            poolIdx = {}
            for o in range(self.nbPools):
                poolIdx[self.poolNames[o]] = o

            # Flow graph

            # Edges from the components to the products
            self.upperBoundComponentToProduct = [[0 for _ in range(self.nbProducts)]
                                                 for _ in range(self.nbComponents)]
            self.costComponentToProduct = [[0 for _ in range(self.nbProducts)]
                                           for _ in range(self.nbComponents)]
            # Edges from the components to the pools
            self.upperBoundFractionComponentToPool = [[0 for _ in range(self.nbPools)]
                                                      for _ in range(self.nbComponents)]
            self.costComponentToPool = [[0 for _ in range(self.nbPools)]
                                        for _ in range(self.nbComponents)]
            # Edges from the pools to the products
            self.upperBoundPoolToProduct = [[0 for _ in range(self.nbProducts)]
                                            for _ in range(self.nbPools)]
            self.costPoolToProduct = [[0 for _ in range(self.nbProducts)]
                                      for _ in range(self.nbPools)]

            # Bound and cost on the edges
            for edge in problem["component_to_product_bound"]:
                self.upperBoundComponentToProduct[componentsIdx[edge["component"]]] \
                    [productIdx[edge["product"]]] = edge["bound"]
                if len(edge) > 3:
                    self.costComponentToProduct[componentsIdx[edge["component"]]] \
                        [productIdx[edge["product"]]] = edge["cost"]

            for edge in problem["component_to_pool_fraction"]:
                self.upperBoundFractionComponentToPool[componentsIdx[edge["component"]]] \
                    [poolIdx[edge["pool"]]] = edge["fraction"]
                if len(edge) > 3:
                    self.costComponentToPool[componentsIdx[edge["component"]]] \
                        [poolIdx[edge["pool"]]] = edge["cost"]

            for edge in problem["pool_to_product_bound"]:
                self.upperBoundPoolToProduct[poolIdx[edge["pool"]]] \
                    [productIdx[edge["product"]]] = edge["bound"]
                if len(edge) > 3:
                    self.costPoolToProduct[poolIdx[edge["pool"]]] \
                        [productIdx[edge["product"]]] = edge["cost"]


def main(instance_file, output_file, time_limit):
    data = PoolingInstance(instance_file)

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Decision variables 

        # Flow from the components to the products
        flowComponentToProduct = [[model.float(0, data.upperBoundComponentToProduct[c][p])
            for p in range(data.nbProducts)] for c in range(data.nbComponents)]

        # Fraction of the total flow in pool o coming from the component c
        fractionComponentToPool = [[model.float(0, data.upperBoundFractionComponentToPool[c][o])
                                    for o in range(data.nbPools)]
                                   for c in range(data.nbComponents)]

        # Flow from the pools to the products
        flowPoolToProduct = [
            [model.float(0, data.upperBoundPoolToProduct[o][p])
             for p in range(data.nbProducts)] for o in range(data.nbPools)]

        # Flow from the components to the products and passing by the pools
        flowComponentToProductByPool = [
            [[fractionComponentToPool[c][o] * flowPoolToProduct[o][p]
              for p in range(data.nbProducts)] for o in range(data.nbPools)]
            for c in range(data.nbComponents)]

        # Constraints

        # Proportion
        for o in range(data.nbPools):
            proportion = model.sum(fractionComponentToPool[c][o]
                                   for c in range(data.nbComponents))
            model.constraint(proportion == 1)

        # Component supply
        for c in range(data.nbComponents):
            flowToProducts = model.sum(flowComponentToProduct[c][p]
                                       for p in range(data.nbProducts))
            flowToPools = model.sum(flowComponentToProductByPool[c][o][p]
                for p in range(data.nbProducts) for o in range(data.nbPools))
            totalOutFlow = model.sum(flowToPools, flowToProducts)
            model.constraint(totalOutFlow <= data.componentSupplies[c])

        # Pool capacity (bounds on edges)
        for c in range(data.nbComponents):
            for o in range(data.nbPools):
                flowComponentToPool = model.sum(flowComponentToProductByPool[c][o][p]
                                                for p in range(data.nbProducts))
                edgeCapacity = model.prod(data.poolCapacities[o],
                                          fractionComponentToPool[c][o])
                model.constraint(flowComponentToPool <= edgeCapacity)

        # Product capacity
        for p in range(data.nbProducts):
            flowFromPools = model.sum(flowPoolToProduct[o][p] for o in range(data.nbPools))
            flowFromComponents = model.sum(flowComponentToProduct[c][p]
                                           for c in range(data.nbComponents))
            totalInFlow = model.sum(flowFromComponents, flowFromPools)
            model.constraint(totalInFlow <= data.productCapacities[p])
            model.constraint(totalInFlow >= data.demand[p])

        # Product tolerance
        for p in range(data.nbProducts):
            for k in range(data.nbAttributes):
                # Attribute from the components
                attributeFromComponents = model.sum(
                    data.componentQuality[c][k] * flowComponentToProduct[c][p]
                    for c in range(data.nbComponents))

                # Attribute from the pools
                attributeFromPools = model.sum(
                    data.componentQuality[c][k] * flowComponentToProductByPool[c][o][p]
                    for o in range(data.nbPools) for c in range(data.nbComponents))

                # Total flow in the blending
                totalFlowIn = model.sum(flowComponentToProduct[c][p]
                                        for c in range(data.nbComponents)) \
                    + model.sum(flowPoolToProduct[o][p] for o in range(data.nbPools))

                totalAttributeIn = model.sum(attributeFromComponents, attributeFromPools)
                model.constraint(totalAttributeIn >= data.minTolerance[p][k] * totalFlowIn)
                model.constraint(totalAttributeIn <= data.maxTolerance[p][k] * totalFlowIn)

        # Objective function

        # Cost of the flows from the components directly to the products
        directFlowCost = model.sum(
            data.costComponentToProduct[c][p] * flowComponentToProduct[c][p]
            for c in range(data.nbComponents) for p in range(data.nbProducts))

        # Cost of the flows from the components to the products passing by the pools
        undirectFlowCost = model.sum(
            (data.costComponentToPool[c][o] + data.costPoolToProduct[o][p]) *
            flowComponentToProductByPool[c][o][p]
            for c in range(data.nbComponents)
            for o in range(data.nbPools) for p in range(data.nbProducts))

        # Gain of selling the final products
        productsGain = model.sum((model.sum(flowComponentToProduct[c][p]
                                            for c in range(data.nbComponents))
                                  + model.sum(flowPoolToProduct[o][p]
                                              for o in range(data.nbPools)))
                                 * data.productPrices[p] for p in range(data.nbProducts))

        # Cost of buying the components
        componentsCost = model.sum(
            (model.sum(flowComponentToProduct[c][p]
                       for p in range(data.nbProducts))
             + model.sum(fractionComponentToPool[c][o] * flowPoolToProduct[o][p]
                         for p in range(data.nbProducts)
                         for o in range(data.nbPools)))
            * data.componentPrices[c] for c in range(data.nbComponents))

        profit = productsGain - componentsCost - (directFlowCost + undirectFlowCost)

        # Maximize the total profit
        model.maximize(profit)

        model.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = time_limit

        optimizer.solve()

        #
        # Write the solution
        #
        if output_file is not None:
            with open(output_file, 'w') as f:
                component_to_poduct = []
                component_to_pool_fraction = []
                pool_to_product = []

                # Solution flows from the components to the products
                for c in range(data.nbComponents):
                    for p in range(data.nbProducts):
                        component_to_poduct.append(
                            {"component": data.componentNames[c],
                             "product": data.productNames[p],
                             "flow": flowComponentToProduct[c][p].value})

                # Solution fraction of the inflow at pool o coming from the component c
                for c in range(data.nbComponents):
                    for o in range(data.nbPools):
                        component_to_pool_fraction.append(
                            {"component": data.componentNames[c],
                             "pool": data.poolNames[o],
                             "flow": fractionComponentToPool[c][o].value})

                # Solution flows from the pools to the products
                for o in range(data.nbPools):
                    for p in range(data.nbProducts):
                        pool_to_product.append(
                            {"pool": data.poolNames[o],
                             "product": data.productNames[p],
                             "flow": flowPoolToProduct[o][p].value})

                json.dump({"objective": profit.value, "solution":
                    {"component_to_pool_fraction": component_to_pool_fraction,
                     "component_to_product": component_to_poduct,
                     "pool_to_product": pool_to_product}}, f)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python pooling.py instance_file [output_file] [time_limit]")
        sys.exit(1)

    instance_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 else None
    time_limit = int(sys.argv[3]) if len(sys.argv) >= 4 else 20
    main(instance_file, output_file, time_limit)
```


Vehicle Routing with Backhauls (VRPB)

```python
import hexaly.optimizer
import sys
import math


def read_elem(filename):
    with open(filename) as f:
        return [str(elem) for elem in f.read().split()]


def main(instance_file, str_time_limit, output_file):

    #
    # Read instance data
    #
    nb_customers, nb_trucks, truck_capacity, dist_matrix_data, dist_depot_data, \
        delivery_demands_data, pickup_demands_data, backhaul_data = read_input_vrpb(
            instance_file)

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Sequence of customers visited by each truck
        customers_sequences = [model.list(nb_customers)
                               for _ in range(nb_trucks)]

        # All customers must be visited by exactly one truck
        model.constraint(model.partition(customers_sequences))

        # Create Hexaly arrays to be able to access them with an "at" operator
        delivery_demands = model.array(delivery_demands_data)
        pickup_demands = model.array(pickup_demands_data)
        dist_matrix = model.array(dist_matrix_data)
        dist_depot = model.array(dist_depot_data)

        # A truck is used if it visits at least one customer
        trucks_used = [(model.count(customers_sequences[k]) > 0)
                       for k in range(nb_trucks)]

        dist_routes = [None] * nb_trucks
        is_backhaul = model.array(backhaul_data.values())
        for k in range(nb_trucks):
            sequence = customers_sequences[k]
            c = model.count(sequence)

            # A pickup cannot be followed by a delivery
            precedency_lambda = model.lambda_function(lambda i: model.or_(model.not_(
                model.at(is_backhaul, sequence[i-1])), model.at(is_backhaul, sequence[i])))
            model.constraint(model.and_(model.range(1, c), precedency_lambda))

            # The quantity needed in each route must not exceed the truck capacity
            delivery_demand_lambda = model.lambda_function(
                lambda j: delivery_demands[j])
            route_pickup_quantity = model.sum(sequence, delivery_demand_lambda)
            model.constraint(route_pickup_quantity <= truck_capacity)

            pickup_demand_lambda = model.lambda_function(
                lambda j: pickup_demands[j])
            route_pickup_quantity = model.sum(sequence, pickup_demand_lambda)
            model.constraint(route_pickup_quantity <= truck_capacity)

            # Distance traveled by each truck
            dist_lambda = model.lambda_function(lambda i:
                                                model.at(dist_matrix,
                                                         sequence[i - 1],
                                                         sequence[i]))
            dist_routes[k] = model.sum(model.range(1, c), dist_lambda) \
                + model.iif(c > 0,
                            dist_depot[sequence[0]] +
                            dist_depot[sequence[c - 1]],
                            0)

        # Total number of trucks used
        nb_trucks_used = model.sum(trucks_used)

        # Total distance traveled
        total_distance = model.sum(dist_routes)

        # Objective: minimize the number of trucks used, then minimize the distance traveled
        model.minimize(nb_trucks_used)
        model.minimize(total_distance)

        model.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = int(str_time_limit)

        optimizer.solve()

        #
        # Write the solution in a file with the following format:
        #  - number of trucks used and total distance
        #  - for each truck the customers visited (omitting the start/end at the depot)
        #
        if output_file is not None:
            with open(output_file, 'w') as f:
                f.write("%d %d\n" %
                        (nb_trucks_used.value, total_distance.value))
                for k in range(nb_trucks):
                    if trucks_used[k].value != 1:
                        continue
                    # Values in sequence are in 0...nbCustomers. +2 is to put it back
                    # in 2...nbCustomers+2 as in the data files (1 being the depot)
                    for customer in customers_sequences[k].value:
                        f.write("%d " % (customer + 2))
                    f.write("\n")


# The input files follow the "CVRPLib" format
def read_input_vrpb(filename):
    file_it = iter(read_elem(filename))

    nb_nodes = 0
    while True:
        token = next(file_it)
        if token == "DIMENSION":
            next(file_it)  # Removes the ":"
            nb_nodes = int(next(file_it))
            nb_customers = nb_nodes - 1
        elif token == "VEHICLES":
            next(file_it)  # Removes the ":"
            nb_trucks = int(next(file_it))
        elif token == "CAPACITY":
            next(file_it)  # Removes the ":"
            truck_capacity = int(next(file_it))
        elif token == "EDGE_WEIGHT_TYPE":
            next(file_it)  # Removes the ":"
            token = next(file_it)
            if token != "EXACT_2D":
                print("Edge Weight Type " + token +
                      " is not supported (only EXACT_2D)")
                sys.exit(1)
        elif token == "NODE_COORD_SECTION":
            break

    customers_x = [None] * nb_customers
    customers_y = [None] * nb_customers
    depot_x = 0
    depot_y = 0
    for n in range(nb_nodes):
        node_id = int(next(file_it))
        if node_id != n + 1:
            print("Unexpected index")
            sys.exit(1)
        if node_id == 1:
            depot_x = int(next(file_it))
            depot_y = int(next(file_it))
        else:
            # -2 because original customer indices are in 2..nbNodes
            customers_x[node_id - 2] = int(next(file_it))
            customers_y[node_id - 2] = int(next(file_it))

    distance_matrix = compute_distance_matrix(customers_x, customers_y)
    distance_depots = compute_distance_depots(
        depot_x, depot_y, customers_x, customers_y)

    token = next(file_it)
    if token != "DEMAND_SECTION":
        print("Expected token DEMAND_SECTION")
        sys.exit(1)

    demands = [None] * nb_customers
    for n in range(nb_nodes):
        node_id = int(next(file_it))
        if node_id != n + 1:
            print("Unexpected index")
            sys.exit(1)
        if node_id == 1:
            if int(next(file_it)) != 0:
                print("Demand for depot should be 0")
                sys.exit(1)
        else:
            # -2 because original customer indices are in 2..nbNodes
            demands[node_id - 2] = int(next(file_it))

    token = next(file_it)
    if token != "BACKHAUL_SECTION":
        print("Expected token BACKHAUL_SECTION")
        sys.exit(1)

    is_backhaul = {i: False for i in range(nb_customers)}
    while True:
        node_id = int(next(file_it))
        if node_id == -1:
            break
        # -2 because original customer indices are in 2..nbNodes
        is_backhaul[node_id - 2] = True
    delivery_demands = [0 if is_backhaul[i] else demands[i]
                        for i in range(nb_customers)]
    pickup_demands = [demands[i] if is_backhaul[i]
                      else 0 for i in range(nb_customers)]

    token = next(file_it)
    if token != "DEPOT_SECTION":
        print("Expected token DEPOT_SECTION")
        sys.exit(1)

    depot_id = int(next(file_it))
    if depot_id != 1:
        print("Depot id is supposed to be 1")
        sys.exit(1)

    end_of_depot_section = int(next(file_it))
    if end_of_depot_section != -1:
        print("Expecting only one depot, more than one found")
        sys.exit(1)

    return nb_customers, nb_trucks, truck_capacity, distance_matrix, distance_depots, \
        delivery_demands, pickup_demands, is_backhaul


# Compute the distance matrix
def compute_distance_matrix(customers_x, customers_y):
    nb_customers = len(customers_x)
    distance_matrix = [[None for i in range(
        nb_customers)] for j in range(nb_customers)]
    for i in range(nb_customers):
        distance_matrix[i][i] = 0
        for j in range(nb_customers):
            dist = compute_dist(
                customers_x[i], customers_x[j], customers_y[i], customers_y[j])
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist
    return distance_matrix


# Compute the distances to depot
def compute_distance_depots(depot_x, depot_y, customers_x, customers_y):
    nb_customers = len(customers_x)
    distance_depots = [None] * nb_customers
    for i in range(nb_customers):
        dist = compute_dist(depot_x, customers_x[i], depot_y, customers_y[i])
        distance_depots[i] = dist
    return distance_depots


def compute_dist(xi, xj, yi, yj):
    exact_dist = math.sqrt(math.pow(xi - xj, 2) + math.pow(yi - yj, 2))
    return int(math.floor(exact_dist + 0.5))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(
            "Usage: python vehicle_routing_backhauls.py input_file [output_file] [time_limit]")
        sys.exit(1)

    instance_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    str_time_limit = sys.argv[3] if len(sys.argv) > 3 else "20"

    main(instance_file, str_time_limit, output_file)
```

Cantilevered Beam

```python
import hexaly.optimizer
import sys

# Constant declaration
P = 1000
E = 10.0e6
L = 36
possibleValues = [0.1, 0.26, 0.35, 0.5, 0.65, 0.75, 0.9, 1.0]

# External function
def evaluate(arguments_values):
    # Argument retrieval
    fH = arguments_values[0]
    fh1 = possibleValues[arguments_values[1]]
    fb1 = arguments_values[2]
    fb2 = arguments_values[3]

    # Big computation
    I = 1.0 / 12.0 * fb2 * pow(fH - 2 * fh1, 3) + 2 * (1.0 / 12.0 * fb1
        * pow(fh1, 3) + fb1 * fh1 * pow(fH - fh1, 2) / 4.0)

    # Constraint computations
    g1 = P * L * fH / (2 * I)
    g2 = P * L**3 / (3 * E * I)

    # Objective function computation
    f = (2 * fh1 * fb1 + (fH - 2 * fh1) * fb2) * L

    return g1, g2, f

def main(evaluation_limit, output_file):
    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        # Declare the optimization model
        model = optimizer.model

        # Numerical decisions
        H = model.float(3.0, 7.0)
        h1 = model.int(0, 7)
        b1 = model.float(2.0, 12.0)
        b2 = model.float(0.1, 2.0)

        # Create and call the external function
        func = model.create_double_array_external_function(evaluate)
        func_call = model.call(func)
        # Add the operands
        func_call.add_operand(H)
        func_call.add_operand(h1)
        func_call.add_operand(b1)
        func_call.add_operand(b2)

        # Enable surrogate modeling
        surrogate_params = func.external_context.enable_surrogate_modeling()

        # Constraint on bending stress
        model.constraint(func_call[0] <= 5000)
        # Constraint on deflection at the tip of the beam
        model.constraint(func_call[1] <= 0.10)

        objective = func_call[2]
        model.minimize(objective)
        model.close()

        # Parameterize the optimizer
        surrogate_params.evaluation_limit = evaluation_limit

        optimizer.solve()

        # Write the solution in a file with the following format:
        # - The value of the minimum found
        # - The location (H; h1; b1; b2) of the minimum
        if output_file is not None:
            with open(output_file, 'w') as f:
                f.write("Objective value: " + str(objective.value) + "\n")
                f.write("Point (H;h1;b1;b2): (" + str(H.value) + ";"
                    + str(possibleValues[h1.value]) + ";" + str(b1.value) + ";"
                    + str(b2.value) + ")")


if __name__ == '__main__':
    output_file = sys.argv[1] if len(sys.argv) > 1 else None
    evaluation_limit = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    main(evaluation_limit, output_file)
```

Surgeries scheduling

```python
import hexaly.optimizer
import sys

def read_instance(filename):
    with open(filename) as f:
        lines = f.readlines()

    first_line = lines[0].split()
    # Number of rooms
    num_rooms = int(first_line[0])
    # Number of nurses
    num_nurses = int(first_line[1])
    # Number of surgeries
    num_surgeries = int(first_line[2])

    # Minimum start of each surgery
    line_min_start = lines[1].split()
    min_start = [int(line_min_start[s]) * 60 for s in range(num_surgeries)]

    # Maximum end of each surgery
    line_max_end = lines[2].split()
    max_end = [int(line_max_end[s]) * 60 for s in range(num_surgeries)]

    # Duration of each surgery
    line_duration = lines[3].split()
    duration = [int(line_duration[s]) for s in range(num_surgeries)]

    # Number of nurses needed for each surgery
    line_nurse_needed = lines[4].split()
    needed_nurses = [int(line_nurse_needed[s]) for s in range(num_surgeries)]

    # Earliest starting shift for each nurse
    line_earliest_shift = lines[5].split()
    shift_earliest_start = [int(line_earliest_shift[s]) * 60 for s in range(num_nurses)]

    # Latest ending shift for each nurse
    line_latest_shift = lines[6].split()
    shift_latest_end = [int(line_latest_shift[s]) * 60 for s in range(num_nurses)]

    # Maximum duration of each nurse's shift
    max_shift_duration = int(lines[7].split()[0]) * 60
    
    #Incompatible rooms for each surgery
    incompatible_rooms = [[0 for r in range(num_rooms)] for s in range(num_surgeries)]
    for s in range(num_surgeries):
        line = lines[8+s].split()
        for r in range(num_rooms):
            incompatible_rooms[s][r] = int(line[r])

    return (num_rooms, num_nurses, num_surgeries, min_start, max_end, 
            needed_nurses, shift_earliest_start, shift_latest_end, 
            max_shift_duration, incompatible_rooms, duration)

def main(instance_file, output_file, time_limit):
    num_rooms, num_nurses, num_surgeries, min_start, max_end, needed_nurses, \
    shift_earliest_start, shift_latest_end, max_shift_duration, \
    incompatible_rooms, duration = read_instance(instance_file)
    
    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Sequence of surgery for each room
        surgery_order = [model.list(num_surgeries) for _ in range(num_rooms)]
        rooms = model.array(surgery_order)

        # Each surgery is scheduled in a room
        model.constraint(model.partition(rooms))

        # Only compatible rooms can be selected for a surgery
        for s in range(num_surgeries):
            for r in incompatible_rooms[s]:
                model.constraint(model.contains(surgery_order[r], s) == 0)

        # For each surgery, the selected room
        # This variable is only used to export the solution
        selected_room = [model.find(rooms, s) for s in range(num_surgeries)]

        # Interval decisions: time range of each surgery
        # Each surgery cannot start before and end after a certain time
        surgeries = [model.interval(min_start[s], max_end[s]) for s in range(num_surgeries)]

        for s in range(num_surgeries):
            # Each surgery has a specific duration
            model.constraint(model.length(surgeries[s]) == duration[s])

        surgery_array = model.array(surgeries)

        # A room can only have one surgery at a time
        for r in range(num_rooms):
            sequence = surgery_order[r]
            sequence_lambda = model.lambda_function(
                lambda s: surgery_array[sequence[s]] < surgery_array[sequence[s + 1]])
            model.constraint(
                model.and_(model.range(0, model.count(sequence) - 1), sequence_lambda)
            )

        # Each surgery needs a specific amount of nurse
        nurse_order = [model.list(num_surgeries) for _ in range(num_nurses)]

        for n in range(num_nurses):
            # Each nurse has an earliest starting shift and latest ending shift to be respected
            sequence = nurse_order[n]
            first_surgery_start = model.iif( 
                model.count(sequence) > 0, 
                model.start(surgery_array[sequence[0]]),
                shift_earliest_start[n]
            )
            last_surgery_end = model.iif(
                model.count(sequence) > 0, 
                model.end(surgery_array[sequence[model.count(sequence)-1]]),
                shift_earliest_start[n]
            )
            
            model.constraint(first_surgery_start >= shift_earliest_start[n])
            model.constraint(last_surgery_end <= shift_latest_end[n])

            # Each nurse cannot work more than a certain amount of hours
            model.constraint(last_surgery_end - first_surgery_start <= max_shift_duration)
            
            # Each nurse can only be at one operation at a time and stays all along the surgery
            sequence_lambda = model.lambda_function(
                lambda s: surgery_array[sequence[s]] < surgery_array[sequence[s + 1]])
            model.constraint(model.and_(
                model.range(0, model.count(sequence) - 1), sequence_lambda))
        
        # Each surgery needs a certain amount of nurses 
        nurse_order_array = model.array(nurse_order)
        for s in range(num_surgeries):
            model.constraint(
                model.sum(
                    model.range(0, num_nurses), 
                    model.lambda_function(lambda n : model.contains(nurse_order_array[n], s))
                )
                >= needed_nurses[s]
            )

        # Minimize the makespan: end of the last task
        makespan = model.max([model.end(surgeries[s]) for s in range(num_surgeries)])
        model.minimize(makespan)

        model.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = time_limit

        optimizer.solve()

        # Write the solution in a file with the following format:
        # - for each surgery, the selected room, the start and end dates, 
        # the nurses working on this operation
        if output_file != None:
            with open(output_file, "w") as f:
                print("Solution written in file", output_file)
                list_nurses = {}
                for n in range(num_nurses):
                    surg_nurse = nurse_order[n].value
                    for s in surg_nurse:
                        if s not in list_nurses:
                            list_nurses[s] = [n]
                        else:
                            list_nurses[s].append(n)
                for s in range(num_surgeries):
                    f.write(str(s) + "\t"
                        + "\t" + str(selected_room[s].value)
                        + "\t" + str(surgeries[s].value.start())
                        + "\t" + str(surgeries[s].value.end()) 
                        + "\t" + str(list_nurses[s]) + "\n")



if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python surgeries_scheduling.py instance_file [output_file] [time_limit]")
        sys.exit(1)

    instance_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 else None
    time_limit = int(sys.argv[3]) if len(sys.argv) >= 4 else 20
    main(instance_file, output_file, time_limit)


```

