## A MILP Model for the Trigger-Arc TSP

### Problem Parameters:

- Let $G = (V, E)$ be a directed graph with:
  - $V = \{0, 1, \dots, N-1\}$ representing the set of nodes (vertices).
  - $E \subseteq V \times V$ representing the set of edges (arcs) with base costs $c_{ij}$ for $(i,j) \in E$.
  - $\delta_{\text{in}}(i) = \{(j,i) \in E\}$ and $\delta_{\text{out}}(i) = \{(i,j) \in E\}$ representing the set of incoming and outgoing edges of node $i$.
- $R_a \subseteq E \times \{a\}, a\in E$ is composed of pairs $R_a = \{(b,a) : b \in E\}$. We call $b$ the trigger of the target arc $a$. If a relation is active, the cost of traversing the target arc $a$ is $c_a + r_{ba}$, the latter being the cost associated with the relation $(b,a)$.

### Decision Variables:

- $x_{ij} \in \{0, 1\}$: binary variable, equal to 1 if edge $(i,j)$ is part of the tour, 0 otherwise.
- $u_i \in [0, N-1]$: continuous variable representing the position of node $i$ in the tour. Note: in the model, we abuse the notation to write $u_a$ for $a \in E$. This corresponds to $u_{i}$ where $a=(i,j)$, i.e., the position of the first node of the arc $a$ in the tour.
- $y_{ba} \in \{0, 1\}$: binary variable, equal to 1 if the relation $(b,a)$ is active, 0 otherwise.
- $z_{a_1a_2} \in \{0, 1\}$: binary variable, used to model precedence constraints between arcs $a_1$ and $a_2$. 

### Objective Function:
We aim to minimize the total cost of the tour, considering the base costs of the arcs and the costs associated with the active relations.

$$
\text{Minimize } \sum_{(i,j) \in E} x_{ij} c_{ij} + \sum_{a \in E} \sum_{(b,a) \in R_a} y_{ba} r_{ba}
$$

### Constraints:

#### 1. Flow Conservation:
Only one incoming and one outgoing edge is allowed for each node.

$$
\sum_{j \in \delta_{\text{out}}(i)} x_{ij} = 1 \quad \forall i \in V
$$
$$
\sum_{j \in \delta_{\text{in}}(i)} x_{ji} = 1 \quad \forall i \in V
$$

#### 2. Subtour Elimination:
Miller-Tucker-Zemlin constraints to avoid subtours.

$$
u_i - u_j + N \cdot x_{ij} \leq N - 1 \quad \forall (i,j) \in E, \, j \neq 0
$$

#### 3. At most one relation is active:
Only one relation can be active for each target arc, and no relation can be active if the target arc is not part of the tour.

$$
\sum_{(b,a) \in R_a} y_{ba} \leq x_a \quad \forall a \in E
$$

#### 4. Strengthen relation deactivation constraint:
Strengthen the previous constraint to ensure that if the target or the trigger is not part of the tour, the relation cannot be active.

$$
y_{ba} \leq x_a \quad \forall a \in E, \, b \in R_a
$$
$$
y_{ba} \leq x_b \quad \forall a \in E, \, b \in R_a
$$

#### 5. Relation is inactive if $b$ follows $a$ in the tour:
A relation $(b,a)$ cannot be active if $b$ follows $a$ in the tour.

$$
u_b + 1 \leq u_a + N \cdot (1 - y_{ba}) \quad \forall a \in E, \, b \in R_a
$$

#### 6. At least one relation is active:
For a given relation $(b,a)$, if $a$ and $b$ are part of the tour ($x_a = x_b = 1$), and $b$ precedes $a$ in the tour ($z_{ab} = 0$), then at least one relation targetting $a$ must be active.

$$
1 - z_{ab} \leq \sum_{(c,a) \in R_a} y_{ca} + (1 - x_a) + (1 - x_b) \quad \forall a \in E, \, b \in R_a
$$

#### 7. Precedence constraints on $z$ variables:
If $a_2$ precedes $a_1$ in the tour, then $z_{a_1 a_2} = 0$.
$$
u_{a_1} \leq u_{a_2} + (N-1) \cdot (1 - z_{a_1 a_2}) \quad \forall (a_1, a_2) \in E \times E
$$

#### 8. Relation precedence constraints:
Given two relations $(b,a)$ and $(c,a)$, if $b$ precedes $c$ which precedes $a$ in the tour ($z_{cb}=z_{ac}=0$, $x_a=x_b=x_c$), then the relation $(b,a)$ cannot be active if $(c,a)$ is active (recall that only one relation can be active for each target arc).

$$
y_{ba} \leq y_{ca} + z_{cb} + z_{ac} + (1 - x_c) + (1 - x_b) + (1 - x_a) \quad \forall a \in E, \, b \in R_a, \, c \in R_a, \, b \neq c
$$

#### 9. Set the starting node:
Set the position of the starting node in the tour to be the first node.
$$
u_0 = 0
$$

