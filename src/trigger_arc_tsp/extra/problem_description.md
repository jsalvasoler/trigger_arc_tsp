Trigger-Arc TSP

Consider a directed graph $G = (N, A)$ with weights on the arcs. The node $0 \in N$ is designated as the starting node (depot). Let $c(a) \in \mathbb{R}^+ \ \forall a \in A$ be defined as the cost of traversing the arc $a$. For each arc $a = (h, k)$, a set of relations $R_a = \{(a_1, a) \mid a_1 \in A \}$ is associated. Finally, let $c(r) \in \mathbb{R}^+ \ \forall r \in R_a$, be the traversal cost of the arc $a$ if the relation $r$ is active. Let $T = (a_1, a_2, a_3, \ldots, a_{|N|})$ be the ordered sequence of arcs starting at node 0 representing a Hamiltonian cycle in $G$. The relation $r = (a_i, a_j)$ is active if and only if the arcs $a_i, a_j \in T$, and the arc $a_i$ precedes the arc $a_j$ in $T$ and there is no active relation $r_1 = (a_i, a_j) \in R_{a_j}$ such that $a_i$ precedes $a_i$ in $T$. It follows that for each arc $a$, at most one relation can be active. As a result, the traversal cost of the arc $a = (h, k)$ will be equal to $c(a)$ if there are no active relations in $R_a$ or $c(r)$ if $r$ is the only active relation in $R_a$.

Finally, the objective is to find the Hamiltonian cycle $T$ that minimizes the total traversal cost of the arcs in $T$