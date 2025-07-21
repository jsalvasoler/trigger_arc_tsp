After evaluating the feasibility of delta evaluation for the three local search neighborhoods (2-opt, swap, and relocate), it appears that delta evaluation is not applicable or advisable for the TA-TSP

🧪 Example: Swap-Two Neighborhood
Given the original tour: A → B → C → D → E → F → G
Swapping nodes C and F results in the new tour: A → B → F → D → E → C → G

A standard delta evaluation would involve replacing old edge costs with new ones:

Δcost = cost(B, F) + cost(F, D) + cost(E, C) + cost(C, G) - cost(B, C) - cost(C, D) - cost(E, F) - cost(F, G)

However, we need to account for the triggered edges, we can partition the edges in the new tour within a 3 regions:

1. Arcs before the modified region
Their own cost remains unchanged.
However, their successors may now be different and edges from this region might become new triggers.
2. Modified arcs (e.g., B→F, E→C)
These arcs may now be triggered by different predecessor arcs.
These arcs may now be active triggers.
3. Arcs following the modified region
Even if the arcs remain the same, their trigger source may have changed, leading to different costs.
This interdependency creates a cascade effect, where a local change affects the cost structure across the tour.

🧮 Implications for Delta Evaluation
Due to the nature of trigger dependencies:

Local moves can result in non-local cost changes.
Delta evaluation would require tracking and recomputing the effective trigger of each arc, not just a few edges.
This defeats the purpose of delta evaluation and introduces significant complexity.
🚫 Complexity of Alternatives
A partial workaround could involve:

Storing and evaluating the base cost (untriggered),
Applying standard delta evaluation on it,
Then recomputing trigger-induced costs after each move.
However, this introduces additional variable for a marginal benefits (instances are lights).

✅ Conclusion
Delta evaluation is not suitable for TA-TSP due to the global and dynamic nature of trigger dependencies. Full cost evaluation ensures correctness, is simpler to implement and maintain, and performs adequately for the targeted instance sizes.