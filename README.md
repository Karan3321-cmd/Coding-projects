# 2D Truss Analysis — Direct Stiffness Method



This script (truss\_Deformation\_analysis.py) calculates and shows deformation on a 2D pin-jointed truss using the
**direct stiffness method**, the same approach used inside commercial
structural analysis software (SAP2000, ABAQUS, ANSYS, etc.), which i have shown as a simplified version in Python.



Given a set of nodes, members, the material, applied loads and supports, the code
calculates:

* **Nodal displacements** — how far each joint moves under load
* **Support reactions** — the forces the supports must provide for equilibrium
* **Member axial forces, stresses and strains** — whether each member is in tension, compression, or carries zero force



The code then plots the original truss alongside its deformed shape (exaggerated), colour-coded by tension/compression/zero-force.

## 

## Engineering Applications

Direct stiffness method (DSM) is the backbone of modern structural and mechanical
engineering analysis. It can be used to calculate and verify integrity of many things such as:

* A bridge truss, roof truss, or crane boom arm can carry its design loads without any member yielding or buckling
* A given structure stays within allowable deflection limits (a sagging beam or swaying tower can be "safe" by strength but can still fail a usability check)
* Support reactions are known in advance, so foundations, bolts, or connections can be sized correctly



The same matrix approach (assemble a global stiffness matrix, apply boundary conditions, solve K·U = F) scales up to beams, frames, plates, and full 3D finite element models. This script is essentially the simplest possible finite element analysis: 2-node, 2-DOF-per-node bar elements.

## 

## Notable parts and changes in the code

* **Plain NumPy arrays for geometry-** Nodes are stored as an `(n, 2)` array of coordinates and elements as an `(m, 2)` array of node-index pairs. This means `nodes\\\[i]` and `nodes\\\[j]` give us a member's endpoint coordinates directly, with no filtering or lookups. It is quick, and it reads exactly like the maths it represents. Pandas DataFrames are reserved for the results tables at the end, which is used to make the results more readable and give a nicely labelled output.
* **`np.outer` for the element stiffness matrix-** A bar element's global stiffness matrix is mathematically the outer product of its direction vector with itself, scaled by `AE/L`:



```python
  v = np.array(\\\[-c, -s, c, s])
  k = (A \\\* E / L) \\\* np.outer(v, v)
  ```



This one line is the main formula for DSM analysis - with no manually typed 4×4 matrix to cross-check, and no risk of a typo in one of sixteen hand-written variables.

* **`K\\\[np.ix\\\_(dofs, dofs)] += k`.** `np.ix\\\_` builds the index grid needed to scatter a small element matrix into the right rows/columns of the big global matrix in a single vectorised assignment, instead of registering every single entry by hand.
* **Loads and supports as dictionaries.** `loads = {2: (0.0, -10000.0)}` and `supports = {0: (True, True), 1: (False, True)}` map node index straight to the physical object being described (a force, a fixed/free direction). The degree-of-freedom bookkeeping (which row/column of the matrix to fix or load) is then generated automatically with a short comprehension, so the input stays readable as engineering data rather than becoming a list of unexplained numbers.
* **Geometry computed once and reused.** Each member's length and direction cosines are calculated once during stiffness assembly and stored in `lengths` / `dir\\\_cosines`, then reused later when recovering member forces, so the trigonometry and square roots do not need to be redone for the same member twice.
* **Tolerance-based zero-force check.** Floating-point arithmetic almost never produces an exact `0.0`, so a truly zero-force member can come out as something like `1e-11 N`. Using `np.isclose(force, 0, atol=1e-6)` classifies these variables correctly as "Zero-force" rather than mislabeling them "Tension" or "Compression" because of a rounding error.
* **Vectorised plotting.** `plt.scatter(\\\*nodes.T, ...)` plots every node marker in one call instead of looping point-by-point, and `zip(nodes\\\[i], nodes\\\[j])` unpacks a member's two endpoints directly for `plt.plot` without pulling x/y out one field at a time.

