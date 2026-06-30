import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. INPUT - geometry, connectivity, material, loads, supports
# ============================================================

# Node coordinates (x, y) in metres - row index = node number - 1
nodes = np.array([
    [0.0, 0.0],   # Node 1
    [3.0, 0.0],   # Node 2
    [3.0, 4.0],   # Node 3
])

# Element connectivity (start node, end node), 0-indexed
elements = np.array([
    [0, 1],   # Element 1
    [1, 2],   # Element 2
    [0, 2],   # Element 3
])

E = 200e9     # Young's modulus, Pa
A = 0.005     # Cross-sectional area, m^2

# Applied loads: {node index: (Fx, Fy)} in Newtons
loads = {2: (0.0, -10000.0)}   # 10 kN downward at Node 3

# Supports: {node index: (fix_x, fix_y)}
supports = {
    0: (True, True),    # Node 1: pinned
    1: (False, True),   # Node 2: roller (y restrained)
}


# ============================================================
# 2. ASSEMBLE THE GLOBAL STIFFNESS MATRIX
# ============================================================

n_dof = nodes.shape[0] * 2
K = np.zeros((n_dof, n_dof))
lengths, dir_cosines = [], []

for i, j in elements:
    dx, dy = nodes[j] - nodes[i]
    L = np.hypot(dx, dy)
    c, s = dx / L, dy / L
    lengths.append(L)
    dir_cosines.append((c, s))

    # k_local in global axes = (AE/L) * [-c -s c s]^T [-c -s c s]
    v = np.array([-c, -s, c, s])
    k = (A * E / L) * np.outer(v, v)

    dofs = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
    K[np.ix_(dofs, dofs)] += k


# ============================================================
# 3. LOADS AND BOUNDARY CONDITIONS
# ============================================================

F = np.zeros(n_dof)
for node, (fx, fy) in loads.items():
    F[2 * node: 2 * node + 2] = (fx, fy)

fixed_dofs = [
    2 * n + d
    for n, (fx, fy) in supports.items()
    for d, is_fixed in enumerate((fx, fy)) if is_fixed
]
free_dofs = np.setdiff1d(np.arange(n_dof), fixed_dofs)


# ============================================================
# 4. SOLVE FOR DISPLACEMENTS AND REACTIONS
# ============================================================

U = np.zeros(n_dof)
U[free_dofs] = np.linalg.solve(K[np.ix_(free_dofs, free_dofs)], F[free_dofs])
reactions = K @ U - F


# ============================================================
# 5. MEMBER FORCES, STRESS, STRAIN
# ============================================================

member_rows = []
for e, (i, j) in enumerate(elements):
    L = lengths[e]
    c, s = dir_cosines[e]
    dofs = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]

    force = (A * E / L) * np.dot([-c, -s, c, s], U[dofs])
    stress = force / A
    strain = stress / E
    # np.isclose absorbs floating-point noise around true zero-force members
    force_type = "Zero-force" if np.isclose(force, 0, atol=1e-6) else (
        "Tension" if force > 0 else "Compression")

    member_rows.append((e + 1, i + 1, j + 1, L, force, stress, strain, force_type))

member_df = pd.DataFrame(member_rows, columns=[
    "Element", "Start Node", "End Node", "Length (m)",
    "Axial Force (N)", "Stress (Pa)", "Strain", "Force Type"
])

disp_df = pd.DataFrame({
    "Node": np.arange(1, len(nodes) + 1),
    "Ux (m)": U[0::2],
    "Uy (m)": U[1::2],
})

reaction_df = pd.DataFrame([
    (n + 1, "X" if d == 0 else "Y", reactions[2 * n + d])
    for n, (fx, fy) in supports.items()
    for d, is_fixed in enumerate((fx, fy)) if is_fixed
], columns=["Node", "Direction", "Reaction Force (N)"])


# ============================================================
# 6. PRINT RESULTS
# ============================================================

print("\n================ NODAL DISPLACEMENTS ================\n", disp_df)
print("\n================ SUPPORT REACTIONS ================\n", reaction_df)
print("\n================ MEMBER RESULTS ================\n", member_df)


# ============================================================
# 7. PLOT ORIGINAL AND DEFORMED TRUSS
# ============================================================

scale = 500   # exaggerate tiny real-world deflections so they're visible
deformed = nodes + scale * U.reshape(-1, 2)
colour_map = {"Tension": "blue", "Compression": "red", "Zero-force": "green"}

plt.figure(figsize=(9, 7))

for e, (i, j) in enumerate(elements):
    plt.plot(*zip(nodes[i], nodes[j]), "k--", linewidth=2,
              label="Original Truss" if e == 0 else "")
    plt.plot(*zip(deformed[i], deformed[j]), color=colour_map[member_df.loc[e, "Force Type"]],
              linewidth=3, label="Deformed Truss" if e == 0 else "")

    mid_x, mid_y = (nodes[i] + nodes[j]) / 2
    plt.text(mid_x, mid_y + 0.15,
              f"E{e + 1}\n{member_df.loc[e, 'Axial Force (N)']:.1f} N",
              fontsize=9, ha="center")

plt.scatter(*nodes.T, color="black", s=70, zorder=5, label="Original Nodes")
plt.scatter(*deformed.T, color="red", s=70, zorder=5, label="Deformed Nodes")

for n, (x, y) in enumerate(nodes):
    plt.text(x - 0.12, y - 0.15, f"N{n + 1}", fontsize=10, color="black")

for node, (fx, fy) in loads.items():
    x, y = nodes[node]
    plt.arrow(x, y + 0.8, 0, -0.6, head_width=0.12, head_length=0.15,
               color="green", length_includes_head=True)
    plt.text(x + 0.15, y + 0.45, f"{abs(fy) / 1000:.0f} kN Load", fontsize=10, color="green")

support_labels = {0: "Pinned Support", 1: "Roller Support"}
for n, label in support_labels.items():
    x, y = nodes[n]
    plt.text(x, y - 0.35, label, fontsize=9, ha="center")

plt.title("2D Truss Analysis: Original and Deformed Shape")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.tight_layout()
plt.savefig("truss_result.png", dpi=150)
plt.show()
