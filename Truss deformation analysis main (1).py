# 2D TRUSS ANALYSIS USING DIRECT STIFFNESS METHOD
# Uses: NumPy, Pandas, Matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# STEP 1: DEFINE THE TRUSS DATA

# Node table: node number, x-position, y-position
node_df = pd.DataFrame({
    "Node": [1, 2, 3],
    "X": [0.0, 3.0, 3.0],
    "Y": [0.0, 0.0, 4.0]
})

# Element table: element number, start node, end node
element_df = pd.DataFrame({
    "Element": [1, 2, 3],
    "Start": [1, 2, 1],
    "End": [2, 3, 3]
})

# Material and section properties
E = 200e9        # Young's Modulus in Pa
A = 0.005        # Cross-sectional area in m^2


# STEP 2: SETUP

number_of_nodes = len(node_df)
total_dof = number_of_nodes * 2

# Each node has 2 DOFs:
# Node 1 = DOF 0,1
# Node 2 = DOF 2,3
# Node 3 = DOF 4,5

K_global = np.zeros((total_dof, total_dof))
F = np.zeros(total_dof)


# STEP 3: APPLY LOADS

# Downward load of 10,000 N at Node 3
F[2 * (3 - 1) + 1] = -10000


# STEP 4: DEFINE SUPPORT CONDITIONS

# Node 1 pinned: x and y fixed
# Node 2 roller: y fixed
fixed_dofs = [0, 1, 3]


# STEP 5: ASSEMBLE GLOBAL STIFFNESS MATRIX

for _, element in element_df.iterrows():

    element_number = int(element["Element"])
    start_node = int(element["Start"])
    end_node = int(element["End"])

    x1 = node_df.loc[node_df["Node"] == start_node, "X"].values[0]
    y1 = node_df.loc[node_df["Node"] == start_node, "Y"].values[0]
    x2 = node_df.loc[node_df["Node"] == end_node, "X"].values[0]
    y2 = node_df.loc[node_df["Node"] == end_node, "Y"].values[0]

    # Member length
    L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # Direction cosines
    c = (x2 - x1) / L
    s = (y2 - y1) / L

    # Element stiffness matrix in global coordinates
    k = (A * E / L) * np.array([
        [ c*c,  c*s, -c*c, -c*s],
        [ c*s,  s*s, -c*s, -s*s],
        [-c*c, -c*s,  c*c,  c*s],
        [-c*s, -s*s,  c*s,  s*s]
    ])

    # Global DOF numbers for this element
    dofs = [
        2 * (start_node - 1),
        2 * (start_node - 1) + 1,
        2 * (end_node - 1),
        2 * (end_node - 1) + 1
    ]

    # Add element stiffness into global stiffness matrix
    for i in range(4):
        for j in range(4):
            K_global[dofs[i], dofs[j]] += k[i, j]


# STEP 6: APPLY BOUNDARY CONDITIONS

all_dofs = np.arange(total_dof)
free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
F_reduced = F[free_dofs]


# STEP 7: SOLVING DISPLACEMENTS

U = np.zeros(total_dof)

U_free = np.linalg.solve(K_reduced, F_reduced)

U[free_dofs] = U_free


# STEP 8: CALCULATING REACTIONS

reactions = np.dot(K_global, U) - F


# STEP 9: CALCULATE MEMBER FORCES, STRESS AND STRAIN

member_results = []

for _, element in element_df.iterrows():

    element_number = int(element["Element"])
    start_node = int(element["Start"])
    end_node = int(element["End"])

    x1 = node_df.loc[node_df["Node"] == start_node, "X"].values[0]
    y1 = node_df.loc[node_df["Node"] == start_node, "Y"].values[0]
    x2 = node_df.loc[node_df["Node"] == end_node, "X"].values[0]
    y2 = node_df.loc[node_df["Node"] == end_node, "Y"].values[0]

    L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    c = (x2 - x1) / L
    s = (y2 - y1) / L

    dofs = [
        2 * (start_node - 1),
        2 * (start_node - 1) + 1,
        2 * (end_node - 1),
        2 * (end_node - 1) + 1
    ]

    element_displacements = U[dofs]

    axial_force = (A * E / L) * np.dot(
        np.array([-c, -s, c, s]),
        element_displacements
    )

    stress = axial_force / A
    strain = stress / E

    if axial_force > 0:
        force_type = "Tension"
    elif axial_force < 0:
        force_type = "Compression"
    else:
        force_type = "Zero-force"

    member_results.append([
        element_number,
        start_node,
        end_node,
        L,
        axial_force,
        stress,
        strain,
        force_type
    ])


member_result_df = pd.DataFrame(
    member_results,
    columns=[
        "Element",
        "Start Node",
        "End Node",
        "Length (m)",
        "Axial Force (N)",
        "Stress (Pa)",
        "Strain",
        "Force Type"
    ]
)


# STEP 10: CREATE DISPLACEMENT RESULT TABLE

displacement_results = []

for _, node in node_df.iterrows():

    node_number = int(node["Node"])

    ux = U[2 * (node_number - 1)]
    uy = U[2 * (node_number - 1) + 1]

    displacement_results.append([
        node_number,
        ux,
        uy
    ])

displacement_df = pd.DataFrame(
    displacement_results,
    columns=["Node", "Ux (m)", "Uy (m)"]
)


# STEP 11: CREATE REACTION RESULT TABLE

reaction_results = []

for dof in fixed_dofs:

    node_number = (dof // 2) + 1

    if dof % 2 == 0:
        direction = "X"
    else:
        direction = "Y"

    reaction_results.append([
        node_number,
        direction,
        reactions[dof]
    ])

reaction_df = pd.DataFrame(
    reaction_results,
    columns=["Node", "Direction", "Reaction Force (N)"]
)


# STEP 12: PRINT RESULTS

print("\n================ GLOBAL STIFFNESS MATRIX ================\n")
print(pd.DataFrame(K_global))

print("\n================ NODAL DISPLACEMENTS ================\n")
print(displacement_df)

print("\n================ SUPPORT REACTIONS ================\n")
print(reaction_df)

print("\n================ MEMBER RESULTS ================\n")
print(member_result_df)


# STEP 13: PLOT ORIGINAL AND DEFORMED TRUSS

# The actual deformation is tiny, so this scale makes it visible
scale = 500

node_df["Ux"] = U[0::2]
node_df["Uy"] = U[1::2]

node_df["X_def"] = node_df["X"] + scale * node_df["Ux"]
node_df["Y_def"] = node_df["Y"] + scale * node_df["Uy"]


plt.figure(figsize=(9, 7))


# Plotting original truss

for _, element in element_df.iterrows():

    element_number = int(element["Element"])
    start_node = int(element["Start"])
    end_node = int(element["End"])

    x_original = [
        node_df.loc[node_df["Node"] == start_node, "X"].values[0],
        node_df.loc[node_df["Node"] == end_node, "X"].values[0]
    ]

    y_original = [
        node_df.loc[node_df["Node"] == start_node, "Y"].values[0],
        node_df.loc[node_df["Node"] == end_node, "Y"].values[0]
    ]

    plt.plot(
        x_original,
        y_original,
        "k--",
        linewidth=2,
        label="Original Truss" if element_number == 1 else ""
    )


# Plotting deformed truss

for _, element in element_df.iterrows():

    element_number = int(element["Element"])
    start_node = int(element["Start"])
    end_node = int(element["End"])

    x_deformed = [
        node_df.loc[node_df["Node"] == start_node, "X_def"].values[0],
        node_df.loc[node_df["Node"] == end_node, "X_def"].values[0]
    ]

    y_deformed = [
        node_df.loc[node_df["Node"] == start_node, "Y_def"].values[0],
        node_df.loc[node_df["Node"] == end_node, "Y_def"].values[0]
    ]

    force_type = member_result_df.loc[
        member_result_df["Element"] == element_number,
        "Force Type"
    ].values[0]

    # Red = compression, blue = tension
    if force_type == "Tension":
        colour = "blue"
    elif force_type == "Compression":
        colour = "red"
    else:
        colour = "green"

    plt.plot(
        x_deformed,
        y_deformed,
        color=colour,
        linewidth=3,
        label="Deformed Truss" if element_number == 1 else ""
    )


# Plot nodes

plt.scatter(
    node_df["X"],
    node_df["Y"],
    color="black",
    s=70,
    zorder=5,
    label="Original Nodes"
)

plt.scatter(
    node_df["X_def"],
    node_df["Y_def"],
    color="red",
    s=70,
    zorder=5,
    label="Deformed Nodes"
)


# Label nodes

for _, node in node_df.iterrows():

    plt.text(
        node["X"] - 0.12,
        node["Y"] - 0.15,
        f"N{int(node['Node'])}",
        fontsize=10,
        color="black"
    )


# Label elements and axial forces

for _, element in element_df.iterrows():

    element_number = int(element["Element"])
    start_node = int(element["Start"])
    end_node = int(element["End"])

    x1 = node_df.loc[node_df["Node"] == start_node, "X"].values[0]
    y1 = node_df.loc[node_df["Node"] == start_node, "Y"].values[0]
    x2 = node_df.loc[node_df["Node"] == end_node, "X"].values[0]
    y2 = node_df.loc[node_df["Node"] == end_node, "Y"].values[0]

    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

    force = member_result_df.loc[
        member_result_df["Element"] == element_number,
        "Axial Force (N)"
    ].values[0]

    plt.text(
        mid_x,
        mid_y + 0.15,
        f"E{element_number}\n{force:.1f} N",
        fontsize=9,
        ha="center"
    )


# Plot applied load arrow at Node 3

node_3_x = node_df.loc[node_df["Node"] == 3, "X"].values[0]
node_3_y = node_df.loc[node_df["Node"] == 3, "Y"].values[0]

plt.arrow(
    node_3_x,
    node_3_y + 0.8,
    0,
    -0.6,
    head_width=0.12,
    head_length=0.15,
    color="green",
    length_includes_head=True
)

plt.text(
    node_3_x + 0.15,
    node_3_y + 0.45,
    "10 kN Load",
    fontsize=10,
    color="green"
)


# Add simple support labels

plt.text(0.0, -0.35, "Pinned Support", fontsize=9, ha="center")
plt.text(3.0, -0.35, "Roller Support", fontsize=9, ha="center")


# Final graph settings-

plt.title("2D Truss Analysis: Original and Deformed Shape")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.show()