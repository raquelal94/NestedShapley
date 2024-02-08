"""
Example of basic nested shapley. This example shows how to compute the Nested Shapley value.
The ordered tree is mannually generated.
"""
import random

import numpy as np
from src.NestedShapley import *
from itertools import combinations

# Insert number of agents, and list of agents
agents = ["A", "B", "C", "D", "E"]
N = len(agents)

# Define your characteristic function and input data
def characteristic_function(players_contributions, synergies, coalition):
    """
    Calculate the total value of a coalition of players, considering individual contributions,
    synergies, and penalties.

    Parameters:
    - players_contributions: A dictionary where keys are player identifiers and values are their contributions.
    - synergies: A dictionary where keys are tuples representing pairs or groups of players, and values are
                 the additional value these groups generate when working together.
    - coalition: A list of player identifiers representing the coalition.

    Returns:
    - The total value (worth) of the coalition, considering synergies and penalties.
    """
    # Start with the sum of individual contributions
    total_value = sum(players_contributions[player] for player in coalition if player in players_contributions)

    # Add value from synergies
    for synergy_group, value in synergies.items():
        if all(player in synergy_group for player in coalition):
            if len(synergy_group) == len(coalition):
                print(synergy_group, value)
                total_value += value
                break

    return total_value

np.random.seed(40)
players_contributions = {agent: np.random.randint(1, 20) for agent in agents}
combinations_players = [list(combinations(agents, num)) for num in range(len(agents)+1)]
combinations_players = [item for sublist in combinations_players for item in sublist]
synergies = {pair: np.random.randint(1, 10) for pair in combinations_players if len(pair)>=2}

# Create list of data to unpack later
data = [agents, players_contributions, synergies]

def calculate_nested_tree(data, Ci, N_M):
    agents, players_contributions, synergies = data
    VS = get_VS_matrix(Ci)
    A = np.dot(VS, N_M)

    # Initialise the array for storing the char.function values
    v = np.zeros(len(A))

    # Loop through the potential combinations as defined with A
    for row in range(len(A)):
        # obtain the coalition that correspond to row in A
        # Important! make sure the list agents has the correct order as they appear in the tree
        coalition_index = A[row]
        coalition = tuple(element for element, index in zip(agents, coalition_index) if index == 1)

        # calculate the characteristic function for coalition and store it in v
        v[row] = characteristic_function(players_contributions, synergies, coalition)

    # Calculate Nested Shapley

    nodal_shapley_node = get_shapley_node(Ci, A, VS, v)
    corrected_shapley_node = get_approx_shapley_node(Ci, nodal_shapley_node)
    nested_shapley_agents = associate_shapley_agent(N_M=N_M, approx_shapley_node=corrected_shapley_node)
    for i, agent in enumerate(agents):
        print(f"Agent {agent}: {nested_shapley_agents[i]}")

    return nodal_shapley_node, corrected_shapley_node, nested_shapley_agents, v, A

def print_S(A, agents):
    order_S = []
    for row in range(len(A)):
        # obtain the coalition that correspond to row in A
        # Important! make sure the list agents has the correct order as they appear in the tree
        coalition_index = A[row]
        coalition = list(element for element, index in zip(agents, coalition_index) if index == 1)
        coalition.sort()
        coalition = tuple(coalition)

        order_S.append(coalition)

    # Transform to an array
    order_S = np.array(order_S, dtype=object)

    return order_S

#%% Ordered Tree 1
# First, define the Ci matrix that indicates the parent and children nodes. Row=parent node, Column=children
# In this example we have a tree with q=3, with the second layer composed of 3 nodes (2,3,4). Only node 2 has 3 children nodes (5,6,7)
Ci_list = [[2,3,4],
           [5,6,7],
           [],
           [],
           [],
           [],
           []
           ]
Ci = get_Ci_matrix(Ci_list)

# Then assign the agents to the nodes with N_M, Row: node, Column: agents
# The first node is associated to all agents, node 2 only with A, B, C, node 3 only with D, and node 4 only with E
N_M_list = [[1,2,3,4,5],
            [1,2,3],
            [4],
            [5],
            [1],
            [2],
            [3]]
N_M = get_NM_matrix(N_M_list)

# Generate the characteristic functions that we need to generate based on the Nested Shapley value definition
# The matrix VS indicates the combination of nodes that needs to be considered (row: combination, column: node)
# The matrix A translates VS into the combination of agents based on the combination of nodes in VS (row: characteristic function, column: agent)

nodal_shapley_node, corrected_shapley_node, nested_shapley_agents, v, A = calculate_nested_tree(data, Ci, N_M)

S = print_S(A, agents)

#%% Ordered Tree 2
# Same tree as 1 but with the agents in another order. In this case, the agents are still associated to the same nodes as in example 1
agents_ = ["B", "C", "A", "E", "D"]
data_2 = [agents_, players_contributions, synergies]

Ci_list = [[2,3,4],
           [5,6,7],
           [],
           [],
           [],
           [],
           []
           ]
Ci = get_Ci_matrix(Ci_list)

N_M_list = [[1,2,3,4,5],
            [1,2,3],
            [4],
            [5],
            [1],
            [2],
            [3]]
N_M = get_NM_matrix(N_M_list)

nodal_shapley_node_2, corrected_shapley_node_2, nested_shapley_agents_2, v_2, A_2 = calculate_nested_tree(data_2, Ci, N_M)

S = print_S(A_2, agents_)

#%% Ordered tree 3
# Same tree as previous examples, but agents are not associated to the same nodes
agents_ = ["A", "D", "E", "B", "C"]
data_3 = [agents_, players_contributions, synergies]

Ci_list = [[2,3,4],
           [5,6,7],
           [],
           [],
           [],
           [],
           []
           ]
Ci = get_Ci_matrix(Ci_list)

N_M_list = [[1,2,3,4,5],
            [1,2,3],
            [4],
            [5],
            [1],
            [2],
            [3]]
N_M = get_NM_matrix(N_M_list)

nodal_shapley_node_3, corrected_shapley_node_3, nested_shapley_agents_3, v_3, A_3 = calculate_nested_tree(data_3, Ci, N_M)

S = print_S(A_3, agents_)

#%% Ordered tree 4
# Different tree, associated by contribution
agents_ = ["A", "C", "B", "D", "E"]
data_4 = [agents_, players_contributions, synergies]

Ci_list = [[2,3,4,5],
           [6,7],
           [],
           [],
           [],
           [],
           []
           ]
Ci = get_Ci_matrix(Ci_list)

N_M_list = [[1,2,3,4,5],
            [1,2],
            [3],
            [4],
            [5],
            [1],
            [2]
            ]
N_M = get_NM_matrix(N_M_list)

nodal_shapley_node_4, corrected_shapley_node_4, nested_shapley_agents_4, v_4, A_4 = calculate_nested_tree(data_4, Ci, N_M)

S = print_S(A_4, agents_)
