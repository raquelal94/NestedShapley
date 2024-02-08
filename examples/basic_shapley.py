import numpy as np
from src.ShapleyValue import shapley, CoalitionalStructure
from itertools import combinations

# Insert number of agents, and list of agents
agents = ["A", "B", "C", "D", "E"]
N = len(agents)

# Generate matrix A (row: characteristic function, column: agent)
A = CoalitionalStructure(number_Players=N)

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
                total_value += value
                break

    return total_value

np.random.seed(40)
players_contributions = {agent: np.random.randint(1, 20) for agent in agents}
combinations_players = [list(combinations(agents, num)) for num in range(len(agents)+1)]
combinations_players = [item for sublist in combinations_players for item in sublist]
synergies = {pair: np.random.randint(1, 10) for pair in combinations_players if len(pair)>=2}

# Initialise the array for storing the char.function values
v = np.zeros(len(A))

# Loop through the potential combinations as defined with A
for row in range(len(A)):
    # obtain the coalition that correspond to row in A
    coalition_index = A[row]
    coalition = tuple(element for element, index in zip(agents, coalition_index) if index == 1)

    # calculate the characteristic function for coalition and store it in v
    v[row] = characteristic_function(players_contributions, synergies, coalition)

# Calculate shapley
shapley_agents = shapley(n=N, v_list=v)
for i, agent in enumerate(agents):
    print(f"Agent {agent}: {shapley_agents[i]}")