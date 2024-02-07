import numpy as np
from itertools import combinations
from src.ShapleyValue import shapley

# Matrix
def create_Ci_list(layers, special_layer=None, special_children=None):
    # Initialize variables
    current_index = 1  # Keeps track of the last node index assigned
    tree_structure = []
    parent_nodes = [1]  # Start with the root node

    # Generate the tree
    for layer_index, children_count in enumerate(layers):
        new_parent_nodes = []  # Prepare to track the next layer of parent nodes
        for parent_index, parent in enumerate(parent_nodes):
            # Check if the current layer and node are special
            if special_layer is not None and layer_index == special_layer:
                if special_children and parent_index < len(special_children):
                    # Use the special number of children for this node
                    children_count = special_children[parent_index]
                else:
                    # Use the default number of children for the rest of the nodes in this layer
                    children_count = layers[layer_index]

            # The children of the current node start at current_index + 1
            children_start = current_index + 1
            children_end = children_start + children_count

            # Add the children range to the tree structure if children_count is not zero
            if children_count > 0:
                tree_structure.append(list(range(children_start, children_end)))
                # Update the current_index to the last child added
                current_index = children_end - 1
                # Add the new children to the list of parent nodes for the next layer
                new_parent_nodes.extend(range(children_start, children_end))
            else:
                # For nodes with no children, add an empty list
                tree_structure.append([])

        # Update parent nodes for the next layer
        parent_nodes = new_parent_nodes

    return tree_structure
def get_Ci_matrix(Ci_list):
    Ci = np.zeros((len(Ci_list), len(Ci_list)))
    for i in range(len(Ci_list)): #row
        for node in Ci_list[i]: # child node of node i
            Ci[i][node-1] = 1
    return Ci

def create_N_M_list(Ci_list, Ci):
    # Initialise a list to hold the agents for each node
    agents_per_node = [[] for _ in range(len(Ci_list))]

    # Identify agent nodes (rows with all zeros)
    agent_indices = np.where(~Ci.any(axis=1))[0]
    agents = {value:agent+1 for agent,value in enumerate(agent_indices)}  # Agent numbering starts from 1

    # Map each agent to its parent nodes directly
    for agent_node in agent_indices:
    # agent_node = agent_indices[0]
        for i in range(len(Ci)):
            if Ci[i, agent_node]== 1:
                agents_per_node[i].append(agents[agent_node])
                break

    # Fill out the agents_per_node of the last nodes with their corresponding agent
    for row, agent in agents.items():
        agents_per_node[row].append(agent)

    # Get indexes in agets_per_node that are still empty
    empty_list_indices = [index for index, agents in enumerate(agents_per_node) if not agents]
    for row in reversed(empty_list_indices):
        children = np.where(Ci[row]==1)[0]
        nodes_to_visit = list(children)
        while nodes_to_visit:
            current_node = nodes_to_visit.pop() + 1
            agents_per_node_current = agents_per_node[current_node]
            if len(agents_per_node_current)==0:
                nodes_to_visit.extend(np.where(Ci[current_node] == 1)[0])
            else:
                agents_per_node[row].extend(agents_per_node_current)
        # eliminate_duplicates
        agents_per_node[row] = list(set(agents_per_node[row]))
        # order the agents_per_node for that particular node
        agents_per_node[row] = sorted(agents_per_node[row])

    # get the nodes for node 1
    children_first_node = np.where(Ci[0]==1)[0]
    agents_per_node[0] = []
    for children in children_first_node:
        agents_per_node[0].extend(agents_per_node[children])

    return agents_per_node
def get_NM_matrix(N_M_list):
    cls = max(len(sublist) for sublist in N_M_list)
    N_M = np.zeros((len(N_M_list), cls))
    for i in range(len(N_M)): # rows - nodes
        for agent in N_M_list[i]: # agent in node i
            N_M[i][agent-1] = 1
    return N_M

# Value Functions
def get_VS_nodes(Ci):
    """
    This function creates the matrix VS who contains all the v(S) for all S contined in C_i where the columns indicate the nodes who need to be combined
    :param Ci:
    :return:
    """
    comb=[]
    for i in range(len(Ci)): # node starting from 2
        arr = Ci[i]
        indices = np.where(arr == 1)[0]
        for i in range(1,len(indices)+1):
            for indices_combination in combinations(indices, i):
                new_arr = np.zeros((len(arr)))
                for col in indices_combination:
                    new_arr[col] = 1
                comb.append(new_arr)

    # new_arr = np.zeros((len(arr))) # create one array for V(N)
    # new_arr[0] = 1
    # comb.insert(0,new_arr)

    VS_node = np.array(comb)
    return VS_node

def get_VS_agents(VS_node,N_M):
    """
    obtain the combination of agents whose value function needs to be calculated
    :param VS_node:
    :param N_M:
    :return: matrix where the rows indicate the combination of nodes in VS_Node, and the columns the agents
    """
    VS_agents = np.dot(VS_node,N_M)
    return VS_agents

def get_VS_matrix(Ci):
    """
    This function creates the matrix VS who contains all the v(S) for all S contined in C_i where the columns indicate the nodes who need to be combined
    :param Ci:
    :return:
    """
    comb=[]
    for i in range(len(Ci)): # node starting from 2
        arr = Ci[i]
        indices = np.where(arr == 1)[0]
        for i in range(1,len(indices)+1):
            for indices_combination in combinations(indices, i):
                new_arr = np.zeros((len(arr)))
                for col in indices_combination:
                    new_arr[col] = 1
                comb.append(new_arr)

    # new_arr = np.zeros((len(arr))) # create one array for V(N)
    # new_arr[0] = 1
    # comb.insert(0,new_arr)

    return np.array(comb)


# Shapley

def get_shapley_node(Ci, VS_agents, VS_node, v):
    """
    Function calculating the shapley values of each node considering their parent nodes

    :param Ci: Matrix containing the child nodes for each node
    :param VS_agents: Matrix containing the combination of agents considering VS
    :param v: vector with the characteristic functions according to the order set by A
    :return: vector with the approximated shapley values for the nodes
    """
    # matrix to contain the shapley* values of the different nodes
    shapley_node = np.zeros(len(Ci))

    # set shapley_ast_matrix of node 1 as the total v(N)
    indices = np.where(np.all(VS_agents == 1, axis=1))[0]
    shapley_node[0] = v[indices]

    # filter out those nodes that are parent nodes
    Ci_ = Ci[np.any(Ci, axis=1)]

    # for each of the combinations, obtain the corresponding vector from v and calculate the shapley values. Save the results in shapley_ast_matrix
    for arr in Ci_:
        #arr = Ci[0]
        nodes = np.where(arr == 1)[0]
        #VS_Ci = VS[np.any(VS[:, list(nodes)] == 1, axis=1),:]
        bool_list = np.any(VS_node[:, list(nodes)] == 1, axis=1)
        indices = np.where(bool_list)[0]
        v_ = v[indices] # vector containing the v(2), v(3), v(2,3)
        shp_values = shapley(len(nodes), v_) # call the function of shapleys with the considered combinations from v_
        for i,n in enumerate(nodes):
            shapley_node[n] = shp_values[i]

    return shapley_node

def get_approx_shapley_node(Ci, shapley_node):
    """
    Function calculating the approximated shapley values of the different nodes

    :param Ci: matrix containing the child nodes of each node
    :param shapley_node: vector containing the shapley values of the different nodes
    :return: vector with the approximated shapley values of each node
    """

    # initiate the matrix to keep the approximated values
    approx_shapley_node = np.zeros(len(Ci))

    # set the value for the initial node
    approx_shapley_node[0] = shapley_node[0]

    for i in range(1, len(Ci)): # loop through columns
        j = np.where(Ci[:, i] == 1)[0] # get parent node of node i (col) which will be the index of the row

        # make the sum of all shapley values of the child nodes of j
        child_nodes = np.where(Ci[j] == 1)[1] # get child nodes
        sum_child_nodes = np.sum(shapley_node[child_nodes])
        # print(f"parent node = {j}, shapley_j = {shapley_node[j]}, sum_child_nodes = {sum_child_nodes}")
        # print(f"shapley child nodes of j = {shapley_node[child_nodes]}")
        # compute the approximated value for the node i
        if sum_child_nodes==0:
            phi_ast = np.abs(shapley_node[i]) / np.abs(shapley_node[child_nodes]).sum() * shapley_node[j]
        else:
            phi_ast = shapley_node[i]/sum_child_nodes * shapley_node[j]

        approx_shapley_node[i] = phi_ast

    return approx_shapley_node

def associate_shapley_agent(N_M, approx_shapley_node):
    """
    Function to associate the shapley values of the nodes with their respective agent
    :param N_M:
    :param approx_shapley_node:
    :return: vector with size of the agents
    """
    # get the index of the rows in NM which contain one single one
    row_sums = np.sum(N_M, axis=1)
    indices_nodes = np.where(row_sums == 1)[0] # Find the indices where the sum is just one

    # indices_nodes contains the nodes with one agent. For each the nodes, retrieve its approx_shapley and associate it to the agent contained within the node
    approx_shapley_agents = np.zeros(len(indices_nodes))
    for node in indices_nodes:
        # retrieve the agent corresponding to node
        agent = np.where(N_M[node] == 1)[0]

        # associate the approx_shapley value of the node with the agent
        approx_shapley_agents[agent] = approx_shapley_node[node]

    return approx_shapley_agents