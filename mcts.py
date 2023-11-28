import numpy as np
import tensorflow as tf

class Node(object):

    def __init__(self, prior):
        """
        Node in MCTS
        prior: The prior on the node, computed from policy network
        """
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_representation = None
        self.reward = 0
        self.expanded = False

    def value(self):
        """
        Compute value of a node
        """
        if self.visit_count == 0:
            return 0
        else:
            return self.value_sum / self.visit_count


def run_mcts(config, root, network, min_max_stats):
    """
    Main loop for MCTS for config.num_simulations simulations

    root: the root node
    network: the network
    min_max_stats: the min max stats object for the simulation

    Hint:
    The MCTS should capture selection, expansion and backpropagation
    """
    for i in range(config.num_simulations):
        history = []
        node = root
        search_path = [node]

        while node.expanded:
            action, node = select_child(config, node, min_max_stats)
            history.append(action)
            search_path.append(node)
        parent = search_path[-2]
        action = history[-1]
        value = expand_node(node, list(
            range(config.action_space_size)), network, parent.hidden_representation, action)
        backpropagate(search_path, value,
                      config.discount, min_max_stats)


def select_action(config, num_moves, node, network, test=False):
    """
    Select an action to take

    If in train mode: action selection should be performed stochastically
    with temperature t
    If in test mode: action selection should be performed with argmax
    """
    visit_counts = [
        (child.visit_count, action) for action, child in node.children.items()
    ]
    if not test:
        t = config.visit_softmax_temperature_fn(num_moves=num_moves)
        action = softmax_sample(visit_counts, t)
    else:
        action = softmax_sample(visit_counts, 0)
    return action


def select_child(config, node, min_max_stats):
    """
    TODO: Implement this function
    Select a child in the MCTS
    This should be done using the UCB score, which uses the
    normalized Q values from the min max stats
    """

    best_action = 0
    best_child = node.children[best_action]
    best_ucb_score = ucb_score(config, node, best_child, min_max_stats)
    
    for curr_action in node.children:
        curr_child = node.children[curr_action]
        curr_ucb_score = ucb_score(config, node, curr_child, min_max_stats)

        if curr_ucb_score > best_ucb_score:
            best_action = curr_action
            best_child = curr_child
            best_ucb_score = curr_ucb_score


    return best_action, best_child


def ucb_score(config, parent, child, min_max_stats):
    """
    Compute UCB Score of a child given the parent statistics
    """
    pb_c = np.log((parent.visit_count + config.pb_c_base + 1)
                  / config.pb_c_base) + config.pb_c_init
    pb_c *= np.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c*child.prior
    if child.visit_count > 0:
        value_score = min_max_stats.normalize(
            child.reward + config.discount*child.value())
    else:
        value_score = 0
    return prior_score + value_score


def expand_root(node, actions, network, current_state):
    """
    TODO: Implement this function
    Expand the root node given the current state

    This should perform initial inference, and calculate a softmax policy over children
    You should set the attributes hidden representation, the reward, the policy and children of the node
    Also, set node.expanded to be true
    For setting the nodes children, you should use node.children and  instantiate
    with the prior from the policy

    Return: the value of the root
    """
    # get hidden state representation
    value, reward, policy_logits, hidden_representation = network.initial_inference(current_state)
    node.hidden_representation = hidden_representation
    node.reward = reward

    # Extract softmax policy and set node.policy
    node.policy = tf.nn.softmax(policy_logits[0])

    # instantiate node's children with prior values, obtained from the predicted policy
    for action in actions:
        child = Node(node.policy[action])
        node.children[action] = child

    # set node as expanded
    node.expanded = True

    return value


def expand_node(node, actions, network, parent_state, parent_action):
    """
    TODO: Implement this function
    Expand a node given the parent state and action
    This should perform recurrent_inference, and store the appropriate values
    The function should look almost identical to expand_root

    Return: value
    """
    # get hidden state representation
    value, reward, policy_logits, hidden_representation = network.recurrent_inference(parent_state, parent_action)
    node.hidden_representation = hidden_representation
    node.reward = reward

    # Extract softmax policy and set node.policy
    node.policy = tf.nn.softmax(policy_logits[0])

    # instantiate node's children with prior values, obtained from the predicted policy
    for action in actions:
        child = Node(node.policy[action])
        node.children[action] = child

    # set node as expanded
    node.expanded = True

    return value


def backpropagate(path, value, discount, min_max_stats):
    """
    Backpropagate the value up the path

    This should update a nodes value_sum, and its visit count

    Update the value with discount and reward of node
    """

    discounted_reward = 0
    for node in reversed(path):
        # YOUR CODE HERE
        node.visit_count += 1

        discounted_reward *= discount
        discounted_reward += node.reward

        node.value_sum = discounted_reward

        min_max_stats.update(node.value())


def add_exploration_noise(config, node):
    """
    Add exploration noise by adding dirichlet noise to the prior over children
    This is governed by root_dirichlet_alpha and root_exploration_fraction
    """
    actions = list(node.children.keys())
    noise = np.random.dirichlet([config.root_dirichlet_alpha]*len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1-frac) + n*frac


def visit_softmax_temperature(num_moves):
    """
    This function regulates exploration vs exploitation when selecting actions
    during self-play.
    Given the current number of games played by the learning algorithm, return the
    temperature value to be used by MCTS.

    You are welcome to devise a more complicated temperature scheme
    """
    return 1


def softmax_sample(visit_counts, temperature):
    """
    Sample an actions

    Input: visit_counts as list of [(visit_count, action)] for each child
    If temperature == 0, choose argmax
    Else: Compute distribution over visit_counts and sample action as in writeup
    """

    # YOUR CODE HERE
    if temperature == 0:
        visit_counts.sort(key=lambda x: x[0], reverse=True)
        best_tuple = visit_counts[0]
        best_action = best_tuple[1]

        return best_action
    else:
        visit_counts_num = [visit_tuple[0] for visit_tuple in visit_counts]
        visit_counts_unnorm = np.power(visit_counts_num, 1/temperature)
        visit_counts_norm = visit_counts_unnorm / np.sum(visit_counts_unnorm)

        sampled_index = int(tf.random.categorical(np.expand_dims(visit_counts_norm, axis=0), 1))
        sampled_tuple = visit_counts[sampled_index]
        sampled_action = sampled_tuple[1]

        return sampled_action