# LATS (Language Agent Tree Search)

LATS combines Monte Carlo Tree Search (MCTS) with LLM-based reasoning to solve complex problems through iterative exploration, evaluation, and backpropagation of value estimates.

The system starts when the user provides a problem statement and search configuration: number of iterations, expansion width (children per node), and exploration constant for UCB1. A root node is created representing the initial problem state with no actions taken.

Each MCTS iteration begins with a selection phase. Starting from the root, the system traverses the tree by selecting the child with the highest UCB1 score at each level: UCB1 = value/visits + C * sqrt(ln(parent_visits)/visits). For unvisited children (visits=0), selection prioritizes them first. This traversal continues until reaching a leaf node (a node with no children or an unvisited child). The selection step is purely computational—no LLM call, just UCB1 math over the tree structure stored in a tree state store.

At the selected leaf, the expansion phase generates new child nodes. An action generator agent receives the problem, the sequence of actions from root to the current node (the trajectory), and the current state. It produces W candidate next-actions (where W is the expansion width), each with a description of the action and the resulting state after taking it. Each candidate becomes a new child node in the tree.

Next, the evaluation phase scores each new child. A state evaluator agent receives the problem, the full trajectory including the new action, and the resulting state. It assigns a value estimate from 0 to 1 indicating how promising this state is for eventually solving the problem, along with a brief rationale. The evaluator also determines if the state represents a terminal solution (problem solved) or a dead end (no viable path forward).

The backpropagation phase then updates value estimates up the tree. Starting from each evaluated child, the system walks back to the root, incrementing the visit count at each ancestor and updating the running average value. This is a deterministic step using the tree store—no LLM call.

A termination gate checks three conditions: (1) a terminal solution was found, (2) the iteration limit was reached, or (3) all leaf nodes are dead ends. If none apply, the loop continues with another selection-expansion-evaluation-backpropagation cycle.

When search terminates, a solution extraction step finds the highest-value terminal node (or the highest-value leaf if no solution was found) and extracts the full action sequence from root to that node. A solution formatter agent takes this trajectory and the original problem and produces a clean final answer.

The input is a LATSConfig containing the problem text, number of iterations, expansion width, and UCB1 exploration constant. The output is a LATSResult containing the final answer, the best trajectory as a list of action-state pairs, total nodes explored, total iterations completed, and whether a terminal solution was found. The tree store retains the full search tree for inspection.
