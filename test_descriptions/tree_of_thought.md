# Tree-of-Thought Agent

The Tree-of-Thought agent solves complex reasoning problems by exploring multiple thinking paths organized as a tree. It branches at each level, evaluates competing approaches, prunes weak candidates, and expands only the most promising lines of reasoning.

The system begins when a user submits a problem along with configuration: the branching factor (candidates per node), tree depth (expansion levels), and beam width (top candidates to keep per level). These parameters are stored in a tree state store that tracks all nodes across the session.

The first step is thought generation. A thought generator agent receives the problem context and the reasoning path so far (root to current node) and produces B candidate next-thoughts, where B is the branching factor. Each candidate is a partial reasoning step advancing toward a solution. This is not a fan-out; the generator produces all B candidates in one call as a list.

Next, each candidate thought must be scored. An evaluator agent receives the problem, the reasoning path leading to the candidate, and the candidate thought itself. It assigns a score from 0 to 10 along with a brief rationale. The evaluator runs once per candidate in a sequential evaluation loop, appending each scored candidate to the current level's results.

After all candidates at a level are scored, a pruning gate checks whether the current depth has reached the configured maximum. If not, the system selects the top-K candidates (where K is the beam width) based on their scores, discards the rest, and loops back to the thought generation step, using each surviving candidate as a new parent node for the next level of expansion. This select-and-expand loop is the core iteration of the architecture.

When the maximum depth is reached, a best-leaf selector step scans all leaf nodes in the tree, identifies the one with the highest cumulative path score, and extracts the full reasoning chain from root to that leaf. A final answer formatter agent takes this best chain and the original problem and produces a clean, coherent final answer.

The input is a ProblemInput containing the problem text, branching factor, tree depth, and beam width. The output is a TreeOfThoughtResult containing the final answer, the best reasoning chain as a list of thought nodes (each with text, score, depth, and parent ID), and the total nodes explored. The tree state store retains all nodes for inspection.
