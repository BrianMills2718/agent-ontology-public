# Voyager-Style Exploration Agent

The Voyager agent is an open-ended exploration agent that discovers, learns, and accumulates skills over time. Unlike goal-directed agents, Voyager autonomously proposes its own objectives, attempts them, and stores successful solutions in a persistent skill library for future reuse.

The system starts when the user provides an environment description and an optional initial objective. A curriculum agent examines the current state of the skill library and the environment, then proposes the next exploration objective. This objective should be novel—something not already covered by existing skills—and achievable given the agent's current capabilities. The curriculum agent outputs a structured objective containing a task description, success criteria, and difficulty estimate.

Next, a skill retrieval step queries the skill library (a vector store) using the proposed objective as the search query. It returns the top-K most relevant existing skills, each containing a name, description, and implementation code. These retrieved skills provide context and reusable building blocks for the current task.

The code generation agent then receives the objective, retrieved skills, and environment description. It writes implementation code to accomplish the objective, potentially calling or adapting retrieved skills. The output is a code block with a skill name, description, and the implementation.

An execution step runs the generated code in a sandboxed environment and captures the result: success/failure status, output logs, and any error messages. This is a deterministic step with no LLM call—just code execution.

A verification gate checks whether the execution succeeded and the success criteria from the objective were met. If verification fails, the system enters a retry loop: a self-reflection agent analyzes the failure (error messages, execution logs, the code that failed) and produces a diagnosis with suggested fixes. The code generation agent then receives this feedback along with the original context and generates an improved version. This retry loop runs up to 3 times.

If the skill succeeds (either on first attempt or after retries), a skill storage step adds the new skill to the skill library with its name, description, implementation, and a success flag. The curriculum agent is then called again to propose the next objective, creating the outer exploration loop.

The exploration loop continues until a maximum number of skills have been discovered or a configured iteration limit is reached.

The input is an ExplorationConfig containing the environment description, optional initial objective, max skills to discover, max retries per skill, and top-K for skill retrieval. The output is an ExplorationResult containing the list of discovered skills, total attempts, success rate, and the final state of the skill library. The skill library store retains all skills across iterations.
