PLANER_PROMPT = """You are a task decomposer, and you need
to break down the user's problem into multiple simple subtasks.
Please split out multiple subtask items so that sufficient
information can be obtained to solve the problem.
The return format is as follows:
```
Plan: the problem to be solved by the current subtask
#E[id] = tool_name[tool_inputs]
Plan: the problem to be solved by the current subtask
#E[id] = tool_name[tool_inputs]
```
1. #E[id] is used to store the execution result of the plan
id and can be used as a placeholder.
2. The content implemented by each #E[id] should strictly
correspond to the problem currently planned to be solved.
3. Tool parameters can be entered as normal text, or
#E[dependency_id], or both.
4. The tool name must be selected from the tool:
{tool_description}.
Note: Each plan should be followed by only one #E.
Start! """

WORKER_PROMPT = """
Thought: {thought}\nResponse: {action_resp}\n
"""

SOLVER_PROMPT = """Solve the following task or problem.
To assist you, we provide some plans and corresponding evidences
that might be helpful. Notice that some of these information
contain noise so you should trust them with caution.\n
{question}\n{worker_log}\nNow begin to solve the task or problem.
Respond with the answer directly with no extra words.{question}\n
"""

REFORMAT_PROMPT = """Response Format Error: {err_msg}. Please reply again:
"""
