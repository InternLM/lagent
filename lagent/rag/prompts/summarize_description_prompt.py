# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License.

# Based on the code from Microsoft GraphRag project (https://github.com/microsoft/GraphRag),
# variable names and some logic statements have been modified for clarity and performance.
# Original Author: Microsoft Corporation
# Modified by [kxZhou612] on [2024.10.15]

DEFAULT_SUMMATY_PROMPT = """
You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two nodes, and a list of descriptions, all related to the same node or group of nodes.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected 
from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the node names so we have the full context.
Under the premise of meeting the above conditions, try to keep the response concise and summarizing. 
If you are not familiar with the mentioned content, try not to add information beyond the descriptions provided to avoid
giving incorrect information.

#######
-Data-
Nodes: {node_name}
Description List: {description_list}
#######
Output:
"""