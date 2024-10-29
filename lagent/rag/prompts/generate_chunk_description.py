CHUNK_DESCRIPTION_PROMPT = """
You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given an input text and a list of identified entities, create a summary that highlights key points and critical information.
Please prioritize the information from the entities, but if the entities are few or incomplete, extract key details from the input text to ensure the summary remains comprehensive and informative.
Merge information from both the text and entities into a single, coherent summary that is concise but complete.
Make sure the summary is written in third person, and avoid adding any information beyond what is provided to maintain accuracy.

#######
-Data-
Entities: {entities}
Text: {input_text}
#######
Output:
"""