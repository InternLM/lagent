from typing import List, Optional


def filter_suffix(response: str, suffixes: Optional[List[str]] = None) -> str:
    """Filter response with suffixes.

    Args:
        response (str): generated response by LLMs.
        suffixes (str): a list of suffixes to be deleted.

    Return:
        str: a clean response.
    """
    if suffixes is None:
        return response
    for item in suffixes:
        # if response.endswith(item):
        #     response = response[:len(response) - len(item)]
        if item in response:
            response = response.split(item)[0]
    return response
