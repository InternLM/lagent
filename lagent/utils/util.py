from typing import List, Optional, Union


def filter_suffix(response: Union[str, List[str]],
                  suffixes: Optional[List[str]] = None) -> str:
    """Filter response with suffixes.

    Args:
        response (Union[str, List[str]]): generated responses by LLMs.
        suffixes (str): a list of suffixes to be deleted.

    Return:
        str: a clean response.
    """
    if suffixes is None:
        return response
    batched = True
    if isinstance(response, str):
        response = [response]
        batched = False
    processed = []
    for resp in response:
        for item in suffixes:
            # if response.endswith(item):
            #     response = response[:len(response) - len(item)]
            if item in resp:
                resp = resp.split(item)[0]
        processed.append(resp)
    if not batched:
        return processed[0]
    return processed
