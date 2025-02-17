from typing import List
import os


def add_html_breaks_every_n_words(string_list: List[str], n: int = 10) -> List[str]:
    """
    Insert HTML break tags into strings every n words.

    This function formats each string by adding a `<br>` tag after every `n`
    words to enhance readability in HTML.

    Parameters
    ----------
    string_list : List[str]
        List of input strings.
    n : int, optional
        Number of words between breaks, by default 10.

    Returns
    -------
    List[str]
        List of formatted strings with HTML breaks.
    """
    new_string_list: List[str] = []
    for text in string_list:
        words = text.split()
        words_with_breaks = [words[i:i + n] for i in range(0, len(words), n)]
        formatted_text = '<br>'.join(' '.join(group)
                                     for group in words_with_breaks)
        new_string_list.append(formatted_text)
    return new_string_list


def read_apikey() -> str:
    """
    Retrieve the OpenAI API key from a designated file.

    This function reads the API key from a file located at
    '../data/openai_api_key.txt'.

    Returns
    -------
    str
        The OpenAI API key.

    Raises
    ------
    ValueError
        If the key file does not contain exactly one line.
    """
    file_path = "../data/openai_api_key.txt"
    with open(file_path, 'r') as file:
        lines = file.readlines()

    if len(lines) == 1:
        return lines[0].strip()
    else:
        raise ValueError(
            "The OpenAI API key file should contain only a single line with "
            "the API key."
        )
