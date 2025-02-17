from typing import Any, List, Tuple, Dict
import pandas as pd


def get_index(
    lastname: str,
    final_answers_comb: pd.DataFrame,
    question_id: int = 14
) -> int:
    """
    Find the index of a student by last name for a specific question.

    This function searches for the student with the given last name within
    the specified question's data.

    Parameters
    ----------
    lastname : str
        Last name of the student.
    final_answers_comb : pd.DataFrame
        DataFrame containing student answers.
    question_id : int, optional
        Question number to filter on, by default 14.

    Returns
    -------
    int
        Index of the student in the DataFrame.

    Raises
    ------
    ValueError
        If the last name is not found for the specified question.
    """
    df_q = final_answers_comb[final_answers_comb["Q #"] == question_id]
    for i, name in enumerate(df_q.LastName):
        if name == lastname:
            return i
    raise ValueError(f"Lastname '{lastname}' not found for question "
                     f"{question_id}.")


def get_name(index: int, final_answers_comb: pd.DataFrame) -> str:
    """
    Retrieve the full name of a student given their index.

    This function combines the first and last names of a student based on
    their index in the DataFrame.

    Parameters
    ----------
    index : int
        Index of the student in the DataFrame.
    final_answers_comb : pd.DataFrame
        DataFrame containing student answers.

    Returns
    -------
    str
        Full name of the student.
    """
    subset = final_answers_comb[
        final_answers_comb["Q #"] ==
        final_answers_comb["Q #"].unique()[0]
    ][["FirstName", "LastName"]]
    name_list = subset.iloc[index].tolist()
    return f"{name_list[0]} {name_list[1]}"


def get_suspect_counts(
    suspect_matches: List[Tuple[int, Any]]
) -> Dict[Any, int]:
    """
    Count the number of occurrences for each suspect in the match list.

    This function tallies how many times each suspect appears in the list
    of matches.

    Parameters
    ----------
    suspect_matches : List[Tuple[int, Any]]
        List of tuples containing suspect indices and associated values.

    Returns
    -------
    Dict[Any, int]
        Dictionary mapping suspect IDs to their count of occurrences.
    """
    suspects_dict = {}
    for match in suspect_matches:
        suspect_id = match[0]
        suspects_dict[suspect_id] = suspects_dict.get(suspect_id, 0) + 1
    return suspects_dict


def label_pooled_questions(
    df: pd.DataFrame,
    question_label: str = "Q #",
    question_col_name: str = "Q Text"
) -> pd.DataFrame:
    """
    Assign unique labels to questions based on their text to handle duplicates.

    This function modifies the DataFrame to ensure each question has a unique
    identifier by appending a suffix when duplicates are detected.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing questions and labels.
    question_label : str, optional
        Column to modify with new labels, by default "Q #".
    question_col_name : str, optional
        Column containing question texts, by default "Q Text".

    Returns
    -------
    pd.DataFrame
        DataFrame with updated question labels.
    """
    result_df = df.copy()
    unique_labels = result_df[question_label].unique()

    for label in unique_labels:
        subset = result_df[result_df[question_label] == label]
        unique_questions = subset[question_col_name].unique()

        if len(unique_questions) > 1:
            new_labels = {
                val: (f"{label}.{i + 1}" if isinstance(label, str)
                      else label + (0.1 * (i + 1)))
                for i, val in enumerate(unique_questions)
            }
            for question, new_label in new_labels.items():
                mask = ((result_df[question_label] == label) &
                        (result_df[question_col_name] == question))
                result_df.loc[mask, question_label] = new_label

    return result_df
