import os
import warnings
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.spatial import distance
import warnings

import matplotlib
matplotlib.use('Agg')


def calculate_difference_matrix(
    vectors: List[np.ndarray],
    metric: str = "euclidean"
) -> np.ndarray:
    """
    Compute a difference matrix using the specified metric.

    This function calculates pairwise distances between vectors using either
    Euclidean or cosine metrics.
    The resulting matrix represents the pairwise differences or similarities
    based on the chosen metric.

    Parameters
    ----------
    vectors : List[np.ndarray]
        List of vectors to compute differences.
    metric : str, optional
        Distance metric to use ("euclidean" or "cosine"), by default "euclidean".

    Returns
    -------
    np.ndarray
        2D difference or similarity matrix.

    Raises
    ------
    ValueError
        If an invalid metric is specified.
    """
    vectors = np.array(vectors)

    if metric in ["euclidean", "cosine"]:
        diff_matrix = distance.pdist(vectors, metric=metric)
        diff_matrix = distance.squareform(diff_matrix)
        if metric == "cosine":
            diff_matrix = 1 - diff_matrix
        return diff_matrix
    else:
        raise ValueError("Invalid metric specified. Use 'euclidean' or 'cosine'.")


def invert_nonzero(array: np.ndarray) -> np.ndarray:
    """
    Invert non-zero elements of the array while keeping zeros unchanged.

    This function performs element-wise inversion on non-zero elements,
    ensuring that zero values remain unaffected.

    Parameters
    ----------
    array : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Array with non-zero elements inverted.
    """
    result_array = np.copy(array)
    nonzero_indices = array != 0
    result_array[nonzero_indices] = 1 / array[nonzero_indices]
    return result_array


def highpass_threshold_matrix(
    matrix: np.ndarray,
    threshold: float = 3.0
) -> np.ndarray:
    """
    Apply a high-pass filter to the matrix by setting values below the
    threshold to zero.

    This function zeros out all elements in the matrix that are below the
    specified threshold, effectively retaining only higher values.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix.
    threshold : float, optional
        Threshold value, by default 3.0.

    Returns
    -------
    np.ndarray
        Thresholded matrix.
    """
    mask = matrix < threshold
    matrix[mask] = 0
    return matrix


def remove_duplicate_rows_by_col(
    df: pd.DataFrame,
    column_label: str,
    parallelize: bool = False
) -> pd.DataFrame:
    """
    Remove duplicate rows within subsets defined by unique column values.

    This function groups the DataFrame by the specified column and removes
    duplicate rows within each group.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to process.
    column_label : str
        Column to group by.
    parallelize : bool, optional
        Whether to parallelize the operation, by default False.

    Returns
    -------
    pd.DataFrame
        DataFrame with duplicates removed.
    """
    def clean_subset(value: Any) -> pd.DataFrame:
        subset_df = df[df[column_label] == value]
        clean_df = subset_df.drop_duplicates()
        return clean_df

    unique_values = df[column_label].unique()

    if parallelize:
        results = Parallel(n_jobs=-1)(
            delayed(clean_subset)(value) for value in unique_values
        )
    else:
        results = [clean_subset(value) for value in unique_values]

    final_df = pd.concat(results, ignore_index=True)
    return final_df


def find_smallest_diff_pairs(
    vectors: List[np.ndarray],
    num_pairs: int = 10,
    metric: str = "euclidean"
) -> List[Tuple[int, int, float]]:
    """
    Identify pairs of vectors with the smallest differences.

    This function finds the top `num_pairs` pairs of vectors that have the
    smallest distances based on the specified metric.

    Parameters
    ----------
    vectors : List[np.ndarray]
        List of vectors to compare.
    num_pairs : int, optional
        Number of smallest pairs to return, by default 10.
    metric : str, optional
        Distance metric to use ("euclidean" or "cosine"), by default "euclidean".

    Returns
    -------
    List[Tuple[int, int, float]]
        List of tuples containing indices and their norm difference.
    """
    num_vectors = len(vectors)
    diff_matrix = calculate_difference_matrix(vectors, metric=metric)
    np.fill_diagonal(diff_matrix, np.inf)
    indices = np.argsort(diff_matrix, axis=None)
    pairs = np.column_stack(np.unravel_index(indices, diff_matrix.shape))
    unique_pairs: List[Tuple[int, int, float]] = []
    seen: set = set()

    for _, (idx1, idx2) in enumerate(pairs):
        if idx1 < idx2 and (idx1, idx2) not in seen:
            unique_pairs.append((idx1, idx2, diff_matrix[idx1, idx2]))
            seen.add((idx1, idx2))
        if len(unique_pairs) >= num_pairs:
            break

    return unique_pairs


def find_smallest_diff_pairs_nm(
    list1: List[np.ndarray],
    list2: List[np.ndarray],
    num_pairs: int = 10,
    metric: str = "euclidean"
) -> List[Tuple[int, int, float]]:
    """
    Find pairs with the smallest differences between two lists of vectors.

    This function identifies the top `num_pairs` pairs across two distinct lists
    of vectors based on the smallest distances.

    Parameters
    ----------
    list1 : List[np.ndarray]
        First list of vectors.
    list2 : List[np.ndarray]
        Second list of vectors.
    num_pairs : int, optional
        Number of smallest pairs to return, by default 10.
    metric : str, optional
        Distance metric to use ("euclidean" or "cosine"), by default "euclidean".

    Returns
    -------
    List[Tuple[int, int, float]]
        List of tuples containing indices and their norm difference.

    Raises
    ------
    ValueError
        If num_pairs is neither int nor 'all'.
    """
    vectors1 = np.array(list1)
    vectors2 = np.array(list2)
    diff_matrix = distance.cdist(vectors1, vectors2, metric=metric)
    indices = np.argsort(diff_matrix, axis=None)
    pairs = np.column_stack(np.unravel_index(indices, diff_matrix.shape))
    results: List[Tuple[int, int, float]] = [
        (idx1, idx2, diff_matrix[idx1, idx2]) for idx1, idx2 in pairs
    ]

    if isinstance(num_pairs, int):
        return results[:num_pairs]
    elif isinstance(num_pairs, str) and num_pairs == "all":
        return results
    else:
        raise ValueError("Invalid value for num_pairs.")


def find_closest_vectors(
    vectors: List[np.ndarray],
    reference_index: int,
    num_results: int = 10,
    metric: str = "euclidean"
) -> List[Tuple[int, float]]:
    """
    Locate vectors closest to a specified reference vector.

    This function computes the distances from the reference vector to all other
    vectors and returns the closest ones.

    Parameters
    ----------
    vectors : List[np.ndarray]
        List of vectors to search.
    reference_index : int
        Index of the reference vector in the list.
    num_results : int, optional
        Number of closest vectors to return, by default 10.
    metric : str, optional
        Distance metric to use ("euclidean" or "cosine"), by default "euclidean".

    Returns
    -------
    List[Tuple[int, float]]
        List of tuples containing index and distance to the reference vector.
    """
    ref_vector = vectors[reference_index]
    vectors = np.array(vectors)
    distances = distance.cdist([ref_vector], vectors, metric=metric)[0]
    distances[reference_index] = np.inf
    indices = np.argsort(distances)
    results = [(idx, distances[idx]) for idx in indices[:num_results]]
    return results


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
                val: f"{label}.{i + 1}"
                if isinstance(label, str)
                else label + (0.1 * (i + 1))
                for i, val in enumerate(unique_questions)
            }
            for question, new_label in new_labels.items():
                mask = (result_df[question_label] == label) & (
                    result_df[question_col_name] == question
                )
                result_df.loc[mask, question_label] = new_label

    return result_df


def average_nonzero(matrix: np.ndarray) -> np.ndarray:
    """
    Calculate the average of non-zero elements in each row.

    This function computes the mean of non-zero values for every row in the
    matrix, ignoring zeros.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix.

    Returns
    -------
    np.ndarray
        Array containing the average of non-zero values per row.
    """
    non_zero_counts = np.count_nonzero(matrix, axis=1)
    sums = np.sum(matrix, axis=1)
    averages = np.divide(
        sums,
        non_zero_counts,
        out=np.zeros_like(sums),
        where=non_zero_counts != 0
    )
    return averages.reshape(-1, 1)


def zero_non_bottom_n(matrix: np.ndarray, n: int = 20) -> np.ndarray:
    """
    Retain only the smallest n non-zero values in each column.

    This function zeroes out all elements except the smallest `n` non-zero
    values in every column of the matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix.
    n : int, optional
        Number of smallest non-zero values to retain per column, by default 20.

    Returns
    -------
    np.ndarray
        Modified matrix with only the smallest n values per column.
    """
    result_matrix = np.zeros_like(matrix)

    for idx in range(matrix.shape[1]):
        column = matrix[:, idx]
        non_zero_indices = column.nonzero()[0]
        if len(non_zero_indices) > n:
            smallest_indices = non_zero_indices[
                np.argsort(column[non_zero_indices])[:n]
            ]
            result_matrix[smallest_indices, idx] = column[smallest_indices]
        else:
            result_matrix[non_zero_indices, idx] = column[non_zero_indices]
    return result_matrix


def calculate_distance_matrix(
    student_embeddings: List[List[float]],
    gpt_answer_embeddings: List[List[float]]
) -> np.ndarray:
    """
    Compute the distance matrix between student and GPT embeddings.

    This function calculates pairwise Euclidean distances between student
    embeddings and GPT-generated answer embeddings.

    Parameters
    ----------
    student_embeddings : List[List[float]]
        Student embeddings.
    gpt_answer_embeddings : List[List[float]]
        GPT-generated answer embeddings.

    Returns
    -------
    np.ndarray
        Distance matrix of shape (n_students, n_gpt_answers).
    """
    students = np.array(student_embeddings)
    gpt_answers = np.array(gpt_answer_embeddings)
    distance_matrix = distance.cdist(students, gpt_answers, metric='euclidean')
    return distance_matrix


def compute_similarity_matrix(
    embedding_dict: Dict[Any, List[np.ndarray]],
    metric: str = 'euclidean',
    n_top: int = 20
) -> np.ndarray:
    """
    Aggregate similarity matrices across all questions.

    This function sums the inverted non-zero distance matrices across all
    questions to form a comprehensive similarity matrix.

    Parameters
    ----------
    embedding_dict : Dict[Any, List[np.ndarray]]
        Dictionary mapping question IDs to their embeddings.
    metric : str, optional
        Distance metric to use ('euclidean' or 'cosine'), by default 'euclidean'.
    n_top : int, optional
        Number of top similarities to retain per student per question,
        by default 20.

    Returns
    -------
    np.ndarray
        Summed similarity matrix across all questions.
    """
    similarity_matrices: List[np.ndarray] = []
    for q_id, embeddings in embedding_dict.items():
        diff_matrix = calculate_difference_matrix(embeddings, metric=metric)
        diff_matrix_filt = zero_non_bottom_n(diff_matrix, n=n_top)
        similarity_matrix = invert_nonzero(diff_matrix_filt)
        similarity_matrices.append(similarity_matrix)

    sum_similarity_matrices = np.array(similarity_matrices).sum(axis=0)
    return sum_similarity_matrices


def extract_pairs_above_threshold(
    similarity_matrix: np.ndarray,
    threshold: float = 20.0
) -> List[Tuple[int, int, float]]:
    """
    Extract pairs with similarity scores above a specified threshold.

    This function identifies all pairs in the similarity matrix that exceed
    the given threshold.

    Parameters
    ----------
    similarity_matrix : np.ndarray
        Summed similarity matrix.
    threshold : float, optional
        Similarity score threshold, by default 20.0.

    Returns
    -------
    List[Tuple[int, int, float]]
        List of tuples containing indices and their similarity score.
    """
    pair_indices = np.vstack(np.where(similarity_matrix > threshold)).T
    pair_scores = [similarity_matrix[idx[0], idx[1]]
                   for idx in pair_indices]

    unique_pairs: Dict[Tuple[int, int], float] = {}
    for (i, j), score in zip(pair_indices, pair_scores):
        if i < j:
            unique_pairs[(i, j)] = score
        else:
            unique_pairs[(j, i)] = score

    sorted_pairs = sorted(unique_pairs.items(), key=lambda x: x[1],
                          reverse=True)

    return [(i, j, score) for ((i, j), score) in sorted_pairs]


def compare_all_embedding_pairs(
    embedding_dict: Dict[Any, List[np.ndarray]],
    metric: str = 'euclidean',
    n_top: int = 20,
    percentile: float = 95.0
) -> Tuple[np.ndarray, float, List[Tuple[int, int, float]]]:
    """
    Compare embeddings to identify highly similar pairs.

    This function computes the summed similarity matrix, determines a
    threshold based on the specified percentile, and extracts pairs
    exceeding this threshold.
    
    Parameters
    ----------
    embedding_dict : Dict[Any, List[np.ndarray]]
        Dictionary mapping question IDs to their embeddings.
    metric : str, optional
        Distance metric to use ('euclidean' or 'cosine'), by default 'euclidean'.
    n_top : int, optional
        Number of top similarities to retain per item, by default 20.
    percentile : float, optional
        Percentile to determine similarity threshold, by default 95.0.
    
    Returns
    -------
    Tuple[np.ndarray, float, List[Tuple[int, int, float]]]
        Summed similarity matrix, similarity threshold, and list of high
        similarity pairs.
    """
    print("Computing summed similarity matrix across all items...")
    sum_similarity_matrix = compute_similarity_matrix(
        embedding_dict,
        metric=metric,
        n_top=n_top
    )
    print("Summed similarity matrix computed.")

    threshold = compute_threshold_percentile(sum_similarity_matrix,
                                               percentile=percentile)

    if threshold is None:
        high_similarity_pairs = []
        print("No similarity threshold computed due to all-zero similarity "
              "matrix.")
    else:
        print("Extracting high similarity pairs based on the threshold...")
        high_similarity_pairs = extract_pairs_above_threshold(
            sum_similarity_matrix, threshold=threshold
        )
        print(f"Found {len(high_similarity_pairs)} high similarity pairs.")

    return sum_similarity_matrix, threshold, high_similarity_pairs


def compute_threshold_percentile(
    matrix: np.ndarray,
    percentile: float = 95.0
) -> Optional[float]:
    """
    Compute the threshold value based on the given percentile.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix from which to compute the threshold.
    percentile : float, optional
        Percentile to determine the threshold, by default 95.0.

    Returns
    -------
    Optional[float]
        Threshold value corresponding to the specified percentile.
    """
    flattened = matrix.flatten()
    flattened = flattened[flattened != 0]
    if flattened.size == 0:
        print("Similarity matrix is all zeros...")
        return None
    threshold = np.percentile(flattened, percentile)
    return threshold


def plot_similarity_histogram(
    sum_similarity_matrix: np.ndarray,
    threshold_percentile: float = 95.0,
    bins: int = 100,
    save_dir: Optional[str] = None,
) -> None:
    """
    Plot a histogram of similarity scores with a threshold line.

    Parameters
    ----------
    sum_similarity_matrix : np.ndarray
        Summed similarity matrix.
    threshold_percentile : float, optional
        Percentile to determine the threshold, by default 95.0.
    bins : int, optional
        Number of bins for the histogram, by default 100.
    """
    threshold = compute_threshold_percentile(sum_similarity_matrix,
                                               percentile=threshold_percentile)
    plt.hist(sum_similarity_matrix.flatten(), bins=bins, alpha=0.75)
    plt.axvline(threshold, color='r', linestyle='dashed', linewidth=1)
    plt.title(
        f"Similarity Scores Histogram (Threshold: "
        f"{threshold_percentile}th Percentile)"
    )
    plt.xlabel("Similarity Score")
    plt.ylabel("Frequency")

    if save_dir:
        histogram_path = os.path.join(save_dir, 'similarity_hisogram.svg')
        plt.savefig(histogram_path)
    else:
        matplotlib.use('TkAgg')
        plt.show()


def pagerank(
    adjacency_matrix: np.ndarray,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1.0e-6,
    personalization: Optional[Dict[int, float]] = None
) -> Dict[int, float]:
    """
    Compute PageRank scores from a similarity adjacency matrix using the
    power iteration method. This function constructs an undirected graph
    from the adjacency matrix and applies the networkx pagerank algorithm.

    Parameters
    ----------
    adjacency_matrix : np.ndarray
        Square matrix representing pairwise similarities among nodes.
    alpha : float
        Damping parameter for PageRank.
    max_iter : int
        Maximum number of iterations in power method.
    tol : float
        Error tolerance to check convergence.
    personalization : Optional[Dict[int, float]]
        Optional dictionary of personalized PageRank values.

    Returns
    -------
    Dict[int, float]
        Dictionary mapping node indices to their PageRank scores.
    """
    G = nx.from_numpy_array(adjacency_matrix)
    return nx.pagerank(
        G,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        personalization=personalization
    )
