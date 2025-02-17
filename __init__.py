from .embedder import Zapper
from .analysis import (
    calculate_difference_matrix,
    invert_nonzero,
    highpass_threshold_matrix,
    remove_duplicate_rows_by_col,
    find_smallest_diff_pairs,
    find_smallest_diff_pairs_nm,
    find_closest_vectors,
    average_nonzero,
    zero_non_bottom_n,
    calculate_distance_matrix
)
from .querying import (
    get_embedding,
    query_gpt
)
from .utils.labels import (
    get_index,
    get_name,
    get_suspect_counts,
    label_pooled_questions,
)
from .utils.text_processing import (
    add_html_breaks_every_n_words,
    read_apikey
)
from .utils.plotting import (
    plot_scatter_3d,
    plot_network
)

__all__ = [
    "Zapper",
    "get_embedding",
    "query_gpt",
    "calculate_difference_matrix",
    "invert_nonzero",
    "highpass_threshold_matrix",
    "remove_duplicate_rows_by_col",
    "find_smallest_diff_pairs",
    "find_smallest_diff_pairs_nm",
    "find_closest_vectors",
    "average_nonzero",
    "zero_non_bottom_n",
    "calculate_distance_matrix",
    "plot_scatter_3d",
    "plot_network",
    "get_index",
    "get_name",
    "get_suspect_counts",
    "label_pooled_questions",
    "add_html_breaks_every_n_words",
    "read_apikey"
]
