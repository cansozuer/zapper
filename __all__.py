#%# __init__.py
from .embedder import Embedder
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
from .utils.dimensionality_reduction import (
    plot_reduced_dim_vector_3d
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

__all__ = [
    "Embedder",
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
    "plot_reduced_dim_vector_3d",
    "get_index",
    "get_name",
    "get_suspect_counts",
    "label_pooled_questions",
    "add_html_breaks_every_n_words",
    "read_apikey"
]
#%# embedder.py
#%# embedder.py
import os
import sys
import math
import time
import pickle
from typing import Optional, Dict, Any, Tuple, List
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import torch.nn as nn

from .analysis import (
    remove_duplicate_rows_by_col,
    calculate_distance_matrix,
    compute_similarity_matrix,
    extract_pairs_above_threshold,
    label_pooled_questions,
    compare_all_embedding_pairs,
    plot_similarity_histogram,
)
from .utils.labels import get_name, get_suspect_counts
from .utils.text_processing import add_html_breaks_every_n_words
from .querying import get_embedding, query_gpt
from .utils.dimensionality_reduction import plot_reduced_dim_vector_3d
from .embeddinggenerator import EmbeddingGenerator


class Embedder:
    """
    The Embedder class orchestrates the loading and preprocessing of data, generation of student
    and GPT embeddings, dimensionality reduction, plotting, and similarity analysis.
    """
    def __init__(
        self,
        excel_path: Optional[str] = None,
        n_gpt_answers: int = 50,
        embedding_model: str = "text-embedding-3-large",
        gpt_model: str = "gpt-4o-mini",
        n_words: int = 50,
        reduction_method: str = 'umap',
        embeddings_args: Dict[str, Any] = None,
        ignore_pooled: bool = False
    ) -> None:
        """
        Initialize Embedder with data paths and model configurations.

        Parameters
        ----------
        excel_path : Optional[str]
            Path to the Excel file containing answers.
        n_gpt_answers : int
            Number of GPT-generated answers per question.
        embedding_model : str
            Text embedding model to utilize.
        gpt_model : str
            GPT model for generating answers.
        n_words : int
            Number of words per chunk to create embeddings for.
        embeddings_args : Dict[str, Any]
            Dictionary that contains embedding arguments including 'method' and 'deep_sets_args'.
        ignore_pooled : bool
            If True, questions originating from question pools (with subnumbered questions) are ignored and removed.
        """
        self.excel_path = excel_path
        self.n_gpt_answers = n_gpt_answers
        self.embedding_model = embedding_model
        self.gpt_model = gpt_model
        self.n_words = n_words
        self.reduction_method = reduction_method
        self.embeddings_args = embeddings_args or {}
        self.ignore_pooled = ignore_pooled

        self.embedding_size = self._get_embedding_size()

        if 'deep_sets_args' in self.embeddings_args:
            self.embeddings_args['deep_sets_args']['input_size'] = self.embedding_size

        self.df: Optional[pd.DataFrame] = None
        self._read_and_clean()

    def save(self, filename: Optional[str] = None, save_dir: Optional[str] = None) -> None:
        """
        Serialize the Embedder instance to a pickle file.

        Parameters
        ----------
        filename : Optional[str]
            Desired filename for the pickle file.
        save_dir : Optional[str]
            Directory path to save the file.
        """
        current_time = time.strftime("%d-%m-%Y_%H-%M-%S", time.localtime())
        filename = filename or f'embedder_instance_{current_time}.pkl'
        if save_dir:
            filename = os.path.join(save_dir, filename)
        with open(filename, 'wb') as file_handle:
            pickle.dump(self, file_handle)
        print(f"Embedder instance saved to {filename}")

    @classmethod
    def load(cls, filepath: str) -> 'Embedder':
        """
        Deserialize an Embedder instance from a pickle file.

        Parameters
        ----------
        filepath : str
            Path to the pickle file.

        Returns
        -------
        Embedder
            The loaded Embedder instance.

        Raises
        ------
        ValueError
            If the loaded object is not an Embedder instance.
        """
        with open(filepath, 'rb') as file_handle:
            instance = pickle.load(file_handle)
        if not isinstance(instance, cls):
            raise ValueError("The loaded object is not an instance of Embedder.")
        return instance

    def _get_embedding_size(self) -> int:
        """
        Determine the embedding size based on the selected embedding model.

        Returns
        -------
        int
            The dimensionality of the embedding vectors.

        Raises
        ------
        ValueError
            If an unsupported embedding model is specified.
        """
        embedding_size_mapping = {
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        try:
            return embedding_size_mapping[self.embedding_model]
        except KeyError:
            raise ValueError(f"Unsupported embedding model: {self.embedding_model}")

    def _read_and_clean(self) -> None:
        """
        Load and preprocess the Excel dataset.

        Raises
        ------
        ValueError
            If the Excel path is not provided.
        """
        if not self.excel_path:
            raise ValueError("Excel path is not provided.")
        self.df = pd.read_excel(self.excel_path)
        self.df = self.df[self.df['Q Type'] == 'WR']
        self.df = remove_duplicate_rows_by_col(self.df, 'Username')
        self.df = label_pooled_questions(self.df)
        
        if self.ignore_pooled:
            self.df = self.df[self.df['Q #'].astype(float) % 1 == 0]

        self.df = self.df.sort_values(by=['Org Defined ID', 'Q #'], ascending=[True, True])

    def _preprocess_answers_questions(self) -> None:
        """
        Clean and organize answers and questions from the dataset.
        """
        self.df['Answer'] = self.df['Answer'].apply(
            lambda x: '' if isinstance(x, float) and math.isnan(x) else x
        )
        self.student_answers: Dict[Any, List[str]] = {
            q_num: self.df[self.df['Q #'] == q_num].Answer.tolist()
            for q_num in self.df['Q #'].unique()
        }
        self.question_ids: np.ndarray = self.df['Q #'].unique()
        self.question_stems: List[str] = [
            text.replace('\xa0', ' ') for text in self.df['Q Text'].unique().tolist()
        ]

    def _generate_multiple_student_answer_embeddings(self) -> None:
        """
        Compute embeddings for each student's answers (multiple embeddings per answer).
        Uses EmbeddingGenerator for maintainability.
        This method only generates the chunked embeddings list; it does not combine them.
        """
        eg = EmbeddingGenerator(
            embedding_model=self.embedding_model,
            n_words=self.n_words,
            **self.embeddings_args
        )
        self.student_answer_embeddings_list: Dict[Any, List[List[np.ndarray]]] = {}
        for question_id in self.question_ids:
            answers = self.student_answers[question_id]
            print(
                f"Generating multiple student embeddings: Question {question_id}", 
                end='\r', 
                flush=True
            )
            chunked_list = eg.generate_embeddings(answers)
            self.student_answer_embeddings_list[question_id] = chunked_list
            print()
        sys.stdout.flush()

    def _combine_student_answer_embeddings(self) -> None:
        """
        Combine the chunked student answer embeddings into a single embedding per answer.
        """
        eg = EmbeddingGenerator(
            embedding_model=self.embedding_model,
            n_words=self.n_words,
            **self.embeddings_args
        )
        self.student_answer_embeddings: Dict[Any, List[np.ndarray]] = {}
        for question_id, chunked_list in self.student_answer_embeddings_list.items():
            print(
                f"Combining student embeddings: Question {question_id}", 
                end='\r', 
                flush=True
            )
            final_embeds = eg.combine_embeddings(chunked_list)
            self.student_answer_embeddings[question_id] = final_embeds
            print()
        sys.stdout.flush()

    def _generate_gpt_answers(self, temperature: float = 0.6) -> None:
        """
        Generate GPT-based answers for each question.

        Parameters
        ----------
        temperature : float
            Sampling temperature for GPT.
        """
        self.gpt_answers: Dict[Any, List[str]] = {}
        for i, question_stem in enumerate(self.question_stems):
            answers: List[str] = []
            for j in range(self.n_gpt_answers):
                print(
                    f"Generating GPT answers: Question {i + 1}/{len(self.question_ids)}" 
                    f"Answer {j + 1}/{self.n_gpt_answers}", 
                    end='\r', 
                    flush=True
                )
                answer = query_gpt(prompt=question_stem, model=self.gpt_model, temperature=temperature)
                answers.append(answer)
            self.gpt_answers[self.question_ids[i]] = answers
            print()
        sys.stdout.flush()

    def _generate_mulitple_gpt_answer_embeddings(self) -> None:
        """
        Compute embeddings for GPT-generated answers (multiple embeddings per answer).
        Uses EmbeddingGenerator for maintainability.
        This method only generates the chunked embeddings list; it does not combine them.
        """
        eg = EmbeddingGenerator(
            embedding_model=self.embedding_model,
            n_words=self.n_words,
            **self.embeddings_args
        )
        self.gpt_answer_embeddings_list: Dict[Any, List[List[np.ndarray]]] = {}
        for question_id in self.question_ids:
            answers = self.gpt_answers[question_id]
            sys.stdout.write(
                f"\rGenerating multiple GPT embeddings: Question {question_id}"
            )
            sys.stdout.flush()
            chunked_list = eg.generate_embeddings(answers)
            self.gpt_answer_embeddings_list[question_id] = chunked_list
            print()
        sys.stdout.flush()

    def _combine_gpt_answer_embeddings(self) -> None:
        """
        Combine the chunked GPT answer embeddings into a single embedding per answer.
        """
        eg = EmbeddingGenerator(
            embedding_model=self.embedding_model,
            n_words=self.n_words,
            **self.embeddings_args
        )
        self.gpt_answer_embeddings: Dict[Any, List[np.ndarray]] = {}
        for question_id, chunked_list in self.gpt_answer_embeddings_list.items():
            sys.stdout.write(
                f"\rCombining GPT embeddings: Question {question_id}"
            )
            sys.stdout.flush()
            final_embeds = eg.combine_embeddings(chunked_list)
            self.gpt_answer_embeddings[question_id] = final_embeds
            print()
        sys.stdout.flush()

    def _reduce_dimensions(
        self,
        vector_list: List[np.ndarray],
        n_components: int = 10,
        method: str = 'umap'
    ) -> np.ndarray:
        """
        Perform dimensionality reduction on a set of vectors.

        Parameters
        ----------
        vector_list : List[np.ndarray]
            List of vectors to reduce.
        n_components : int, optional
            Number of dimensions to reduce to, by default 10.
        method : str, optional
            Dimensionality reduction method ('pca', 'tsne', 'umap'), by default 'umap'.

        Returns
        -------
        np.ndarray
            Reduced dimension vectors.
        """
        if len(vector_list) < n_components:
            raise ValueError(
                f"Not enough vectors ({len(vector_list)}) for {n_components} components."
            )
        X = np.vstack(vector_list)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if method == 'pca':
            reducer = PCA(n_components=n_components)
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, learning_rate='auto', init='random')
        elif method == 'umap':
            reducer = UMAP(n_components=n_components)
        else:
            raise ValueError("Method must be 'pca', 'tsne', or 'umap'.")

        X_reduced = reducer.fit_transform(X_scaled)
        return X_reduced

    def _generate_reduced_embeddings(
        self,
        n_components: int = 10,
        method: Optional[str] = None
    ) -> None:
        """
        Generate reduced-dimensional embeddings for both student and GPT answers.

        Parameters
        ----------
        n_components : int, optional
            Number of dimensions for reduction, by default 10.
        method : str, optional
            Dimensionality reduction method ('pca', 'tsne', 'umap'), by default 'umap'.
        """
        method = method if method is not None else self.reduction_method
        self.reduced_student_embeddings: Dict[Any, np.ndarray] = {}
        self.reduced_gpt_embeddings: Dict[Any, np.ndarray] = {}
        self.scalers: Dict[Any, StandardScaler] = {}
        self.reducers: Dict[Any, Any] = {}
        for question_id in self.question_ids:
            student_vectors = self.student_answer_embeddings[question_id]
            self.scalers[question_id] = StandardScaler()
            X_scaled = self.scalers[question_id].fit_transform(np.vstack(student_vectors))
            if method == 'pca':
                reducer = PCA(n_components=n_components)
            elif method == 'tsne':
                reducer = TSNE(n_components=n_components, learning_rate='auto', init='random')
            elif method == 'umap':
                reducer = UMAP(n_components=n_components)
            else:
                raise ValueError("Method must be 'pca', 'tsne', or 'umap'.")
            X_reduced = reducer.fit_transform(X_scaled)
            self.reducers[question_id] = reducer
            self.reduced_student_embeddings[question_id] = X_reduced

            gpt_vectors = self.gpt_answer_embeddings[question_id]
            X_scaled_gpt = self.scalers[question_id].transform(np.vstack(gpt_vectors))
            Y_reduced = self.reducers[question_id].transform(X_scaled_gpt)
            self.reduced_gpt_embeddings[question_id] = Y_reduced

    def _generate_plots(
        self,
        components: List[int] = [0, 1, 2],
        save_dir: Optional[str] = None,
        method: Optional[str] = None,
    ) -> None:
        """
        Create 3D scatter plots of the reduced embeddings.

        Parameters
        ----------
        components : List[int], optional
            Indices of the components to plot, by default [0, 1, 2].
        save_dir : Optional[str], optional
            Directory to save the generated plots, by default None.
        """
        method = method if method is not None else self.reduction_method
        self.plots: List[Any] = []
        for question_id in self.question_ids:
            X_reduced = self.reduced_student_embeddings[question_id]
            Y_reduced = self.reduced_gpt_embeddings[question_id]
            student_answers = self.student_answers[question_id]
            gpt_answers = self.gpt_answers[question_id]
            question_df = self.df[self.df['Q #'] == question_id]
            full_names = [f"{row['FirstName']} {row['LastName']}" for _, row in question_df.iterrows()]
            wrapped_student_answers = []
            for name, ans in zip(full_names, student_answers):
                wrapped = textwrap.wrap(ans, width=79, break_long_words=False)
                final_text = f"{name}:<br>{'<br>'.join(wrapped)}"
                wrapped_student_answers.append(final_text)
            wrapped_gpt_answers = []
            for ans in gpt_answers:
                wrapped = textwrap.wrap(ans, width=79, break_long_words=False)
                final_text = f"GPT Answer:<br>{'<br>'.join(wrapped)}"
                wrapped_gpt_answers.append(final_text)
            fig = plot_reduced_dim_vector_3d(
                X_reduced,
                components=components,
                student_answers=wrapped_student_answers,
                comparison_reduced=Y_reduced,
                comparison_answers=wrapped_gpt_answers,
                method=method
            )
            self.plots.append(fig)
            if save_dir:
                plot_path = os.path.join(save_dir, f'plot_question_{question_id}.html')
                fig.write_html(plot_path)
                print(f"Plot saved to {plot_path}")

    def _generate_distance_matrices(self) -> None:
        """
        Compute distance matrices between student and GPT embeddings for each question.
        """
        self.distance_matrices: Dict[Any, np.ndarray] = {}
        for question_id in self.question_ids:
            student_embeds = self.student_answer_embeddings[question_id]
            gpt_embeds = self.gpt_answer_embeddings[question_id]
            distance_matrix = calculate_distance_matrix(student_embeds, gpt_embeds)
            self.distance_matrices[question_id] = distance_matrix

    def _get_similarity_df(
        self,
        suspect_pairs: List[Tuple[int, int, float]]
    ) -> pd.DataFrame:
        """
        Create a DataFrame from suspected similar pairs.

        Parameters
        ----------
        suspect_pairs : List[Tuple[int, int, float]]
            List of tuples containing student indices and their similarity score.

        Returns
        -------
        pd.DataFrame
            DataFrame with student ID pairs and similarity scores.
        """
        data = []
        for i, j, score in suspect_pairs:
            student_id_1 = self.df.iloc[i]['Org Defined ID']
            student_id_2 = self.df.iloc[j]['Org Defined ID']
            data.append({
                'Student_ID_1': student_id_1,
                'Student_ID_2': student_id_2,
                'Similarity_Score': score
            })
        similarity_df = pd.DataFrame(data)
        self.similarity_df = similarity_df
        return similarity_df

    def _between_student_similarity_comparison(
        self,
        metric: str = 'euclidean',
        n_top: int = 20,
        similarity_percentile: float = 95.0,
        save_dir: Optional[str] = None
    ) -> Tuple[pd.DataFrame, List[Tuple[str, str, float]]]:
        """
        Execute similarity comparison and handle post-processing tasks.

        Parameters
        ----------
        metric : str
            Distance metric for similarity comparison.
        n_top : int
            Number of top similarities to retain per student per question.
        similarity_percentile : float
            Percentile to determine the similarity threshold.
        save_dir : Optional[str]
            Directory to save the similarity DataFrame.

        Returns
        -------
        Tuple[pd.DataFrame, List[Tuple[str, str, float]]]
            DataFrame of similarity scores and list of suspect name tuples with scores.
        """
        sum_similarity_matrix, threshold, high_similarity_pairs = compare_all_embedding_pairs(
            embedding_dict=self.student_answer_embeddings,
            metric=metric,
            n_top=n_top,
            percentile=similarity_percentile
        )

        plot_similarity_histogram(
            sum_similarity_matrix=sum_similarity_matrix,
            threshold_percentile=similarity_percentile,
            bins=100,
            save_dir=save_dir
        )

        similarity_df_id = self.construct_similarity_dataframe(high_similarity_pairs)

        similarity_list_name = self.report_suspect_names(high_similarity_pairs)
        similarity_df_name = pd.DataFrame(
            similarity_list_name, 
            index=None, 
            columns=['student_name1', 'student_name2', 'similarity_score']
        )

        if save_dir:
            df_id_path = os.path.join(save_dir, 'similarity_scores_id.csv')
            similarity_df_id.to_csv(df_id_path, index=False)

            df_name_path = os.path.join(save_dir, 'similarity_scores_name.csv')
            similarity_df_name.to_csv(df_name_path, index=False)

            print(f"Similarity scores saved to {df_id_path} and {df_name_path}")

        return similarity_df_id, similarity_df_name

    def compare_student_answers(
        self,
        metric: str = 'euclidean',
        n_top: int = 20,
        similarity_percentile: float = 95.0
    ) -> Tuple[pd.DataFrame, List[Tuple[str, str, float]]]:
        """
        Placeholder method for interface consistency.

        Parameters
        ----------
        metric : str
            Distance metric to use ('euclidean' or 'cosine').
        n_top : int
            Number of top similarities to retain per student per question.
        similarity_percentile : float
            Percentile to determine the similarity threshold.

        Returns
        -------
        Tuple[pd.DataFrame, List[Tuple[str, str, float]]]
            Similarity DataFrame and list of suspect name tuples.
        """
        pass

    def construct_similarity_dataframe(
        self,
        suspect_pairs: List[Tuple[int, int, float]]
    ) -> pd.DataFrame:
        """
        Generate a DataFrame from high similarity pairs.

        Parameters
        ----------
        suspect_pairs : List[Tuple[int, int, float]]
            List of tuples containing student indices and similarity scores.

        Returns
        -------
        pd.DataFrame
            DataFrame with student ID pairs and similarity scores.
        """
        data = []
        for i, j, score in suspect_pairs:
            student_id_1 = self.df.iloc[i]['Org Defined ID']
            student_id_2 = self.df.iloc[j]['Org Defined ID']
            data.append({
                'Student_ID_1': student_id_1,
                'Student_ID_2': student_id_2,
                'Similarity_Score': score
            })
        similarity_df = pd.DataFrame(data)
        return similarity_df

    def report_suspect_names(
        self,
        suspect_pairs: List[Tuple[int, int, float]]
    ) -> List[Tuple[str, str, float]]:
        """
        Output names and similarity scores of suspect pairs.

        Parameters
        ----------
        suspect_pairs : List[Tuple[int, int, float]]
            List of tuples containing student indices and similarity scores.

        Returns
        -------
        List[Tuple[str, str, float]]
            List of tuples with student names and their similarity scores.
        """
        suspect_names: List[Tuple[str, str, float]] = []
        for i, j, score in suspect_pairs:
            name1 = get_name(i, self.df)
            name2 = get_name(j, self.df)
            suspect_names.append((name1, name2, score))
            print(f"[{name1}, {name2}] Similarity Score: {round(score, 5)}")
        return suspect_names

    def full_process(
        self,
        temperature: float = 0.6,
        n_components: int = 10,
        method: str = 'umap',
        save_dir: Optional[str] = None,
        metric: str = 'euclidean',
        n_top: int = 20,
        similarity_percentile: float = 95.0
    ) -> Tuple[pd.DataFrame, List[Tuple[str, str, float]]]:
        """
        Execute the complete pipeline for embedding, plotting, and similarity analysis.

        Parameters
        ----------
        temperature : float
            Sampling temperature for GPT.
        n_components : int
            Number of components for dimensionality reduction.
        method : str
            Dimensionality reduction method ('pca', 'tsne', 'umap').
        save_dir : Optional[str]
            Directory to save plots and similarity scores.
        metric : str
            Distance metric for similarity comparison.
        n_top : int
            Number of top similarities to retain per student per question.
        similarity_percentile : float
            Percentile to determine the similarity threshold.

        Returns
        -------
        Tuple[pd.DataFrame, List[Tuple[str, str, float]]]
            Similarity DataFrame and list of suspect name tuples.
        """
        self._preprocess_answers_questions()
        self._generate_multiple_student_answer_embeddings()
        self._combine_student_answer_embeddings()
        self._generate_gpt_answers(temperature=temperature)
        self._generate_mulitple_gpt_answer_embeddings()
        self._combine_gpt_answer_embeddings()
        self._generate_reduced_embeddings(n_components=n_components, method=method)
        self._generate_plots(save_dir=save_dir)
        self._generate_distance_matrices()

        similarity_df, suspect_names = self._between_student_similarity_comparison(
            metric=metric,
            n_top=n_top,
            similarity_percentile=similarity_percentile,
            save_dir=save_dir
        )

        print("Full processing complete.")
        return similarity_df, suspect_names
#%# deepsets.py
from typing import Optional

import torch
import torch.nn as nn
import numpy as np

class DeepSets(nn.Module):
    """
    This class applies a DeepSets-based distillation of a list of embeddings into a single embedding.
    
    Parameters
    ----------
    input_size : int
        The dimension of each embedding vector.
    phi_hidden_size : int, optional
        The hidden size for the phi network, by default 128.
    rho_hidden_size : int, optional
        The hidden size for the rho network, by default 128.
    output_size : int, optional
        The dimension of the distilled embedding, by default same as input_size.
    device : str, optional
        Device to run the model on, e.g., 'cpu' or 'cuda', by default 'cpu'.
    """
    def __init__(
        self,
        input_size: int,
        phi_hidden_size: int = 128,
        rho_hidden_size: int = 128,
        output_size: Optional[int] = None,
        device: str = 'cpu'
    ):
        super(DeepSets, self).__init__()
        self.device = device
        
        self.phi_network = nn.Sequential(
            nn.Linear(input_size, phi_hidden_size),
            nn.ReLU(),
            nn.Linear(phi_hidden_size, phi_hidden_size),
            nn.ReLU()
        ).to(device)
        
        self.rho_network = nn.Sequential(
            nn.Linear(phi_hidden_size, rho_hidden_size),
            nn.ReLU(),
            nn.Linear(rho_hidden_size, output_size if output_size else input_size)
        ).to(device)

    def forward(self, embeddings_list: np.ndarray) -> np.ndarray:
        """
        Apply the DeepSets algorithm to distill a list of embeddings into a single embedding.
        
        Parameters
        ----------
        embeddings_list : np.ndarray
            A numpy array of shape (N, D) where N is the number of embeddings and D is the embedding dimension.
        
        Returns
        -------
        np.ndarray
            A single distilled embedding of shape (output_size,).
        """
        with torch.no_grad():
            x = torch.tensor(embeddings_list, dtype=torch.float32, device=self.device)
            x_phi = self.phi_network(x)
            x_sum = torch.sum(x_phi, dim=0, keepdim=True)
            out = self.rho_network(x_sum)
            return out.cpu().numpy().flatten()
#%# analysis.py
import os
import warnings
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.spatial import distance
import warnings

import matplotlib
matplotlib.use('Agg') 


def calculate_difference_matrix(vectors: List[np.ndarray], metric: str = "euclidean") -> np.ndarray:
    """
    Compute a difference matrix using the specified metric.

    This function calculates pairwise distances between vectors using either Euclidean or cosine metrics.
    The resulting matrix represents the pairwise differences or similarities based on the chosen metric.

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

    This function performs element-wise inversion on non-zero elements, ensuring that zero values remain unaffected.

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


def highpass_threshold_matrix(matrix: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Apply a high-pass filter to the matrix by setting values below the threshold to zero.

    This function zeros out all elements in the matrix that are below the specified threshold, effectively retaining only higher values.

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

    This function groups the DataFrame by the specified column and removes duplicate rows within each group.

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

    This function finds the top `num_pairs` pairs of vectors that have the smallest distances based on the specified metric.

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

    This function identifies the top `num_pairs` pairs across two distinct lists of vectors based on the smallest distances.

    Parameters
    ----------
    list1 : List[np.ndarray]
        First list of vectors.
    list2 : List[np.ndarray]
        Second list of vectors.
    num_pairs : int or str, optional
        Number of smallest pairs to return or 'all', by default 10.
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

    This function computes the distances from the reference vector to all other vectors and returns the closest ones.

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

    This function modifies the DataFrame to ensure each question has a unique identifier by appending a suffix when duplicates are detected.

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
                val: f"{label}.{i + 1}" if isinstance(label, str) else label + (0.1 * (i + 1))
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

    This function computes the mean of non-zero values for every row in the matrix, ignoring zeros.

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

    This function zeroes out all elements except the smallest `n` non-zero values in every column of the matrix.

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
            smallest_indices = non_zero_indices[np.argsort(column[non_zero_indices])[:n]]
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

    This function calculates pairwise Euclidean distances between student embeddings and GPT-generated answer embeddings.

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

    This function sums the inverted non-zero distance matrices across all questions to form a comprehensive similarity matrix.

    Parameters
    ----------
    embedding_dict : Dict[Any, List[np.ndarray]]
        Dictionary mapping question IDs to their embeddings.
    metric : str, optional
        Distance metric to use ('euclidean' or 'cosine'), by default 'euclidean'.
    n_top : int, optional
        Number of top similarities to retain per student per question, by default 20.

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

    This function identifies all pairs in the similarity matrix that exceed the given threshold.

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
    pair_scores = [similarity_matrix[idx[0], idx[1]] for idx in pair_indices]

    unique_pairs: Dict[Tuple[int, int], float] = {}
    for (i, j), score in zip(pair_indices, pair_scores):
        if i < j:
            unique_pairs[(i, j)] = score
        else:
            unique_pairs[(j, i)] = score

    sorted_pairs = sorted(unique_pairs.items(), key=lambda x: x[1], reverse=True)

    return [(i, j, score) for ((i, j), score) in sorted_pairs]


def compare_all_embedding_pairs(
    embedding_dict: Dict[Any, List[np.ndarray]],
    metric: str = 'euclidean',
    n_top: int = 20,
    percentile: float = 95.0
) -> Tuple[np.ndarray, float, List[Tuple[int, int, float]]]:
    """
    Compare embeddings to identify highly similar pairs.

    This function computes the summed similarity matrix, determines a threshold based on the specified percentile, and extracts pairs exceeding this threshold.

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
        Summed similarity matrix, similarity threshold, and list of high similarity pairs.
    """
    print("Computing summed similarity matrix across all items...")
    sum_similarity_matrix = compute_similarity_matrix(
        embedding_dict,
        metric=metric,
        n_top=n_top
    )
    print("Summed similarity matrix computed.")

    threshold = compute_threshold_percentile(sum_similarity_matrix, percentile=percentile)

    print("Extracting high similarity pairs based on the threshold...")
    high_similarity_pairs = extract_pairs_above_threshold(sum_similarity_matrix, threshold=threshold)
    print(f"Found {len(high_similarity_pairs)} high similarity pairs.")

    return sum_similarity_matrix, threshold, high_similarity_pairs


def compute_threshold_percentile(matrix: np.ndarray, percentile: float = 95.0) -> float:
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
    float
        Threshold value corresponding to the specified percentile.
    """
    flattened = matrix.flatten()
    flattened = flattened[flattened != 0]
    if not flattened:
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
    threshold = compute_threshold_percentile(sum_similarity_matrix, percentile=threshold_percentile)
    plt.hist(sum_similarity_matrix.flatten(), bins=bins, alpha=0.75)
    plt.axvline(threshold, color='r', linestyle='dashed', linewidth=1)
    plt.title(f"Similarity Scores Histogram (Threshold: {threshold_percentile}th Percentile)")
    plt.xlabel("Similarity Score")
    plt.ylabel("Frequency")

    if save_dir:
        histogram_path = os.path.join(save_dir, 'similarity_hisogram.svg')
        plt.savefig(histogram_path)
    else:
        matplotlib.use('TkAgg') 
        plt.show()
#%# embeddinggenerator.py
#%# embeddinggenerator.py
import sys
import numpy as np
from typing import List, Optional, Tuple, Dict, Any

from .deepsets import DeepSets
from .querying import get_embedding


class EmbeddingGenerator:
    """
    The EmbeddingGenerator class handles the logic of generating embeddings from a list of answers.
    It splits answers into chunks, obtains embeddings for each chunk, and (in a separate step) distills them into a single embedding
    using a method specified by the embeddings_args dictionary. If 'method' is 'deepsets', it initializes
    a DeepSets model using the parameters in 'deep_sets_args'.

    Methods
    -------
    generate_embeddings(answers: List[str]) -> List[List[np.ndarray]]:
        Generate embeddings for a list of answers, returning only the chunked embeddings.

    combine_embeddings(chunked_embeddings_list: List[List[np.ndarray]]) -> List[np.ndarray]:
        Combine previously generated chunked embeddings into final embeddings.
    """

    def __init__(
        self,
        embedding_model: str,
        n_words: int,
        method: str = 'max',
        deep_sets_args: Optional[Dict[str, Any]] = None,
        batch_nans: int = 1,
    ) -> None:
        """
        Initialize the EmbeddingGenerator.

        Parameters
        ----------
        embedding_model : str
            The embedding model to use.
        n_words : int
            Number of words per chunk.
        method : str, optional
            Method to combine chunk embeddings into a final embedding.
            Options are 'deepsets', 'max', 'min', 'mean', 'concat'. Default is 'mean'.
        deep_sets_args : Dict[str, Any], optional
            Arguments for initializing the DeepSets model if method='deepsets'.
        batch_nans : int, optional
            Number of answers to embed simultaneously in a single batch of get_embedding calls.
        """
        supported_methods = {'deepsets', 'max', 'min', 'mean', 'concat'}
        if method not in supported_methods:
            raise ValueError(
                f"Unsupported method '{method}'. Supported methods are: {supported_methods}"
            )

        self.embedding_model = embedding_model
        self.n_words = n_words
        self.method = method
        self.batch_nans = batch_nans

        if self.method == 'deepsets':
            if deep_sets_args is None:
                raise ValueError(
                    "deep_sets_args must be provided when method is 'deepsets'."
                )
            self.deepsets = DeepSets(**deep_sets_args)
        else:
            self.deepsets = None

    def _split_text_into_chunks(self, text: str) -> List[str]:
        """
        Split a given text into chunks of n_words.

        Parameters
        ----------
        text : str
            The text to split.

        Returns
        -------
        List[str]
            A list of text chunks, each containing up to n_words words.
        """
        words = text.split()
        chunks = [' '.join(words[i:i + self.n_words]) for i in range(0, len(words), self.n_words)]
        return chunks if chunks else ['']

    def _combine_embeddings(self, chunk_embeddings: np.ndarray) -> np.ndarray:
        """
        Combine chunk embeddings into a single embedding using the specified method.

        Parameters
        ----------
        chunk_embeddings : np.ndarray
            An array of shape (num_chunks, embedding_dim) containing chunk embeddings.

        Returns
        -------
        np.ndarray
            A single embedding vector.

        Notes
        -----
        - 'deepsets' uses a learned combination.
        - 'max', 'min', 'mean' are simple statistical combinations.
        - 'concat' concatenates all chunk embeddings in order, resulting in a vector whose length is 
          (number_of_chunks * embedding_dim).
        """
        if self.method == 'deepsets':
            return self.deepsets.forward(chunk_embeddings)
        elif self.method == 'max':
            return np.max(chunk_embeddings, axis=0)
        elif self.method == 'min':
            return np.min(chunk_embeddings, axis=0)
        elif self.method == 'mean':
            return np.mean(chunk_embeddings, axis=0)
        elif self.method == 'concat':
            return np.concatenate(chunk_embeddings, axis=0)

    def generate_embeddings(
        self, answers: List[str]
    ) -> List[List[np.ndarray]]:
        """
        Generate embeddings for a list of answers using batch processing.

        This method splits each answer into chunks of n_words. It then processes the answers in batches of size
        batch_nans. For each batch, it retrieves embeddings for all chunks from all answers in that batch, and then
        organizes them back into a list of chunk embeddings per answer.

        Parameters
        ----------
        answers : List[str]
            The list of answers to generate embeddings for.

        Returns
        -------
        List[List[np.ndarray]]
            A list containing, for each answer, a list of chunk embeddings.
        """
        chunked_embeddings_list: List[List[np.ndarray]] = []
        total_answers = len(answers)
        i = 0

        while i < total_answers:
            batch_answers = answers[i:i + self.batch_nans]

            batch_chunks = []
            chunk_counts = []
            for ans in batch_answers:
                chunks = self._split_text_into_chunks(ans)
                batch_chunks.extend(chunks)
                chunk_counts.append(len(chunks))

            if batch_chunks:
                batch_embeddings = get_embedding(batch_chunks, model=self.embedding_model)
            else:
                batch_embeddings = []

            idx = 0
            for c_count in chunk_counts:
                ans_embeds = batch_embeddings[idx:idx + c_count]
                chunked_embeddings_list.append(ans_embeds)
                idx += c_count

            i += self.batch_nans

        sys.stdout.write("\n")
        sys.stdout.flush()
        return chunked_embeddings_list

    def combine_embeddings(
        self, chunked_embeddings_list: List[List[np.ndarray]]
    ) -> List[np.ndarray]:
        """
        Combine previously generated chunked embeddings into final embeddings.

        Parameters
        ----------
        chunked_embeddings_list : List[List[np.ndarray]]
            A list of lists of chunk embeddings (output of generate_embeddings).

        Returns
        -------
        List[np.ndarray]
            A list containing the final embedding for each answer after combination.
        """
        all_final_embeddings: List[np.ndarray] = []
        total_answers = len(chunked_embeddings_list)

        for index, chunk_embeddings in enumerate(chunked_embeddings_list):
            print(
                f"Combining embeddings for answer {index + 1}/{total_answers}", 
                end='\r', 
                flush=True
            )
            chunk_embeddings_array = np.vstack(chunk_embeddings)
            final_embedding = self._combine_embeddings(chunk_embeddings_array)
            all_final_embeddings.append(final_embedding)

        sys.stdout.write("\n")
        sys.stdout.flush()
        return all_final_embeddings
#%# querying.py
from typing import Optional, List

import numpy as np
import openai

from .utils.text_processing import read_apikey


api_key: str = 'sk-proj-9gp6UXdknTMKuyTRkQ7XT3BlbkFJB2uA4ehQm5hswLZiarEo'
openai.api_key = api_key

client = openai.OpenAI(
    api_key=openai.api_key,
)


def get_embedding(text: [str, List[str]], model: str = "text-embedding-3-large") -> [np.ndarray, List[np.ndarray]]:
    """
    Generate embedding vectors for the given text(s) using the specified model.
    This function now accepts either a single string or a list of strings.

    Parameters
    ----------
    text : str or List[str]
        Input text(s) to embed.
    model : str, optional
        Embedding model to use, by default "text-embedding-3-large".

    Returns
    -------
    np.ndarray or List[np.ndarray]
        If input was a single string, returns a single np.ndarray embedding vector.
        If input was a list of strings, returns a list of np.ndarray embedding vectors.

    Raises
    ------
    ValueError
        If an unknown model is specified.
    """
    if isinstance(text, str):
        texts = [text.replace("\n", " ")]
    else:
        texts = [t.replace("\n", " ") for t in text]

    if model == "text-embedding-ada-002":
        vector_length = 1536
    elif model == "text-embedding-3-large":
        vector_length = 3072
    else:
        raise ValueError(f"Unknown model: {model}")

    try:
        embedding = client.embeddings.create(input=texts, model=model)
        embeddings = [np.array(e.embedding) for e in embedding.data]
    except:
        embeddings = [np.zeros(vector_length) for _ in texts]

    if isinstance(text, str):
        return embeddings[0]
    else:
        return embeddings


def query_gpt(
    prompt: str,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.6
) -> str:
    """
    Obtain a GPT-generated response for the given prompt.

    This function sends a prompt to the GPT model and retrieves the generated response.

    Parameters
    ----------
    prompt : str
        The input prompt to send to GPT.
    model : str, optional
        GPT model to use, by default "gpt-3.5-turbo".
    temperature : float, optional
        Sampling temperature for the response, by default 0.6.

    Returns
    -------
    str
        Generated text response from GPT.
    """
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=temperature,
    )
    response = chat_completion.choices[0].message.content
    return response
#%# text_processing.py
from typing import List
import os


def add_html_breaks_every_n_words(string_list: List[str], n: int = 10) -> List[str]:
    """
    Insert HTML break tags into strings every n words.

    This function formats each string by adding a `<br>` tag after every `n` words to enhance readability in HTML.

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
        formatted_text = '<br>'.join(' '.join(group) for group in words_with_breaks)
        new_string_list.append(formatted_text)
    return new_string_list


def read_apikey() -> str:
    """
    Retrieve the OpenAI API key from a designated file.

    This function reads the API key from a file located at '../data/openai_api_key.txt'.

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
            "The OpenAI API key file should contain only a single line with the API key."
        )
#%# __init__.py
# utils submodule __init__.py

from .dimensionality_reduction import (
    plot_reduced_dim_vector_3d
)

from .labels import (
    get_index,
    get_name,
    get_suspect_counts,
    label_pooled_questions
)
from .text_processing import (
    add_html_breaks_every_n_words,
    read_apikey
)

__all__ = [
    "plot_reduced_dim_vector_3d",
    "get_index",
    "get_name",
    "get_suspect_counts",
    "label_pooled_questions",
    "add_html_breaks_every_n_words",
    "read_apikey"
]
#%# labels.py
from typing import Any, List, Tuple, Dict
import pandas as pd


def get_index(
    lastname: str,
    final_answers_comb: pd.DataFrame,
    question_id: int = 14
) -> int:
    """
    Find the index of a student by last name for a specific question.

    This function searches for the student with the given last name within the specified question's data.

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
    for i, name in enumerate(
        final_answers_comb[final_answers_comb["Q #"] == question_id].LastName
    ):
        if name == lastname:
            return i
    raise ValueError(f"Lastname '{lastname}' not found for question {question_id}.")


def get_name(index: int, final_answers_comb: pd.DataFrame) -> str:
    """
    Retrieve the full name of a student given their index.

    This function combines the first and last names of a student based on their index in the DataFrame.

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
    name_list = (
        final_answers_comb[["FirstName", "LastName"]]
        [final_answers_comb["Q #"] == final_answers_comb["Q #"].unique()[0]]
        .iloc[index]
        .tolist()
    )
    return f"{name_list[0]} {name_list[1]}"


def get_suspect_counts(suspect_matches: List[Tuple[int, Any]]) -> Dict[Any, int]:
    """
    Count the number of occurrences for each suspect in the match list.

    This function tallies how many times each suspect appears in the list of matches.

    Parameters
    ----------
    suspect_matches : List[Tuple[int, Any]]
        List of tuples containing suspect indices and associated values.

    Returns
    -------
    Dict[Any, int]
        Dictionary mapping suspect IDs to their count of occurrences.
    """
    suspects_dict: Dict[Any, int] = {}
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

    This function modifies the DataFrame to ensure each question has a unique identifier by appending a suffix when duplicates are detected.

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
                val: f"{label}.{i + 1}" if isinstance(label, str) else label + (0.1 * (i + 1))
                for i, val in enumerate(unique_questions)
            }
            for question, new_label in new_labels.items():
                mask = (result_df[question_label] == label) & (
                    result_df[question_col_name] == question
                )
                result_df.loc[mask, question_label] = new_label

    return result_df
#%# dimensionality_reduction.py
from typing import List, Optional
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import plotly.graph_objects as go


def reduce_dimensions(
    vector_list: List[np.ndarray],
    method: str = 'pca',
    n_components: int = 3
) -> np.ndarray:
    """
    Reduce the dimensionality of vectors using a specified method.

    This function applies PCA, t-SNE, or UMAP to project high-dimensional data into a lower-dimensional space.

    Parameters
    ----------
    vector_list : List[np.ndarray]
        List of data vectors to reduce.
    method : str, optional
        Dimensionality reduction technique ('pca', 'tsne', 'umap'), by default 'pca'.
    n_components : int, optional
        Target number of dimensions, by default 3.

    Returns
    -------
    np.ndarray
        Reduced dimension representation of the input vectors.

    Raises
    ------
    ValueError
        If an invalid reduction method is specified.
    """
    X = np.vstack(vector_list)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, learning_rate='auto', init='random')
    elif method == 'umap':
        reducer = UMAP(n_components=n_components)
    else:
        raise ValueError("Method must be 'pca', 'tsne', or 'umap'.")

    X_reduced = reducer.fit_transform(X_scaled)
    return X_reduced


def plot_reduced_dim_vector_3d(
    X_reduced: np.ndarray,
    components: List[int] = [0, 1, 2],
    student_answers: Optional[List[str]] = None,
    comparison_reduced: Optional[np.ndarray] = None,
    comparison_answers: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    method: str = 'umap'
) -> go.Figure:
    """
    Create an interactive 3D scatter plot of reduced-dimensional vectors.

    This function visualizes the reduced vectors using Plotly, differentiating between 
    student and comparison (e.g., GPT) data. The plot is customized by removing the colorbar, 
    removing the legend, hiding axis tick labels, and ensuring the aspect ratio is a perfect cube.

    Parameters
    ----------
    X_reduced : np.ndarray
        Reduced dimensional data for students.
    components : List[int], optional
        Indices of the components to plot, by default [0, 1, 2].
    student_answers : Optional[List[str]], optional
        Text data for hover information, by default None.
    comparison_reduced : Optional[np.ndarray], optional
        Reduced dimensional data for comparison (e.g., GPT answers), by default None.
    comparison_answers : Optional[List[str]], optional
        Text data for comparison hover information, by default None.
    save_path : Optional[str], optional
        File path to save the plot, by default None.
    method : str, optional
        Dimensionality reduction method used ('pca', 'tsne', 'umap'), by default 'pca'.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive 3D scatter plot with the specified customizations.
    """
    method = method.lower()
    if method == 'pca':
        x_label, y_label, z_label = "PCA 1", "PCA 2", "PCA 3"
    elif method == 'tsne':
        x_label, y_label, z_label = "TSNE 1", "TSNE 2", "TSNE 3"
    elif method == 'umap':
        x_label, y_label, z_label = "UMAP 1", "UMAP 2", "UMAP 3"
    else:
        x_label, y_label, z_label = "Component 1", "Component 2", "Component 3"

    hover_texts = (
        [f"Student {i}: {ans}" for i, ans in enumerate(student_answers)]
        if student_answers
        else None
    )

    student_trace = go.Scatter3d(
        x=X_reduced[:, components[0]],
        y=X_reduced[:, components[1]],
        z=X_reduced[:, components[2]],
        mode='markers',
        marker=dict(
            size=3,
            color=X_reduced[:, components[0]],
            colorscale='Viridis',
            opacity=0.8,
            showscale=False,  
        ),
        text=hover_texts,
        hoverinfo='text',
        showlegend=False 
    )

    data = [student_trace]

    if comparison_reduced is not None:
        comparison_hover_texts = (
            [f"GPT Answer {i}: {ans}" for i, ans in enumerate(comparison_answers)]
            if comparison_answers
            else None
        )

        gpt_trace = go.Scatter3d(
            x=comparison_reduced[:, components[0]],
            y=comparison_reduced[:, components[1]],
            z=comparison_reduced[:, components[2]],
            mode='markers',
            marker=dict(size=3, color='black', opacity=0.8),
            text=comparison_hover_texts,
            hoverinfo='text',
            showlegend=False  
        )
        data.append(gpt_trace)

    fig = go.Figure(data=data)

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title=x_label,
                showticklabels=False 
            ),
            yaxis=dict(
                title=y_label,
                showticklabels=False  
            ),
            zaxis=dict(
                title=z_label,
                showticklabels=False  
            ),
            aspectmode='cube' 
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=800,
        showlegend=False 
    )

    if save_path:
        fig.write_html(save_path)

    return fig

