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
import networkx as nx
import plotly.graph_objects as go

from .analysis import (
    remove_duplicate_rows_by_col,
    calculate_distance_matrix,
    compute_similarity_matrix,
    extract_pairs_above_threshold,
    label_pooled_questions,
    compare_all_embedding_pairs,
    plot_similarity_histogram,
    pagerank
)
from .utils.labels import get_name, get_suspect_counts
from .utils.text_processing import add_html_breaks_every_n_words
from .utils.plotting import plot_network, plot_scatter_3d
from .querying import get_embedding, query_gpt
from .embeddinggenerator import EmbeddingGenerator


class Zapper:
    """
    The Zapper class orchestrates the loading and preprocessing of data, generation of student
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
        Initialize Zapper with data paths and model configurations.

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
        reduction_method : str
            Dimensionality reduction method.
        embeddings_args : Dict[str, Any]
            Dictionary that contains embedding arguments including 'method' and
            'deep_sets_args'.
        ignore_pooled : bool
            If True, questions originating from question pools (with subnumbered
            questions) are ignored and removed.
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
            self.embeddings_args['deep_sets_args']['input_size'] = (
                self.embedding_size
            )

        self.df: Optional[pd.DataFrame] = None
        self.student_answers: Optional[Dict[Any, List[str]]] = None
        self.student_answer_embeddings_list: Optional[Dict[Any,
                                              List[List[np.ndarray]]]] = None
        self.student_answer_embeddings: Optional[Dict[Any,
                                          List[np.ndarray]]] = None
        self.gpt_answers: Optional[Dict[Any, List[str]]] = None
        self.gpt_answer_embeddings_list: Optional[Dict[Any,
                                           List[List[np.ndarray]]]] = None
        self.gpt_answer_embeddings: Optional[Dict[Any,
                                      List[np.ndarray]]] = None
        self.reduced_student_embeddings: Optional[Dict[Any,
                                           np.ndarray]] = None
        self.reduced_gpt_embeddings: Optional[Dict[Any,
                                      np.ndarray]] = None
        self.scalers: Optional[Dict[Any, StandardScaler]] = None
        self.reducers: Optional[Dict[Any, Any]] = None
        self.plots: Optional[List[Any]] = None
        self.distance_matrices: Optional[Dict[Any, np.ndarray]] = None
        self.similarity_df: Optional[pd.DataFrame] = None
        self.similarity_df_id: Optional[pd.DataFrame] = None
        self.similarity_df_name: Optional[pd.DataFrame] = None
        self.suspect_names: Optional[List[Tuple[str, str, float]]] = None

        self._read_and_clean()

    def save(self, filename: Optional[str] = None,
             save_dir: Optional[str] = None) -> None:
        """
        Serialize the Zapper instance to a pickle file.

        Parameters
        ----------
        filename : Optional[str]
            Desired filename for the pickle file.
        save_dir : Optional[str]
            Directory path to save the file.
        """
        current_time = time.strftime("%d-%m-%Y_%H-%M-%S", time.localtime())
        filename = filename or f'zapper_instance_{current_time}.pkl'
        if save_dir:
            filename = os.path.join(save_dir, filename)
        with open(filename, 'wb') as file_handle:
            pickle.dump(self, file_handle)
        print(f"Zapper instance saved to {filename}")

    @classmethod
    def load(cls, filepath: str) -> 'Zapper':
        """
        Deserialize a Zapper instance from a pickle file.

        Parameters
        ----------
        filepath : str
            Path to the pickle file.

        Returns
        -------
        Zapper
            The loaded Zapper instance.

        Raises
        ------
        ValueError
            If the loaded object is not a Zapper instance.
        """
        with open(filepath, 'rb') as file_handle:
            instance = pickle.load(file_handle)
        if not isinstance(instance, cls):
            raise ValueError("The loaded object is not an instance of Zapper.")
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
            raise ValueError(f"Unsupported embedding model: "
                             f"{self.embedding_model}")

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

        self.df = self.df.sort_values(
            by=['Org Defined ID', 'Q #'], ascending=[True, True]
        )
        self._anonymize_map = self._build_anonymize_map()

    def _build_anonymize_map(self) -> Dict[str, str]:
        """
        Build a dictionary mapping full student names to anonymized labels.

        The method collects all unique 'FirstName LastName' from self.df,
        sorts them, and assigns each name a unique 'Student X' label. 
        This ensures consistent anonymization across different plots.
        """
        full_names = (self.df['FirstName'] + ' ' + self.df['LastName']
                      ).tolist()
        unique_names = sorted(set(full_names))
        return {name: f"Student {i}" for i, name in enumerate(unique_names)}

    def _preprocess_answers_questions(self) -> None:
        """
        Clean and organize answers and questions from the dataset.
        """
        self.df['Answer'] = self.df['Answer'].apply(
            lambda x: '' if isinstance(x, float) and math.isnan(x) else x
        )
        self.student_answers = {
            q_num: self.df[self.df['Q #'] == q_num].Answer.tolist()
            for q_num in self.df['Q #'].unique()
        }
        self.question_ids = self.df['Q #'].unique()
        self.question_stems = [
            text.replace('\xa0', ' ') for text in
            self.df['Q Text'].unique().tolist()
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
        self.student_answer_embeddings_list = {}
        for question_id in self.question_ids:
            answers = self.student_answers[question_id]
            print(
                f"Generating multiple student embeddings: Question "
                f"{question_id}",
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
        self.student_answer_embeddings = {}
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
        self.gpt_answers = {}
        for i, question_stem in enumerate(self.question_stems):
            answers = []
            for j in range(self.n_gpt_answers):
                print(
                    f"Generating GPT answers: Question {i + 1}/"
                    f"{len(self.question_ids)} Answer {j + 1}/"
                    f"{self.n_gpt_answers}",
                    end='\r',
                    flush=True
                )
                answer = query_gpt(
                    prompt=question_stem, model=self.gpt_model,
                    temperature=temperature
                )
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
        self.gpt_answer_embeddings_list = {}
        for question_id in self.question_ids:
            answers = self.gpt_answers[question_id]
            sys.stdout.write(
                f"\rGenerating multiple GPT embeddings: Question "
                f"{question_id}"
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
        self.gpt_answer_embeddings = {}
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
            Dimensionality reduction method ('pca', 'tsne', 'umap'), 
            by default 'umap'.

        Returns
        -------
        np.ndarray
            Reduced dimension vectors.
        """
        if len(vector_list) < n_components:
            raise ValueError(
                f"Not enough vectors ({len(vector_list)}) for "
                f"{n_components} components."
            )
        X = np.vstack(vector_list)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if method == 'pca':
            reducer = PCA(n_components=n_components)
        elif method == 'tsne':
            reducer = TSNE(
                n_components=n_components, learning_rate='auto', init='random'
            )
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
            Dimensionality reduction method ('pca', 'tsne', 'umap'), 
            by default 'umap'.
        """
        method = method if method is not None else self.reduction_method
        self.reduced_student_embeddings = {}
        self.reduced_gpt_embeddings = {}
        self.scalers = {}
        self.reducers = {}
        for question_id in self.question_ids:
            student_vectors = self.student_answer_embeddings[question_id]
            self.scalers[question_id] = StandardScaler()
            X_scaled = self.scalers[question_id].fit_transform(
                np.vstack(student_vectors)
            )
            if method == 'pca':
                reducer = PCA(n_components=n_components)
            elif method == 'tsne':
                reducer = TSNE(
                    n_components=n_components,
                    learning_rate='auto',
                    init='random'
                )
            elif method == 'umap':
                reducer = UMAP(n_components=n_components)
            else:
                raise ValueError("Method must be 'pca', 'tsne', or 'umap'.")
            X_reduced = reducer.fit_transform(X_scaled)
            self.reducers[question_id] = reducer
            self.reduced_student_embeddings[question_id] = X_reduced

            gpt_vectors = self.gpt_answer_embeddings[question_id]
            X_scaled_gpt = self.scalers[question_id].transform(
                np.vstack(gpt_vectors)
            )
            Y_reduced = self.reducers[question_id].transform(X_scaled_gpt)
            self.reduced_gpt_embeddings[question_id] = Y_reduced

    def _generate_distance_matrices(self) -> None:
        """
        Compute distance matrices between student and GPT embeddings
        for each question.
        """
        self.distance_matrices = {}
        for question_id in self.question_ids:
            student_embeds = self.student_answer_embeddings[question_id]
            gpt_embeds = self.gpt_answer_embeddings[question_id]
            distance_matrix = calculate_distance_matrix(student_embeds,
                                                        gpt_embeds)
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

    def _compare_student_pairs(
        self,
        metric: str = 'euclidean',
        n_top: int = 20,
        similarity_percentile: float = 95.0,
        save_dir: Optional[str] = None
    ) -> None:
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
        """
        sum_similarity_matrix, threshold, high_similarity_pairs = \
            compare_all_embedding_pairs(
                embedding_dict=self.student_answer_embeddings,
                metric=metric,
                n_top=n_top,
                percentile=similarity_percentile
            )
        self.sum_similarity_matrix = sum_similarity_matrix
        plot_similarity_histogram(
            sum_similarity_matrix=sum_similarity_matrix,
            threshold_percentile=similarity_percentile,
            bins=100,
            save_dir=save_dir
        )
        similarity_df_id = self.construct_similarity_dataframe(
            high_similarity_pairs
        )
        similarity_list_name = self.report_suspect_names(high_similarity_pairs)
        similarity_df_name = pd.DataFrame(
            similarity_list_name,
            index=None,
            columns=['student_name1', 'student_name2', 'similarity_score']
        )
        self.similarity_df_id = similarity_df_id
        self.suspect_names = similarity_list_name
        self.similarity_df_name = similarity_df_name
        if save_dir:
            df_id_path = os.path.join(save_dir, 'similarity_scores_id.csv')
            similarity_df_id.to_csv(df_id_path, index=False)
            df_name_path = os.path.join(save_dir, 'similarity_scores_name.csv')
            similarity_df_name.to_csv(df_name_path, index=False)
            print(f"Similarity scores saved to {df_id_path} and {df_name_path}")

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
        suspect_names = []
        for i, j, score in suspect_pairs:
            name1 = get_name(i, self.df)
            name2 = get_name(j, self.df)
            suspect_names.append((name1, name2, score))
            print(f"[{name1}, {name2}] Similarity Score: "
                  f"{round(score, 5)}")
        return suspect_names

    def _student_similarity_network(
        self,
        max_rows: int = 50,
        output_html: Optional[str] = "student_similarity_network.html",
        seed: int = 42,
        edge_color: str = 'rgba(125,125,125,0.7)',
        edge_width: int = 2,
        node_symbol: str = 'circle',
        node_size: int = 8,
        node_color: str = 'rgb(255, 0, 0)',
        node_line_color: str = 'rgb(50,50,50)',
        node_line_width: float = 0.5,
        showlegend: bool = False,
        hovermode: str = 'closest',
        scene_showbackground: bool = False,
        scene_showticklabels: bool = False,
        scene_title: str = '',
        margin_l: int = 40,
        margin_r: int = 40,
        margin_b: int = 85,
        margin_t: int = 100,
        anonymize_names: bool = False
    ) -> None:
        """
        Plot a 3D interactive graph from the similarity_df_name attribute and save
        it to an HTML file.

        Parameters
        ----------
        max_rows : int
            Maximum number of rows to read from the top of the DataFrame.
        output_html : Optional[str]
            Path to save the output HTML file.
        seed : int
            Seed for the spring layout algorithm.
        edge_color : str
            Color of the edges in the graph.
        edge_width : int
            Width of the edges in the graph.
        node_symbol : str
            Symbol used for the nodes in the graph.
        node_size : int
            Size of the nodes in the graph.
        node_color : str
            Color of the nodes in the graph.
        node_line_color : str
            Color of the node borders.
        node_line_width : float
            Width of the node borders.
        showlegend : bool
            Whether to display the legend.
        hovermode : str
            Mode for hover interactions.
        scene_showbackground : bool
            Whether to show the background in the 3D scene.
        scene_showticklabels : bool
            Whether to show tick labels in the 3D scene axes.
        scene_title : str
            Title for the 3D scene axes.
        margin_l : int
            Left margin of the plot.
        margin_r : int
            Right margin of the plot.
        margin_b : int
            Bottom margin of the plot.
        margin_t : int
            Top margin of the plot.
        anonymize_names : bool
            If True, replaces node names with numeric identifiers.
        """
        if self.similarity_df_name is None or self.similarity_df_name.empty:
            return
        df = self.similarity_df_name.head(max_rows)
        if df.shape[1] < 2:
            return
        edges = list(df.iloc[:, :2].dropna().itertuples(index=False, name=None))
        if not edges:
            return
        plot_network(
            edges=edges,
            seed=seed,
            edge_color=edge_color,
            edge_width=edge_width,
            node_symbol=node_symbol,
            node_size=node_size,
            node_color=node_color,
            node_line_color=node_line_color,
            node_line_width=node_line_width,
            showlegend=showlegend,
            hovermode=hovermode,
            scene_showbackground=scene_showbackground,
            scene_showticklabels=scene_showticklabels,
            scene_title=scene_title,
            margin_l=margin_l,
            margin_r=margin_r,
            margin_b=margin_b,
            margin_t=margin_t,
            output_html=output_html,
            anonymize_names=anonymize_names,
            name_map=self._anonymize_map
        )

    def _run_pagerank(
        self,
        power: float = 20,
        alpha: float = 0.90,
        max_iter: int = 200,
        tol: float = 1.0e-6,
        personalization: Optional[Dict[int, float]] = None,
        csv_filename: str = "pagerank_scores.csv",
        hist_filename: str = "pagerank_hist.png",
        bins: int = 50,
        save_dir: Optional[str] = None
    ) -> None:
        """
        Apply the PageRank algorithm to the summed similarity matrix and
        save the resulting scores and their histogram.

        This method uses the summed similarity matrix as an adjacency matrix
        for PageRank. It saves the scores to a CSV file and plots a histogram
        with the specified number of bins.

        Parameters
        ----------
        alpha : float
            Damping parameter for PageRank.
        max_iter : int
            Maximum number of iterations in the power method.
        tol : float
            Error tolerance to check convergence.
        personalization : Optional[Dict[int, float]]
            Optional dictionary of personalized PageRank values.
        csv_filename : str
            Filename for saving the PageRank scores as CSV.
        hist_filename : str
            Filename for saving the histogram of scores.
        bins : int
            Number of bins in the histogram.
        save_dir : Optional[str]
            Directory where CSV and histogram files are saved.
        """
        if not hasattr(self, 'sum_similarity_matrix'):
            raise ValueError("Summed similarity matrix not found. "
                             "Run the comparison first.")

        scores_dict = pagerank(
            adjacency_matrix=self.sum_similarity_matrix**power,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            personalization=personalization
        )
        sorted_scores = sorted(scores_dict.items(),
                               key=lambda x: x[1], reverse=True)
        names_scores = []
        for idx, score in sorted_scores:
            student_name = get_name(idx, self.df)
            names_scores.append((student_name, score))
        df = pd.DataFrame(names_scores, columns=["Student_Name", "PageRank_Score"])

        final_csv_path = csv_filename
        final_hist_path = hist_filename
        if save_dir:
            final_csv_path = os.path.join(save_dir, csv_filename)
            final_hist_path = os.path.join(save_dir, hist_filename)

        df.to_csv(final_csv_path, index=False)
        plt.figure()
        plt.hist(df["PageRank_Score"], bins=bins)
        plt.savefig(final_hist_path)
        plt.close()

    def _generate_plots(
        self,
        components: List[int] = [0, 1, 2],
        save_dir: Optional[str] = None,
        method: Optional[str] = None,
        student_marker_size: int = 3,
        student_colorscale: str = 'Viridis',
        student_opacity: float = 0.8,
        student_show_scale: bool = False,
        gpt_marker_size: int = 3,
        gpt_color: str = 'black',
        gpt_opacity: float = 0.8,
        layout_height: int = 800,
        anonymize_names: bool = False,
        hide_answers: str = 'none',
        partial_answer_length: int = 10
    ) -> None:
        """
        Create 3D scatter plots of the reduced embeddings and a similarity network.

        Parameters
        ----------
        components : List[int], optional
            Indices of the components to plot, by default [0, 1, 2].
        save_dir : Optional[str], optional
            Directory to save the generated plots, by default None.
        method : Optional[str], optional
            Method used for dimensionality reduction, by default None.
        student_marker_size : int, optional
            Size of the student markers in the plot, by default 3.
        student_colorscale : str, optional
            Colorscale for the student markers, by default 'Viridis'.
        student_opacity : float, optional
            Opacity of the student markers, by default 0.8.
        student_show_scale : bool, optional
            Whether to show the color scale for student markers, by default False.
        gpt_marker_size : int, optional
            Size of the GPT markers in the plot, by default 3.
        gpt_color : str, optional
            Color of the GPT markers, by default 'black'.
        gpt_opacity : float, optional
            Opacity of the GPT markers, by default 0.8.
        layout_height : int, optional
            Height of the plot in pixels, by default 800.
        anonymize_names : bool, optional
            If True, replaces student names with numerical identifiers.
        hide_answers : {'none', 'partial', 'full'}, optional
            Determines how much of the student answers to display.
        partial_answer_length : int, optional
            Number of words to display when hide_answers is 'partial'.
        """
        method = method if method is not None else self.reduction_method
        self.plots = []
        for question_id in self.question_ids:
            X_reduced = self.reduced_student_embeddings[question_id]
            Y_reduced = self.reduced_gpt_embeddings[question_id]
            student_answers = self.student_answers[question_id]
            gpt_answers = self.gpt_answers[question_id]
            question_df = self.df[self.df['Q #'] == question_id]
            full_names = [
                f"{row['FirstName']} {row['LastName']}"
                for _, row in question_df.iterrows()
            ]
            wrapped_student_answers = []
            for name, ans in zip(full_names, student_answers):
                wrapped = textwrap.wrap(ans, width=79,
                                        break_long_words=False)
                final_text = f"{name}:<br>{'<br>'.join(wrapped)}"
                wrapped_student_answers.append(final_text)
            wrapped_gpt_answers = []
            for ans in gpt_answers:
                wrapped = textwrap.wrap(ans, width=79,
                                        break_long_words=False)
                final_text = f"GPT Answer:<br>{'<br>'.join(wrapped)}"
                wrapped_gpt_answers.append(final_text)
            fig = plot_scatter_3d(
                X_reduced,
                components=components,
                student_answers=wrapped_student_answers,
                comparison_points=Y_reduced,
                comparison_answers=wrapped_gpt_answers,
                method=method,
                student_marker_size=student_marker_size,
                student_colorscale=student_colorscale,
                student_opacity=student_opacity,
                student_show_scale=student_show_scale,
                gpt_marker_size=gpt_marker_size,
                gpt_color=gpt_color,
                gpt_opacity=gpt_opacity,
                layout_height=layout_height,
                anonymize_names=anonymize_names,
                hide_answers=hide_answers,
                partial_answer_length=partial_answer_length,
                name_map=self._anonymize_map
            )
            self.plots.append(fig)
            if save_dir:
                plot_path = os.path.join(
                    save_dir, f'plot_question_{question_id}.html'
                )
                fig.write_html(plot_path)
                print(f"Plot saved to {plot_path}")
        self._student_similarity_network(anonymize_names=anonymize_names)
        self._run_pagerank()

    def full_process(
        self,
        temperature: float = 0.6,
        n_components: int = 10,
        method: str = 'umap',
        save_dir: Optional[str] = None,
        metric: str = 'euclidean',
        n_top: int = 20,
        similarity_percentile: float = 95.0
    ) -> None:
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
        self._generate_distance_matrices()
        self._compare_student_pairs(
            metric=metric,
            n_top=n_top,
            similarity_percentile=similarity_percentile,
            save_dir=save_dir
        )
        self._generate_plots(save_dir=save_dir)
        print("Full processing complete.")
