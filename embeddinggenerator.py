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
