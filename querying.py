from typing import Optional, List

import numpy as np
import openai

from .utils.text_processing import read_apikey


api_key: str = ''
openai.api_key = api_key

client = openai.OpenAI(
    api_key=openai.api_key,
)


def get_embedding(text: Union[str, List[str]],
                  model: str = "text-embedding-3-large"
                 ) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Generate embedding vectors for the given text(s) using the specified
    model.
    
    Parameters
    ----------
    text : str or List[str]
        Input text(s) to embed.
    model : str, optional
        Embedding model to use, by default "text-embedding-3-large".
    
    Returns
    -------
    np.ndarray or List[np.ndarray]
        If input was a single string, returns a single np.ndarray embedding
        vector.
        If input was a list of strings, returns a list of np.ndarray embedding
        vectors.
    
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
