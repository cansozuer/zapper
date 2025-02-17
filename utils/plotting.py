import os
import textwrap
from typing import List, Optional, Tuple, Dict

import numpy as np
import networkx as nx
import plotly.graph_objects as go


def plot_scatter_3d(
    data_points: np.ndarray,
    components: List[int] = [0, 1, 2],
    student_answers: Optional[List[str]] = None,
    comparison_points: Optional[np.ndarray] = None,
    comparison_answers: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    method: str = 'umap',
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
    partial_answer_length: int = 10,
    name_map: Optional[Dict[str, str]] = None
) -> go.Figure:
    """
    Create an interactive 3D scatter plot of vectors.

    This function visualizes vectors using Plotly, differentiating between
    student and comparison (e.g., GPT) data. The plot is customized by
    removing the colorbar, removing the legend, hiding axis tick labels, and
    ensuring the aspect ratio is a perfect cube.

    Parameters
    ----------
    data_points : np.ndarray
        Vector data for students.
    components : List[int], optional
        Indices of the components to plot, by default [0, 1, 2].
    student_answers : Optional[List[str]], optional
        Text data for hover information, by default None.
    comparison_points : Optional[np.ndarray], optional
        Vector data for comparison (e.g., GPT answers), by default None.
    comparison_answers : Optional[List[str]], optional
        Text data for comparison hover information, by default None.
    save_path : Optional[str], optional
        File path to save the plot, by default None.
    method : str, optional
        Method used for data preparation, by default 'umap'.
    student_marker_size : int, optional
        Size of the student markers in the plot, by default 3.
    student_colorscale : str, optional
        Colorscale for the student markers, by default 'Viridis'.
    student_opacity : float, optional
        Opacity of the student markers, by default 0.8.
    student_show_scale : bool, optional
        Whether to show the color scale for student markers,
        by default False.
    gpt_marker_size : int, optional
        Size of the GPT markers in the plot, by default 3.
    gpt_color : str, optional
        Color of the GPT markers, by default 'black'.
    gpt_opacity : float, optional
        Opacity of the GPT markers, by default 0.8.
    layout_height : int, optional
        Height of the plot in pixels, by default 800.
    anonymize_names : bool, optional
        If True, replaces original names with enumerated identifiers.
    hide_answers : {'none', 'partial', 'full'}, optional
        Determines how much of the student answers to display.
    partial_answer_length : int, optional
        Number of words to display when hide_answers is 'partial'.
    name_map : Optional[Dict[str, str]], optional
        A dictionary mapping original names to anonymized labels. If provided
        and anonymize_names is True, names will be replaced accordingly.

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
        x_label, y_label, z_label = ("Component 1", "Component 2",
                                     "Component 3")

    if student_answers is not None:
        processed_student_answers = []
        for ans in student_answers:
            original_label = ans.split(':', 1)[0] if ':' in ans else ''
            real_ans = ans.split(':', 1)[1] if ':' in ans else ans
            if anonymize_names and name_map is not None:
                label = name_map.get(original_label, original_label)
            else:
                label = original_label
            if hide_answers == 'none':
                final_ans = real_ans
            elif hide_answers == 'partial':
                final_ans = ' '.join(real_ans.split()[
                    :partial_answer_length])
            else:
                final_ans = ''
            processed_student_answers.append(
                f"{label}: {final_ans}" if final_ans else label
            )
        hover_texts = processed_student_answers
    else:
        hover_texts = None

    student_trace = go.Scatter3d(
        x=data_points[:, components[0]],
        y=data_points[:, components[1]],
        z=data_points[:, components[2]],
        mode='markers',
        marker=dict(
            size=student_marker_size,
            color=data_points[:, components[0]],
            colorscale=student_colorscale,
            opacity=student_opacity,
            showscale=student_show_scale,
        ),
        text=hover_texts,
        hoverinfo='text',
        showlegend=False
    )

    data = [student_trace]

    if comparison_points is not None:
        comparison_hover_texts = (
            [f"GPT Answer {i}: {ans}" for i, ans in 
             enumerate(comparison_answers)]
            if comparison_answers
            else None
        )
        gpt_trace = go.Scatter3d(
            x=comparison_points[:, components[0]],
            y=comparison_points[:, components[1]],
            z=comparison_points[:, components[2]],
            mode='markers',
            marker=dict(
                size=gpt_marker_size,
                color=gpt_color,
                opacity=gpt_opacity
            ),
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
        height=layout_height,
        showlegend=False
    )

    if save_path:
        fig.write_html(save_path)

    return fig


def plot_network(
    edges: List[Tuple[str, str]],
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
    output_html: Optional[str] = None,
    anonymize_names: bool = False,
    name_map: Optional[Dict[str, str]] = None
) -> None:
    """
    Create and display a 3D interactive network graph using Plotly.

    This function constructs a network graph from the provided edges,
    computes a 3D spring layout, and visualizes the graph with customizable
    aesthetics. The resulting plot is displayed interactively and can
    optionally be saved to an HTML file.

    Parameters
    ----------
    edges : List[Tuple[str, str]]
        List of edge tuples representing connections between nodes.
    seed : int, optional
        Seed for the spring layout algorithm, by default 42.
    edge_color : str, optional
        Color of the edges in the graph, by default
        'rgba(125,125,125,0.7)'.
    edge_width : int, optional
        Width of the edges in the graph, by default 2.
    node_symbol : str, optional
        Symbol used for the nodes in the graph, by default 'circle'.
    node_size : int, optional
        Size of the nodes in the graph, by default 8.
    node_color : str, optional
        Color of the nodes in the graph, by default 'rgb(255, 0, 0)'.
    node_line_color : str, optional
        Color of the node borders, by default 'rgb(50,50,50)'.
    node_line_width : float, optional
        Width of the node borders, by default 0.5.
    showlegend : bool, optional
        Whether to display the legend, by default False.
    hovermode : str, optional
        Mode for hover interactions, by default 'closest'.
    scene_showbackground : bool, optional
        Whether to show the background in the 3D scene, by default False.
    scene_showticklabels : bool, optional
        Whether to show tick labels in the 3D scene axes,
        by default False.
    scene_title : str, optional
        Title for the 3D scene axes, by default ''.
    margin_l : int, optional
        Left margin of the plot, by default 40.
    margin_r : int, optional
        Right margin of the plot, by default 40.
    margin_b : int, optional
        Bottom margin of the plot, by default 85.
    margin_t : int, optional
        Top margin of the plot, by default 100.
    output_html : Optional[str], optional
        Path to save the output HTML file, by default None.
    anonymize_names : bool, optional
        If True, replaces node names with enumerated identifiers.
    name_map : Optional[Dict[str, str]], optional
        A dictionary mapping original names to anonymized labels. If provided
        and anonymize_names is True, node labels will be replaced accordingly.

    Side Effects
    ------------
    Displays an interactive plot.
    """
    G = nx.Graph()
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, dim=3, seed=seed)
    x_nodes = [pos[node][0] for node in G.nodes()]
    y_nodes = [pos[node][1] for node in G.nodes()]
    z_nodes = [pos[node][2] for node in G.nodes()]
    edge_x = []
    edge_y = []
    edge_z = []
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode='lines',
        line=dict(color=edge_color, width=edge_width),
        hoverinfo='none'
    )
    node_labels = list(G.nodes())
    if anonymize_names and name_map is not None:
        node_labels = [name_map.get(label, label) for label in node_labels]
    node_trace = go.Scatter3d(
        x=x_nodes,
        y=y_nodes,
        z=z_nodes,
        mode='markers+text',
        marker=dict(
            symbol=node_symbol,
            size=node_size,
            color=node_color,
            line=dict(color=node_line_color, width=node_line_width)
        ),
        text=node_labels,
        textposition="top center",
        hoverinfo='text'
    )
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=showlegend,
            hovermode=hovermode,
            scene=dict(
                xaxis=dict(
                    showbackground=scene_showbackground,
                    showticklabels=scene_showticklabels,
                    title=scene_title
                ),
                yaxis=dict(
                    showbackground=scene_showbackground,
                    showticklabels=scene_showticklabels,
                    title=scene_title
                ),
                zaxis=dict(
                    showbackground=scene_showbackground,
                    showticklabels=scene_showticklabels,
                    title=scene_title
                )
            ),
            margin=dict(l=margin_l, r=margin_r, b=margin_b, t=margin_t)
        )
    )
    fig.show()
    try:
        if output_html:
            dir_name = os.path.dirname(output_html)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            fig.write_html(output_html)
    except Exception:
        pass
