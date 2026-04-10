"""Build and visualize an initialized ARG graph without GAT training.

Example:
    python graph/visualize_arg_graph.py \
        --dataset I \
        --focus_class "golden retriever" \
        --semantic_mode clip \
        --output_png graph/arg_imagenet_preview.png
"""

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import networkx as nx
import torch

CURRENT_DIR = Path(__file__).resolve().parent
TCA_ROOT = CURRENT_DIR.parent
if str(TCA_ROOT) not in sys.path:
    sys.path.insert(0, str(TCA_ROOT))

from graph.utils import (
    build_class_membership_edges,
    build_semantic_edges,
    combine_edges,
    encode_attributes_with_clip,
    get_unique_attributes,
    load_attributes,
)


ATTRIBUTES_PATH = {
    "Flower102": "Attributes/OxfordFlowers_2_V.json",
    "Food101": "Attributes/Food101_2_V.json",
    "DTD": "Attributes/DescribableTextures_2_V.json",
    "OxfordPets": "Attributes/OxfordPets_2_V.json",
    "Caltech101": "Attributes/Caltech101_2_V.json",
    "SUN397": "Attributes/SUN397_2_V.json",
    "UCF101": "Attributes/UCF101_2_V.json",
    "StanfordCars": "Attributes/StanfordCars_2_V.json",
    "Aircraft": "Attributes/FGVCAircraft_2_V.json",
    "EuroSAT": "Attributes/EuroSAT_2_V.json",
    "I": "Attributes/ImageNet_2_V.json",
}


def _pick_focus_class(attributes, focus_class):
    if focus_class is not None:
        if focus_class not in attributes:
            raise ValueError(f"focus_class '{focus_class}' not found in attributes file")
        return focus_class
    return sorted(attributes.keys())[0]


def _build_local_attribute_view(attributes, focus_class, num_attributes, neighbor_classes):
    focus_attrs = attributes[focus_class][:num_attributes]
    focus_set = set(focus_attrs)

    scored_neighbors = []
    for class_name, attrs in attributes.items():
        selected = attrs[:num_attributes]
        overlap = len(set(selected).intersection(focus_set))
        if class_name == focus_class:
            continue
        if overlap > 0:
            scored_neighbors.append((overlap, class_name))

    scored_neighbors.sort(reverse=True)
    kept_neighbors = [name for _, name in scored_neighbors[:neighbor_classes]]

    subset = {focus_class: attributes[focus_class]}
    for class_name in kept_neighbors:
        subset[class_name] = attributes[class_name]

    return subset, focus_attrs


def _to_undirected_edge_set(edge_index):
    edges = set()
    if edge_index.numel() == 0:
        return edges
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for u, v in zip(src, dst):
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        edges.add((a, b))
    return edges


def build_graph(args):
    device = torch.device("cpu")
    attributes = load_attributes(ATTRIBUTES_PATH[args.dataset])
    focus_class = _pick_focus_class(attributes, args.focus_class)

    subset_attributes, focus_attrs = _build_local_attribute_view(
        attributes=attributes,
        focus_class=focus_class,
        num_attributes=args.num_attributes,
        neighbor_classes=args.neighbor_classes,
    )

    unique_attrs = get_unique_attributes(subset_attributes)

    if args.semantic_mode == "clip":
        from clip import load as clip_load

        clip_model, _, _ = clip_load(args.arch, device=device)
        clip_model.eval()
        node_feats = encode_attributes_with_clip(unique_attrs, clip_model, device)
        sem_edges = build_semantic_edges(node_feats, threshold=args.similarity_threshold)
    else:
        sem_edges = torch.zeros(2, 0, dtype=torch.long)

    class_edges = build_class_membership_edges(
        unique_attrs,
        subset_attributes,
        num_attributes=args.num_attributes,
    )
    edge_index = combine_edges(sem_edges, class_edges)

    g = nx.Graph()
    for idx, attr in enumerate(unique_attrs):
        g.add_node(idx, label=attr, is_focus=(attr in set(focus_attrs)))

    for u, v in _to_undirected_edge_set(edge_index):
        g.add_edge(u, v)

    return {
        "graph": g,
        "unique_attrs": unique_attrs,
        "focus_class": focus_class,
        "focus_attrs": focus_attrs,
        "subset_class_count": len(subset_attributes),
        "semantic_edge_count": len(_to_undirected_edge_set(sem_edges)),
        "class_edge_count": len(_to_undirected_edge_set(class_edges)),
        "total_edge_count": g.number_of_edges(),
    }


def _select_plot_graph(g, plot_mode):
    if plot_mode == "full":
        return g.copy()

    if plot_mode == "connected":
        connected_nodes = [n for n, d in g.degree() if d > 0]
        return g.subgraph(connected_nodes).copy()

    if plot_mode == "largest":
        connected_nodes = [n for n, d in g.degree() if d > 0]
        connected_graph = g.subgraph(connected_nodes).copy()
        if connected_graph.number_of_nodes() == 0:
            return connected_graph
        components = list(nx.connected_components(connected_graph))
        largest_nodes = max(components, key=len)
        return connected_graph.subgraph(largest_nodes).copy()

    raise ValueError(f"Unsupported plot_mode: {plot_mode}")


def save_visualization(result, output_png, label_top_k, plot_mode):
    full_graph = result["graph"]
    g = _select_plot_graph(full_graph, plot_mode)

    if g.number_of_nodes() == 0:
        raise ValueError("Selected plot graph has no nodes. Try plot_mode=full.")

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(g, seed=42, k=0.9 / max(g.number_of_nodes(), 1) ** 0.5, iterations=300)

    degrees = dict(g.degree())
    node_sizes = [140 + 45 * min(degrees[n], 10) for n in g.nodes()]
    node_colors = ["#D1495B" if g.nodes[n]["is_focus"] else "#2D6A8A" for n in g.nodes()]

    nx.draw_networkx_edges(g, pos, alpha=0.32, width=1.0, edge_color="#7A7A7A")
    nx.draw_networkx_nodes(g, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)

    if label_top_k > 0 and g.number_of_nodes() > 0:
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:label_top_k]
        labels = {n: g.nodes[n]["label"] for n, _ in top_nodes}
        for n in g.nodes():
            if g.nodes[n]["is_focus"]:
                labels[n] = g.nodes[n]["label"]
        nx.draw_networkx_labels(g, pos, labels=labels, font_size=9)

    plt.title(f"ARG Initialization Graph (No GAT Training) - mode: {plot_mode}")
    plt.axis("off")
    plt.tight_layout()

    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_stats(result, output_json):
    g = result["graph"]
    components = list(nx.connected_components(g)) if g.number_of_nodes() > 0 else []
    largest_component = max((len(c) for c in components), default=0)

    stats = {
        "focus_class": result["focus_class"],
        "focus_attrs": result["focus_attrs"],
        "subset_class_count": result["subset_class_count"],
        "num_nodes": g.number_of_nodes(),
        "num_edges": g.number_of_edges(),
        "semantic_edge_count": result["semantic_edge_count"],
        "class_edge_count": result["class_edge_count"],
        "avg_degree": (2.0 * g.number_of_edges() / max(g.number_of_nodes(), 1)),
        "num_connected_components": len(components),
        "largest_component_size": largest_component,
    }

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="I", choices=sorted(ATTRIBUTES_PATH.keys()))
    parser.add_argument("--focus_class", type=str, default=None)
    parser.add_argument("--num_attributes", type=int, default=2)
    parser.add_argument("--neighbor_classes", type=int, default=30)
    parser.add_argument("--semantic_mode", type=str, default="clip", choices=["clip", "none"])
    parser.add_argument("--arch", type=str, default="RN50")
    parser.add_argument("--similarity_threshold", type=float, default=0.75)
    parser.add_argument("--label_top_k", type=int, default=25)
    parser.add_argument("--plot_mode", type=str, default="connected", choices=["full", "connected", "largest"])
    parser.add_argument("--output_png", type=str, default="graph/arg_graph_preview.png")
    parser.add_argument("--output_json", type=str, default="graph/arg_graph_stats.json")
    args = parser.parse_args()

    result = build_graph(args)
    save_visualization(result, args.output_png, args.label_top_k, args.plot_mode)
    save_stats(result, args.output_json)


if __name__ == "__main__":
    main()
