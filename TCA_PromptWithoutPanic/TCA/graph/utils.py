import json

import torch


def load_attributes(json_path):
    """Load a {class_name: [attr1, attr2, ...]} map from a TCA json file."""
    with open(json_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {class_name: value.split() for class_name, value in raw.items()}


def get_unique_attributes(attributes):
    """Return sorted unique attributes across all classes."""
    seen = set()
    for attr_list in attributes.values():
        seen.update(attr_list)
    return sorted(seen)


def encode_attributes_with_clip(unique_attrs, clip_model, device):
    """Encode standalone attributes with frozen CLIP text encoder."""
    from clip import tokenize

    embeddings = []
    clip_model.eval()
    with torch.no_grad():
        for attr in unique_attrs:
            tokens = tokenize([attr]).to(device)
            emb = clip_model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.squeeze(0))
    return torch.stack(embeddings, dim=0)


def build_semantic_edges(node_embeddings, threshold=0.75):
    """Build edges for attribute pairs above cosine similarity threshold."""
    sim = node_embeddings @ node_embeddings.T
    mask = (sim > threshold) & (sim < 1.0 - 1e-6)
    return mask.nonzero(as_tuple=False).T


def build_class_membership_edges(unique_attrs, attributes, num_attributes):
    """Connect attributes that appear within the same class attribute list."""
    attr_to_idx = {attr: idx for idx, attr in enumerate(unique_attrs)}
    src = []
    dst = []

    for class_attrs in attributes.values():
        selected = class_attrs[:num_attributes]
        idxs = [attr_to_idx[attr] for attr in selected if attr in attr_to_idx]
        for i in idxs:
            for j in idxs:
                if i != j:
                    src.append(i)
                    dst.append(j)

    if not src:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


def combine_edges(*edge_indices):
    """Concatenate and deduplicate edge_index tensors."""
    valid = [edge for edge in edge_indices if edge is not None and edge.numel() > 0]
    if not valid:
        return torch.zeros(2, 0, dtype=torch.long)
    combined = torch.cat(valid, dim=1)
    return torch.unique(combined, dim=1)
