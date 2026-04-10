"""Sanity checks for ARG embedding artifacts."""

import argparse

import torch


def main(path):
    data = torch.load(path, map_location="cpu")
    attrs = data.get("unique_attrs", [])
    refined = data.get("refined_embeddings")
    raw = data.get("raw_embeddings")

    if refined is None or raw is None:
        raise ValueError("Artifact must contain 'raw_embeddings' and 'refined_embeddings'.")

    if refined.shape != raw.shape:
        raise ValueError("Raw and refined embeddings must have the same shape.")

    cos = torch.nn.functional.cosine_similarity(raw, refined, dim=-1)
    print(f"attributes: {len(attrs)}")
    print(f"embedding_dim: {refined.shape[-1]}")
    print(f"mean_cosine(raw, refined): {cos.mean().item():.4f}")
    print(f"min_cosine(raw, refined): {cos.min().item():.4f}")
    print(f"max_cosine(raw, refined): {cos.max().item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=str, help="Path to ARG .pt output")
    args = parser.parse_args()
    main(args.artifact)
