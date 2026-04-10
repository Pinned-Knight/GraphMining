"""Offline ARG-TCA builder for refined attribute embeddings."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from clip import load as clip_load, tokenize
from graph.attribute_graph import AttributeGAT
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


def contrastive_loss(refined_embs, class_member_edges, tau=0.07):
    embs = F.normalize(refined_embs, dim=-1)
    sim = (embs @ embs.T) / tau

    n_attrs = embs.size(0)
    pos_mask = torch.zeros(n_attrs, n_attrs, device=embs.device, dtype=torch.bool)
    if class_member_edges.numel() > 0:
        pos_mask[class_member_edges[0], class_member_edges[1]] = True

    sim = sim - torch.diag(torch.diag(sim))
    exp_sim = torch.exp(sim)

    loss = torch.tensor(0.0, device=embs.device)
    count = 0
    for i in range(n_attrs):
        pos_idx = pos_mask[i].nonzero(as_tuple=True)[0]
        if pos_idx.numel() == 0:
            continue

        num = exp_sim[i, pos_idx].sum()
        den = exp_sim[i].sum() + 1e-8
        loss = loss - torch.log((num / den) + 1e-8)
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=embs.device)
    return loss / count


def build_and_save(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    attributes = load_attributes(ATTRIBUTES_PATH[args.dataset])
    unique_attrs = get_unique_attributes(attributes)

    clip_model, _, _ = clip_load(args.arch, device=device)
    clip_model.eval()

    node_feats = encode_attributes_with_clip(unique_attrs, clip_model, device)
    sem_edges = build_semantic_edges(node_feats, args.similarity_threshold).to(device)
    class_edges = build_class_membership_edges(unique_attrs, attributes, args.num_attributes).to(device)
    edge_index = combine_edges(sem_edges, class_edges).to(device)

    gat = AttributeGAT(
        clip_dim=node_feats.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.gat_layers,
        num_heads=args.gat_heads,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(gat.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_loss = float("inf")
    best_refined = node_feats

    for _ in range(args.epochs):
        optimizer.zero_grad()
        refined = gat(node_feats, edge_index)
        loss = contrastive_loss(refined, class_edges, tau=args.temperature)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_refined = refined.detach().clone()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.dataset}_{args.arch.replace('/', '-')}_{args.num_attributes}attr.pt"

    # Class anchors are saved to simplify downstream diagnostics.
    class_anchors = {}
    with torch.no_grad():
        for class_name in attributes.keys():
            class_anchors[class_name] = (
                clip_model.encode_text(tokenize([class_name]).to(device)).squeeze(0).cpu()
            )

    torch.save(
        {
            "dataset": args.dataset,
            "arch": args.arch,
            "num_attributes": args.num_attributes,
            "unique_attrs": unique_attrs,
            "raw_embeddings": node_feats.detach().cpu(),
            "refined_embeddings": best_refined.detach().cpu(),
            "class_anchors": class_anchors,
            "similarity_threshold": args.similarity_threshold,
            "best_loss": best_loss,
        },
        output_file,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=sorted(ATTRIBUTES_PATH.keys()))
    parser.add_argument("--arch", type=str, default="RN50")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_attributes", type=int, default=2)
    parser.add_argument("--similarity_threshold", type=float, default=0.75)
    parser.add_argument("--gat_layers", type=int, default=2)
    parser.add_argument("--gat_heads", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--output_dir", type=str, default="Attributes/arg_embeddings")
    build_and_save(parser.parse_args())
