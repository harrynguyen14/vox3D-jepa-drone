"""JEPA representation-quality metrics, logged alongside the training loss.

Loss alone doesn't catch collapse: a JEPA can drive cosine loss near zero
by mapping every input to (nearly) the same embedding, which is useless as
a representation. These are cheap, label-free diagnostics computed
directly on each batch's embeddings (see idea.md section 9f for sources):

- embedding_std: mean per-dimension std across the batch. Near zero
  (papers report ~1e-7 as the collapse threshold) means collapse. This is
  the most reliable of the three for catching collapse directly — verified
  locally that effective_rank can stay "healthy" even when std has
  dropped to 1e-3 (residual noise still spreads across many directions),
  so treat effective_rank as a secondary signal, not a substitute for std.
- effective_rank (RankMe): exp(entropy of normalized singular values) of
  the centered embedding matrix. Ranges [1, min(batch, embed_dim)] — near
  1 means all variance sits on one direction (collapse); higher is better
  used dimensionality. Known limitation (matches the Graph-JEPA finding
  cited below): can read healthy on a near-collapsed batch because it
  can't distinguish "many directions of real structure" from "many
  directions of tiny residual noise" — always check embedding_std too.
- scenario_separation: mean cosine distance between the *mean* embedding
  of different scenarios in the batch, divided by the mean cosine distance
  *within* a scenario. >1 means different scenes are more separated than
  same-scene noise — a direct, domain-specific sanity check, since both
  embedding_std and effective_rank can look healthy while the
  representation still carries no usable scene information (see idea.md).
"""
import torch


@torch.no_grad()
def embedding_std(z: torch.Tensor) -> float:
    """Mean per-dimension std across the batch. z: (B, D)."""
    if z.shape[0] < 2:
        return float("nan")
    return z.float().std(dim=0).mean().item()


@torch.no_grad()
def effective_rank(z: torch.Tensor, eps: float = 1e-12) -> float:
    """RankMe: exp(entropy of normalized singular values). z: (B, D)."""
    if z.shape[0] < 2:
        return float("nan")
    z = z.float() - z.float().mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(z)
    p = singular_values / (singular_values.sum() + eps)
    p = p.clamp(min=eps)
    entropy = -(p * p.log()).sum()
    return entropy.exp().item()


@torch.no_grad()
def scenario_separation(z: torch.Tensor, scenario_ids: list[str]) -> float:
    """Ratio of between-scenario to within-scenario embedding spread.

    z: (B, D) embeddings. scenario_ids: length-B list naming each sample's
    scenario (e.g. "stairs", "corridor_1"). Returns NaN if the batch
    contains fewer than 2 distinct scenarios (nothing to compare).
    """
    unique = sorted(set(scenario_ids))
    if len(unique) < 2:
        return float("nan")

    z = torch.nn.functional.normalize(z.float(), dim=-1)
    group_means = []
    within_dists = []
    for s in unique:
        idx = [i for i, sid in enumerate(scenario_ids) if sid == s]
        group = z[idx]
        mean = group.mean(dim=0, keepdim=True)
        group_means.append(mean)
        if group.shape[0] > 1:
            within_dists.append((1 - (group @ mean.T).squeeze(-1)).mean())

    group_means = torch.cat(group_means, dim=0)  # (n_scenarios, D)
    between = 1 - (group_means @ group_means.T)
    off_diag = between[~torch.eye(len(unique), dtype=torch.bool, device=z.device)]
    between_mean = off_diag.mean()

    if not within_dists:
        return float("nan")
    within_mean = torch.stack(within_dists).mean()
    return (between_mean / within_mean.clamp(min=1e-8)).item()
