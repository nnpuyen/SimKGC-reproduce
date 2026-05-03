import numpy as np
import torch

def uniformity_np_unique(vectors_np, eps=1e-12):
    # vectors_np: numpy array (n, d)
    if vectors_np.shape[0] < 2:
        return 0.0
    unique_rows, unique_indices = np.unique(vectors_np, axis=0, return_index=True)
    if unique_indices.shape[0] < 2:
        return 0.0
    uniq = vectors_np[unique_indices]
    # pairwise distances
    pd = np.linalg.norm(uniq[:, None, :] - uniq[None, :, :], axis=-1)
    mask = ~np.eye(uniq.shape[0], dtype=bool)
    dists = pd[mask]
    exp_term = np.exp(-2 * (dists ** 2))
    mean_exp = exp_term.mean()
    return -np.log(mean_exp + eps)


def uniformity_id_dedup(vectors, ids, eps=1e-12):
    # vectors: torch tensor (n, d), ids: list of ids length n
    if vectors.size(0) < 2:
        return 0.0
    seen = set()
    uniq_idx = []
    for i, idv in enumerate(ids):
        if idv not in seen:
            seen.add(idv)
            uniq_idx.append(i)
    if len(uniq_idx) < 2:
        return 0.0
    uniq = vectors[uniq_idx].cpu().numpy()
    return uniformity_np_unique(uniq, eps=eps)


def main():
    dim = 64
    # Simulate entity E1 repeated 3 times with slight dropout noise, and E2 once
    batch_size = 4
    base_E1 = np.random.randn(dim)
    base_E1 /= np.linalg.norm(base_E1)
    base_E2 = np.random.randn(dim)
    base_E2 /= np.linalg.norm(base_E2)

    # Create 3 noisy copies for E1
    noise_scale = 1e-3
    v0 = base_E1 + np.random.randn(dim) * noise_scale
    v1 = base_E1 + np.random.randn(dim) * noise_scale
    v2 = base_E1 + np.random.randn(dim) * noise_scale
    v3 = base_E2 + np.random.randn(dim) * noise_scale

    vectors = np.stack([v0, v1, v2, v3], axis=0)
    ids = ['E1', 'E1', 'E1', 'E2']

    u_np = uniformity_np_unique(vectors)
    u_id = uniformity_id_dedup(torch.from_numpy(vectors.astype(np.float32)), ids)

    print(f"Uniformity (np.unique on vectors): {u_np:.6e}")
    print(f"Uniformity (dedup by id):      {u_id:.6e}")

    # Show that dedup reduces loss
    print('\nExplanation: higher uniformity loss => vectors too close => penalty')

if __name__ == '__main__':
    main()
