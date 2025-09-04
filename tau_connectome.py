import numpy as np
from collections import deque

# Reproducibility
rng = np.random.default_rng(0)

# =========================
# 1) Brain-ish synthetic connectome
# =========================
def make_brainish_connectome(
    N=84, n_modules=6, density=0.06, lambda_dist=0.12, module_strength=3.0,
    shortcut_frac=0.02, directed=True, hemispheres=True
):
    """
    Generate a synthetic "brain-like" structural connectome.

    This function constructs a weighted adjacency matrix intended to mimic
    qualitative properties of empirical human connectomes, including:
    hemispheric separation, modular organization, distance-dependent
    connectivity, sparse interhemispheric links with callosal shortcuts,
    and log-normal weight distributions. The resulting graph is suitable
    for simulation or algorithm testing where realistic but non-biological
    connectomes are needed.

    Parameters
    ----------
    N : int, default=84
        Number of brain regions (nodes) in the connectome.
        Roughly aligned with the Desikan–Killiany parcellation scale.
    n_modules : int, default=6
        Number of functional modules (e.g., lobes or networks).
        Each node is assigned to one of these modules.
    density : float, default=0.06
        Desired connection density, expressed as a fraction of the
        possible edges (N * (N - 1)) that should be present.
    lambda_dist : float, default=0.12
        Distance decay constant controlling how strongly connection
        probability decreases with Euclidean distance.
        Smaller values emphasize local connections.
    module_strength : float, default=3.0
        Intra-module connectivity multiplier.
        Edges between nodes in the same module are strengthened by
        a factor of (1 + module_strength).
    shortcut_frac : float, default=0.02
        Fraction of existing edges to rewire into long-range
        "shortcuts," producing small-world properties.
    directed : bool, default=True
        If True, output can represent a directed graph. Currently not
        explicitly used, but included for API compatibility.
    hemispheres : bool, default=True
        If True, nodes are divided into left/right hemispheres, with:
          - x-coordinate separation,
          - reduced interhemispheric edge probability,
          - a few guaranteed long callosal links,
          - medial temporal/entorhinal regions assigned to module 0.

    Returns
    -------
    SC : (N, N) ndarray of float
        Weighted structural connectivity matrix.
        Edge weights follow a log-normal distribution scaled by
        distance and module similarity.
    labels : list of str, length N
        Region names in the format "Reg-{i}_M{module_id}".
    module_id : (N,) ndarray of int
        Module assignment of each node (0..n_modules-1).
    pos : (N, 2) ndarray of float
        2D coordinates of each node on the synthetic cortical sheet.

    Notes
    -----
    The construction follows these steps:
    1. Assign random 2D coordinates to nodes.
    2. Optionally split into hemispheres with a spatial gap.
    3. Randomly assign modules, forcing some medial nodes into module 0.
    4. Compute distance-based similarity with exponential decay.
    5. Boost intra-module connectivity.
    6. Suppress interhemispheric connectivity, except for callosal links.
    7. Select top edges until target density is met.
    8. Rewire a fraction of edges into long-range shortcuts.
    9. Assign log-normal weights scaled by distance similarity.

    This produces a sparse, modular, small-world, weighted connectome
    resembling key qualitative features of empirical brain networks.
    """
    pos = rng.random((N, 2)) # random coordinates in [0,1]x[0,1]
    if hemispheres:
        # Split into left/right halves by sorting x-coordinates. The idea is to force a left and right half of the brain with a gap in between (the interhemispheric fissure).
        left = np.argsort(pos[:, 0])[: N // 2]
        right = np.setdiff1d(np.arange(N), left)
        pos[left, 0] *= 0.45 # Compress left hemisphere into [0, 0.45] range
        pos[right, 0] = 0.55 + 0.45 * pos[right, 0] # Compress right hemisphere into [0.55, 1.0]

    module_id = rng.integers(0, n_modules, size=N) # random module per node.
    if hemispheres:
        # Force some medial regions into module 0 (e.g., entorhinal/MTL). In real brains, medial temporal / entorhinal cortex sits near the midline.
        midline_x = 0.275
        d = np.abs(pos[:, 0] - midline_x) # distance from midline
        pick = np.argsort(d)[: max(6, N // 14)] # approx 6 closest nodes
        module_id[pick] = 0

    dists = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)) # NxN matrix
    np.fill_diagonal(dists, np.inf) # self-distances = inf (ignore)

    S = np.exp(-dists / lambda_dist) # distance decay. This encodes the "wiring cost" principle: closer nodes are more likely to be connected.
    intra = (module_id[:, None] == module_id[None, :]).astype(float)
    S *= (1.0 + module_strength * intra) # boost intra-module edges. This encodes the "modular organization principle": nodes in the same functional module are more likely to connect, independent of distance.

    if hemispheres:
        # This creates few, weak inter-hemispheric connections, just like in real brains.
        left_mask = pos[:, 0] < 0.5
        hemi = np.where(left_mask, 0, 1) # 0=left, 1=right
        inter = (hemi[:, None] != hemi[None, :]).astype(float)
        S *= (1.0 - 0.85 * inter) # suppress across-hemisphere edges
        # Ensure each node gets at least 2 strong inter-hemispheric edges. Real brains have commissural fibers that guarantee cross-hemisphere connectivity.
        for i in range(N):
            candidates = np.where(hemi != hemi[i])[0] # opposite hemisphere nodes
            if candidates.size > 0:
                # Sort opposite-hemisphere nodes by distance (farthest first)
                far = candidates[np.argsort(dists[i, candidates])[::-1]] # Sort those candidates by distance (descending -> farthest first)
                for j in far[:2]:
                    # Guarantee minimum similarity. This mimics callosal pathways: relatively sparse, long-distance, but ensuring hemispheric integration.
                    S[i, j] = max(S[i, j], 0.15)

    target_m = int(density * N * (N - 1)) # total edges desired
    idx = np.argsort(S.ravel())[::-1] # sort all pairs by strength
    A = np.zeros((N, N), float)
    count = 0 # keeps track of how many edges we’ve added so far.
    for k in idx:
        i, j = divmod(k, N)
        if i == j or S[i, j] <= 0:
            continue
        if A[i, j] == 0:
            A[i, j] = 1.0
            count += 1
            if count >= target_m:
                break # stop once density is reached (target number of edges)

    # Watts–Strogatz–style rewiring to inject a few long-range “shortcut” edges and make the graph more small-world. This reduces average shortest-path length without massively changing degree distribution.
    if shortcut_frac > 0:
        E = np.argwhere(A > 0) # list of edges
        n_short = int(shortcut_frac * len(E)) # number to rewire
        if n_short > 0:
            pick = rng.choice(len(E), n_short, replace=False) # choose edges to rewire
            for idx_e in pick:
                i, j = E[idx_e]
                A[i, j] = 0.0 # remove the chosen edge
                far_targets = np.argsort(dists[i])[::-1] # sort all nodes by distance from i (farthest first)
                for cand in far_targets:
                    if cand != i and A[i, cand] == 0: # pick the farthest not yet connected
                        A[i, cand] = 1.0 # add the new long-range shortcut
                        break

    mask = A > 0 # Build a boolean mask of existing edges. True where an edge is present, False otherwise.
    W = np.zeros_like(A) # Initialize the weighted adjacency matrix with zeros.
    w_raw = rng.lognormal(mean=-1.0, sigma=0.8, size=mask.sum()) # Sample raw edge weights from a log-normal distribution (heavy-tailed, positive), which resembles empirical tract strength distributions (many weak, few strong connections).
    W[mask] = w_raw # Assign those sampled weights only to edges that exist in A. Non-edges stay zero.
    W *= (S / (np.max(S) + 1e-8)) # shorter distances and intra-module pairs (which had larger S) get proportionally stronger weights.
    np.fill_diagonal(W, 0.0) # Ensure no self-connections in the weighted matrix.

    labels = [f"Reg-{i}_M{module_id[i]}" for i in range(N)] # Human-readable region names, embedding the module label (e.g., Reg-12_M3).
    SC = W
    return SC, labels, module_id, pos

# =========================
# 2) Hopfield utilities (with tau-dependent noise + silencing)
# =========================
def sign(x):
    """
    Bipolar sign function: returns +1 if x >= 0, else -1.
    Used to update Hopfield neurons into active (+1) or inactive (-1) states.
    """
    return np.where(x >= 0, 1, -1)

def make_correlated_patterns(P, N, rho=0.0):
    """
    enerate P binary patterns (+1/-1) of length N with correlation rho.

    Correlation is added so the patterns are not all random and independent, but instead share some similarity, which makes the recall problem harder and more realistic (closer to how real brains deal with overlapping memories).

    Each pattern is created by mixing a common base pattern with Gaussian noise:
      - rho = 0 → independent patterns
      - rho > 0 → patterns biased toward the base
      - rho < 0 → patterns biased opposite the base

    Returns
    -------
    (P, N) ndarray of int : binary patterns
    """
    base = sign(rng.normal(size=N)) # Draw a random "base" pattern of length N (values are +-1)
    patt = []   # List to hold all generated patterns
    scale = np.sqrt(max(1e-8, 1 - rho**2)) # Scaling factor so that rho controls correlation strength properly (ensures variance stays normalized).
    for _ in range(P):
        # Generate P patterns
        # Mix the base pattern with Gaussian noise
        # - rho * base -> correlated part
        # - scale * noise -> independent random part
        z = rho * base + scale * rng.normal(size=N)
        patt.append(sign(z)) # Binarize into +-1 using sign function
    return np.stack(patt)

def hebbian(patterns):
    """
    Compute Hebbian weight matrix from binary patterns.

    - If two neurons are often active together in stored patterns, they get a positive weight.
    - If they’re often in opposite states, they get a negative weight.
    - This matrix W encodes the memories so the Hopfield network can retrieve them later.

    Parameters
    ----------
    patterns : (P, N) ndarray
        P stored patterns of length N with values +-1.

    Returns
    -------
    W : (N, N) ndarray
        Symmetric Hebbian connectivity matrix with zero diagonal.
    """
    N = patterns.shape[1] # Number of neurons per pattern (length of each pattern)
    # Hebbian learning rule:
    # Multiply pattern matrix with its transpose to capture co-activations.
    # (patterns.T @ patterns) gives correlations between neurons across all patterns.
    # Divide by N to normalize weights (keep them in a manageable range).
    W = (patterns.T @ patterns) / N
    np.fill_diagonal(W, 0.0) # Remove self-connections (a neuron shouldn't connect to itself).
    return W

def stochastic_recall_tau_noise(W_eff, cue, x_tau, steps=15, T_base=0.15, c_noise=0.35,
                                dead_mask=None, silent_mask=None, async_update=False):
    """
    Recall a pattern in a Hopfield-like memory network with noise that depends on tau.

    The network tries to recover a stored pattern starting from a noisy cue.
    Each neuron is updated using a "temperature" value that controls
    how much randomness it has:
        T_i = T_base + c_noise * x_tau[i]
    Higher tau means more noise in that neuron’s activity.

    Options:
    - dead_mask: permanently turn off selected units (set to -1 once).
    - silent_mask: keep selected units silent at every step (set to -1 repeatedly).
    - async_update: update neurons one by one in random order (asynchronous),
      otherwise update all neurons together (synchronous).
      Both updating styles are included so the function can be used in different contexts (fast experiments (synchronous) or biologically-grounded simulations (asynchronous)) without duplicating code.

    Parameters
    ----------
    W_eff : (N, N) ndarray
        Connection weights between units.
    cue : (N,) ndarray
        Starting state of the network (+-1).
    x_tau : (N,) ndarray
        Tau values per unit, used to scale noise.
    steps : int, default=15
        Number of update rounds.
    T_base : float, default=0.15
        Minimum noise level (temperature).
    c_noise : float, default=0.35
        How strongly tau adds extra noise.
    dead_mask : (N,) bool ndarray or None
        Units permanently set to -1.
    silent_mask : (N,) bool ndarray or None
        Units forced to -1 at each step.
    async_update : bool, default=False
        Whether to update neurons one at a time (True) or all at once (False).

    Returns
    -------
    s : (N,) ndarray
        The final recalled binary pattern (+-1).
    """
    s = cue.copy() # Copy the input pattern so we don't overwrite the original "cue"
    # Each neuron gets its own temperature T_i:
    # Higher T_i means more randomness (noisier updates).
    # The temperature is clipped to stay within [1e-4, 2.0] for stability.
    T_i = np.clip(T_base + c_noise * x_tau, 1e-4, 2.0)
    # Beta = 1/T, the "inverse temperature" that controls how deterministic the neuron is.
    # Large beta (low T) means very deterministic, almost always follow the input.
    # Small beta (high T) means very noisy, neuron flips more randomly.
    beta_i = 1.0 / T_i

    # ========================
    # Case 1: Asynchronous updates
    # ========================
    if async_update:
        for _ in range(steps):
            for i in rng.permutation(len(s)): # Update neurons one by one, in random order each step
                h_i = W_eff[i] @ s # Compute total input to neuron i (weighted sum from others)
                # Compute probability that neuron i should be +1
                # Sigmoid function of input strength * beta
                p_i = 1.0 / (1.0 + np.exp(-2.0 * beta_i[i] * h_i))
                # Stochastically update neuron i:
                # - with probability p_i -> set to +1
                # - otherwise -> set to -1
                s[i] = 1 if rng.random() < p_i else -1
            if dead_mask is not None: s[~dead_mask] = -1  # Apply masks if provided dead mask
            if silent_mask is not None: s[silent_mask] = -1 # Apply mask if provided silent mask
        return s
    
    # ========================
    # Case 2: Synchronous updates
    # ========================
    for _ in range(steps):
        h = W_eff @ s # Compute inputs for all neurons at once
        p = 1.0 / (1.0 + np.exp(-2.0 * beta_i * h)) # Compute probability of each neuron being +1
        # Update all neurons in parallel:
        # Compare random number to probability p is +1 or -1
        s = np.where(rng.random(len(s)) < p, 1, -1)
        # Apply masks again as above
        if dead_mask is not None: s[~dead_mask] = -1
        if silent_mask is not None: s[silent_mask] = -1
    return s

def overlap(a, b):
    return float((a * b).mean())

# =========================
# 3) Tau dynamics & coupling
# =========================
# mass-conserving random-walk Laplacian on symmetrized SC
def build_L_rw(SC):
    """
    Create a random-walk Laplacian from a connectivity matrix.

    This makes a matrix (L_rw) that describes how activity would
    diffuse or spread across the network, conserving total mass.

    Parameters
    ----------
    SC : (N, N) ndarray
        Connectivity matrix between regions.

    Returns
    -------
    L_rw : (N, N) ndarray
        Random-walk Laplacian, used to model diffusion.
    A : (N, N) ndarray
        Symmetrized version of the connectivity matrix.
    """
    A = 0.5 * (SC + SC.T)  # Make the matrix symmetric (so connections are the same both ways).
    deg = A.sum(axis=1) # Degree of each node = total connection strength to all other nodes. "How many roads lead out" from each region.
    deg[deg == 0.0] = 1.0 # Prevent division by zero (for isolated nodes with no connections).
    P = A / deg[:, None] # Build a transition matrix P where each row sums to 1. This means: from each node, probabilities of "walking" to neighbors.
    # Random-walk Laplacian:
    # L_rw = I - P
    # This matrix describes how things spread across the network, while conserving total "mass".
    L_rw = np.eye(SC.shape[0]) - P
    return L_rw, A

# --- eigenvector centrality (for hub-weighted vulnerability)
def eigenvector_centrality(A_und, tol=1e-8, maxit=1000):
    """
    Compute eigenvector centrality for an undirected network.

    Eigenvector centrality assigns each node a score based on how well
    it is connected to other high-scoring nodes. It is the leading
    eigenvector of the adjacency matrix.

    The purpose is to quantify the influence of each node in the network, not just by counting its neighbors, but by considering how influential those neighbors are too.

    Parameters
    ----------
    A_und : (N, N) ndarray
        Symmetric adjacency matrix of the network.
    tol : float, default=1e-8
        Convergence tolerance for the iterative power method.
    maxit : int, default=1000
        Maximum number of iterations.

    Returns
    -------
    v : (N,) ndarray
        Normalized eigenvector centrality values in [0, 1] for each node.
    """
    N = A_und.shape[0] # Number of nodes in the network
    v = rng.random(N); v /= (np.linalg.norm(v) + 1e-12) # Start with a random guess for the centrality of each node. Normalize so the vector has length 1 (avoids overflow).
    for _ in range(maxit):
        # Power iteration: repeatedly multiply by the adjacency matrix
        # This finds the "dominant eigenvector" of A_und
        v_new = A_und @ v # Update guess: centrality of a node = sum of neighbors' scores
        nrm = np.linalg.norm(v_new) # Compute its length (norm)
        if nrm < 1e-12: break # if result is basically zero, stop
        v_new /= nrm # Normalize new vector (keep it stable)
        if np.linalg.norm(v_new - v) < tol: v = v_new; break # Check if the estimate converged
        v = v_new # Otherwise, continue with updated guess
    v = np.abs(v) # Take absolute values (ignore sign differences)
    v = (v - v.min()) / (v.max() - v.min() + 1e-12) # Normalize scores into the range [0, 1] for easier comparison
    return v

# --- gated logistic with pulse seeding and regional xmax
def tau_step_rw(x, L_rw, alpha_i, beta_diff, k_auto_i, xmax_i,
                t, seed_idx=None, seed_amp=0.0, seed_t0=0, seed_dur=3,
                x_thresh=0.02, dt=1.0):
    """
    Update tau levels in the brain network for one time step.

    The change in tau comes from:
    - Decay (tau breaks down at rate alpha_i)
    - Diffusion (spreads to neighbors via L_rw, scaled by beta_diff)
    - Auto-amplification (grows above a threshold, limited by xmax_i)
    - Seeding (optional external input at a region for a set time)

    Parameters
    -------
    - x = current tau levels in each brain region
    - L_rw = matrix that tells how tau can spread between regions
    - alpha_i = how fast tau naturally decays
    - beta_diff = how strongly tau diffuses between regions
    - k_auto_i = how strongly tau amplifies itself locally
    - xmax_i = maximum tau level allowed in each region
    - t = current time
    - seed_* = parameters for optional external tau "injection"

    Returns
    -------
    x_new : (N,) ndarray
        Tau levels after the update, clipped between 0 and xmax_i.
    """
    s = np.zeros_like(x) # Start with no external seeding (extra tau input)
    # If we are in the seeding window, add tau to the chosen region
    if seed_idx is not None and seed_dur > 0 and (seed_t0 <= t < seed_t0 + seed_dur):
        s[seed_idx] = seed_amp
    # Local auto-amplification:
    # - only happens if tau is above a small threshold
    # - grows faster when tau is low, slows down as tau approaches its maximum
    auto = k_auto_i * np.clip(x - x_thresh, 0.0, None) * (1.0 - x / np.maximum(xmax_i, 1e-6)) 
    dx = (-alpha_i * x - beta_diff * (L_rw @ x) + auto + s) * dt # Compute change in tau (dx)
    x_new = np.clip(x + dx, 0.0, xmax_i) # Update tau levels and keep them within [0, xmax_i]
    return x_new

# synapse mapping with hub-weighted, late-superlinear pruning + silencing
def apply_pathology(W_base, x, centrality, xmax_i,
                    a_weak=0.55, k_prune=0.40, p_del=2.0, eta=1.2,
                    lam_c=0.35, death_thresh=1.05, rho_sil=0.5):
    """
    Apply tau-related pathology to the brain network.

    This modifies the baseline connectivity matrix (W_base) based on
    tau burden (x) and node centrality, modeling four effects:
      1. Weakening: connections from damaged nodes become weaker.
      2. Pruning: highly damaged node pairs lose connections entirely.
      3. Death: nodes above a tau threshold are permanently disconnected.
      4. Silencing: nodes may be temporarily shut down with probability
         depending on tau level.

    Parameters
    ----------
    W_base : (N, N) ndarray
        Baseline connectivity matrix.
    x : (N,) ndarray
        Tau levels per node.
    centrality : (N,) ndarray
        Eigenvector centrality values per node.
    xmax_i : (N,) ndarray
        Maximum tau levels for each node.
    a_weak, k_prune, p_del, eta, lam_c, death_thresh, rho_sil : float
        Model parameters controlling weakening, pruning, death, and silencing.

    Returns
    -------
    W_eff : (N, N) ndarray
        Connectivity after applying pathology.
    dead_mask : (N,) bool ndarray
        True for alive nodes, False for dead nodes.
    silent_mask : (N,) bool ndarray
        True for functionally silenced nodes (temporary).
    """
    # Calculate damage for each node
    # Damage increases with tau level (x) and is boosted if the node is central/important in the network.
    damage = (np.clip(x, 0, None) ** eta) * (1.0 + lam_c * centrality)
    # Weakening effect
    # Connections from damaged nodes get weaker (scaled down).
    weakening = 1.0 / (1.0 + a_weak * damage)
    F = np.outer(weakening, weakening)
    # Pruning effect
    # If two nodes are both highly damaged, their connection may be removed entirely (set close to zero).
    dij = np.outer(damage, damage)
    prune = np.clip(k_prune * (dij ** p_del), 0.0, 0.85)
    W_eff = W_base * F * (1.0 - prune) # Apply weakening and pruning to the baseline connectivity
    np.fill_diagonal(W_eff, 0.0) # no self-connections

    # Neuronal death
    # If tau goes above a threshold, the node is permanently "dead": all its connections are cut and it is marked as inactive.
    dead_mask = np.ones_like(x, dtype=bool)
    if death_thresh is not None:
        dead = (x >= death_thresh)
        if np.any(dead):
            W_eff[dead, :] = 0.0
            W_eff[:, dead] = 0.0
            dead_mask[dead] = False

    # tau-dependent functional silencing (not permanent like death)
    # Even if a node is not dead, high tau can make it unreliable. With some probability (based on tau level), we mark it as silent for this step of the simulation.
    p_sil = np.clip(rho_sil * (np.clip(x / np.maximum(xmax_i, 1e-9), 0, 1) ** eta), 0.0, 0.90)
    silent_mask = rng.random(len(x)) < p_sil
    return W_eff, dead_mask, silent_mask

# =========================
# 4) Build synthetic brain SC & anatomical mask
# =========================
# We simplified the weighted connectome into a binary, undirected version to filter out noisy weak edges and make the network easier to analyze with standard graph-theory tools.
SC, labels, module_id, pos = make_brainish_connectome(
    N=84, n_modules=6, density=0.06, lambda_dist=0.12, module_strength=3.0,
    shortcut_frac=0.02, directed=True, hemispheres=True
)
N = SC.shape[0] # Number of nodes in the network

nonzero = SC[SC > 0] # Extract all existing (nonzero) connection weights
thr = np.percentile(nonzero, 5) if nonzero.size > 0 else 0.0 # Define a threshold: the 5th percentile of connection weights. Weak connections below this value will be ignored.
B = (SC >= thr).astype(float) # Create a binary matrix B: mark an edge as present (1) if its weight is >= threshold, else absent (0)
M = ((B + B.T) > 0).astype(float) # Create a symmetrized matrix M: mark an undirected edge as present if there is a connection in either direction
np.fill_diagonal(M, 0.0) # Remove self-connections

# =========================
# 5) Seed selection: "entorhinal-like" nodes (1 per hemisphere)
# =========================
left_idx = np.where(pos[:, 0] < 0.5)[0] # Find all nodes that belong to the left hemisphere (x-position < 0.5)
right_idx = np.where(pos[:, 0] >= 0.5)[0] # Find all nodes that belong to the right hemisphere (x-position ≥ 0.5)
mtl = np.where(module_id == 0)[0] # Find all nodes assigned to module 0 (treated as the medial temporal lobe / entorhinal system)

def pick_seeds(candidates, side_idx, k=1):
    """
    Pick seed nodes from a given brain hemisphere.

    The function first tries to select up to k nodes that are both in
    the candidate list and on the given side. Preference is given to
    nodes closest to the midline (x approx 0.25 for left, x approx 0.75 for right).
    If not enough candidates are available, it fills the remaining slots
    with other nodes from the same side, again chosen by midline proximity.

    We use pick_seeds to choose realistic starting regions for pathology or activity, prioritizing candidates near the brain’s midline, since these are biologically plausible “seed” areas.

    Parameters
    ----------
    candidates : array-like
        List of candidate node indices.
    side_idx : array-like
        Indices of all nodes in the chosen hemisphere (left or right).
    k : int, default=1
        Number of seeds to pick.

    Returns
    -------
    chosen : (k,) ndarray of int
        Indices of the chosen seed nodes.
    """
    side = np.intersect1d(candidates, side_idx) # Get nodes that are both in the candidate list and on this hemisphere
    need = k; chosen = [] # How many seeds we still need to pick
    mid = 0.25 if np.mean(pos[side_idx, 0]) < 0.5 else 0.75 # Midline reference: approx 0.25 for left side, approx 0.75 for right side
    if side.size > 0:
        # pick from candidates on this side, closest to the midline
        d = np.abs(pos[side, 0] - mid)
        take = side[np.argsort(d)[:min(len(side), need)]]
        chosen.extend(take); need -= len(take)
    if need > 0:
        # if not enough picked, fill the rest from other nodes on this side
        pool = np.setdiff1d(side_idx, np.array(chosen, dtype=int))
        if pool.size > 0:
            d2 = np.abs(pos[pool, 0] - mid)
            add = pool[np.argsort(d2)[:min(len(pool), need)]]
            chosen.extend(add)
    return np.array(chosen, dtype=int)

seed_idx = np.unique(np.r_[pick_seeds(mtl, left_idx, 1),
                           pick_seeds(mtl, right_idx, 1)])

# =========================
# 6) Hopfield patterns & masked weights (this block builds the Hopfield memory weights (W0) from patterns, then filters them through the brain’s network structure (M), 
# so memories are only stored along biologically plausible connections, and finally rescales them to keep activity stable.)
# =========================
P = max(12, N // 8) # Number of patterns to store in the Hopfield network (at least 12, or proportional to network size)
rho = 0.0 # Pattern correlation strength (0 = independent patterns)
patterns = make_correlated_patterns(P, N, rho) # Generate P binary patterns of length N
W0 = hebbian(patterns) # Build Hebbian weight matrix from the patterns
W0 = W0 * M # Mask weights so connections only exist where the brain network M has edges
np.fill_diagonal(W0, 0.0) # Remove self-connections
d = M.sum() / (N * (N - 1) + 1e-8) # Compute effective network density
W0 = W0 / max(d, 1e-6) # Normalize weights by density so overall strength stays balanced

# =========================
# 7) Braak-like staging by graph hop distance from seeds (undirected). This block is about turning hop distances from seeds into staging groups.
# =========================
A_und_bin = ((M + M.T) > 0).astype(int) # Build a binary undirected adjacency: mark edge as present (1) if there is a connection in either direction

def hop_dist_from_seeds(A_bin, seeds):
    """
    Compute hop (shortest path) distances from seed nodes.

    Parameters
    ----------
    A_bin : (N, N) ndarray
        Binary adjacency matrix of the network.
    seeds : list or array of int
        Indices of seed nodes.

    Returns
    -------
    dist : (N,) ndarray
        Number of hops from the nearest seed to each node
        (inf if a node is unreachable).
    """
    Nn = A_bin.shape[0] # Number of nodes in the network
    dist = np.full(Nn, np.inf) # Start with all distances set to infinity (unknown/unreachable)
    q = deque() # Queue of nodes to explore
    # Initialize: seed nodes are distance 0 from themselves
    for s0 in seeds:
        dist[s0] = 0; q.append(s0)
        # Breadth-first search to find shortest hop distances
    while q:
        u = q.popleft() # take the next node
        # Look at all neighbors directly connected to this node
        for v in np.where(A_bin[u] > 0)[0]:
            # If this neighbor hasn't been reached yet, set its distance
            if dist[v] == np.inf:
                dist[v] = dist[u] + 1; q.append(v) # one hop further than current node, add it to the queue to explore later
    return dist

hop = hop_dist_from_seeds(A_und_bin, seed_idx) # Compute hop distances from the chosen seeds
stage_edges = [0, 2, 4, 6, 8, np.inf] # Define staging boundaries (in hops)
stage_masks = [] # stage_masks will hold boolean arrays marking which nodes fall into each stage
lo = 0 # Build masks for each stage range
for hi in stage_edges:
    # Mark nodes whose hop distance is between lo and hi
    stage_masks.append((hop >= lo) & (hop <= hi))
    # Next stage will start just after this upper bound
    lo = hi + 1

# =========================
# 8) Parameters & simulation (month units)
# =========================
# Diffusion operator and centrality
L_rw, A_und_w = build_L_rw(SC) # Build the random-walk Laplacian (for diffusion) and adjacency
centrality = eigenvector_centrality(A_und_w) # Compute eigenvector centrality (importance of each region in the network)

# Regional capacity heterogeneity
xmax_base = 1.6
xmax_i = xmax_base * (1.0 + 0.25 * (module_id == 0))  # Base maximum tau level per region, slightly higher in medial temporal lobe
resilient = (hop >= 7) # Mark resilient regions: cortex far from seeds (≥ 7 hops away)
xmax_i[resilient] *= 0.85 # Resilient regions have a lower maximum tau capacity

# Vulnerability heterogeneity (log-normal)
months_per_step = 1.0 # time resolution = 1 month per step
t_half_months = 30.0 # tau half-life = 30 months
alpha_clear_mean = np.log(2) / t_half_months # baseline clearance rate
k_auto_mean = 0.18 # baseline self-amplification rate
sigma_vuln = 0.20 # variability across regions

# Sample heterogeneous clearance and auto-amplification rates for each region
alpha_i = alpha_clear_mean * np.exp(rng.normal(0, sigma_vuln, N))
k_auto_i = k_auto_mean * np.exp(rng.normal(0, sigma_vuln, N))

# Adjust MTL nodes slightly: slower clearance and stronger amplification
alpha_i[module_id == 0] *= 0.9 # slower clearance means higher tau
k_auto_i[module_id == 0] *= 1.1 # stronger amplification in MTL

# Simulation schedule
dt        = months_per_step # time step size
T_steps   = 150 # number of time steps (150 months approx 12.5 years)
seed_amp  = 0.03 # size of seeding pulse
seed_t0   = 0 # seeding starts at t=0
seed_dur  = 6 # seeding lasts for 6 months
x_thresh  = 0.02 # minimum tau level required for self-amplification

# Severity helper (0 early -> 1 late)
def _sev(m, mid=0.22, k=30.0):
    """Sigmoid severity function mapping m is [0,1] to [0,1]."""
    return 1.0 / (1.0 + np.exp(-k * (m - mid)))

# Diffusion ramp: slower early, then mid-course take-off, late taper to spare sensory/motor
beta_base = 0.022 # slightly slower baseline diffusion

def beta_diff_schedule(mean_tau):
    """
    Adaptive diffusion rate schedule based on average tau burden.

    The diffusion strength (beta) increases once mean tau exceeds ~0.12
    but later tapers off when mean tau gets high, preventing runaway growth.

    Parameters
    ----------
    mean_tau : float
        Current average tau level across all regions.

    Returns
    -------
    beta : float
        Adjusted diffusion rate for this time step.
    """
    sev = _sev(mean_tau, mid=0.12, k=25.0) # Convert mean tau into a severity value using a sigmoid
    ramp  = 1.0 + 0.60 * sev # Ramp-up: diffusion becomes stronger as tau burden increases
    taper = 0.30 + 0.70/(1.0 + np.exp(30*(mean_tau - 0.38))) # Tapering: once mean_tau passes approx 0.38, diffusion strength is reduced (prevents unbounded growth at late stages)
    return beta_base * ramp * taper # Final scheduled diffusion rate

# Synapse impact (hub-weighted; time-varying via severity)
a_weak_base  = 0.28 # Base weakening factor for synapses
k_prune_base = 0.18 # Base pruning factor for connections
p_del        = 2.1 # Exponent controlling how strongly joint damage drives pruning
eta          = 1.6 # Nonlinearity for how tau burden translates into "damage"
lam_c        = 0.35 # Weight for centrality: hubs (important regions) are penalized more
death_th     = 1.05 # Threshold above which a node is considered "dead"

# Silencing ramp with global load (low early -> higher late), softened
# (reserve keeps functional units online until sev approx 0.4)
def rho_sil_schedule(mean_tau):
    """Tau-dependent schedule for silencing probability (0–0.8). 
    It maps the average tau level to a probability (5–80%) that regions become temporarily silenced, rising steeply once tau burden is moderate.
    """
    sev = _sev(mean_tau, mid=0.18, k=35.0)
    return np.clip(0.05 + 0.55 * (sev**1.2), 0.0, 0.80)

# Recall base noise ramps late (models breakdown of compensation)
T_base0  = 0.09 # Baseline noise level during recall
c_noise0 = 0.08 # Scaling factor: how much tau increases recall noise

def recall_noise_schedule(mean_tau):
    """
    Schedule recall noise parameters as a function of tau burden.

    As average tau increases, both the baseline noise (T_base) and the
    tau-dependent noise scaling (c_noise) rise, reflecting loss of 
    compensatory mechanisms and greater variability in recall.

    As tau pathology worsens, the network becomes noisier and less reliable in recall, with both baseline and tau-driven noise rising sharply after a threshold.

    Parameters
    ----------
    mean_tau : float
        Current average tau level across regions.

    Returns
    -------
    T_base : float
        Baseline recall noise level.
    c_noise : float
        Scaling factor for tau-dependent noise.
    """
    sev = _sev(mean_tau, mid=0.20, k=28.0) # Convert mean tau into a severity score (0–1), centered at 0.20. Steeper curve (k=28) makes noise increase mainly after moderate tau levels.
    T_base = T_base0 * (1.0 + 2.0 * (sev**1.3)) # Baseline recall noise grows with severity: starts near T_base0, then triples (approx +200%) as tau burden rises.
    c_noise = c_noise0 * (0.40 + 1.60 * (sev**1.3)) # Tau-dependent noise scaling also increases: starts at ~40% of c_noise0, then rises to approx 200% as severity grows.
    return T_base, c_noise

# Probe corruption decreases with reserve early, then increases (attention lapses)
# (models errors in how a memory is probed)
# Starts low with reserve, but rises as tau burden grows
flip_p0 = 0.15

def flip_p_schedule(mean_tau):
    """Schedule for recall flip probability (0.05–0.25) increasing with tau burden."""
    sev = _sev(mean_tau, mid=0.22, k=25.0)
    return np.clip(flip_p0 * (0.7 + 0.8 * sev), 0.05, 0.25)

steps_rec = 20 # Number of update steps per recall attempt (simulation depth of retrieval process)

# =========================
# 9) Run with reserve-aware operators + validation panels
# =========================
x = np.zeros(N) # Initialize tau levels (all zero at start)
acc_hist, ovl_hist, mean_tau = [], [], [] # History trackers for accuracy, overlap, and mean tau over time
stage_means_over_time = [[] for _ in stage_masks] # Track mean tau values within each stage group across time

# Main simulation loop
for t in range(T_steps):
    # Update current diffusion strength (dynamic with tau burden)
    beta_now = beta_diff_schedule(np.mean(x))
    # Evolve tau levels by one step (diffusion, clearance, amplification, seeding)
    x = tau_step_rw(x, L_rw, alpha_i, beta_now, k_auto_i, xmax_i,
                    t, seed_idx=seed_idx, seed_amp=seed_amp, seed_t0=seed_t0, seed_dur=seed_dur,
                    x_thresh=x_thresh, dt=dt)

    # Record global tau statistics
    mt = float(np.mean(x)) # mean tau across all regions
    max_tau = float(np.max(x)) # maximum tau in any region
    frac_01 = float(np.mean(x > 0.1)) # fraction of regions with tau > 0.1
    frac_03 = float(np.mean(x > 0.3)) # fraction of regions with tau > 0.3
    print(f"t={t:3d} mean={mt:.3f} max={max_tau:.3f} >0.1={frac_01:.2f} >0.3={frac_03:.2f}")

    # Record mean tau for each stage group (defined by hop distance from seeds)
    for si, mask in enumerate(stage_masks):
        stage_means_over_time[si].append(float(np.mean(x[mask])) if np.any(mask) else np.nan)

    # Time-varying synaptic damage parameters (reserve): gentle early, steep late
    sev = _sev(mt, mid=0.20, k=28.0)
    a_weak_eff = a_weak_base * (0.5 + 1.8 * (sev**1.5)) # weakening factor
    k_prune_eff = k_prune_base * (0.5 + 2.0 * (sev**2.0)) # pruning factor
    rho_sil = rho_sil_schedule(mt) # silencing probability

    # Apply pathology to update effective connectivity and masks
    W_eff, dead_mask, silent_mask = apply_pathology(
        W0, x, centrality, xmax_i,
        a_weak=a_weak_eff, k_prune=k_prune_eff, p_del=p_del, eta=eta, lam_c=lam_c,
        death_thresh=death_th, rho_sil=rho_sil
    )

    # Recall noise schedule (reserve-aware)
    # Noise parameters adapt to tau burden
    T_base_now, c_noise_now = recall_noise_schedule(mt)
    flip_p_now = flip_p_schedule(mt)

    accs, ovs = [], []
    for mu in range(P):
        # Corrupt the cue by flipping some bits with probability flip_p_now
        flips = (rng.random(N) < flip_p_now)
        cue = patterns[mu].copy()
        cue[flips] *= -1
        # Recall the memory under current pathology and noise
        s = stochastic_recall_tau_noise(
            W_eff, cue, x, steps=steps_rec, T_base=T_base_now, c_noise=c_noise_now,
            dead_mask=dead_mask, silent_mask=silent_mask, async_update=False
        )
        # Measure overlap with all stored patterns and choose best match
        overlaps = np.array([overlap(s, patterns[k]) for k in range(P)])
        pred = int(np.argmax(overlaps))
        accs.append(pred == mu) # 1 if correct recall
        ovs.append(overlaps[mu]) # overlap with true pattern

    # Record recall performance for this time step
    acc_hist.append(float(np.mean(accs)))
    ovl_hist.append(float(np.mean(ovs)))
    mean_tau.append(mt)

# Summary reporting
print("\n==== SUMMARY (Reserve-aware variant) ====")
print("N (nodes)    :", N)
print("Seeds (idx)  :", seed_idx.tolist())
print("Time (steps) :", list(range(T_steps)))
print("Mean tau     :", [round(v,3) for v in mean_tau])
print("Accuracy     :", [round(v,3) for v in acc_hist])
print("Overlap      :", [round(v,3) for v in ovl_hist])

# Correlation between tau accumulation and recall accuracy
if np.std(mean_tau) > 1e-6 and np.std(acc_hist) > 1e-6:
    corr_tau_acc = float(np.corrcoef(mean_tau, acc_hist)[0,1])
else:
    corr_tau_acc = float('nan')
print(f"Corr(mean_tau, accuracy) over time: {corr_tau_acc:.3f}")

# Functional collapse time = first step where accuracy < 60%
collapse_t = next((t for t, a in enumerate(acc_hist) if a < 0.60), None)
print("Functional-collapse time (acc<0.60):", collapse_t)

# Report tau stage means at selected time points
marks = [0, 24, 60, 96, 120, 149]
for mark in marks:
    if mark < len(mean_tau):
        means_at_mark = [np.nanmean([stage_means_over_time[s][mark]]) for s in range(len(stage_masks))]
        print(f"t={mark:3d} stage_means:", [round(v,3) if np.isfinite(v) else None for v in means_at_mark])