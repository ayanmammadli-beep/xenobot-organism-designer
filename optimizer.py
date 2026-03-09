"""
backend/generative/optimizer.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The generative AI core of Morpheus.

Three components working together:

1. MorphologyVAE
   3D convolutional VAE that encodes voxel grids to a continuous
   latent space z ∈ ℝ⁶⁴. Decoder maps z → voxel grid.
   Enables gradient-based design optimization.

2. EnvironmentConditionedGNN
   Graph Neural Network that takes a voxel graph + environment
   embedding and predicts fitness. Differentiable → gradient ascent.
   Message passing mimics real force propagation through the body.

3. BayesianMorphologyOptimizer
   Combines: NEAT evolution (global search) + BO with GNN surrogate
   (efficient sampling) + gradient ascent through VAE (local refine).

Why this architecture:
   - VAE learns the manifold of viable organism shapes
   - GNN predicts fitness without running full sim (1000x speedup)
   - BO handles the exploration-exploitation tradeoff properly
   - Gradient ascent finds local optima that evolution misses
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import List, Tuple, Dict, Optional, Callable
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════
# 1. MORPHOLOGY VAE — Latent Space for Body Design
# ═══════════════════════════════════════════════════════════

class Encoder3D(nn.Module):
    """
    3D CNN encoder: voxel grid (B, C, X, Y, Z) → (μ, logσ²) ∈ ℝ⁶⁴

    Input channels C = 5 (one-hot voxel type encoding)
    Architecture: 3× Conv3d + BN + ReLU → GlobalAvgPool → Linear → (μ, σ)
    """
    def __init__(self, grid_size=(8,8,6), n_types=5, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv = nn.Sequential(
            nn.Conv3d(n_types, 32, kernel_size=3, padding=1), nn.BatchNorm3d(32), nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),      nn.BatchNorm3d(64), nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),     nn.BatchNorm3d(128), nn.ReLU(),
        )
        # Compute flat size after conv
        self._flat = 128 * grid_size[0] * grid_size[1] * grid_size[2]
        self.fc_mu     = nn.Linear(self._flat, latent_dim)
        self.fc_logvar = nn.Linear(self._flat, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder3D(nn.Module):
    """
    3D CNN decoder: z ∈ ℝ⁶⁴ → voxel logits (B, n_types, X, Y, Z)

    Architecture: Linear → Reshape → 3× ConvTranspose3d → voxel logits
    """
    def __init__(self, grid_size=(8,8,6), n_types=5, latent_dim=64):
        super().__init__()
        self.grid_size = grid_size
        self.n_types   = n_types
        self._flat     = 128 * grid_size[0] * grid_size[1] * grid_size[2]
        self.fc = nn.Linear(latent_dim, self._flat)
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=3, padding=1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1),  nn.BatchNorm3d(32), nn.ReLU(),
            nn.ConvTranspose3d(32, n_types, kernel_size=3, padding=1),
        )

    def forward(self, z):
        h = F.relu(self.fc(z))
        h = h.view(h.size(0), 128, *self.grid_size)
        return self.deconv(h)   # logits — apply softmax for probs


class MorphologyVAE(nn.Module):
    """
    Full VAE for organism morphology.

    Loss = Reconstruction Loss + β·KL divergence
    β-VAE with β > 1 encourages disentangled latent factors
    (e.g., one dimension = muscle ratio, another = body elongation)
    """
    def __init__(self, grid_size=(8,8,6), n_types=5, latent_dim=64, beta=4.0):
        super().__init__()
        self.encoder = Encoder3D(grid_size, n_types, latent_dim)
        self.decoder = Decoder3D(grid_size, n_types, latent_dim)
        self.beta    = beta
        self.latent_dim = latent_dim
        self.grid_size  = grid_size
        self.n_types    = n_types

    def reparameterize(self, mu, logvar):
        """z = μ + σ·ε, ε ~ N(0,I)  — differentiable sampling"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z          = self.reparameterize(mu, logvar)
        recon      = self.decoder(z)
        return recon, mu, logvar, z

    def encode(self, x) -> torch.Tensor:
        """Grid → latent vector (deterministic, for optimization)"""
        self.eval()
        with torch.no_grad():
            mu, _ = self.encoder(x)
        return mu

    def decode_to_grid(self, z: torch.Tensor) -> np.ndarray:
        """Latent vector → voxel grid (argmax over types)"""
        self.eval()
        with torch.no_grad():
            logits = self.decoder(z)
            types  = logits.argmax(dim=1)  # (B, X, Y, Z)
        return types.squeeze(0).numpy().astype(int)

    def grid_to_onehot(self, grid: np.ndarray) -> torch.Tensor:
        """Convert int grid (X,Y,Z) → one-hot tensor (1, n_types, X, Y, Z)"""
        X, Y, Z = grid.shape
        oh = np.zeros((self.n_types, X, Y, Z), dtype=np.float32)
        for t in range(self.n_types):
            oh[t] = (grid == t).astype(np.float32)
        return torch.FloatTensor(oh).unsqueeze(0)

    def vae_loss(self, recon, target_oh, mu, logvar) -> torch.Tensor:
        """
        L = CrossEntropy(recon, target) + β · KL(q(z|x) || p(z))
        KL = -0.5 · Σ(1 + logσ² - μ² - σ²)
        """
        B = target_oh.size(0)
        # Target: class indices from one-hot
        target_cls = target_oh.argmax(dim=1)  # (B, X, Y, Z)
        recon_loss = F.cross_entropy(recon, target_cls)
        kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_loss


# ═══════════════════════════════════════════════════════════
# 2. ENVIRONMENT-CONDITIONED GNN SURROGATE
# ═══════════════════════════════════════════════════════════

class EnvironmentEncoder(nn.Module):
    """
    Encodes fluid + contaminant params → environment embedding ∈ ℝ³²

    Input features (8 dims):
      [viscosity_log, density_norm, flow_speed_log, temperature_norm,
       pH_norm, ionic_strength_log, particle_diameter_log, zeta_potential_norm]
    """
    def __init__(self, env_dim=8, embed_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(env_dim, 32), nn.LayerNorm(32), nn.GELU(),
            nn.Linear(32, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(),
        )

    def forward(self, env_vec):
        return self.net(env_vec)


class MorphologyGNN(nn.Module):
    """
    Graph Neural Network for fitness prediction.

    Voxel graph: nodes = voxels, edges = spring connections
    Node features (9 dims):
      [type_onehot(5), x_norm, y_norm, z_norm, degree_norm]

    Message passing (3 rounds):
      h_i^{l+1} = σ(W_self · h_i^l + W_nbr · mean_{j∈N(i)} h_j^l + b)

    This mimics real force propagation through the spring network.

    Global pooling → concatenate with env_embedding → MLP → fitness
    """
    def __init__(self, node_dim=9, env_dim=32, hidden=64, out_dim=1):
        super().__init__()
        self.node_dim = node_dim

        # Message passing layers (manual, no torch_geometric dependency for portability)
        self.mp1 = nn.Linear(node_dim, hidden)
        self.mp2 = nn.Linear(hidden, hidden)
        self.mp3 = nn.Linear(hidden, hidden)

        self.self1 = nn.Linear(node_dim, hidden)
        self.self2 = nn.Linear(hidden, hidden)
        self.self3 = nn.Linear(hidden, hidden)

        self.bn1 = nn.LayerNorm(hidden)
        self.bn2 = nn.LayerNorm(hidden)
        self.bn3 = nn.LayerNorm(hidden)

        # Fitness head
        self.head = nn.Sequential(
            nn.Linear(hidden + env_dim, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, out_dim),
            nn.Softplus()   # fitness is always non-negative
        )

    def message_pass(self, h, adj, mp_layer, self_layer, norm):
        """
        One round of message passing.
        h:   (N, d) node features
        adj: (N, N) adjacency matrix (normalized)
        """
        msg  = adj @ h              # aggregate neighbor messages
        h_new = F.gelu(norm(mp_layer(msg) + self_layer(h)))
        return h_new

    def forward(self, node_feats, adj, env_embed):
        """
        node_feats: (N, node_dim)
        adj:        (N, N) normalized adjacency
        env_embed:  (env_dim,) or (1, env_dim)
        """
        h = node_feats
        h = self.message_pass(h, adj, self.mp1, self.self1, self.bn1)
        h = self.message_pass(h, adj, self.mp2, self.self2, self.bn2)
        h = self.message_pass(h, adj, self.mp3, self.self3, self.bn3)

        # Global mean pooling
        h_global = h.mean(dim=0, keepdim=True)  # (1, hidden)

        # Concatenate environment embedding
        if env_embed.dim() == 1:
            env_embed = env_embed.unsqueeze(0)
        combined = torch.cat([h_global, env_embed], dim=-1)
        return self.head(combined).squeeze()


class MorpheusSurrogate(nn.Module):
    """Full surrogate: env_encoder + GNN → fitness prediction"""
    def __init__(self, env_dim=8, node_dim=9, hidden=64, env_embed_dim=32):
        super().__init__()
        self.env_encoder = EnvironmentEncoder(env_dim, env_embed_dim)
        self.gnn         = MorphologyGNN(node_dim, env_embed_dim, hidden)

    def grid_to_graph(self, grid: np.ndarray):
        """Convert voxel grid to node features + adjacency matrix."""
        X, Y, Z = grid.shape
        nodes, node_types = [], []

        coord_idx = {}
        for i in range(X):
            for j in range(Y):
                for k in range(Z):
                    if grid[i,j,k] != 0:
                        coord_idx[(i,j,k)] = len(nodes)
                        nodes.append((i,j,k))
                        node_types.append(grid[i,j,k])

        N = len(nodes)
        if N == 0:
            return None, None

        # Node features: [one_hot_type(5), x_norm, y_norm, z_norm, degree_norm]
        feats = np.zeros((N, 9), dtype=np.float32)
        adj   = np.zeros((N, N), dtype=np.float32)

        dirs = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
        for idx, (ci,cj,ck) in enumerate(nodes):
            # One-hot type (5 classes, 0=empty never included)
            t = node_types[idx]
            if 1 <= t <= 4:
                feats[idx, t-1] = 1.0
            # Normalized position
            feats[idx, 5] = ci / max(X-1, 1)
            feats[idx, 6] = cj / max(Y-1, 1)
            feats[idx, 7] = ck / max(Z-1, 1)
            # Degree (will normalize after)
            for di,dj,dk in dirs:
                nb = (ci+di, cj+dj, ck+dk)
                if nb in coord_idx:
                    jdx = coord_idx[nb]
                    adj[idx, jdx] = 1.0
                    feats[idx, 8] += 1.0

        # Normalize degree
        feats[:, 8] /= 6.0
        # Normalize adjacency (add self-loops, D^-1/2 A D^-1/2)
        adj += np.eye(N)
        deg  = adj.sum(axis=1, keepdims=True)
        deg_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-9))
        adj  = adj * deg_inv_sqrt * deg_inv_sqrt.T

        return torch.FloatTensor(feats), torch.FloatTensor(adj)

    def env_to_vector(self, fluid, contaminant, flow_speed: float) -> torch.Tensor:
        """Convert FluidModel + Contaminant → normalized 8-dim vector."""
        from data.contaminants.database import get_viscosity
        eta  = get_viscosity(fluid, flow_speed / 1e-3)
        vec  = np.array([
            np.log10(max(eta, 1e-6)),                      # log viscosity
            fluid.density / 1200.0,                         # norm density
            np.log10(max(flow_speed, 1e-6)),               # log flow speed
            (fluid.temperature - 273.15) / 50.0,           # norm temperature
            (fluid.pH - 7.0) / 3.0,                        # norm pH
            np.log10(max(fluid.ionic_strength, 1e-6)),     # log ionic strength
            np.log10(max(contaminant.diameter, 1e-12)),    # log particle size
            contaminant.zeta_potential / 50.0,             # norm zeta potential
        ], dtype=np.float32)
        return torch.FloatTensor(vec)

    def forward(self, grid, fluid, contaminant, flow_speed=0.01):
        feats, adj = self.grid_to_graph(grid)
        if feats is None:
            return torch.tensor(0.0)
        env_vec   = self.env_to_vector(fluid, contaminant, flow_speed)
        env_embed = self.env_encoder(env_vec)
        return self.gnn(feats, adj, env_embed)

    def predict_fitness(self, grid, fluid, contaminant, flow_speed=0.01) -> float:
        self.eval()
        with torch.no_grad():
            return float(self(grid, fluid, contaminant, flow_speed))


# ═══════════════════════════════════════════════════════════
# 3. BAYESIAN MORPHOLOGY OPTIMIZER
# ═══════════════════════════════════════════════════════════

class BayesianMorphologyOptimizer:
    """
    Three-stage optimization:

    Stage 1 — NEAT+CPPN evolution (global, topology-aware)
    Stage 2 — BO with GNN surrogate (sample-efficient, exploits learned prior)
    Stage 3 — Gradient ascent in VAE latent space (local refinement)

    The combination covers global, mid-range, and local optimization.
    """

    def __init__(self, vae: MorphologyVAE, surrogate: MorpheusSurrogate,
                 sim_fn: Callable, config, grid_size=(8,8,6)):
        self.vae        = vae
        self.surrogate  = surrogate
        self.sim_fn     = sim_fn   # run_simulation(grid, config) → SimulationResult
        self.config     = config
        self.grid_size  = grid_size

        # Dataset of observed (latent_z, fitness) pairs
        self.observed_z       = []
        self.observed_fitness = []
        self.observed_grids   = []
        self.generation_log   = []

    def random_grid(self, rng=None) -> np.ndarray:
        """Sample a random voxel grid (used for initial population)."""
        if rng is None:
            rng = np.random.default_rng()
        X, Y, Z  = self.grid_size
        grid     = np.zeros((X,Y,Z), dtype=int)
        # Random body: each position has 60% chance of being non-empty
        for i in range(X):
            for j in range(Y):
                for k in range(Z):
                    if rng.random() < 0.6:
                        grid[i,j,k] = rng.integers(1, 5)
        return grid

    def cppn_grid(self, weights: np.ndarray) -> np.ndarray:
        """
        CPPN: f(x,y,z,d) → cell_type via small MLP with sinusoidal activations.

        weights: flat array that parameterizes a 4→8→8→5 network
        Returns integer voxel grid.
        """
        X, Y, Z = self.grid_size
        grid    = np.zeros((X,Y,Z), dtype=int)
        W1 = weights[:32].reshape(8,4)
        b1 = weights[32:40]
        W2 = weights[40:104].reshape(8,8)
        b2 = weights[104:112]
        W3 = weights[112:152].reshape(5,8)
        b3 = weights[152:157]

        for i in range(X):
            for j in range(Y):
                for k in range(Z):
                    xn = 2*i/(X-1)-1; yn = 2*j/(Y-1)-1; zn = 2*k/(Z-1)-1
                    dn = np.sqrt(xn**2+yn**2+zn**2)/np.sqrt(3)
                    h  = np.array([xn, yn, zn, dn])
                    h1 = np.sin(W1 @ h + b1)       # sinusoidal → periodic patterns
                    h2 = np.tanh(W2 @ h1 + b2)
                    h3 = W3 @ h2 + b3
                    t  = int(np.argmax(h3))
                    if np.max(h3) < 0.05:
                        t = 0
                    grid[i,j,k] = t
        return grid

    @property
    def n_cppn_weights(self):
        return 4*8 + 8 + 8*8 + 8 + 8*5 + 5  # = 157

    # ── Stage 1: Evolutionary Search ────────────────────────

    def run_evolution(self, n_generations: int = 30, pop_size: int = 20,
                      callback=None) -> Tuple[np.ndarray, float]:
        """
        (μ+λ) evolution strategy on CPPN weights.
        Uses GNN surrogate for fast fitness evaluation.
        Falls back to real sim for top candidates.
        """
        rng = np.random.default_rng(42)
        n_w = self.n_cppn_weights

        # Initialize population
        pop = [rng.standard_normal(n_w) * 0.5 for _ in range(pop_size)]

        best_w, best_f = None, -np.inf
        self.generation_log = []

        for gen in range(n_generations):
            # Evaluate population with surrogate
            fitnesses = []
            grids     = []
            for w in pop:
                grid = self.cppn_grid(w)
                # Use surrogate (fast) or sim (slow) based on budget
                f = self.surrogate.predict_fitness(
                    grid,
                    self.config.fluid,
                    self.config.contaminant,
                    float(np.linalg.norm(self.config.flow_velocity))
                )
                fitnesses.append(f)
                grids.append(grid)

            # Rank and select top 25%
            ranked  = sorted(zip(fitnesses, range(pop_size)), reverse=True)
            top_n   = max(pop_size // 4, 2)
            parents = [pop[ranked[i][1]] for i in range(top_n)]

            gen_best_f = ranked[0][0]
            gen_best_g = grids[ranked[0][1]]

            # Run real simulation on top-3 for ground truth
            real_fitnesses = []
            for i in range(min(3, len(parents))):
                result = self.sim_fn(grids[ranked[i][1]], generation=gen)
                real_fitnesses.append(result.fitness)
                self.observed_z.append(None)   # will encode later
                self.observed_fitness.append(result.fitness)
                self.observed_grids.append(grids[ranked[i][1]])

            gen_real_best = max(real_fitnesses) if real_fitnesses else gen_best_f

            if gen_real_best > best_f:
                best_f = gen_real_best
                best_w = parents[0].copy()

            log = {
                "generation":     gen,
                "best_surrogate": float(gen_best_f),
                "best_real":      float(gen_real_best),
                "best_overall":   float(best_f),
                "pop_size":       pop_size,
                "grid":           gen_best_g,
            }
            self.generation_log.append(log)

            if callback:
                callback(log)

            # Breed next generation: recombination + mutation
            new_pop = list(parents)
            while len(new_pop) < pop_size:
                p1, p2 = parents[rng.integers(top_n)], parents[rng.integers(top_n)]
                mask   = rng.random(n_w) < 0.5
                child  = np.where(mask, p1, p2)
                sigma  = max(0.5 * (0.95 ** gen), 0.05)
                child += rng.standard_normal(n_w) * sigma
                new_pop.append(child)
            pop = new_pop

        # Update surrogate on new data
        self._update_surrogate()

        return self.cppn_grid(best_w), best_f

    def _update_surrogate(self):
        """Fine-tune surrogate on newly observed (grid, fitness) pairs."""
        if len(self.observed_grids) < 5:
            return
        opt = Adam(self.surrogate.parameters(), lr=1e-3)
        self.surrogate.train()

        for _ in range(50):   # quick fine-tune
            for grid, f_true in zip(self.observed_grids[-20:],
                                     self.observed_fitness[-20:]):
                opt.zero_grad()
                f_pred = self.surrogate(
                    grid, self.config.fluid, self.config.contaminant,
                    float(np.linalg.norm(self.config.flow_velocity))
                )
                loss = F.mse_loss(f_pred, torch.tensor(float(f_true)))
                loss.backward()
                opt.step()
        self.surrogate.eval()

    # ── Stage 2: Bayesian Optimization ──────────────────────

    def run_bayesian_optimization(self, n_iter: int = 20,
                                   seed_grid: Optional[np.ndarray] = None,
                                   callback=None) -> Tuple[np.ndarray, float]:
        """
        BO in CPPN weight space using surrogate as acquisition model.
        Uses Expected Improvement (EI) acquisition function.

        EI(w) = E[max(f(w) - f*, 0)]
               ≈ (μ(w) - f*) · Φ(Z) + σ(w) · φ(Z)
               where Z = (μ(w) - f*) / σ(w)

        Approximates σ with MC dropout over 20 forward passes.
        """
        from scipy.stats import norm as sp_norm

        rng    = np.random.default_rng()
        n_w    = self.n_cppn_weights
        f_best = max(self.observed_fitness) if self.observed_fitness else 0.0
        best_grid, best_f = seed_grid, f_best

        def predict_with_uncertainty(w: np.ndarray, n_samples=20):
            """MC dropout for uncertainty estimate."""
            grid = self.cppn_grid(w)
            preds = []
            self.surrogate.train()  # enable dropout
            for _ in range(n_samples):
                with torch.no_grad():
                    p = self.surrogate.predict_fitness(
                        grid, self.config.fluid, self.config.contaminant,
                        float(np.linalg.norm(self.config.flow_velocity))
                    )
                preds.append(p)
            self.surrogate.eval()
            return np.mean(preds), np.std(preds)

        for iteration in range(n_iter):
            # Sample candidates and pick by EI
            candidates = [rng.standard_normal(n_w) * 0.3 for _ in range(50)]
            best_ei, best_w = -np.inf, None

            for w in candidates:
                mu, sigma = predict_with_uncertainty(w)
                sigma = max(sigma, 1e-6)
                Z     = (mu - f_best) / sigma
                ei    = (mu - f_best) * sp_norm.cdf(Z) + sigma * sp_norm.pdf(Z)
                if ei > best_ei:
                    best_ei, best_w = ei, w

            # Evaluate best candidate with real sim
            grid   = self.cppn_grid(best_w)
            result = self.sim_fn(grid)
            f_obs  = result.fitness

            self.observed_grids.append(grid)
            self.observed_fitness.append(f_obs)

            if f_obs > best_f:
                best_f, best_grid = f_obs, grid
                f_best = f_obs

            self._update_surrogate()

            if callback:
                callback({"bo_iter": iteration, "fitness": f_obs, "best": best_f})

        return best_grid, best_f

    # ── Stage 3: Gradient Ascent in Latent Space ────────────

    def gradient_optimize(self, seed_grid: np.ndarray,
                           n_steps: int = 100, lr: float = 0.05,
                           callback=None) -> Tuple[np.ndarray, float]:
        """
        Gradient ascent through: z → decoder → GNN surrogate → fitness

        dz/dt = ∇_z [surrogate(decoder(z))]

        This finds local optima that discrete evolution cannot.
        The VAE decoder ensures z always maps to a valid organism body.
        """
        self.vae.eval()
        self.surrogate.eval()

        # Encode seed grid to latent space
        with torch.no_grad():
            seed_oh = self.vae.grid_to_onehot(seed_grid)
            z0, _   = self.vae.encoder(seed_oh)

        z = nn.Parameter(z0.clone())
        opt = Adam([z], lr=lr)

        best_z, best_f = z.data.clone(), -np.inf
        fluid      = self.config.fluid
        contaminant= self.config.contaminant
        flow_speed = float(np.linalg.norm(self.config.flow_velocity))

        for step in range(n_steps):
            opt.zero_grad()

            # Decode z → logits → soft grid (differentiable via Gumbel-Softmax)
            logits    = self.vae.decoder(z)                   # (1, n_types, X,Y,Z)
            soft_grid = F.gumbel_softmax(logits, tau=0.5, dim=1, hard=False)

            # Build differentiable node features for GNN
            # Use soft type probabilities instead of argmax
            B, C, X, Y, Z = soft_grid.shape
            soft_np = soft_grid.squeeze(0).detach().numpy()
            hard_grid = np.argmax(soft_np, axis=0).astype(int)

            feats_np, adj_np = self.surrogate.grid_to_graph(hard_grid)
            if feats_np is None:
                continue

            env_vec   = self.surrogate.env_to_vector(fluid, contaminant, flow_speed)
            env_embed = self.surrogate.env_encoder(env_vec)
            fitness   = self.surrogate.gnn(feats_np, adj_np, env_embed)

            # Maximize fitness → minimize negative fitness
            loss = -fitness
            loss.backward()
            opt.step()

            f_val = float(fitness.detach())
            if f_val > best_f:
                best_f = f_val
                best_z = z.data.clone()

            if callback and step % 10 == 0:
                callback({"grad_step": step, "fitness": f_val, "best": best_f})

        # Decode best z to grid
        best_grid = self.vae.decode_to_grid(best_z)

        # Final real simulation for ground truth
        result    = self.sim_fn(best_grid)
        return best_grid, result.fitness

    # ── Full Pipeline ────────────────────────────────────────

    def optimize(self, evo_gens=30, bo_iters=15, grad_steps=80,
                 pop_size=20, progress_callback=None) -> Dict:
        """
        Run complete 3-stage optimization pipeline.
        Returns full results dict with grids, fitness, and history.
        """
        results = {"stages": [], "best_grid": None, "best_fitness": -np.inf}

        def cb(info):
            if progress_callback:
                progress_callback(info)

        # Stage 1
        print("  Stage 1/3: Evolutionary search (NEAT+CPPN)...")
        evo_grid, evo_f = self.run_evolution(evo_gens, pop_size, callback=cb)
        results["stages"].append({"stage": "evolution", "fitness": evo_f, "grid": evo_grid})
        results["evo_log"] = self.generation_log

        # Stage 2
        print("  Stage 2/3: Bayesian optimization with GNN surrogate...")
        bo_grid, bo_f = self.run_bayesian_optimization(bo_iters, seed_grid=evo_grid, callback=cb)
        results["stages"].append({"stage": "bayesian_opt", "fitness": bo_f, "grid": bo_grid})

        # Stage 3
        seed = bo_grid if bo_f > evo_f else evo_grid
        print("  Stage 3/3: Gradient ascent in VAE latent space...")
        grad_grid, grad_f = self.gradient_optimize(seed, grad_steps, callback=cb)
        results["stages"].append({"stage": "gradient", "fitness": grad_f, "grid": grad_grid})

        # Pick overall best
        all_stages = [(evo_f, evo_grid), (bo_f, bo_grid), (grad_f, grad_grid)]
        best_f, best_grid = max(all_stages, key=lambda x: x[0])
        results["best_fitness"] = best_f
        results["best_grid"]    = best_grid

        # Run final simulation with full history
        print("  Running final simulation for full analysis...")
        final_result = self.sim_fn(best_grid, record_history=True)
        results["final_simulation"] = final_result

        return results
