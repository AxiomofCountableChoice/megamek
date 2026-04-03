# BattleTech RL AI: Architecture & Formal Specification


## 1. Environment Specification: A Zero-Sum Markov Game
We formally define the game of BattleTech as a discrete-time, two-player, zero-sum Markov Game denoted by the tuple $\mathcal{M} = \langle \mathcal{G}, \mathcal{A}^1, \mathcal{A}^2, P, R, \gamma \rangle$.


### 1a. State Space ($\mathcal{G}$)
A state $g \in \mathcal{G}$ is defined as the tuple $(p, B)$.
* **The Spatial Graph ($B$):** Conceptually an augmented heterogeneous graph $B = (\mathcal{V}, \mathcal{E})$. $\mathcal{V} = \mathcal{V}_U \cup \mathcal{V}_H$, where $\mathcal{V}_U$ represents unit nodes and $\mathcal{V}_H$ represents terrain hex nodes. $\mathcal{E} = \mathcal{E}_V \cup \mathcal{E}_L \cup \mathcal{E}_M$, representing physical adjacency ($\mathcal{E}_V$), Line of Sight declarations ($\mathcal{E}_L$), and movement commands/TMM encoding ($\mathcal{E}_M$).
* **The Phase Vector ($p$):** A tuple acting as a granular state machine: $p = (\phi_{main}, \phi_{sub}, \iota, \tau, \omega)$.
    * $\phi_{main} \in \{1, \dots, 7\}$ denotes the primary game phase.
    * $\phi_{sub}$ denotes the sub-state, handling interstitial interrupts.
    * $\iota \in \{1, 2\}$ denotes the active player.
    * $\tau$ encodes the turn number and rounds left in the phase.
    * $\omega$ encodes objective-based scores.


### 1b. Graph Covariates
To accurately capture the state, nodes and edges in $B$ are enriched with specific covariates:
* **Hex Nodes ($\mathcal{V}_H$):** Elevation, Terrain Type (Woods, Water, Pavement), Fire/Smoke presence, Objective Status (e.g., Control Zone, Extraction Point).
* **Entity Nodes ($\mathcal{V}_U$):** Current Heat, Heat Capacity, Armor/Structure (per location), Engine/Gyro critical hits, Pilot Consciousness.
* **Edges ($\mathcal{E}$):** Distance, Cover modifiers provided by intervening hexes, active TMMs.


### 1c. Action Space ($\mathcal{A}$)
The action space is hierarchical, phase-dependent, and sparse. The MegaMek engine deterministically resolves edge-case ambiguity (such as Line of Sight pathing), ensuring a strictly alternating Markov Game without interactive interrupts.
1.  **Initiative Phase:** Handled automatically by the engine; action space is null.
2.  **Movement Phase (Ground):** Hierarchical sequence. $a_{move} = (a^{(unit)}, a^{(type)}, a^{(hex_1)}, a^{(hex_2)}, \dots, a^{(stop)})$.
3.  **Movement Phase (Aerospace):** Ignored for this formulation.
4.  **Weapon Attack Phase:** Pure autoregressive sequence over engine-validated visible targets. $a_{attack} = (a^{(unit)}, a^{(tgt_1)}, a^{(w_{1,1})}, \dots, a^{(stop)})$.
5.  **Physical Attack Phase:** Hierarchical targeting for Melee/DFA. $a_{phys} = (a^{(unit)}, a^{(tgt)}, a^{(type)})$.
6.  **Heat Phase:** Deterministic resolution; action space is null.
7.  **End Phase:** Deterministic clean-up; action space is null.


### 1d. Action Features (Pointer Embeddings)
To leverage a pointer network, the valid sub-actions (e.g., weapons, targets) are represented dynamically via their covariates.
* **Weapon Statistics:** Base Damage, Heat Generated, Range Brackets, Is_Cluster, Cluster Size.
* **Target State Embedding:** A compressed representation of the target (e.g., Pristine, Heavily Damaged, Open Torso, Targets Facing Arc - are we in their rear arc for example).
* **Attacker State Embedding:** Attacker's heat level and structural integrity (enabling "Hail Mary" recognition).


### 1e. Reward Signals ($R$)
The environment enforces a strict zero-sum constraint to ensure a Markov Nash Equilibrium:
$$R^1(g, a^1, a^2) = -R^2(g, a^1, a^2)$$
* **Sparse Terminal Rewards:** $+100$ (Win), $-100$ (Loss), $0$ (Tie).
* **Dense Shaping Rewards:** Designed to prevent deviation from the Nash Equilibrium while providing step-wise gradients. To ensure stable gradients across wildly varying match sizes, all components are intrinsically scaled by their maximum match constraints. $\beta$ and $\alpha$ serve purely as strategic priority weights:
$$R_t = \left( \frac{\Delta \text{BV}_{opp} - \Delta \text{BV}_{self}}{\text{TotalMatchBV}} \right) + \beta \left( \frac{\Delta \text{Obj}}{\text{MaxObj}} \right) - \alpha \left( \frac{\text{Heat}_{self}}{\text{MaxHeat}_{self}} - \frac{\text{Heat}_{opp}}{\text{MaxHeat}_{opp}} \right)$$


### 1f. Transition Dynamics ($P$)
The transition measure $P$ represents the MegaMek engine. Due to strict turn-taking ($\iota$), if Player 1 is active, $\mathcal{A}^2(g_t)$ is masked to $\{\emptyset\}$, preserving the Markov Game definition.


---


## 2. RL Algorithmic Considerations


### 2a. Asynchronous Actor-Critic Structure (IMPALA)
The agent utilizes an Importance Weighted Actor-Learner Architecture (IMPALA). To maximize hardware utilization, the system decouples trajectory generation from gradient computation, operating asynchronously.
* **The Actors:** Multiple CPU threads run headless environments, sampling actions from a slightly delayed behavior policy $\mu$ and pushing trajectories to a queue.
* **The Learner:** The GPU continuously pulls from the queue, updating the target policy $\pi_\theta$ and critic $V_\omega$.
* **V-Trace Off-Policy Correction:** Because the behavior policy $\mu$ lags behind the target policy $\pi_\theta$, the system utilizes V-trace clipping to strictly bound the variance of the importance sampling weights, guaranteeing mathematical convergence despite the asynchronous lag.


### 2b. Value Ensembles & Compute Optimization
To provide epistemic exploration without bottlenecking hardware, the critic architecture is heavily optimized:
* **Parameterization ($E$):** The number of value heads $E$ acts as a hyperparameter. Setting $E=1$ mathematically zeroes out the variance bonus, seamlessly reverting to a pure entropy-driven exploration model for strict hardware constraints.
* **Latent Caching:** To prevent computational bloat when $E > 1$, the heavy CNN and Context MLP forward passes are computed only once. The ensemble heads $\{V_{\omega_1}, \dots, V_{\omega_E}\}$ are lightweight MLPs that branch directly from the cached global latent state $z$.


### 2c. Epistemic Exploration Bonus
The variance across the value ensemble quantifies the agent's epistemic uncertainty. In the IMPALA formulation, this variance is injected directly into the **V-trace target** ($v_s$) prior to calculating the policy gradient, acting as an intrinsic exploration bonus:
$$v^{enhanced}_s = v_s + \lambda \sqrt{\text{Var}(V_{\omega_i}(z_s))}$$
By inflating the value target of highly uncertain states, the resulting V-trace advantage natively directs the actor's policy updates toward unmapped strategy spaces.


### 2d. Learning Curriculum
To ensure the agent does not overfit to pure attrition, the curriculum systematically expands the state space:
1.  **Static Gunnery:** 1v1 stationary targets (weapon optimization).
2.  **Maneuver & Fire:** Pathfinding, LOS, and TMM mechanics.
3.  **Objective Control:** Introduction of extraction zones and King-of-the-Hill hexes, forcing the agent to weigh scaled $\Delta \text{BV}$ against scaled $\Delta \text{ObjectiveScore}$.
4.  **The Baseline Duel:** Reactive tactics against the Princess AI.
5.  **Self-Play:** Symmetrical progression toward the Nash Equilibrium across varying objective types.


---


## 3. First Pass of Model Representations


### 3a. CNN Structure
To ensure computational feasibility, $B$ is projected into a Global Multi-Channel Spatial Tensor $T \in \mathbb{R}^{W \times H \times C}$.
* Un-pooled CNN layers process separate channels for Terrain (Elevation, Woods), Entities (Friendly/Enemy footprints), and Phase Masks (valid LOS). Outputs a spatial patch sequence $M$.


### 3b. Late Fusion Embedding ($z$)
* The phase metadata $p$ is processed via an MLP: $p \rightarrow z_{context}$.
* **Late Fusion:** The global state representation is formed: $z = \text{Flatten}(M) \oplus z_{context}$.


### 3c. Value Ensemble Architecture (Critic Heads)
To support the epistemic exploration bonus without introducing massive computational overhead, the architecture utilizes a branched ensemble of critics:
* **Input:** The ensemble takes the cached global latent state $z$ as its sole input, completely avoiding re-computation of the heavy CNN and Context MLP layers.
* **Ensemble Size ($E$):** A hyperparameter defining the number of independent value heads. For strict desktop performance, $E=1$. To enable epistemic exploration, $E$ is typically set between $3$ and $5$.
* **Structure:** Each head $i \in \{1, \dots, E\}$ is a lightweight, independent Multi-Layer Perceptron (MLP) with its own weights $\omega_i$.
* **Output:** Each head outputs a scalar value $V_{\omega_i}(z)$ representing the expected discounted return from the current state. The variance across these $E$ outputs generates the intrinsic exploration bonus.


### 3d. Pointer Embedding MLP
For valid sub-actions $c$ with covariates $x_c$ (from Section 1d), a shared MLP creates dense embeddings:
$$e_c = \text{ActionMLP}(x_c)$$


### 3e. Autoregressive Head (Transformer Decoder)
Sequence generation uses a Decoder-only architecture.
* Initializes with $z$.
* At step $k$, hidden state $h_k$ is updated via cross-attention over the spatial memory $M$.
* Computes dot-product attention between $h_k$ and valid sub-action embeddings $e_c$, applying a softmax to sample the next token.


### 3f. Full Loss Statement & Gradient Updates (IMPALA V-Trace Formulation)
The total loss minimizes the policy gradient loss, the ensemble critic loss, and the entropy regularization bonus, corrected for off-policy generation via V-trace.
$$L_{total}(\theta, \omega) = \hat{\mathbb{E}}_{s} \left[ L_{actor}(\theta) + \frac{c_1}{E} \sum_{i=1}^E L_{critic}(\omega_i) - c_2 L_{entropy}(\theta) \right]$$


**Note on MDP Timesteps vs. Autoregressive Generation:**
The MDP timestep $t$ in the V-trace formulation corresponds strictly to a complete environment interaction (a macro-action where MegaMek advances the state and returns a reward). The sub-action sampling happens internally on the GPU. The action probability $\pi_\theta(a_t|z_t)$ evaluated during the loss calculation is the joint probability of the entire sampled token sequence: $\pi_\theta(a_t|z_t) = \prod_{k} \pi_\theta(c_k | z_t, c_{<k})$.


**1. The V-Trace Target ($v_s$):**
We calculate the $n$-step V-trace target for state $s$, utilizing the mean of the ensemble $\bar{V}_\omega(z) = \frac{1}{E} \sum_{i=1}^E V_{\omega_i}(z)$ as the baseline:
$$v_s = \bar{V}_\omega(z_s) + \sum_{t=s}^{s+n-1} \gamma^{t-s} \left( \prod_{i=s}^{t-1} c_i \right) \delta_t V$$
Where the temporal difference error is $\delta_t V = \bar{\rho}_t (r_t + \gamma \bar{V}_\omega(z_{t+1}) - \bar{V}_\omega(z_t))$, and the clipped importance weights are $\bar{\rho}_t = \min\left(\bar{\rho}, \frac{\pi_\theta(a_t|z_t)}{\mu(a_t|z_t)}\right)$ and $c_i = \min\left(\bar{c}, \frac{\pi_\theta(a_i|z_i)}{\mu(a_i|z_i)}\right)$.


**2. Enhanced Target & Stop-Gradients (Epistemic Exploration):**
To provide the actor with an intrinsic exploration bonus without creating recursive gradient loops, the V-trace target is augmented with the epistemic variance across the value ensemble. This variance is computed using frozen weights (**stop-gradients**):
$$v^{enhanced}_s = v_s + \lambda \sqrt{\text{Var}(V_{\omega_i}(z_s))}$$


**3. Actor Loss (Policy Gradient with V-Trace):**
The actor updates its weights to maximize the probability of actions that resulted in a high enhanced advantage. The targets $v^{enhanced}$ and ensemble mean $\bar{V}_\omega$ are treated as constants here:
$$L_{actor}(\theta) = - \rho_s \log \pi_\theta(a_s|z_s) \left( r_s + \gamma v^{enhanced}_{s+1} - \bar{V}_\omega(z_s) \right)$$


**4. Critic Loss (Ensemble Mean Squared Error):**
Each head in the ensemble is updated independently to match the standard V-trace target:
$$L_{critic}(\omega_i) = \frac{1}{2} \left( V_{\omega_i}(z_s) - v_s \right)^2$$


**5. Entropy Bonus:**
$$L_{entropy}(\theta) = -\sum_{a \in \mathcal{A}_{valid}} \pi_\theta(a|z_s) \log \pi_\theta(a|z_s)$$


---


## 4. Future State Representation


### 4a. GNN Formulation
When hardware permits, the CNN will be replaced by a Heterogeneous Graph Neural Network (GNN) operating directly on $B = (\mathcal{V}, \mathcal{E})$, allowing for parameter-free scaling to arbitrarily large hex maps and unit counts.


### 4b. League Training
To explore the "Strategy Space" and prevent policy collapse, the self-play curriculum will expand into League Training. The agent will train against a historical database of frozen checkpoints (exploiters and main agents), guaranteeing robustness across the full policy space.


---


## 5. Approach for Desktop Formulation


### 5a. MegaMek API Contract
MegaMek must operate headlessly and expose a dual-purpose API:
* **The Forward Model:** Expose a deterministic `Checkpoint/Restore` API to simulate $g_{t+1}$ statelessly for idealized MCTS aspirations.
* **The Action Masking Engine:** At every step $k$ of the autoregressive sequence, the Java server MUST emit a **hierarchical Action Mask** (structured as a nested dictionary or tree) alongside the state tensor. Because the sampling is autoregressive, a flat boolean array is insufficient. The mask must allow the Python wrapper to dynamically traverse valid branches (e.g., locking `Valid Targets` based on the `Selected Unit`, and locking `Valid Weapons` based on the `Selected Target`).


### 5b. Explicit Feature Heuristics
To accelerate early learning, the network will ingest pre-computed tactical heuristics drawn from the MegaMek CASPAR/Princess bots and the BattleTech: Aces ruleset:
`Is_In_Optimal_Range_Bracket`, `Has_Optimal_TMM`, `Target_Is_Vulnerable`.


### 5c. Bootstrapping (Behavioral Cloning via Auto-Generation)
Starting RL from random weights in a sparse, high-dimensional space results in catastrophic initial variance. To "warm-start" the policy:
* **Trajectory Generation:** MegaMek is run headlessly at hyper-speed, playing thousands of matches of the built-in Princess AI against itself.
* **Data Extraction:** The $(s, a)$ pairs from these matches are intercepted and logged into a massive dataset.
* **Behavioral Cloning:** Prior to RL training, the Transformer network is trained to replicate Princess AI's logic via standard supervised cross-entropy loss. This provides a baseline competency in movement, line-of-sight, and weapon bracketing before the IMPALA optimization begins.


### 5d. Software Architecture (Asynchronous Queueing)
A Python-based Asynchronous Environment Manager will manage the headless MegaMek instances.
* **Actors:** CPU threads step environments using the behavior policy $\mu$, pushing $(s, a, \mu(a|s), r, \text{done}, \text{action\_mask\_tree})$ tuples into a high-speed shared memory queue.
* **Learner:** A dedicated GPU thread continuously pulls mini-batches from the queue, applies the V-trace correction using the current $\pi_\theta$, computes gradients, and asynchronously broadcasts the updated weights back to the Actors.


### 5e. Inference Time Considerations
By abandoning the nested Micro/Macro MCTS during live play and relying purely on the Transformer's autoregressive Ancestral Sampling, the resulting model object will be incredibly lightweight. Once trained, the forward pass requires minimal compute, enabling deployment on consumer CPUs without requiring dedicated GPU support.


---


## 6. Key Literature & References
The architecture is mathematically grounded in the following foundational Deep RL methodologies:
1.  **IMPALA & V-Trace:** Espeholt, L., et al. (2018). *IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures*.
2.  **Bi-Level RL & Entropy Regularization:** Zeng, S., et al. (2026). *A Hessian-Free Actor-Critic Algorithm for Bi-Level Reinforcement Learning with Applications to LLM Fine-Tuning*. (Provides the mathematical proof for single-loop, finite-time convergence in actor-critic setups).
3.  **Non-Linear Function Approximation:** Dong, J., et al. (2022). *Provably Efficient Convergence of Primal-Dual Actor-Critic with Nonlinear Function Approximation*.
4.  **Autoregressive Actions & Supervised Bootstrapping:** Vinyals, O., et al. (2019). *Grandmaster level in StarCraft II using multi-agent reinforcement learning* (AlphaStar).
5.  **Epistemic Exploration via Value Ensembles:** Osband, I., et al. (2016). *Deep Exploration via Bootstrapped DQN*.


---