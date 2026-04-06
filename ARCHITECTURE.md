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
* **Hex Nodes ($\mathcal{V}_H$):** Elevation, Terrain Type (Woods, Water, Pavement), Fire/Smoke presence, Objective Status.
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

### 1d. Action Features
While spatial targets are handled natively by the graph structure, non-spatial sub-actions (e.g., specific weapon selections) are represented dynamically via their covariates: Base Damage, Heat Generated, Range Brackets, Is_Cluster, and Cluster Size.

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
The agent utilizes an Importance Weighted Actor-Learner Architecture (IMPALA). To maximize hardware utilization, the system decouples trajectory generation from gradient computation.
* **The Actors:** Multiple CPU threads run headless environments, sampling actions from a slightly delayed behavior policy $\mu$ and pushing trajectories to a queue.
* **The Learner:** The GPU continuously pulls from the queue, updating the target policy $\pi_\theta$ and critic $V_\omega$.
* **V-Trace Off-Policy Correction:** Because the behavior policy $\mu$ lags behind the target policy $\pi_\theta$, the system utilizes V-trace clipping to strictly bound the variance of the importance sampling weights, guaranteeing mathematical convergence despite the asynchronous lag.

### 2b. Value Ensembles & Compute Optimization
To provide epistemic exploration without bottlenecking hardware, the critic architecture is heavily optimized:
* **Parameterization ($E$):** The number of value heads $E$ acts as a hyperparameter. Setting $E=1$ mathematically zeroes out the variance bonus, reverting to a pure entropy-driven exploration model.
* **Latent Caching:** The heavy Graph Encoder and Latent Fusion forward passes are computed only once. The ensemble heads $\{V_{\omega_1}, \dots, V_{\omega_E}\}$ branch directly from the cached global latent state $z$.

### 2c. Epistemic Exploration Bonus
The variance across the value ensemble quantifies the agent's epistemic uncertainty. In the IMPALA formulation, this variance is injected directly into the **V-trace target** ($v_s$) prior to calculating the policy gradient:
$$v^{enhanced}_s = v_s + \lambda \sqrt{\text{Var}(V_{\omega_i}(z_s))}$$

### 2d. Learning Curriculum
1.  **Static Gunnery:** 1v1 stationary targets (weapon optimization).
2.  **Maneuver & Fire:** Pathfinding, LOS, and TMM mechanics.
3.  **Objective Control:** Introduction of extraction zones and King-of-the-Hill hexes, forcing the agent to weigh scaled $\Delta \text{BV}$ against scaled $\Delta \text{ObjectiveScore}$.
4.  **The Baseline Duel:** Reactive tactics against the Princess AI.
5.  **Self-Play:** Symmetrical progression toward the Nash Equilibrium.

---

## 3. Network Architecture Representations

### 3a. Heterogeneous Graph Attention Network (HAN) Encoder
To ensure computational feasibility, translation invariance, and zero-padding efficiency across arbitrary map sizes, $B$ is explicitly modeled using a Heterogeneous Graph Attention Network (Wang et al., 2019).
* **Node Embeddings:** Separate linear projections map the raw covariates of $\mathcal{V}_H$ and $\mathcal{V}_U$ into a shared hidden dimension.
* **Intra-Meta-Path Attention:** The network computes attention weights $\alpha_{ij}$ specifically across defined physical edges ($\mathcal{E}_{adj}, \mathcal{E}_{occ}, \mathcal{E}_{LoS}$), preventing the need to implicitly learn raycasting physics.
* **Semantic Attention:** The network learns a semantic weight vector to fuse the different meta-paths for each node.
* **Output:** The HAN outputs a dense, context-aware embedding matrix for all nodes currently on the board: $H_{updated} \in \mathbb{R}^{|\mathcal{V}| \times d}$.

### 3b. Graph Readout & Latent Fusion ($z$)
Because the Critic evaluates the global game state, the dynamically sized node matrix $H_{updated}$ must be compressed into a fixed-size graph embedding before fusing with the phase context.
* **Readout Function (Global Attention Pooling):** A third, global layer of attention (Li et al., 2015) computes a scalar importance score for every node, allowing the network to dynamically focus on critical entities (e.g., highly damaged Mechs, contested objectives) while ignoring empty hexes:
$$z_{graph} = \sum_{i \in \mathcal{V}} \text{softmax}(W_{gate} h_i) \odot (W_{feat} h_i)$$
* **Phase Context:** The phase metadata $p$ is processed via an MLP: $p \rightarrow z_{context}$.
* **Latent Fusion:** The final global state representation is formed via concatenation:
$$z = z_{graph} \oplus z_{context}$$

### 3c. Value Ensemble Architecture (Critic Heads)
* **Input:** The ensemble takes the cached global latent state $z$ as its sole input.
* **Structure:** Each head $i \in \{1, \dots, E\}$ is an independent MLP with weights $\omega_i$.
* **Output:** Each head outputs $V_{\omega_i}(z)$. The variance across these outputs drives the epistemic exploration bonus.

### 3d. Action-Conditioned Pointer Execution (Autoregressive Actor Tree)
The actor policy utilizes an autoregressive, sequence-to-sequence decoder to navigate a dynamically generated **Action Tree** rather than a flat permutation array. Because MegaMek encompasses Movement (pathing, facing, modes) and Combat (target selection, weapon assignments), a flat action space suffers from combinatorial explosion (millions of permutations).

To solve this, actions are modeled as spatial branches within the heterogeneous graph.

#### The Hierarchical Action Tree ($\mathcal{T}$) and Action Nodes ($V_A$)
Given a valid action sequence length $N$ yielding discrete sub-actions $a_k$, the complete sequential execution is defined as $A = (a_0, a_1, \dots, a_{N-1})$. 

The Action Mask from Java represents a bounded subset of valid sequences: $\mathcal{T} = \{A_1, A_2, \dots, A_M\}$.
This forms a tree structure where each step $k$ branches based on the prefix condition $A_{<k}$. 

In PyTorch Geometric (`env.py`), an **Action Node** mathematically represents a valid branch choice within this tree: $(k, a_k \mid A_{<k})$. 
* **Movement Example**: 
  * $k=0$: $\mathcal{C}_0$ is the set of valid Root Actions (e.g. Unique `Target_Hex` indices).
  * $k=1$: Conditioned on the chosen hex $a_0$, $\mathcal{C}_1$ is the set of valid continuations (e.g. `[Final_Facing, MP_Used, Is_Jump]`).

Instead of a flat array, `env.py` instantiates graph nodes for all $a_k \in \mathcal{C}_k$ across the tree, mapping them using a `step_indices` tracking tensor.

#### Evaluated Action Embeddings ($e_{ak}$)
To evaluate sequence targets spatially, the Actor utilizes an `ActionMLP` against the Global state to map candidates into continuous embeddings. Let $e_{a_k^\ast}$ denote the evaluated spatial tensor specifically belonging to the ground-truth node $a_k^\ast$ chosen by Princess at step $k$:

$$e_{a_k^\ast} = \text{ActionMLP}( H_{unit} \oplus H_{target} \oplus X_{a_k^\ast} )$$

#### Autoregressive Execution Loop (Teacher Forcing)
In Sequence-to-Sequence learning, we aim to maximize the conditional probability of predicting the exact ground-truth sequence $A^\ast = (a_0^\ast, \dots, a_{N^\ast-1}^\ast)$ generated by Princess AI:

$$P(A^\ast | z) = \prod_{k=0}^{N^\ast-1} P(a_k = a_k^\ast \mid A_{<k}^\ast, z)$$

**Teacher Forcing** implies that during training, we actively substitute the network's potential mistake trajectories at step $k$ with the *True* chosen prefix $A_{<k}^\ast$ to perfectly condition the subsequent state generation. We feed this sequence directly into a Causal Transformer:

$$s_k = \text{TransformerDecoder}\left(z, [e_{a_0^\ast}, \dots, e_{a_{k-1}^\ast}] \right)$$

For the active step $k$, we utilize $s_k$ to evaluate dot-product Logits across all valid candidate Action Nodes available specifically at that tier ($\mathcal{C}_k$):
$$\text{Logits}(a_k) = s_k^T \cdot e_{a_k} \quad \forall a_k \in \mathcal{C}_k$$

#### Behavioral Cloning Target Objective ($\mathcal{L}$)
The objective function for a single graph activation trajectory $i$ is inherently the **Segmented Cross-Entropy Loss** applied independently at each sequence step and summed over the hierarchy length $N^{(i)}$:

$$\mathcal{L}_{BC}^{(i)} = -\sum_{k=0}^{N^{(i)}-1} \log \left( \frac{\exp((s_k^{(i)})^T \cdot e_{a_{k}^\ast}^{(i)})}{\sum_{a_j \in \mathcal{C}_{k}^{(i)}} \exp((s_k^{(i)})^T \cdot e_{a_j}^{(i)})} \right)$$

For PyTorch DataLoader execution, we optimize the expected loss over a randomized mini-batch size $\mathcal{B}$:
$$\mathcal{L}_{Batch} = \frac{1}{\mathcal{B}} \sum_{i=1}^{\mathcal{B}} \mathcal{L}_{BC}^{(i)}$$

This mathematically limits evaluations to the specific valid hierarchical branches drafted by the Action Mask, perfectly resolving combinatorial explosion while enforcing the constraints of causal Multi-Stage combat logic!

### 3e. Full Loss Statement & Gradient Updates (IMPALA V-Trace Formulation)
The total loss minimizes the policy gradient loss, the ensemble critic loss, and the entropy regularization bonus, corrected for off-policy generation via V-trace.
$$L_{total}(\theta, \omega) = \hat{\mathbb{E}}_{s} \left[ L_{actor}(\theta) + \frac{c_1}{E} \sum_{i=1}^E L_{critic}(\omega_i) - c_2 L_{entropy}(\theta) \right]$$

**Note on MDP Timesteps vs. Autoregressive Generation:**
The MDP timestep $t$ in the V-trace formulation corresponds strictly to a complete environment interaction. The sub-action sampling happens internally on the GPU. The action probability $\pi_\theta(a_t|z_t)$ evaluated during the loss calculation is the joint probability of the entire sampled token sequence: $\pi_\theta(a_t|z_t) = \prod_{k} \pi_\theta(c_k | z_t, c_{<k})$.

**1. The V-Trace Target ($v_s$):**
$$v_s = \bar{V}_\omega(z_s) + \sum_{t=s}^{s+n-1} \gamma^{t-s} \left( \prod_{i=s}^{t-1} c_i \right) \delta_t V$$
Where $\delta_t V = \bar{\rho}_t (r_t + \gamma \bar{V}_\omega(z_{t+1}) - \bar{V}_\omega(z_t))$, $\bar{\rho}_t = \min\left(\bar{\rho}, \frac{\pi_\theta(a_t|z_t)}{\mu(a_t|z_t)}\right)$, and $c_i = \min\left(\bar{c}, \frac{\pi_\theta(a_i|z_i)}{\mu(a_i|z_i)}\right)$.

**2. Enhanced Target & Stop-Gradients (Epistemic Exploration):**
$$v^{enhanced}_s = v_s + \lambda \sqrt{\text{Var}(V_{\omega_i}(z_s))}$$

**3. Actor Loss (Policy Gradient with V-Trace):**
$$L_{actor}(\theta) = - \rho_s \log \pi_\theta(a_s|z_s) \left( r_s + \gamma v^{enhanced}_{s+1} - \bar{V}_\omega(z_s) \right)$$

**4. Critic Loss (Ensemble Mean Squared Error):**
$$L_{critic}(\omega_i) = \frac{1}{2} \left( V_{\omega_i}(z_s) - v_s \right)^2$$

**5. Entropy Bonus:**
$$L_{entropy}(\theta) = -\sum_{a \in \mathcal{A}_{valid}} \pi_\theta(a|z_s) \log \pi_\theta(a|z_s)$$

---

## 4. League Training & Policy Robustness
To explore the "Strategy Space" and prevent policy collapse, the self-play curriculum utilizes League Training. The agent trains against a historical database of frozen checkpoints (exploiters and main agents), guaranteeing robustness across the full policy space without degenerating into highly specific counter-strategies.

---

## 5. Approach for Desktop Formulation

### 5a. MegaMek API Contract
MegaMek must operate headlessly and expose a dual-purpose API:
* **The Forward Model:** Expose a deterministic `Checkpoint/Restore` API to simulate $g_{t+1}$ statelessly.
* **The Action Masking Engine:** At every step $k$ of the autoregressive sequence, the Java server MUST emit a **hierarchical Action Mask** (structured as a nested dictionary or tree) alongside the state tensor to dynamically lock/unlock valid sub-actions.

### 5b. Explicit Feature Heuristics
To accelerate early learning, the network will ingest pre-computed tactical heuristics drawn from the MegaMek CASPAR/Princess bots: `Is_In_Optimal_Range_Bracket`, `Has_Optimal_TMM`, `Target_Is_Vulnerable`.

### 5c. Bootstrapping (Behavioral Cloning via Auto-Generation)
To warm-start the policy and reduce initial variance:
* MegaMek is run headlessly to generate thousands of Princess vs. Princess matches.
* Prior to RL training, the Transformer network is trained to replicate Princess AI's logic via standard supervised cross-entropy loss, establishing baseline competency in movement, line-of-sight, and weapon bracketing.

### 5d. Software Architecture & IPC Delta Serialization
A Python-based Asynchronous Environment Manager handles the headless MegaMek instances. To prevent the multiprocessing queue from bottlenecking on massive graph serializations, the IPC pipeline utilizes **Delta Encoding**:
* **Initialization:** Java sends the static topology ($\mathcal{V}_H$, $\mathcal{E}_{adj}$) once per match. The Python Actor caches this in RAM.
* **Step Payload (Deltas):** At each step, Java transmits only the dynamic state ($\mathcal{V}_U$ covariates, $\mathcal{E}_{occ}$, $\mathcal{E}_{LoS}$) and the current Action Mask Tree.
* **Actors:** CPU threads push these lightweight, flat arrays into a high-speed shared memory queue.
* **Learner:** A dedicated GPU thread pulls the flat arrays, dynamically reconstructs the PyG `HeteroData` batches using its own static cache, applies V-trace corrections, computes gradients, and asynchronously broadcasts updated weights back to the Actors.

### 5e. Inference Time Considerations
By relying purely on the Actor's autoregressive Ancestral Sampling, the resulting model object is incredibly lightweight. Once trained, the forward pass requires minimal compute, enabling deployment on consumer CPUs without requiring dedicated GPU support.

---

## 6. Key Literature & References
1.  **IMPALA & V-Trace:** Espeholt, L., et al. (2018). *IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures*.
2.  **SBEED (Primal-Dual Optimization):** Dai, B., et al. (2018). *SBEED: Convergent Reinforcement Learning with Nonlinear Function Approximation*. (ICML 2018). Provides the primal-dual framework addressing the double-sample issue and guaranteeing stable convergence for non-linear function approximation.
3.  **Non-Linear Function Approximation:** Dong, J., et al. (2022). *Provably Efficient Convergence of Primal-Dual Actor-Critic with Nonlinear Function Approximation*.
4.  **Autoregressive Actions:** Vinyals, O., et al. (2019). *Grandmaster level in StarCraft II using multi-agent reinforcement learning*.
5.  **Epistemic Exploration:** Osband, I., et al. (2016). *Deep Exploration via Bootstrapped DQN*.
6.  **Heterogeneous Graph Representation:** Wang, X., et al. (2019). *Heterogeneous Graph Attention Network*.
7.  **Global Attention Pooling:** Li, Y., et al. (2015). *Gated Graph Sequence Neural Networks*.