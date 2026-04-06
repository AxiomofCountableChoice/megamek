# BattleTech RL AI: Architecture & Formal Specification

## 1. Environment Specification: A Zero-Sum Markov Game
We formally define the game of BattleTech as a discrete-time, two-player, zero-sum Markov Game denoted by the tuple $\mathcal{M} = \langle \mathcal{G}, \mathcal{A}^1, \mathcal{A}^2, P, R, \gamma \rangle$.

### 1a. State Space ($\mathcal{G}$)
A state $g \in \mathcal{G}$ is defined as the tuple $(p, B)$.
* **The Spatial Graph ($B$):** Conceptually an augmented heterogeneous graph $B = (\mathcal{V}, \mathcal{E})$. $\mathcal{V} = \mathcal{V}_H \cup \mathcal{V}_U \cup \mathcal{V}_W$, representing terrain hex nodes, unit nodes, and instantiated weapon nodes. $\mathcal{E}$ represents the discrete, directional edges capturing structural adjacency, occupancy, threat geometry, and movement mechanics.
* **The Phase Vector ($p$):** A tuple acting as a granular state machine: $p = (\phi_{main}, \phi_{sub}, \iota, \tau, \omega)$.
    * $\phi_{main} \in \{1, \dots, 7\}$ denotes the primary game phase.
    * $\phi_{sub}$ denotes the sub-state, handling interstitial interrupts.
    * $\iota \in \{1, 2\}$ denotes the active player.
    * $\tau$ encodes the turn number and rounds left in the phase.
    * $\omega$ encodes objective-based scores.

### 1b. Graph Covariates
To accurately capture the state without parameter sparsity, nodes and edges in $B$ are enriched with specific covariates, while spatial physics are compressed into discrete edge types:
* **Hex Nodes ($\mathcal{V}_H$):** Elevation, Terrain Type (Woods, Water, Pavement), Fire/Smoke presence, Objective Status.
* **Unit Nodes ($\mathcal{V}_U$):** Current Heat, Heat Capacity, Armor/Structure arrays, Engine/Gyro critical hits, Pilot Consciousness, Is_Active_Unit, Is_Enemy.
* **Weapon Nodes ($\mathcal{V}_W$):** Base Damage, Heat Generated, Range Brackets, Cluster properties, Is_Operational.
* **Edges ($\mathcal{E}$):** Discretized and binned structural relationships (e.g., elevation deltas, target movement modifiers, cover bins) to maximize HGT routing efficiency.

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

## 3. The Edge-Aware Heterogeneous Graph Architecture

To accommodate the critical covariates embedded in BattleTech's spatial physics without falling victim to combinatorial feature explosion or parameter sparsity, the state graph $B$ is explicitly modeled as a directed Heterogeneous Graph. The forward pass is computed via a Heterogeneous Graph Transformer (HGT) (Hu et al., 2020).

### 3a. The Heterogeneous Graph Topology: $B = (\mathcal{V}, \mathcal{E})$

#### 1. Node Types ($\mathcal{V}$)
The spatial board and its entities are decomposed into three distinct node types.
* **Hex Nodes ($\mathcal{V}_H$):** Represents static terrain. Features: `[Elevation, Is_Water, Is_Pavement, Is_Objective]`.
* **Unit Nodes ($\mathcal{V}_U$):** Represents physical Mechs/Vehicles. Features: `[Current_Heat, Heat_Cap, Armor_Arrays, Internal_Arrays, Is_Active_Unit, Is_Enemy]`.
* **Weapon Nodes ($\mathcal{V}_W$):** Represents specific, instantiated physical components bolted to a unit. Features: `[Base_Damage, Heat_Gen, Min_Range, Short, Med, Long, Cluster_Size, Is_Operational]`.

#### 2. Edge Types ($\mathcal{E}$) & Adjacency Rules
Continuous spatial physics are compressed into discrete, directional, binned edge types $\phi(e)$ to maximize HGT routing efficiency.

**A. Structural Edges (Static & Semi-Static)**
* **Adjacency (Bidirectional but Asymmetric):** Connects adjacent $\mathcal{V}_H \to \mathcal{V}_H$. Encodes elevation changes natively. ($\mathcal{E}_{adj\_flat}$, $\mathcal{E}_{adj\_up\_[Delta]}$, $\mathcal{E}_{adj\_down\_[Delta]}$).
* **Occupancy:** $\mathcal{V}_U \xrightarrow{\mathcal{E}_{occ}} \mathcal{V}_H$. Defines exactly which hex a unit occupies.
* **Arsenal:** $\mathcal{V}_U \xrightarrow{\mathcal{E}_{equip}} \mathcal{V}_W$. Defines the active weapons. Deleted if a weapon is destroyed.

**B. Threat Geometry (Enemy $\to$ Hex)**
Computed relative to the current phase, these edges embed the enemy's spatial threat directly into the terrain.
* **Direct Threat:** $\mathcal{V}_{Enemy} \xrightarrow{\mathcal{E}_{threat\_[Cover\_Bin]}} \mathcal{V}_H$. Binned by intervening cover.
* **Indirect Fire (IDF):** $\mathcal{V}_{Enemy} \xrightarrow{\mathcal{E}_{threat\_IDF}} \mathcal{V}_H$. Projected by LRM/Arrow IV carriers to hexes illuminated by a friendly spotter's $\mathcal{E}_{threat}$ edge.
* **Deadzone:** $\mathcal{V}_{Enemy} \xrightarrow{\mathcal{E}_{threat\_deadzone}} \mathcal{V}_H$. Projected to hexes within the minimum-range bubble of the enemy's primary weapon payload.
* **Zone of Control (ZoC):** $\mathcal{V}_{Enemy} \xrightarrow{\mathcal{E}_{ZoC\_[Range\_Profile]}} \mathcal{V}_H$. Projects a 1-hop edge to any hex the enemy can physically reach next turn, binned by payload.

**C. Movement Actions (Active Unit $\to$ Hex)**
Generated exclusively during the Movement Phase for the active player's unit.
* **Ground Movement:** $\mathcal{V}_{Active} \xrightarrow{\mathcal{E}_{move\_[Type]\_[TMM\_Bin]}} \mathcal{V}_H$. (e.g., `walk_low`, `run_high`).
* **Jump Movement:** $\mathcal{V}_{Active} \xrightarrow{\mathcal{E}_{jump\_[TMM\_Bin]}} \mathcal{V}_H$. Separated to intrinsically encode the $+2$ attacker penalty and heat generation.

**D. Fire Control (Active Unit $\to$ Enemy Unit)**
Generated exclusively during the Weapon Attack Phase.
* **Line of Sight:** $\mathcal{V}_{Active} \xrightarrow{\mathcal{E}_{LoS\_[Distance\_Bin]\_[Cover\_Bin]}} \mathcal{V}_{Enemy}$. Encodes discrete range and cover modifiers.

### 3b. The Heterogeneous Graph Transformer (HGT) Forward Pass
The architecture relies on an HGT layer, utilizing discrete node types $\tau(n)$ and edge types $\phi(e)$ to select specific learned weight matrices, dynamically routing messages across the topology. For any directed edge $e$ connecting source $s$ to target $t$, the relation is defined as $\langle \tau(s), \phi(e), \tau(t) \rangle$.

**1. Node-Type Specific Projections:**
Because Query and Key matrices are strictly asymmetric ($W_{K-\tau} \neq W_{Q-\tau}$), the network preserves directionality even between homogeneous nodes.
$$Q(t) = h_t^{(l-1)} W_{Q\text{-}\tau(t)}$$
$$K(s) = h_s^{(l-1)} W_{K\text{-}\tau(s)}$$
$$V(s) = h_s^{(l-1)} W_{V\text{-}\tau(s)}$$

**2. Edge-Type Specific Attention:**
An edge-specific weight matrix $W^{ATT}_{\phi(e)}$ is sandwiched between the Query and Key vectors.
$$\text{Attention}(s, e, t) = \underset{\forall s \in \mathcal{N}(t)}{\text{Softmax}} \left( \frac{K(s) W^{ATT}_{\phi(e)} Q(t)^T}{\sqrt{d}} \cdot \mu_{\langle \tau(s), \phi(e), \tau(t) \rangle} \right)$$

**3. Edge-Type Specific Message Formulation:**
The payload passed from $s$ to $t$ is modulated by the specific edge connecting them.
$$\text{Message}(s, e, t) = V(s) W^{MSG}_{\phi(e)}$$

**4. Target-Specific Aggregation & Update:**
$$h_t^{(l)} = \text{GELU} \left( \left( \sum_{s \in \mathcal{N}(t)} \text{Attention}(s, e, t) \cdot \text{Message}(s, e, t) \right) W_{A\text{-}\tau(t)} \right) + h_t^{(l-1)}$$

### 3c. Graph Readout & Latent Fusion ($z$)
Because the Critic evaluates the global game state, the dynamically sized node matrix $H_{updated}$ must be compressed into a fixed-size graph embedding before fusing with the phase context.
* **Readout Function (Global Attention Pooling):** A third, global layer of attention computes a scalar importance score for every node, allowing the network to dynamically focus on critical entities while ignoring empty hexes:
$$z_{graph} = \sum_{i \in \mathcal{V}} \text{softmax}(W_{gate} h_i) \odot (W_{feat} h_i)$$
* **Phase Context:** The phase metadata $p$ is processed via an MLP: $p \rightarrow z_{context}$.
* **Latent Fusion:** The final global state representation is formed via concatenation:
$$z = z_{graph} \oplus z_{context}$$

### 3d. Value Ensemble Architecture (Critic Heads)
* **Input:** The ensemble takes the cached global latent state $z$ as its sole input.
* **Structure:** Each head $i \in \{1, \dots, E\}$ is an independent MLP with weights $\omega_i$.
* **Output:** Each head outputs $V_{\omega_i}(z)$. The variance across these outputs drives the epistemic exploration bonus.

### 3e. Action-Conditioned Pointer Execution (Autoregressive Actor Tree)
The actor policy utilizes an autoregressive, sequence-to-sequence decoder to navigate a dynamically generated **Action Tree** rather than a flat permutation array. Because MegaMek encompasses Movement (pathing, facing, modes) and Combat (target selection, weapon assignments), a flat action space suffers from combinatorial explosion.

#### The Hierarchical Action Tree ($\mathcal{T}$) and Action Nodes ($V_A$)
Given a valid action sequence length $N$ yielding discrete sub-actions $a_k$, the complete sequential execution is defined as $A = (a_0, a_1, \dots, a_{N-1})$. 

The Action Mask from Java represents a bounded subset of valid sequences: $\mathcal{T} = \{A_1, A_2, \dots, A_M\}$.
This forms a tree structure where each step $k$ branches based on the prefix condition $A_{<k}$. 

In PyTorch Geometric (`env.py`), an **Action Node** mathematically represents a valid branch choice within this tree: $(k, a_k \mid A_{<k})$. 
* **Movement Example**: 
  * $k=0$: $\mathcal{C}_0$ is the set of valid Root Actions (e.g. Unique `Target_Hex` indices).
  * $k=1$: Conditioned on the chosen hex $a_0$, $\mathcal{C}_1$ is the set of valid continuations (e.g. `[Final_Facing, MP_Used, Is_Jump]`).

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

### 3f. Full Loss Statement & Gradient Updates (IMPALA V-Trace Formulation)
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
6.  **Heterogeneous Graph Transformer:** Hu, Z., et al. (2020). *Heterogeneous Graph Transformer*.
7.  **Global Attention Pooling:** Li, Y., et al. (2015). *Gated Graph Sequence Neural Networks*.