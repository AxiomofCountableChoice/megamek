# Reinforcement Learning Agent for BattleTech Classic

## Abstract
In this document we specify a Markov Game with which to define a (two-player) game of BattleTech classic, under the TotalWar ruleset (ignoring naval / aerospace elements), upon which a reinforcement learning agent can be specified, with the intention of building such a model to play BattleTech classic.

---

## 1. BattleTech as a two-player zero-sum Markov game

### 1.1 Definition of the state-space
BattleTech is a game that features multiple stochastic elements, and additionally features multiple phases per turn, that require differing sets of actions from the players. BattleTech classic features 7 phases per turn:

1. **_Initiative Phase_**: Here the order of play is determined for the current turn, with the winner of initiative phase acting last in each of the subsequent phases - giving them an information advantage.
2. **_Movement Phase (Ground)_**: Movement of ground units, important considerations here include accrual of Target Movement Modifiers (TMM), positioning to avoid or gain Line of Sight (LOS) and to gain cover or take advantage of range bracketing. Additionally important for objective based situations. Additionally, charge or death-from-above (DFA) attacks are made via the movement phase.
3. **_Movement Phase (Aerospace)_**: As above, except for Aerospace units (will not emphasise this aspect of TotalWar).
4. **_Weapon Attack Phase_**: Ranged weapon phase, involves target declarations, twisting / rotating of torsos / turrets and resolution of LOS ambiguity (which itself is a tactical decision). All damage is determined simultaneously at the end of the phase.
5. **_Physical Attack Phase_**: Similar to the above, but specific to physical attacks (such as punching, kicks or use of melee weapons), damage again determined at the end of the phase.
6. **_Heat Phase_**: A phase where effects from heat build-up and dissipation occur.
7. **_End Phase_**: A phase where consciousness rolls and clean-up such as turret and torso position resets. This signals the end of the current turn and the start of the next one.

Given the structure above, there are two important bits of data - the current board configuration as well as data about the current turn and phase. We design the following representation of the game as an augmented heterogeneous graph (to use parlance the ML community will be familiar with).

We define the following sets of nodes:

1. **Entity Nodes ($\mathcal{V}_U$):** are entity nodes, which encode information about units partaking in the game. A set of node covariates are of the form:
   `unitClass_OneHot`, `unitRole`, `facingVector_SinCos`, `isOmnidirectional`, `cruisingSpeed`, `flankingSpeed`, `jumpDistance`, `GunnerySkill`, `PilotingSkill`, `currentHeat`, `heatCapacity`, `mechHeadInternal`, `mechHeadArmour`, $\dots$, `mechCenterTorsoArmourRear`, `vehicleFrontArmour`, $\dots$
2. **Hex Nodes ($\mathcal{V}_H$):** are hex nodes, which encode information about terrain and battlefield geometry. A set of node covariates are of the form:
   `elevationLevel`, `terrainType_OneHot`, `lightForestFlag`, `heavyForestFlag`, `fire/smokeFlag`, `objectiveVPValue`, `isExtractionZone`, $\dots$
3. **Weapon Nodes ($\mathcal{V}_W$):** are weapon nodes, which encode information about weapons mounted to mechs, vehicles or wielded by infantry. A set of node covariates are of the form:
   `minimumRange`, `shortRange`, `mediumRange`, `longRange`, `damageDealt`, `isOperational`, `isCluster`, `numClusters`, `salvosRemaining`, `heatGenerated`, `rapidFire`, $\dots$

We also define the following sets of edges, which are used to encode game specific geometry, we may incorporate more as we progress.

1. Edges for showing two hexes are adjacent, and which of the 6 hex-sides they connect via ($i \in \lbrace 0, \dots, 5 \rbrace$).
   $$ \mathcal{E}_{\text{hexAdj}_i}: \mathcal{V}_H \to \mathcal{V}_H $$
2. Edges for showing what hex a unit occupies.
   $$ \mathcal{E}_{\text{occupies}}: \mathcal{V}_U \to \mathcal{V}_H $$
3. Edges for showing what unit a weapon is equipped to.
   $$ \mathcal{E}_{\text{equips}}: \mathcal{V}_W \to \mathcal{V}_U $$
4. Edges for showing what hexes a unit can move to.
   $$ \mathcal{E}_{\text{moveTypeTMM}_{\text{bracket}_i}}: \mathcal{V}_U \to \mathcal{V}_H $$
5. Edges for showing which hexes an inactivated enemy can move to, which are in movement range of a unit to be activated.
   $$ \mathcal{E}_{\text{movementThreat}}: \mathcal{V}_U \to \mathcal{V}_H $$
6. Edges for showing which hexes are in LOS of enemy units, which are in movement range of a unit to be activated. A consideration required is if it's possible to compute as a function of the unit to be activated, as their height will impact LOS calculations.
   $$ \mathcal{E}_{\text{LOSThreat}}: \mathcal{V}_U \to \mathcal{V}_H $$
7. Edges for showing which enemies are in LOS of a unit to be activated.
   $$ \mathcal{E}_{\text{LOSTarget}}: \mathcal{V}_U \to \mathcal{V}_U $$
8. Edges for showing which hexes a unit to be activated have partial cover from already activated enemy units.
   $$ \mathcal{E}_{\text{partialCover}}: \mathcal{V}_U \to \mathcal{V}_H $$

Defining the sets:
$$ \mathcal{V} = \mathcal{V}_U \cup \mathcal{V}_W \cup \mathcal{V}_H, \quad \mathcal{E} = \bigcup_{e \in \mathcal{T}} \mathcal{E}_{e}, \quad \mathcal{T} = \lbrace \text{hexAdj}_i, \text{occupies}, \text{equips}, \dots \rbrace $$
we have that the graph $(\mathcal{V}, \mathcal{E})$ represents the game board-state at any given time. However, to capture the meta information, we need as well to have information regarding the current phase, information regarding who had initiative during the turn, etc.

We let the space phase meta-data be $\mathcal{P}$, we note that each element $p \in \mathcal{P}$ is a vector which encodes:
* The turn number
* The phase of the turn
* Which player is to take action
* How many rounds are left in the phase
* Scores / information regarding objective based play

Let the space of all valid board-states be $\mathcal{B}$, and let $\mathcal{G} \subset \mathcal{P} \times \mathcal{B}$ be the set of all legal game-states. Then we have that the state of a game can be represented as an element:

$$ (p, B) \in \mathcal{G}, \:\: \exists (\mathcal{V}, \mathcal{E}) \in \mathcal{B}: B = (\mathcal{V}, \mathcal{E}). $$

This allows us to hence define the major component of our state-space, an encoding of the game.

### 1.2 Definition of the action-space
BattleTech is a complicated game to model, as the action space is incredibly sparse, and dependent on the current game-state (e.g. the phase would determine what sorts of actions would be possible, say movement is not allowed during the attack phases, or what combination of weapons and enemy units to attack that the unit selected for firing has in LOS, etc) - and so care is required when attempting to model it. It is not feasible to enumerate every possible action (as this would be an incredibly large number across all legal boards).

However, in BattleTech any action is formed by a set of smaller atomic decisions before the "game state progresses" (i.e. there is a step in the games dynamics under the next-state transition kernel), after a declaration is formally made. For this reason, instead we seek to define a hierarchical structure mapping between the set of legal actions, and a set of features which can be utilised to estimate the best action to take, essentially a form of 'action embedding' in order to allow for dealing with arbitrarily large sets of actions.

We define $\mathcal{A}$ as the space of all actions, and $\mathcal{A}_g$ as the set of legal actions for the given game-state $g \in \mathcal{G}$. This legal action set $\mathcal{A}_g$ has a structure that is a union of disjoint trees, with the root nodes being the unit to activate in the phase, and the nodes being the 'atomic sub-actions', with edges showing the steps between them, and leaf nodes being the `declarationComplete` terminal state. By introducing an arbitrary `declarationState` node, we can create a single tree, with each of the child nodes being a yet-to-be-activated unit. We seek to enumerate the 'sub-actions' per phase now:

1. **Movement Phase**:
   `declarationStart`, `selectActiveUnit`, `selectMovementType`, `destinationHex`, `facing`, `declarationComplete`
2. **Weapon Attack Phase**:
   `declarationStart`, `selectActiveUnit`, `torsoTwist`, `selectTargetUnit_i`, `selectWeapon_j`, `declarationComplete`
   where $i$ and $j$ range over the legal targets and legal-to-fire mounted weapons for the selected unit to activate. 
3. **Physical Attack Phase**:
   `declarationStart`, `selectActiveUnit`, `selectTargetUnit`, `selectAttackType`, `declarationComplete`
   for simplicity, we have ignored both of the physical attacks that require declaration during the movement phase, charging and death-from-above attacks.

Additionally, we allow for deterministic mechanisms to resolve all other declarations. Technically turning on / off heat-sinks and dumping ammo as part of the other heat and end-phases will be ignored for now.

As a final point, we recognise that the action space is actually defined as the Cartesian product of the two players action spaces, specifically we define $\mathcal{A} = \mathcal{A}^1 \times \mathcal{A}^2$, where the non-active players valid action space is simply $\emptyset$.

Letting the active player be $\eta \in \lbrace 0, 1 \rbrace$, then we have that we represent $\mathcal{A}_g^{\eta}$ as the tree-structure described above, having nodes $ \mathcal{V}_{\mathcal{A}_g^{\eta}}$ representing the set of sub-actions, with edges between consecutive sub-actions being given by the valid action-mask that Megamek generates. We associated $\mathcal{A}_g^{\eta}$ with the valid action mask structure.

### 1.3 Specification of the transition dynamics
Considering the formulation of the problem as a turn-based Markov game, we can specify the transition dynamics via the following transition measure:

$$ \mathbb{P}:  \mathcal{G} \times \mathcal{A}^1 \times \mathcal{A}^2 \mapsto \mathcal{M}_{\mathcal{G}} $$

where $\mathcal{M}_{\mathcal{G}}$ is the space of probability measures over $\mathcal{G}$. We provide the following informal justification that this is a valid transition kernel. Conditional on the game state $g \in \mathcal{G}$ and valid action space $\mathcal{A}_g^1 \times \mathcal{A}_g^2$, any declared action $a^1 \times a^2 \in \mathcal{A}_g^1 \times \mathcal{A}_g^2$, will generate either a series of dice-rolls against target numbers that either succeed or fail, or an action completes successfully. This generates a new game-state $g^{\prime}$ according to the successful / failed rolls, leading to the next action to be sampled. This next-state distribution is fully determined by the current game-state $g$ and declared action $a^1 \times a^2$, leading to the Markovian property claimed.

### 1.4 Reward Structure ($R$)
To guarantee a competitive Nash Equilibrium and prevent cooperative local minima (e.g., a "Truce" where agents refuse to act to avoid penalties), the environment strictly enforces a Zero-Sum constraint at every time-step: $\mathcal{R}^{(1)}_t = -\mathcal{R}^{(2)}_t$. Furthermore, while the ideal reward (to match the true game reward and Nash Equilibrium) is just the win-loss state, given how sparse the rewards are, we introduce auxiliary dense rewards.

The rewards cover the change in Battle Value (BV) experienced by players during combat, Victory Points (VP) covering normally the win-loss state but extending to interim scores for objective-based scenarios, and Tertiary Penalties (TP). We introduce hyper-parameters $\beta_{obj}, \beta_{BV}, \beta_{TP} \in [0, \infty)$ (where we would likely have $\beta_{obj} > \beta_{BV} > \beta_{TP}$) provide weightings of each reward:

$$ \mathcal{R}^{(1)}_t = \beta_{BV} \left( \frac{\Delta \text{BV}^{(2)}_t - \Delta \text{BV}^{(1)}_t}{\text{TotalMatchBV}} \right) + \beta_{obj} \left( \frac{\Delta \text{VP}_t^{(1)} - \Delta \text{VP}_t^{(2)}}{\text{MaxVP}} \right) + \beta_{TP} \left( \sum_{i \in \texttt{units}_2}\text{TP}^{(2)}_{i, t} - \sum_{j \in \texttt{units}_1} \text{TP}^{(1)}_{j, t} \right) $$

Let $\mathbb{I}$ denote an indicator function. Tertiary penalties shape behaviour strictly around rule-set de-buffs or unforced Piloting Skill Rolls (PSRs). Any penalty incurred by Player 1 is explicitly awarded to Player 2, preserving the zero-sum mirror and natively rewarding the offensive use of heat-inducing weapons (e.g., Flamers) - for the $i$-th unit for player $\eta \in \lbrace 1, 2 \rbrace$ on the $t$-th round:

$$ \text{TP}^{(\eta)}_{i, t} = \epsilon_{mov}\mathbb{I}_{\lbrace\text{Heat} \ge 5\rbrace} + \epsilon_{acc}\mathbb{I}_{\lbrace\text{Heat} \ge 8\rbrace} + \epsilon_{shut}\mathbb{I}_{\lbrace\text{Heat} \ge 14\rbrace} + \epsilon_{ammo}\mathbb{I}_{\lbrace\text{Heat} \ge 19 \land \text{Ammo}\rbrace} + \epsilon_{psr}\mathbb{I}_{\lbrace\text{UnforcedPSR}\rbrace} $$

### 1.5 Definition of the Markov Game
We formally define the game of BattleTech as the discrete-time, fully-observable Markov Game tuple:

$$ \mathcal{M} = \langle \mathcal{G}, \mathcal{A}^1, \mathcal{A}^2, \mathbb{P}, \mathcal{R}, \gamma \rangle $$

---

## 2. Neural Network Architecture

### 2.1 Heterogeneous Graph Transformer (HGT)
The state graph $g_t = (b_t, p_t)$, of board state $b_t$ and phase meta-data $p_t$ is processed via an HGT layer. We introduce the nodes $n \in \mathcal{V}_{b_t}$ and edges $\varepsilon \in \mathcal{E}_{b_t}$ as the edges and nodes of the current state graph $b_t$. Let $\tau$ and $\phi$ be the node type and edge type identifier functions and $\mathcal{N}_{b_t}(n)$ represent the 1-hop neighbourhood of node $n$ across all edge types.

For discrete node types $\tau(n)$ and edge types $\phi(e)$ select specific learned weight matrices to dynamically route messages. For a directed edge $\varepsilon_{s,\iota}$ from source $n_s$ to target $n_{\iota}$:

$$ Q(n_{\iota}) = h_{n_{\iota}}^{(l-1)} W_{Q\text{-}\tau(n_{\iota})}, \quad K(n_s) = h_{n_s}^{(l-1)} W_{K\text{-}\tau(n_s)}, \quad V(n_s) = h_{n_s}^{(l-1)} W_{V\text{-}\tau(n_s)} $$

$$ \text{Attention}(n_s, \varepsilon_{s,\iota}, n_{\iota}) = \underset{\forall n_s \in \mathcal{N}_{b_t}(n_{\iota})}{\text{Softmax}} \left( \frac{K(n_s) W^{ATT}_{\phi(\varepsilon_{s,\iota})} Q(n_{\iota})^T}{\sqrt{d}} \cdot \mu_{\langle \tau(n_s), \phi(\varepsilon_{s,\iota}), \tau(n_{\iota}) \rangle} \right) $$

$$ h_{n_{\iota}}^{(l)} = \text{GELU} \left( \sum_{n_s \in \mathcal{N}_{b_t}(n_{\iota})} \text{Attention}(n_s, \varepsilon_{s,\iota}, n_{\iota}) \cdot \left(V(n_s) W^{MSG}_{\phi(\varepsilon_{s,\iota})}\right) W_{A\text{-}\tau(n_{\iota})} \right) + h_{n_{\iota}}^{(l-1)} $$

The dynamically sized node matrix is compressed into a fixed-size graph embedding via Global Attention Pooling and concatenated with the phase context MLP to form the global latent state $z_t$:

$$
\begin{aligned}
z_{graph} &= \text{GlobalAttentionPooling}\left(\text{HGT}(b_t)\right)\\
&= \text{GlobalAttentionPooling}\left(\lbrace h_{n_i}\rbrace_{n_i \in \mathcal{V}_{b_t}}\right)\\
&= \sum_{n_i \in \mathcal{V}_{b_t}} \text{softmax}(W_{gate} h_{n_i}) \odot (W_{feat} h_{n_i})\\
\implies z_t &= z_{graph} \oplus \text{MLP}(p_t).
\end{aligned}
$$

### 2.2 Action-Conditioned Pointer Execution (Actor)
The Actor policy $\pi_{\theta}$ navigates the valid action space $\mathcal{A}_{g_t}^{\eta}$ using an autoregressive causal transformer. Candidates sub-actions are mapped to continuous embeddings via an `ActionMLP`:

$$ e_{a_k} = \text{ActionMLP}( h_{\text{active\_unit}} \oplus h_{\text{target\_node}} \oplus x_{a_k} ) \quad \forall a_k \in V_{\mathcal{A}_g^{\eta}} $$

where $x_{a_k}$ represents features that we compute for the given sub-action $a_k$, and $V_{\mathcal{A}_g^{\eta}}$ is the set of nodes in the tree defined by $\mathcal{A}_{g_t}^{\eta}$. The causal transformer processes the latent state $z_t$ and prefix embeddings to generate a query vector $s_k$. Policy logits are evaluated via a pointer-network dot product:

$$ s_k = \text{TransformerDecoder}\left(z_t, [e_{a_0}, \dots, e_{a_{k-1}}] \right), \quad \text{Logits}(a_k) = s_k^T \cdot e_{a_k}. $$

Letting $\varphi(a)$ be the index for the terminal node of action $a$, and $\mathcal{C}(a)$ be the children of node $a$, this then defines the actor policy as:

$$ \pi_{\theta}(a \mid g_t) = \prod_{k=1}^{\varphi(a)} \frac{\exp\left(s_{k}^T \cdot e_{a_k}\right)}{\sum_{\zeta \in \mathcal{C}(a_{k-1})} \exp\left(s_{k}^T \cdot e_{\zeta}\right)} $$

where the root node $a_0$ is fixed and hence ignored in the above conditional probability mechanism.

### 2.3 Aleatoric & Epistemic Uncertainty Considerations
To model the heavy-tailed aleatoric variance of the 2d6 and target-number randomisation mechanism in BattleTech, Distributional RL (e.g., Implicit Quantile Networks - IQN) was considered to model the future-return distribution. However, the additional computational cost and complexity in representation means this will be reserved for future extensions. Note that the value estimator $V_{\omega}$ is defined as:

$$ V_{\omega}(g_t) = \text{ValueMLP}(z_t), \quad z_t = \text{GlobalAttentionPooling}\left(\text{HGT}(b_t) \right) \oplus \text{MLP}(p_t) $$

Epistemic uncertainty is approached via an ensemble of $E$ independent critic heads $V_{\omega_e}(z_t)$. The balance between exploration and exploitation is naturally driven by the posterior concentration of the ensemble variance

$$ \bar{V}_{\omega}(g_t) = \mathbb{E}_E \left[ V_{\omega_e}(g_t) \right], \quad \sigma_{\omega}^2(g_t) = \mathbb{V}_E \left[ V_{\omega_e}(g_t) \right] $$

where $\omega = \oplus_{e=1}^E \omega_e$, and we use a bit of abuse of notation to represent the ensemble parameters.

---

## 3. Optimization Framework: IMPALA

### 3.1 V-Trace & Entropy Smoothing
Aligned with the algorithmic implementation outlined within Espeholt et. al. (IMPALA), we compute clipped importance weights in order to compute the V-trace, and utilise an Entropy regularisation term in order to ensure that we do not have vanishing importance sampling weights. In the following we assume that we have $N$ samples of trajectories

$$ \lbrace g_{t,i}, a_{t,i}, r_{t,i}\rbrace_{t = 1}^{T_i} \sim \mu_i $$

generated under some behaviour policy $\mu_i$, for $i \in 1, \dots, N$, and $\hat{\mathbb{E}}$ is the empirical expectation over samples across these trajectories from some replay-buffer capturing them.

We note that the $\mu_i$ in our case will be the actor network defined above, but instantiated with an older iteration of the model parameters, call then $\tilde{\theta}$, that is $\mu_i = \pi_{\tilde{\theta}}$, similarly the value functions used in the V-trace, they utilise prior iteration parameters $\tilde{\omega}$. For simplicity in the following, we drop the explicit references to each trajectory sample $i$.

#### 3.1.1 The V-Trace Target ($v_t(\omega)$)
Recall the V-trace is defined as

$$ v_t(\omega) = \bar{V}_\omega(g_t) + \sum_{k=t}^{T_i \wedge (t+n-1)} \gamma^{k-t} \left( \prod_{j=t}^{k-1} c_j \right) \delta_k V $$

where $\delta_k V = \bar{\rho}_k (r_k + \gamma \bar{V}_\omega(g_{k+1}) - \bar{V}_\omega(g_k))$, $\bar{\rho}_k = \min\left(\bar{\rho}, \frac{\pi_{\theta}(a_k \mid g_k)}{\mu(a_k \mid g_k)}\right)$, and $c_j = \min\left(\bar{c}, \frac{\pi_{\theta}(a_j \mid g_j)}{\mu(a_j \mid g_j)}\right)$, and $\bar{\rho}, \bar{c} \in (0,1)$ and $\bar{c} \leq \bar{\rho}$. In particular we see the following recursive representation

$$ v_t(\omega) = \bar{V}_{\omega}(g_t) + \delta_t V + \gamma \cdot c_t \left( v_{t+1}(\omega) - \bar{V}_{\omega}(g_{t+1}) \right). $$

#### 3.1.2 Actor Gradient (Policy Gradient with V-Trace)
To incorporate our epistemic exploration bonus, we bake into the advantage function utilised for the actor loss a posterior-variance term determined by the ensemble as a form of bonus toward unexplored state, action pairs. We express the gradient of the actor as

$$ \nabla_{\theta} \hat{L}_{actor}(\theta) = -\hat{\mathbb{E}} \left[ \rho_t \nabla_{\theta} \left[ \log \pi_\theta(a_t \mid g_t) \right] \left( r_t + \gamma \lbrace v_{t+1}(\omega) + \lambda \sqrt{\sigma_{\omega}^2(g_{t+1})} \rbrace - \bar{V}_\omega(g_t) \right) \right]. $$

We see that for larger ensemble variance we have a larger gradient for the actor, and hence the gradient nudges actor parameters towards values that maximise the expected returns, and so which place more mass on state, action pairs that have higher return as estimated by the V-trace or epistemic variance. Furthermore, to encourage that we do not have vanishing mass on actions for our actor policy, we employ the aforementioned entropy regularisation term

$$ \nabla_{\theta} \hat{L}_{entropy}(\theta) = \hat{\mathbb{E}} \left[ \sum_{a \in \mathcal{A}_{valid}} \nabla_{\theta} \left[ \pi_\theta(a \mid g_t) \log \pi_\theta(a \mid g_t) \right] \right]. $$

which is the negative gradient of the entropy of $\pi_{\theta}$, which seeks to maximise its entropy, encouraging the model to not place all its mass onto a single action.

#### 3.1.3 Critic Gradient (Ensemble Mean Squared Error)
For the critic, we apply the usual $L^2$ loss, and may explore instead formulating as an IQN objective in future:

$$ \nabla_{\omega_i}\hat{L}_{critic}(\omega_i) = \hat{\mathbb{E}} \left[ 2 \cdot \left( V_{\omega_i}(g_t) - v_t(\tilde{\omega}) \right) \cdot \nabla_{\omega_i} \left[ V_{\omega_i}(g_t) \right] \right] $$

where $i = 1, \dots, E$ for each critic head in the ensemble. Note that the V-trace $v_t(\tilde{\omega})$ is not computed under the parameters $\omega_i$, which the gradient is taken with respect to. Instead they are with respect to the parameters used by the behaviour policy, or the current parameters prior to the update, the $\tilde{\omega}$ defined earlier.

In the case of instability, an exponential-moving-average of historical and updated critic weights could be computed to update the v-trace targets, akin to the approach of DQN to allow for convergence under a two time-scale induced stochastic-approximation system, e.g. as

$$ \tilde{\omega}_i = (1 - \eta_{\text{m}}) \cdot \tilde{\omega}_i + \eta_{\text{m}} \cdot \omega_i. $$

---

## 4. Bootstrapping, Behavioural Cloning, Curriculums & League Training
Due to the extreme sparsity of the reward space, learning _tabula rasa_ is highly inefficient. To warm-start the network, MegaMek is run headlessly to generate thousands of historical trajectories using its native Princess AI. 

### 4.1 Behavioural Cloning via Auto-Generation
Prior to RL training, the Actor network undergoes Behavioural Cloning via Teacher Forcing. The objective is the Segmented Cross-Entropy Loss applied independently at each sequence step and summed over the hierarchy length $K$:

$$ \mathcal{L}_{BC} = \hat{\mathbb{E}} \left[ -\sum_{k=0}^{K_t-1} \log \left( \frac{\exp((s_{t,k})^T \cdot e_{a_{t,k}^{\ast}})}{\sum_{a_j \in \mathcal{C}(a_{k-1})} \exp((s_{t,k})^T \cdot e_{a_j})} \right) \right] $$

where $\lbrace g_t, a_t = \lbrace a_{t,k}^{\ast} \rbrace_{k=0}^{K_t-1}\rbrace_{t=1}^T$ are the observations used to compute the loss estimate. This establishes baseline competency in movement, line-of-sight pathing, and weapon bracketing before the IMPALA gradients are engaged.

### 4.2 Learning Curriculum & League Training
Following Behavioral Cloning, the RL agent undergoes a phased learning curriculum:
1. **_Static Gunnery:_** 1v1 stationary targets.
2. **_Manoeuvre \& Fire:_** Pathfinding, LOS, and TMM mechanics.
3. **_Objective Control:_** Introduction of extraction zones and King-of-the-Hill hexes.
4. **_The Baseline Duel:_** Reactive tactics against the Princess AI.
5. **_Self-Play \& League Training:_** Symmetrical progression toward the Nash Equilibrium. The self-play curriculum utilizes League Training, pitting the agent against a historical database of frozen checkpoints (exploiters and main agents) to guarantee robustness without degenerating into highly specific counter-strategies.

### 4.3 Future Considerations
Depending on the performance of the learner, there are a few additional pathways to consider in order to improve its performance and ability to plan - particularly given the aleatoric uncertainty in BattleTech, having a MCTS derived planner could be incredibly effective. This would necessitate having a learning structure akin to that of the MuZero learners.

In particular, leveraging an IQN structure in addition to MuZero style dynamics-learner might be the most appropriate way to model the learning process for BattleTech. In particular if it allows for learning without the dense reward shaping, as it is then closer to being able to learn / represent the true Nash Equilibrium, rather than the modified Markov-Game we have introduced above. The recent work of Maes et. al (LeWorldModel), and in particular the use of the SIGReg regularisation term (rather than the $L^2$ regulariser of MuZero) on the dynamics learner and encoder, might be beneficial to incorporate.

One other important fact to note, is that in many ways MuZero approaches the problem very differently from that of traditional RL algorithms that rely on direct Bellman iteration. Here the approach is instead to learn system dynamics and then utilise MCTS on the learnt dynamics to attempt to learn the optimal action from a given state, this has some theoretical justification, but is heavily reliant on accurately learning system dynamics, which themselves are dependent on the actions of the agent policy. This may (in some situations) lead to instability (particularly in very noisy environments like BattleTech) as there is less smoothing / aggregation gained in the learning process as compared to Bellman-iteration related flavours. The value of approaches like that of Dai et al. is that they focus on learning the optimal process for a fixed representation quite well, and hence might be suited to the "fixed framework" of BattleTech.

---

## 5. Software Architecture & Implementation

### 5.1 MegaMek API Contract & Delta Serialization
A Python-based Asynchronous Environment Manager handles the headless MegaMek instances. To prevent the multiprocessing queue from bottlenecking on massive graph serializations, the IPC pipeline utilizes _Delta Encoding_:
* **_Initialization:_** Java sends the static topology ($\mathcal{V}_H$, $\mathcal{E}_{adj}$) once per match. The Python Actor caches this in RAM.
* **_Step Payload (Deltas):_** At each step, Java transmits only the dynamic state ($\mathcal{V}_U$ covariates, $\mathcal{E}_{occ}$, $\mathcal{E}_{LoS}$) and the current hierarchical Action Mask Tree.
* **_Actors:_** CPU threads push these lightweight, flat arrays into a high-speed shared memory queue.
* **_Learner:_** A dedicated GPU thread pulls the arrays, dynamically reconstructs the PyG batches, computes gradients, and asynchronously broadcasts updated weights back to the Actors.

### 5.2 Inference Time Considerations
By relying purely on the Actor's autoregressive Ancestral Sampling, the resulting model object is incredibly lightweight. Once trained, the forward pass requires minimal compute, enabling deployment on consumer CPUs without requiring dedicated GPU support.

---

## 6. References
1. **IMPALA \& V-Trace:** Espeholt, L., et al. (2018). *IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures*.
2. **SBEED (Primal-Dual Optimization):** Dai, B., et al. (2018). *SBEED: Convergent Reinforcement Learning with Nonlinear Function Approximation*. ICML 2018.
3. **Heterogeneous Graph Transformer:** Hu, Z., et al. (2020). *Heterogeneous Graph Transformer*. WWW 2020.
4. **Autoregressive Actions in RL:** Vinyals, O., et al. (2019). *Grandmaster level in StarCraft II using multi-agent reinforcement learning*. Nature.
5. **Epistemic Exploration Ensembles:** Osband, I., et al. (2016). *Deep Exploration via Bootstrapped DQN*. NIPS 2016.
6. **Implicit Quantile Networks:** Dabney, W., et al. (2018). *Implicit Quantile Networks for Distributional Reinforcement Learning*. ICML 2018.
7. **MuZero:** Schrittwieser, J., et al. (2020). *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model*. Nature 2020.
8. **LeWorldModel:** Maes, L., et al. (2026). *LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels*. Arxiv Pre-print 2026.