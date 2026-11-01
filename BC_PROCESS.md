# Implementation Blueprint: Streamlined Hybrid Imitation Learning (HIL) & Perturbed State Initialization

## Context & Objectives
To solve the sample inefficiency of pure Reinforcement Learning in BattleTech on local compute while preventing the agent from permanently plateauing at the sub-optimal heuristics of the baseline opponent (Princess AI), our pipeline uses a **Streamlined Hybrid Imitation Learning (HIL)** architecture combining:

1. **Offline Behavioral Cloning (BC) Warm-Start (`bc_generator.py` & `train_bc.py`)**: Supervised teacher-forcing over the Unified Action Tree and bootstrapped Monte Carlo Critic pre-training across all $E=8$ heads.
2. **Perturbed State Initialization (`PSI`)**: Launching a fraction ($P_{\text{PSI}} = 0.25$) of live RL rollouts directly into mid-game `.sav.gz` snapshots (Rounds 4–12) to break early-game horizon bottlenecks.
3. **Dual-Buffer Co-Training with Annealed Supervised BC (`impala_master.py`)**: Co-training off-policy V-Trace RL ($\mathcal{L}_{\text{RL}}$) alongside an annealed supervised BC anchor ($\lambda_{\text{BC}}(t)\mathcal{L}_{\text{BC}}$) that stabilizes the HGT Encoder and Actor while the Critic Ensemble calibrates, before decaying cleanly to $0$.

---

## 1. Perturbed State Initialization (PSI)

**The Goal:** Break state-distribution correlation in the replay buffer. If the agent always spawns at Round 1, it only experiences high-heat, close-range brawling, internal structure crits, and physical attacks after surviving long sequences of opening maneuvers.

**Implementation Architecture:**
1. **Mid-Game Snapshot Archive (`data/midgame_saves/`):**
   * During headless matches (`PrincessOfflineTrainer.java` with `-autoSaveMidGame`), `RLServerManager.java` intercepts `END_REPORT` at Rounds 4, 8, and 12 and saves compressed `.sav.gz` snapshots.
2. **Orchestrator Reset Sampling (`self_play_orchestrator.py`):**
   * With probability `--p-psi` (default `0.25`), `megamek_runner` launches MegaMek with `-savegame <random_midgame_save.sav.gz>`.
   * Otherwise, it launches a fresh Round 1 match (`-randomMap -randomBV ...`).
3. **Player-Identity Remapping & True Initial BV Preservation (`PrincessOfflineTrainer.java` & `RLDataPipeline.java`):**
   * Because `.sav.gz` snapshots serialize the original player names from when the save was created (e.g., `"Princess_Alpha"` and `"Princess_Beta"`), `PrincessOfflineTrainer.java` immediately remaps the loaded `Player` names in `server.getGame()` after `server.loadGame(gameFile)` to match the active mode (`"RL_Agent_Alpha"` / `"RL_Agent_Beta"` in `-selfplay`, or `"RL_Agent"` / `"Princess_Beta"` in `-autogen`). This prevents MegaMek's `Server.addNewPlayer` from creating unitless ghost players.
   * To ensure $\Delta\text{BV}$ rewards are normalized against the true match-start force BV rather than the damaged mid-game residual BV, `RLDataPipeline.calculateRewards()` computes `initial_bv1` and `initial_bv2` by summing `e.getInitialBV()` across both `game.getEntitiesVector()` and `game.getOutOfGameEntitiesVector()`.

---

## 2. Why the GAIL Adversarial Style Discriminator Was Removed

An earlier design considered a Generative Adversarial Imitation Learning (GAIL) Style Discriminator $D_\psi(\text{sg}(s_0), \text{sg}(e_a), \text{sg}(\text{Context}_a))$ to supply a dense step-level reward $r_{\text{style}}$. Upon mathematical and implementation audit, $D_\psi$ was removed in favor of **Option A (Annealed $\mathcal{L}_{\text{BC}}$ + PSI)** for three fundamental reasons:

1. **Zero Feature-Learning Utility:** Because inputs to $D_\psi$ had to be detached (`stop_gradient`) to prevent adversarial instability in the HGT and Perceiver layers, $\mathcal{L}_{\text{discrim}}$ never updated a single weight in the HGT Encoder or Actor.
2. **Board-State Occupancy Leakage:** Because $D_\psi$ conditioned on $s_0 = \text{Linear}(z_{\text{global}})$, it could trivially distinguish *who was playing* from the board state (e.g., heat distribution or spatial spread accumulated over prior turns) rather than evaluating the quality of the current action $a_k$.
3. **Asymptotic Ceiling Against Outplaying Princess:** Any state-action pair $(g_t, a_t)$ where the RL agent discovers a superior tactical play that Princess would not take is classified by $D_\psi$ as $0$ ("non-Princess"), actively penalizing superhuman play in the reward function.

---

## 3. Dual-Buffer Co-Training & Configurable Annealing (`impala_master.py`)

**The Goal:** Prevent catastrophic forgetting of basic BattleTech mechanics during the initial phase of RL while the 8 Critic heads calibrate their V-Trace value estimates on live rollout returns, then smoothly hand over 100% of control and GPU compute to pure off-policy RL.

### 3.1 The Unified Objective
During each `ImpalaLearner.learn_step()`, the learner updates on:
```math
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{RL}} + \lambda_{\text{BC}}(t)\mathcal{L}_{\text{BC}}
```
where:
* $\mathcal{L}_{\text{RL}} = \mathcal{L}_{\text{Actor, V-Trace}} + 0.5\,\mathcal{L}_{\text{Critic, Bootstrapped}} + c_{\text{ent}}\mathcal{L}_{\text{Entropy}}$ is evaluated on a batch of live trajectories from `ImpalaReplayBuffer`.
* $\mathcal{L}_{\text{BC}} = -\frac{1}{|\mathcal{B}_{\text{exp}}|}\sum_{(g, a^*) \in \mathcal{B}_{\text{exp}}} \log \pi_\theta(a^* \mid g)$ is the full-distribution teacher-forcing cross-entropy loss evaluated on an expert batch from `ExpertReplayBuffer`. Unlike a scalar reward bonus, $\nabla_\theta \mathcal{L}_{\text{BC}}$ flows directly into the HGT Encoder, Perceiver Queries, and Actor Pointer across every candidate branch at every decision step.

### 3.2 Configurable Annealing Schedule (`compute_annealed_weight`)
Both $\lambda_{\text{BC}}(t)$ and the Actor's epistemic exploration weight $\lambda_{\text{epistemic}}(t)$ are governed by `compute_annealed_weight(step, initial, final, warmup_steps, decay_steps, schedule)`:

* **Warmup Phase (`step < warmup_steps`):** Weight remains constant at `initial` (default $\lambda_{\text{BC}} = 1.0$ for `500` steps) while the Critic ensemble warms up on live V-Trace targets.
* **Decay Phase (`warmup_steps <= step < warmup_steps + decay_steps`):** Decays smoothly from `initial` to `final` (default `0.0` over `5000` steps) via one of three selectable schedules:
  * `"cosine"` (default): $\lambda(t) = \lambda_{\text{final}} + \frac{1}{2}(\lambda_{\text{init}} - \lambda_{\text{final}})(1 + \cos(\pi p))$
  * `"linear"`: $\lambda(t) = \lambda_{\text{final}} + (\lambda_{\text{init}} - \lambda_{\text{final}})(1 - p)$
  * `"exponential"`: $\lambda(t) = \lambda_{\text{init}} (\lambda_{\text{final}} / \lambda_{\text{init}})^p$
* **Post-Decay Compute Short-Circuit (`lambda_bc <= 1e-5`):** Once $\lambda_{\text{BC}}(t)$ decays to $0$, `ImpalaLearner.learn_step` skips sampling and evaluating `expert_buffer` completely, dedicating 100% of GPU throughput to V-Trace RL.
