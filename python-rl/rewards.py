class RewardCalculator:
    """
    Computes dense rewards based on Battle Value (BV), Target Priority (TP), 
    and Victory Points (VP) deltas.
    """
    def __init__(self, beta_bv=0.1, beta_tp=0.0, beta_vp=1.0):
        self.beta_bv = beta_bv
        self.beta_tp = beta_tp
        self.beta_vp = beta_vp
        
        self.prev_bv1 = None
        self.prev_bv2 = None
        self.prev_tp1 = 0
        self.prev_tp2 = 0
        self.prev_vp1 = 0
        self.initial_bv1 = None
        self.initial_bv2 = None

    def reset(self):
        self.prev_bv1 = None
        self.prev_bv2 = None
        self.prev_tp1 = 0
        self.prev_tp2 = 0
        self.prev_vp1 = 0
        self.initial_bv1 = None
        self.initial_bv2 = None

    def compute_reward(self, payload):
        reward = 0.0
        if payload and "rewards" in payload:
            rew_dict = payload["rewards"]
            bv1 = rew_dict.get("bv1", 0)
            bv2 = rew_dict.get("bv2", 0)
            tp1 = rew_dict.get("tp1", 0)
            tp2 = rew_dict.get("tp2", 0)
            vp1 = rew_dict.get("vp1", 0) # Already zero-sum from Java
            
            if self.prev_bv1 is None:
                self.prev_bv1, self.prev_bv2 = bv1, bv2
                self.prev_vp1 = vp1
                self.initial_bv1 = bv1 if bv1 > 0 else 5000.0
                self.initial_bv2 = bv2 if bv2 > 0 else 5000.0
                
            delta_bv1 = bv1 - self.prev_bv1
            delta_bv2 = bv2 - self.prev_bv2
            delta_vp1 = vp1 - self.prev_vp1
            
            # Dynamically normalized BV difference using sum-based normalization
            total_initial_bv = self.initial_bv1 + self.initial_bv2
            reward_bv = self.beta_bv * 2.0 * (delta_bv1 - delta_bv2) / total_initial_bv
            reward_vp = self.beta_vp * delta_vp1
            reward_tp = self.beta_tp * (tp1 - tp2)
            
            reward = reward_bv + reward_vp + reward_tp
            
            self.prev_bv1, self.prev_bv2 = bv1, bv2
            self.prev_tp1, self.prev_tp2 = tp1, tp2
            self.prev_vp1 = vp1
            
        return reward
