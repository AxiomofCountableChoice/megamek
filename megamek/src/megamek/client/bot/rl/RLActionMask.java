/*
 * Copyright (c) 2024 - The MegaMek Team. All Rights Reserved.
 *
 * This file is part of MegaMek.
 *
 * MegaMek is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * MegaMek is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with MegaMek. If not, see <http://www.gnu.org/licenses/>.
 */
package megamek.client.bot.rl;

import java.util.List;
public class RLActionMask {
    public Integer active_entity_index;
    public Integer active_entity;
    public List<RLPathMask> valid_paths;
    public List<RLTargetMask> valid_targets; // Restored for Physical Phase
    public List<RLTwistMask> valid_twists; // Added for Weapon Phase Option C
    public RLTargetAction target_action;   // Added for Behavioral Cloning

    public static class RLTwistMask {
        public Integer twist;
        public List<RLTargetMask> valid_targets;
    }

    public static class RLPathMask {
        public Integer path_index;
        public Integer dest_index;
        public Integer dest_facing;
        public Integer mp_used;
        public Boolean is_jump;
    }

    public static class RLTargetMask {
        public Integer target_entity_index;
        public Integer target_entity_id;
        public List<RLWeaponMask> valid_weapons;
        public List<RLPhysicalMask> valid_attacks;
    }

    public static class RLWeaponMask {
        public Integer weapon_id;
        public String weapon_name;
        public Integer to_hit;
    }

    public static class RLPhysicalMask {
        public Integer action_type;
        public String name;
        public Integer to_hit;
    }

    // --- Behavioral Cloning Trajectory Classes ---
    public static class RLTargetAction {
        public Integer torso_twist;
        public List<RLAttack> attacks;
    }

    public static class RLAttack {
        public Integer target_entity_index;
        public Integer weapon_id;
        public Integer physical_action_type;
    }
}
