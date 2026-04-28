package megamek.client.bot.rl;

import java.util.List;
import com.fasterxml.jackson.annotation.JsonInclude;

@JsonInclude(JsonInclude.Include.NON_NULL)
public class RLActionMask {
    public Integer active_entity_index;
    public Integer active_entity;
    public List<RLPathMask> valid_paths;
    public List<RLTargetMask> valid_targets;

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
}
