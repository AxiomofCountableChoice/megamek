package megamek.client.bot.rl;

import java.util.List;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public class RLActionResponse {
    public Integer selected_path_index;
    public List<RLAttack> attacks;
    public RLPhysicalAttack attack;

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class RLAttack {
        public Integer target_id;
        public Integer weapon_id;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class RLPhysicalAttack {
        public Integer target_id;
        public Integer action_type;
    }
}
