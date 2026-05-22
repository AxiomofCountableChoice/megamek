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
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public class RLActionResponse {
    public Integer selected_path_index;
    public Integer selected_entity_id;
    public Integer twist;
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
