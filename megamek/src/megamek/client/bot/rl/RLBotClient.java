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

import java.util.HashMap;
import java.util.Map;
import java.util.Vector;
import java.util.List;
import java.util.ArrayList;

import megamek.client.bot.BotClient;
import megamek.client.bot.PhysicalOption;
import megamek.common.board.BoardLocation;
import megamek.common.units.Entity;
import megamek.common.event.player.GamePlayerChatEvent;
import megamek.common.moves.MovePath;

public class RLBotClient extends BotClient {
    
    private RLDataPipeline dataPipeline;

    public RLBotClient(String playerName, String host, int port, int listenPort) {
        super(playerName, host, port);
        this.dataPipeline = new RLDataPipeline(this);
        this.dataPipeline.listenForPython(listenPort);
    }

    @Override
    public void initialize() {
        // Prepare bot
    }

    @Override
    protected void processChat(GamePlayerChatEvent ge) {}

    @Override
    protected void initMovement() {}

    @Override
    protected void initFiring() {}

    @Override
    protected void calculatePreEndDeclarationsTurn() {}

    @Override
    protected void calculateInfantryVsInfantryCombatTurn() {}

    @Override
    protected MovePath calculateMoveTurn() {
        // Find an unmoved entity and ask Python for its move
        List<Entity> myUnits = getEntitiesOwned();
        System.err.println("RLBOTCLIENT: calculateMoveTurn called. Configured entities: " + getGame().getEntitiesVector().size() + ", Owned entities: " + myUnits.size());
        for (Entity e : myUnits) {
            if (e.isSelectableThisTurn()) {
                System.err.println("RLBOTCLIENT: Picked entity " + e.getDisplayName() + " to move.");
                return continueMovementFor(e);
            }
        }
        System.err.println("RLBOTCLIENT: No selectable entities found.");
        return null;
    }

    @Override
    protected MovePath continueMovementFor(Entity entity) {
        try {
            RLActionMask maskData = new RLActionMask();
            int activeIndex = game.getEntitiesVector().indexOf(entity);
            maskData.active_entity_index = activeIndex;
            
            List<RLActionMask.RLPathMask> serializedMask = new java.util.ArrayList<>();
            List<MovePath> calculatedPaths = dataPipeline.buildMovementMask(entity, serializedMask);
            maskData.valid_paths = serializedMask;

            RLActionResponse response = dataPipeline.queryPython("MOVEMENT", maskData, RLActionResponse.class);
            
            if (response != null && response.selected_path_index != null) {
                int idx = response.selected_path_index;
                if (idx >= 0 && idx < calculatedPaths.size()) {
                    return calculatedPaths.get(idx);
                }
            }
            
            // If no path given, command the entity to stand still / end turn
            sendDone(true);
        } catch (Throwable t) {
            System.err.println("RLBOTCLIENT FATAL THROWABLE in continueMovementFor: " + t.toString());
            t.printStackTrace();
        }
        return null;
    }

    private List<RLActionMask.RLTargetMask> buildFiringMask(Entity shooter) {
        List<RLActionMask.RLTargetMask> targetsMask = new ArrayList<>();
        
        for (Entity target : game.getEntitiesVector()) {
            if (!target.isTargetable() || target.isDestroyed() || !target.isEnemyOf(shooter)) {
                continue;
            }
            
            List<RLActionMask.RLWeaponMask> validWeapons = new ArrayList<>();
            for (megamek.common.equipment.WeaponMounted wm : shooter.getWeaponList()) {
                if (!wm.canFire() || (wm.getLinkedAmmo() != null && wm.getLinkedAmmo().getUsableShotsLeft() == 0)) {
                    continue;
                }

                // Leverage the game's actual attack resolution engine
                megamek.common.ToHitData toHit = megamek.common.actions.WeaponAttackAction.toHit(
                        game, shooter.getId(), target, shooter.getEquipmentNum(wm), false);
                
                if (toHit.getValue() != megamek.common.rolls.TargetRoll.IMPOSSIBLE 
                        && toHit.getValue() != megamek.common.rolls.TargetRoll.AUTOMATIC_FAIL) {
                    
                    RLActionMask.RLWeaponMask wData = new RLActionMask.RLWeaponMask();
                    wData.weapon_id = shooter.getEquipmentNum(wm);
                    wData.weapon_name = wm.getName();
                    wData.to_hit = toHit.getValue();
                    validWeapons.add(wData);
                }
            }
            
            if (!validWeapons.isEmpty()) {
                RLActionMask.RLTargetMask tm = new RLActionMask.RLTargetMask();
                int targetIndex = game.getEntitiesVector().indexOf(target);
                tm.target_entity_index = targetIndex;
                tm.valid_weapons = validWeapons;
                targetsMask.add(tm);
            }
        }
        
        return targetsMask;
    }

    @Override
    protected void calculateFiringTurn() {
        Entity shooter = game.getFirstEntity(getMyTurn());
        if (shooter == null) {
            sendDone(true);
            return;
        }

        RLActionMask maskData = new RLActionMask();
        maskData.active_entity = shooter.getId();
        maskData.valid_targets = buildFiringMask(shooter);

        RLActionResponse response = dataPipeline.queryPython("FIRING", maskData, RLActionResponse.class);

        Vector<megamek.common.actions.EntityAction> actions = new Vector<>();
        if (response != null && response.attacks != null) {
            for (RLActionResponse.RLAttack att : response.attacks) {
                if (att.target_id != null && att.weapon_id != null) {
                    actions.add(new megamek.common.actions.WeaponAttackAction(shooter.getId(), att.target_id, att.weapon_id));
                }
            }
        }
        
        sendAttackData(shooter.getId(), actions);
    }

    @Override
    protected void calculateDeployment() throws Exception {
        dataPipeline.queryPython("DEPLOYMENT", new HashMap<>(), RLActionResponse.class);
        sendDone(true);
    }

    private List<RLActionMask.RLTargetMask> buildPhysicalMask(Entity shooter) {
        List<RLActionMask.RLTargetMask> targetsMask = new ArrayList<>();
        
        for (Entity target : game.getEntitiesVector()) {
            if (!target.isTargetable() || target.isDestroyed() || !target.isEnemyOf(shooter)) continue;
            
            List<RLActionMask.RLPhysicalMask> validAttacks = new ArrayList<>();
            
            // Try PUNCH
            for (int i = 0; i < 2; i++) {
                int arm = (i == 0) ? megamek.common.actions.PunchAttackAction.LEFT : megamek.common.actions.PunchAttackAction.RIGHT;
                int poType = (i == 0) ? megamek.client.bot.PhysicalOption.PUNCH_LEFT : megamek.client.bot.PhysicalOption.PUNCH_RIGHT;
                
                megamek.common.ToHitData th = megamek.common.actions.PunchAttackAction.toHit(
                        game, shooter.getId(), target, arm, false);
                if (th.getValue() != megamek.common.rolls.TargetRoll.IMPOSSIBLE && th.getValue() != megamek.common.rolls.TargetRoll.AUTOMATIC_FAIL) {
                    RLActionMask.RLPhysicalMask att = new RLActionMask.RLPhysicalMask();
                    att.action_type = poType;
                    att.name = (i==0) ? "PUNCH_LEFT" : "PUNCH_RIGHT";
                    att.to_hit = th.getValue();
                    validAttacks.add(att);
                }
            }
            
            // Try KICK
            for (int i = 0; i < 2; i++) {
                int leg = (i == 0) ? megamek.common.actions.KickAttackAction.LEFT : megamek.common.actions.KickAttackAction.RIGHT;
                int poType = (i == 0) ? megamek.client.bot.PhysicalOption.KICK_LEFT : megamek.client.bot.PhysicalOption.KICK_RIGHT;
                
                megamek.common.ToHitData th = megamek.common.actions.KickAttackAction.toHit(
                        game, shooter.getId(), target, leg);
                if (th.getValue() != megamek.common.rolls.TargetRoll.IMPOSSIBLE && th.getValue() != megamek.common.rolls.TargetRoll.AUTOMATIC_FAIL) {
                    RLActionMask.RLPhysicalMask att = new RLActionMask.RLPhysicalMask();
                    att.action_type = poType;
                    att.name = (i==0) ? "KICK_LEFT" : "KICK_RIGHT";
                    att.to_hit = th.getValue();
                    validAttacks.add(att);
                }
            }
            
            // Try PUSH
            megamek.common.ToHitData pushTh = megamek.common.actions.PushAttackAction.toHit(game, shooter.getId(), target);
            if (pushTh.getValue() != megamek.common.rolls.TargetRoll.IMPOSSIBLE && pushTh.getValue() != megamek.common.rolls.TargetRoll.AUTOMATIC_FAIL) {
                RLActionMask.RLPhysicalMask att = new RLActionMask.RLPhysicalMask();
                att.action_type = megamek.client.bot.PhysicalOption.PUSH_ATTACK;
                att.name = "PUSH";
                att.to_hit = pushTh.getValue();
                validAttacks.add(att);
            }
            
            if (!validAttacks.isEmpty()) {
                RLActionMask.RLTargetMask tm = new RLActionMask.RLTargetMask();
                tm.target_entity_id = target.getId();
                tm.valid_attacks = validAttacks;
                targetsMask.add(tm);
            }
        }
        return targetsMask;
    }

    @Override
    protected PhysicalOption calculatePhysicalTurn() {
        Entity shooter = game.getFirstEntity(getMyTurn());
        if (shooter == null) {
            return null;
        }

        RLActionMask maskData = new RLActionMask();
        maskData.active_entity = shooter.getId();
        maskData.valid_targets = buildPhysicalMask(shooter);

        RLActionResponse response = dataPipeline.queryPython("PHYSICAL", maskData, RLActionResponse.class);
        
        if (response != null && response.attack != null) {
            RLActionResponse.RLPhysicalAttack att = response.attack;
            if (att.target_id != null && att.action_type != null) {
                megamek.common.units.Targetable target = game.getEntity(att.target_id);
                if (target != null) {
                    return new PhysicalOption(shooter, target, 0.0, att.action_type, null);
                }
            }
        }

        return null;
    }

    @Override
    protected Vector<BoardLocation> calculateArtyAutoHitHexes() {
        return new Vector<>();
    }

    @Override
    protected void checkMorale() {}

    @Override
    public void changePhase(megamek.common.enums.GamePhase phase) {
        super.changePhase(phase);
        
        // If we are fully automated and pre-deployed by RLServerManager,
        // we never get a Deployment GameTurn, so we must manually skip the phase here.
        if (phase.isDeployment()) {
            boolean hasUndeployed = false;
            for (Entity e : getEntitiesOwned()) {
                if (!e.isDeployed()) {
                    hasUndeployed = true;
                    break;
                }
            }
            if (!hasUndeployed) {
                sendDone(true);
            }
        }
    }

    @Override
    protected void postMovementProcessing() {}
}
