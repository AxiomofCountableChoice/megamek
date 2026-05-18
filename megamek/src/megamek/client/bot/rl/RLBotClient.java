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
import java.util.ArrayList;
import java.util.List;


import megamek.client.bot.BotClient;
import megamek.client.bot.PhysicalOption;
import megamek.common.board.BoardLocation;
import megamek.common.units.Entity;
import megamek.common.event.player.GamePlayerChatEvent;
import megamek.common.moves.MovePath;
import megamek.common.Report;

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
    protected void processChat(GamePlayerChatEvent ge) {
        if (RLDataPipeline.DEBUG_RL_SYNC) {
            System.err.println("[RLBotClient CHAT] " + ge.getMessage());
        }
    }

    @Override
    public String receiveReport(List<Report> reports) {
        return "";
    }

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
        if (RLDataPipeline.DEBUG_RL_SYNC) {
            System.err.println("RL_SYNC_DEBUG [calculateMoveTurn]: Configured entities: " + getGame().getEntitiesVector().size() + ", Owned entities: " + myUnits.size());
        }
        for (Entity e : myUnits) {
            if (e.isSelectableThisTurn()) {
                if (RLDataPipeline.DEBUG_RL_SYNC) {
                    System.err.println("RL_SYNC_DEBUG [calculateMoveTurn]: Picked entity " + e.getDisplayName() + " to move. done=" + e.isDone() + " int=" + e.turnWasInterrupted());
                }
                return continueMovementFor(e);
            }
        }
        if (RLDataPipeline.DEBUG_RL_SYNC) {
            System.err.println("RL_SYNC_DEBUG [calculateMoveTurn]: No selectable entities found.");
        }
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

            RLActionResponse response = dataPipeline.queryPython("MOVEMENT_INFERENCE", maskData, RLActionResponse.class);
            
            if (response != null && response.selected_path_index != null) {
                int idx = response.selected_path_index;
                if (idx >= 0 && idx < calculatedPaths.size()) {
                    if (RLDataPipeline.DEBUG_RL_SYNC) {
                        System.err.println("RL_SYNC_DEBUG [continueMovementFor]: Successfully parsed path index " + idx + " out of " + calculatedPaths.size() + " paths.");
                    }
                    MovePath chosenPath = calculatedPaths.get(idx);
                    if (chosenPath != null) {
                        java.util.ListIterator<megamek.common.moves.MoveStep> it = chosenPath.getSteps();
                        while (it.hasNext()) {
                            megamek.common.moves.MoveStep step = it.next();
                            megamek.common.Hex hex = game.getBoard(chosenPath.getFinalBoardId()).getHex(step.getPosition());
                            if (hex == null) {
                                System.err.println("RL_SYNC_DEBUG [continueMovementFor]: WARNING! Intercepted a MovePath with an off-board hex. Scrubbing path to prevent server crash.");
                                return new MovePath(game, entity);
                            }
                        }
                    }
                    return chosenPath;
                } else {
                    if (RLDataPipeline.DEBUG_RL_SYNC) {
                        System.err.println("RL_SYNC_DEBUG [continueMovementFor]: WARNING! Python returned an out-of-bounds selected_path_index: " + idx + " (max " + calculatedPaths.size() + "). Defaulting to standing still.");
                    }
                }
            } else {
                if (RLDataPipeline.DEBUG_RL_SYNC) {
                    System.err.println("RL_SYNC_DEBUG [continueMovementFor]: WARNING! Python response was null or lacked selected_path_index. Defaulting to standing still.");
                }
            }
            
            // If no path given, command the entity to stand still / end turn
            return new MovePath(game, entity);
        } catch (Throwable t) {
            System.err.println("RL_SYNC_DEBUG [continueMovementFor]: FATAL THROWABLE: " + t.toString());
            t.printStackTrace();
        }
        return new MovePath(game, entity);
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
        maskData.valid_twists = dataPipeline.buildFiringMask(shooter);

        RLActionResponse response = dataPipeline.queryPython("WEAPON_INFERENCE", maskData, RLActionResponse.class);

        Vector<megamek.common.actions.EntityAction> actions = new Vector<>();
        if (response != null) {
            System.out.println("[RLBotClient REPORT] WEAPON_INFERENCE executed. Twist: " + response.twist + ", Attacks: " + (response.attacks != null ? response.attacks.size() : 0));
            if (response.twist != null && response.twist != 0) {
                int newFacing = megamek.client.bot.princess.FireControl.correctFacing(shooter.getFacing() + response.twist);
                actions.add(new megamek.common.actions.TorsoTwistAction(shooter.getId(), newFacing));
            }
            if (response.attacks != null) {
                for (RLActionResponse.RLAttack att : response.attacks) {
                    if (att != null && att.target_id != null && att.weapon_id != null) {
                        if (att.target_id >= 0 && att.target_id < game.getEntitiesVector().size()) {
                            Entity target = game.getEntitiesVector().get(att.target_id);
                            if (target != null) {
                                megamek.common.equipment.Mounted weapon = shooter.getEquipment(att.weapon_id);
                                if (weapon instanceof megamek.common.equipment.WeaponMounted) {
                                    megamek.common.equipment.WeaponMounted wm = (megamek.common.equipment.WeaponMounted) weapon;
                                    if (wm.getType().hasFlag(megamek.common.equipment.WeaponType.F_ARTILLERY) ||
                                        (wm.getType() instanceof megamek.common.weapons.capitalWeapons.CapitalMissileWeapon &&
                                         megamek.common.compute.Compute.isGroundToGround(shooter, target))) {
                                        actions.add(new megamek.common.actions.ArtilleryAttackAction(shooter.getId(), target.getTargetType(), target.getId(), att.weapon_id, game));
                                    } else {
                                        actions.add(new megamek.common.actions.WeaponAttackAction(shooter.getId(), target.getTargetType(), target.getId(), att.weapon_id));
                                    }
                                } else {
                                    actions.add(new megamek.common.actions.WeaponAttackAction(shooter.getId(), target.getId(), att.weapon_id));
                                }
                            }
                        }
                    }
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

    @Override
    protected PhysicalOption calculatePhysicalTurn() {
        Entity shooter = game.getFirstEntity(getMyTurn());
        if (shooter == null) {
            return null;
        }

        RLActionMask maskData = new RLActionMask();
        maskData.active_entity = shooter.getId();
        maskData.valid_targets = dataPipeline.buildPhysicalMask(shooter);

        RLActionResponse response = dataPipeline.queryPython("PHYSICAL_INFERENCE", maskData, RLActionResponse.class);
        
        if (response != null && response.attack != null) {
            RLActionResponse.RLPhysicalAttack att = response.attack;
            if (att.target_id != null && att.action_type != null) {
                if (att.target_id >= 0 && att.target_id < game.getEntitiesVector().size()) {
                    megamek.common.units.Targetable target = game.getEntitiesVector().get(att.target_id);
                    if (target != null) {
                        return new PhysicalOption(shooter, target, 0.0, att.action_type, null);
                    }
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

    private boolean hasStartedGame = false;

    @Override
    public void changePhase(megamek.common.enums.GamePhase phase) {
        super.changePhase(phase);
        
        if (!hasStartedGame && !phase.isLounge()) {
            dataPipeline.broadcastStartGame();
            hasStartedGame = true;
        }

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

    @Override
    public synchronized void die() {
        if (dataPipeline != null) {
            dataPipeline.close();
        }
        super.die();
    }
}
