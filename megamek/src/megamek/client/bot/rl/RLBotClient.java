package megamek.client.bot.rl;

import java.io.InputStream;
import java.net.Socket;
import java.util.HashMap;
import java.util.Map;
import java.util.Vector;
import java.util.List;
import java.util.ArrayList;

import megamek.client.bot.BotClient;
import megamek.client.bot.PhysicalOption;
import megamek.common.BoardLocation;
import megamek.common.Entity;
import megamek.common.event.GamePlayerChatEvent;
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
            Map<String, Object> maskData = new HashMap<>();
            int activeIndex = game.getEntitiesVector().indexOf(entity);
            maskData.put("active_entity_index", activeIndex);
            
            List<Map<String, Object>> serializedMask = new java.util.ArrayList<>();
            List<MovePath> calculatedPaths = dataPipeline.buildMovementMask(entity, serializedMask);
            maskData.put("valid_paths", serializedMask);

            Map<String, Object> response = dataPipeline.queryPython("MOVEMENT", maskData, Map.class);
            
            if (response != null && response.containsKey("selected_path_index")) {
                int idx = ((Number) response.get("selected_path_index")).intValue();
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

    private List<Map<String, Object>> buildFiringMask(Entity shooter) {
        List<Map<String, Object>> targetsMask = new ArrayList<>();
        
        for (Entity target : game.getEntitiesVector()) {
            if (!target.isTargetable() || target.isDestroyed() || !target.isEnemyOf(shooter)) {
                continue;
            }
            
            List<Map<String, Object>> validWeapons = new ArrayList<>();
            for (megamek.common.equipment.WeaponMounted wm : shooter.getWeaponList()) {
                if (!wm.canFire() || (wm.getLinkedAmmo() != null && wm.getLinkedAmmo().getUsableShotsLeft() == 0)) {
                    continue;
                }

                // Leverage the game's actual attack resolution engine
                megamek.common.ToHitData toHit = megamek.common.actions.WeaponAttackAction.toHit(
                        game, shooter.getId(), target, shooter.getEquipmentNum(wm), false);
                
                if (toHit.getValue() != megamek.common.TargetRoll.IMPOSSIBLE 
                        && toHit.getValue() != megamek.common.TargetRoll.AUTOMATIC_FAIL) {
                    
                    Map<String, Object> wData = new HashMap<>();
                    wData.put("weapon_id", shooter.getEquipmentNum(wm));
                    wData.put("weapon_name", wm.getName());
                    wData.put("to_hit", toHit.getValue());
                    validWeapons.add(wData);
                }
            }
            
            if (!validWeapons.isEmpty()) {
                Map<String, Object> tm = new HashMap<>();
                int targetIndex = game.getEntitiesVector().indexOf(target);
                tm.put("target_entity_index", targetIndex);
                tm.put("valid_weapons", validWeapons);
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

        Map<String, Object> maskData = new HashMap<>();
        maskData.put("active_entity", shooter.getId());
        maskData.put("valid_targets", buildFiringMask(shooter));

        Map<String, Object> response = dataPipeline.queryPython("FIRING", maskData, Map.class);

        Vector<megamek.common.actions.EntityAction> actions = new Vector<>();
        if (response != null && response.containsKey("attacks")) {
            List<Map<String, Object>> attacks = (List<Map<String, Object>>) response.get("attacks");
            for (Map<String, Object> att : attacks) {
                int targetId = ((Number) att.get("target_id")).intValue();
                int weaponId = ((Number) att.get("weapon_id")).intValue();
                actions.add(new megamek.common.actions.WeaponAttackAction(shooter.getId(), targetId, weaponId));
            }
        }
        
        sendAttackData(shooter.getId(), actions);
    }

    @Override
    protected void calculateDeployment() throws Exception {
        Map<String, Object> map = dataPipeline.queryPython("DEPLOYMENT", new HashMap<>(), Map.class);
        sendDone(true);
    }

    private List<Map<String, Object>> buildPhysicalMask(Entity shooter) {
        List<Map<String, Object>> targetsMask = new ArrayList<>();
        
        for (Entity target : game.getEntitiesVector()) {
            if (!target.isTargetable() || target.isDestroyed() || !target.isEnemyOf(shooter)) continue;
            
            List<Map<String, Object>> validAttacks = new ArrayList<>();
            
            // Try PUNCH
            for (int i = 0; i < 2; i++) {
                int arm = (i == 0) ? megamek.common.actions.PunchAttackAction.LEFT : megamek.common.actions.PunchAttackAction.RIGHT;
                int poType = (i == 0) ? megamek.client.bot.PhysicalOption.PUNCH_LEFT : megamek.client.bot.PhysicalOption.PUNCH_RIGHT;
                
                megamek.common.ToHitData th = megamek.common.actions.PunchAttackAction.toHit(
                        game, shooter.getId(), target, arm, false);
                if (th.getValue() != megamek.common.TargetRoll.IMPOSSIBLE && th.getValue() != megamek.common.TargetRoll.AUTOMATIC_FAIL) {
                    Map<String, Object> att = new HashMap<>();
                    att.put("action_type", poType);
                    att.put("name", (i==0) ? "PUNCH_LEFT" : "PUNCH_RIGHT");
                    att.put("to_hit", th.getValue());
                    validAttacks.add(att);
                }
            }
            
            // Try KICK
            for (int i = 0; i < 2; i++) {
                int leg = (i == 0) ? megamek.common.actions.KickAttackAction.LEFT : megamek.common.actions.KickAttackAction.RIGHT;
                int poType = (i == 0) ? megamek.client.bot.PhysicalOption.KICK_LEFT : megamek.client.bot.PhysicalOption.KICK_RIGHT;
                
                megamek.common.ToHitData th = megamek.common.actions.KickAttackAction.toHit(
                        game, shooter.getId(), target, leg);
                if (th.getValue() != megamek.common.TargetRoll.IMPOSSIBLE && th.getValue() != megamek.common.TargetRoll.AUTOMATIC_FAIL) {
                    Map<String, Object> att = new HashMap<>();
                    att.put("action_type", poType);
                    att.put("name", (i==0) ? "KICK_LEFT" : "KICK_RIGHT");
                    att.put("to_hit", th.getValue());
                    validAttacks.add(att);
                }
            }
            
            // Try PUSH
            megamek.common.ToHitData pushTh = megamek.common.actions.PushAttackAction.toHit(game, shooter.getId(), target);
            if (pushTh.getValue() != megamek.common.TargetRoll.IMPOSSIBLE && pushTh.getValue() != megamek.common.TargetRoll.AUTOMATIC_FAIL) {
                Map<String, Object> att = new HashMap<>();
                att.put("action_type", megamek.client.bot.PhysicalOption.PUSH_ATTACK);
                att.put("name", "PUSH");
                att.put("to_hit", pushTh.getValue());
                validAttacks.add(att);
            }
            
            if (!validAttacks.isEmpty()) {
                Map<String, Object> tm = new HashMap<>();
                tm.put("target_entity_id", target.getId());
                tm.put("valid_attacks", validAttacks);
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

        Map<String, Object> maskData = new HashMap<>();
        maskData.put("active_entity", shooter.getId());
        maskData.put("valid_targets", buildPhysicalMask(shooter));

        Map<String, Object> response = dataPipeline.queryPython("PHYSICAL", maskData, Map.class);
        
        if (response != null && response.containsKey("attack")) {
            Map<String, Object> att = (Map<String, Object>) response.get("attack");
            int targetId = ((Number) att.get("target_id")).intValue();
            int actionType = ((Number) att.get("action_type")).intValue();
            
            megamek.common.Targetable target = game.getEntity(targetId);
            if (target != null) {
                return new PhysicalOption(shooter, target, 0.0, actionType, null);
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
