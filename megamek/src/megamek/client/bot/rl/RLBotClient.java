package megamek.client.bot.rl;

import java.io.InputStream;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.HashMap;
import java.util.Map;
import java.util.Vector;
import java.util.List;
import java.util.ArrayList;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.msgpack.jackson.dataformat.MessagePackFactory;

import megamek.client.bot.BotClient;
import megamek.client.bot.PhysicalOption;
import megamek.common.BoardLocation;
import megamek.common.Entity;
import megamek.common.event.GamePlayerChatEvent;
import megamek.common.moves.MovePath;
import megamek.logging.MMLogger;

public class RLBotClient extends BotClient {
    private static final MMLogger logger = MMLogger.create(RLBotClient.class);
    
    private ObjectMapper msgpackMapper;
    private ServerSocket pythonServerSocket;
    private Socket pythonSocket;
    private InputStream pythonIn;
    private OutputStream pythonOut;
    private int listenPort;
    
    private List<MovePath> lastCalculatedPaths = new ArrayList<>();

    public RLBotClient(String playerName, String host, int port, int listenPort) {
        super(playerName, host, port);
        this.listenPort = listenPort;
        this.msgpackMapper = new ObjectMapper(new MessagePackFactory());
        
        // Wait for python connection before proceeding
        listenForPython();
    }
    
    private void listenForPython() {
        try {
            pythonServerSocket = new ServerSocket(listenPort);
            pythonServerSocket.setSoTimeout(30000); // 30 seconds to connect
            logger.info("RLBotClient waiting for Python connection on port " + listenPort);
            try {
                pythonSocket = pythonServerSocket.accept();
                pythonSocket.setSoTimeout(10000); // 10 seconds read timeout on actual communications
                pythonIn = pythonSocket.getInputStream();
                pythonOut = pythonSocket.getOutputStream();
                logger.info("Python connected on port " + listenPort);
            } catch (java.net.SocketTimeoutException e) {
                logger.warn("Python client did not connect within 30 seconds. Running without Python actor.");
            }
        } catch (Exception e) {
            logger.error(e, "Error accepting Python connection.");
        }
    }
    
    private Map<String, Object> serializeGameState() {
        Map<String, Object> state = new HashMap<>();
        
        // 1. Covariates (Phase Vector p)
        state.put("phase_main", game.getPhase().name());
        state.put("turn_number", game.getTurnIndex());
        state.put("round_number", game.getRoundCount());
        int myActivations = 0;
        if (getLocalPlayer() != null) {
            state.put("current_player_id", getLocalPlayer().getId());
            for (Entity e : getEntitiesOwned()) {
                if (!e.isDone()) myActivations++;
            }
        }
        state.put("my_activations_left", myActivations);

        // 2. Global Game State
        Map<String, Object> global = new HashMap<>();
        if (game.getPlanetaryConditions() != null) {
            global.put("gravity", game.getPlanetaryConditions().getGravity());
            global.put("temperature", game.getPlanetaryConditions().getTemperature());
            global.put("light_level", game.getPlanetaryConditions().getLight());
        }
        if (game.getBoard() != null) {
            global.put("board_width", game.getBoard().getWidth());
            global.put("board_height", game.getBoard().getHeight());
        }
        state.put("global_state", global);

        // 3. Entities
        Map<Integer, Object> entitiesData = new HashMap<>();
        for (Entity e : game.getEntitiesVector()) {
            Map<String, Object> ed = new HashMap<>();
            ed.put("id", e.getId());
            if (e.getPosition() != null) {
                ed.put("x", e.getPosition().getX());
                ed.put("y", e.getPosition().getY());
            }
            ed.put("facing", e.getFacing());
            ed.put("heat", e.getHeat());
            ed.put("owner_id", e.getOwnerId());
            ed.put("is_active", e.isActive());
            ed.put("is_destroyed", e.isDestroyed());
            ed.put("armor", e.getTotalArmor());
            ed.put("structure", e.getTotalInternal());
            ed.put("elevation", e.getElevation());
            entitiesData.put(e.getId(), ed);
        }
        state.put("entities", entitiesData);

        return state;
    }
    
    private <T> T queryPython(String actionContext, Map<String, Object> mask, Class<T> responseType) {
        if (pythonSocket == null || !pythonSocket.isConnected()) {
            logger.warn("Python not connected! Proceeding with empty sub-action.");
            return null;
        }
        
        try {
            Map<String, Object> payload = new HashMap<>();
            payload.put("context", actionContext);
            payload.put("state", serializeGameState());
            payload.put("mask", mask);
            
            // Send to python
            byte[] bytes = msgpackMapper.writeValueAsBytes(payload);
            System.err.println("RLBOTCLIENT: Serialized payload to " + bytes.length + " bytes. Writing...");
            // Write length integer first (4 bytes)
            pythonOut.write(java.nio.ByteBuffer.allocate(4).putInt(bytes.length).array());
            pythonOut.write(bytes);
            pythonOut.flush();
            System.err.println("RLBOTCLIENT: Flushed payload. Awaiting response length...");
            
            // Read response length
            byte[] lenBytes = new byte[4];
            int read = pythonIn.read(lenBytes);
            System.err.println("RLBOTCLIENT: Read " + read + " bytes for length prefix.");
            if (read < 4) return null;
            int length = java.nio.ByteBuffer.wrap(lenBytes).getInt();
            System.err.println("RLBOTCLIENT: Parsed length as " + length + " bytes. Awaiting payload...");
            
            // Read payload
            byte[] data = new byte[length];
            int offset = 0;
            while (offset < length) {
                int r = pythonIn.read(data, offset, length - offset);
                if (r < 0) break;
                offset += r;
            }
            
            return msgpackMapper.readValue(data, responseType);
            
        } catch (java.net.SocketTimeoutException e) {
            System.err.println("RLBOTCLIENT FATAL: Python SocketTimeoutException: " + e.getMessage());
            logger.warn("Python client timed out responding. Closing connection to prevent hang.");
            try { pythonSocket.close(); } catch (Exception ignored) {}
            return null;
        } catch (Exception e) {
            System.err.println("RLBOTCLIENT FATAL EXCEPTION: " + e.getMessage());
            e.printStackTrace();
            logger.error(e, "Error communicating with python");
            return null;
        }
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

    private List<Map<String, Object>> buildMovementMask(Entity mover) {
        lastCalculatedPaths.clear();
        List<Map<String, Object>> serializedMask = new ArrayList<>();
        
        // Ground path generation
        int maxMove = Math.min(mover.getRunMPwithoutMASC(), mover.getRunMP(megamek.common.MPCalculationSetting.NO_GRAVITY));
        if (maxMove > 0) {
            megamek.common.pathfinder.ShortestPathFinder spfGround = megamek.common.pathfinder.ShortestPathFinder.newInstanceOfOneToAll(maxMove, megamek.common.moves.MovePath.MoveStepType.FORWARDS, game);
            spfGround.run(new megamek.common.moves.MovePath(game, mover, null));
            lastCalculatedPaths.addAll(spfGround.getAllComputedPathsUncategorized());
        }
        
        // Add jump paths if applicable
        if (mover.getAnyTypeMaxJumpMP() > 0) {
            megamek.common.pathfinder.ShortestPathFinder spfJump = megamek.common.pathfinder.ShortestPathFinder.newInstanceOfOneToAll(mover.getAnyTypeMaxJumpMP(), megamek.common.moves.MovePath.MoveStepType.FORWARDS, game);
            spfJump.run(new megamek.common.moves.MovePath(game, mover, null).addStep(megamek.common.moves.MovePath.MoveStepType.START_JUMP));
            lastCalculatedPaths.addAll(spfJump.getAllComputedPathsUncategorized());
        }

        for (int i = 0; i < lastCalculatedPaths.size(); i++) {
            MovePath p = lastCalculatedPaths.get(i);
            
            // Filter out illegal destination states (stacking violations)
            if (!p.isMoveLegal() || megamek.common.Compute.stackingViolation(game, mover.getId(), p.getFinalCoords(), mover.climbMode()) != null) {
                continue;
            }

            Map<String, Object> pm = new HashMap<>();
            pm.put("path_index", i);
            if (p.getFinalCoords() != null) {
                pm.put("dest_x", p.getFinalCoords().getX());
                pm.put("dest_y", p.getFinalCoords().getY());
            }
            pm.put("dest_facing", p.getFinalFacing());
            pm.put("mp_used", p.getMpUsed());
            pm.put("is_jump", p.isJumping());
            serializedMask.add(pm);
        }
        return serializedMask;
    }

    @Override
    protected MovePath continueMovementFor(Entity entity) {
        try {
            Map<String, Object> maskData = new HashMap<>();
            maskData.put("active_entity", entity.getId());
            maskData.put("valid_paths", buildMovementMask(entity));

            Map<String, Object> response = queryPython("MOVEMENT", maskData, Map.class);
            
            if (response != null && response.containsKey("selected_path_index")) {
                int idx = ((Number) response.get("selected_path_index")).intValue();
                if (idx >= 0 && idx < lastCalculatedPaths.size()) {
                    return lastCalculatedPaths.get(idx);
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
                tm.put("target_entity_id", target.getId());
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

        Map<String, Object> response = queryPython("FIRING", maskData, Map.class);

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
        Map<String, Object> map = queryPython("DEPLOYMENT", new HashMap<>(), Map.class);
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

        Map<String, Object> response = queryPython("PHYSICAL", maskData, Map.class);
        
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
