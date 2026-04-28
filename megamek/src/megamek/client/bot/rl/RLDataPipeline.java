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

import java.io.InputStream;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.msgpack.jackson.dataformat.MessagePackFactory;

import megamek.client.bot.BotClient;
import megamek.common.board.Board;
import megamek.common.units.Entity;
import megamek.common.equipment.Mounted;
import megamek.common.Hex;
import megamek.common.game.Game;
import megamek.common.Player;
import megamek.common.units.Terrains;
import megamek.common.equipment.WeaponType;
import megamek.common.moves.MovePath;
import megamek.logging.MMLogger;

public class RLDataPipeline {
    private static final MMLogger logger = MMLogger.create(RLDataPipeline.class);
    
    private ObjectMapper msgpackMapper;
    private BotClient baseClient;
    
    private ServerSocket pythonServerSocket;
    private Socket pythonSocket;
    private InputStream pythonIn;
    private OutputStream pythonOut;
    
    private boolean hasSentTopology = false;

    public RLDataPipeline(BotClient client) {
        this.baseClient = client;
        this.msgpackMapper = new ObjectMapper(new MessagePackFactory());
    }
    
    public void listenForPython(int listenPort) {
        try {
            pythonServerSocket = new ServerSocket(listenPort);
            pythonServerSocket.setSoTimeout(30000); // 30 seconds to connect
            logger.info("RLDataPipeline waiting for Python connection on port " + listenPort);
            try {
                pythonSocket = pythonServerSocket.accept();
                pythonSocket.setSoTimeout(10000); // 10 seconds read timeout on actual communications
                pythonIn = pythonSocket.getInputStream();
                pythonOut = pythonSocket.getOutputStream();
                logger.info("Python connected on port " + listenPort);
            } catch (java.net.SocketTimeoutException e) {
                logger.warn("Python client did not connect within 30 seconds.");
            }
        } catch (Exception e) {
            logger.error(e, "Error accepting Python connection.");
        }
    }
    
    public boolean isConnected() {
        return pythonSocket != null && pythonSocket.isConnected();
    }
    
    public void ensureTopologySent() {
        if (!hasSentTopology && baseClient.getGame() != null && baseClient.getGame().getBoard() != null && isConnected()) {
            sendTopologyToPython();
            hasSentTopology = true;
        }
    }

    private void sendTopologyToPython() {
        try {
            Map<String, Object> payload = new HashMap<>();
            payload.put("context", "TOPOLOGY");
            
            Game game = (Game) baseClient.getGame();
            Board board = game.getBoard();
            int width = board.getWidth();
            int height = board.getHeight();
            payload.put("width", width);
            payload.put("height", height);
            
            List<float[]> hexes = new ArrayList<>();
            List<int[]> edges = new ArrayList<>();
            
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    Hex hex = board.getHex(x, y);
                    if (hex == null) {
                        hexes.add(new float[]{0f, 0f, 0f, 0f, 0f});
                        continue;
                    }
                    
                    float elev = hex.getLevel();
                    float woods = Math.max(hex.terrainLevel(Terrains.WOODS), hex.terrainLevel(Terrains.JUNGLE));
                    if (woods < 0) woods = 0;
                    float water = hex.terrainLevel(Terrains.WATER);
                    if (water < 0) water = 0;
                    float isPavement = hex.hasPavement() ? 1f : 0f;
                    float isBuilding = hex.containsTerrain(Terrains.BUILDING) ? 1f : 0f;
                    
                    hexes.add(new float[]{elev, woods, water, isPavement, isBuilding});
                    
                    int currentIndex = y * width + x;
                    for (int dir = 0; dir < 6; dir++) {
                        Hex adj = board.getHexInDir(x, y, dir);
                        if (adj != null) {
                            int adjIndex = adj.getCoords().getY() * width + adj.getCoords().getX();
                            edges.add(new int[]{currentIndex, adjIndex, dir});
                        }
                    }
                }
            }
            
            payload.put("hex_nodes", hexes);
            payload.put("hex_edges", edges);
            
            sendPayload(payload);
            System.err.println("RLDataPipeline: Topology payload sent.");
        } catch (Exception e) {
            System.err.println("RLDataPipeline: Failed to send topology " + e.getMessage());
            e.printStackTrace();
        }
    }

    public void sendPayload(Map<String, Object> payload) throws Exception {
        byte[] bytes = msgpackMapper.writeValueAsBytes(payload);
        pythonOut.write(java.nio.ByteBuffer.allocate(4).putInt(bytes.length).array());
        pythonOut.write(bytes);
        pythonOut.flush();
    }
    
    public <T> T queryPython(String actionContext, Object mask, Class<T> responseType) {
        if (!isConnected()) {
            return null;
        }
        
        ensureTopologySent();
        
        try {
            Map<String, Object> payload = new HashMap<>();
            payload.put("context", actionContext);
            payload.put("state", serializeGameState());
            payload.put("mask", mask);
            
            sendPayload(payload);
            
            // Read response
            byte[] lenBytes = new byte[4];
            java.io.DataInputStream dis = new java.io.DataInputStream(pythonIn);
            dis.readFully(lenBytes);
            int len = java.nio.ByteBuffer.wrap(lenBytes).getInt();
            
            byte[] dataBytes = new byte[len];
            dis.readFully(dataBytes);
            
            return msgpackMapper.readValue(dataBytes, responseType);
        } catch (Exception e) {
            logger.error("RLDataPipeline: Python query failed: " + e.toString());
            return null;
        }
    }

    public float[] extractEntityFeatures(Entity e) {
        Game game = (Game) baseClient.getGame();
        Player localPlayer = baseClient.getLocalPlayer();
        
        float isMine = (localPlayer != null && e.getOwnerId() == localPlayer.getId()) ? 1f : 0f;
        float x = e.getPosition() != null ? e.getPosition().getX() : -1f;
        float y = e.getPosition() != null ? e.getPosition().getY() : -1f;
        float facing = e.getFacing();
        float heat = e.getHeat();
        float maxHeat = e.getHeatCapacity();
        float armor = e.getTotalArmor();
        float structure = e.getTotalInternal();
        float tmm = megamek.common.compute.Compute.getTargetMovementModifier(game, e.getId()).getValue();
        
        float speedMode = 0f;
        if (e.moved != megamek.common.units.EntityMovementType.MOVE_NONE) {
            if (e.moved == megamek.common.units.EntityMovementType.MOVE_WALK) speedMode = 1f;
            else if (e.moved == megamek.common.units.EntityMovementType.MOVE_RUN) speedMode = 2f;
            else if (e.moved == megamek.common.units.EntityMovementType.MOVE_JUMP) speedMode = 3f;
        }
        
        float gunnery = e.getCrew() != null ? e.getCrew().getGunnery() : 4f;
        float piloting = e.getCrew() != null ? e.getCrew().getPiloting() : 5f;
        
        return new float[]{
            isMine, x, y, facing, heat, maxHeat, armor, structure, tmm, speedMode, gunnery, piloting
        };
    }

    public float[] extractWeaponFeatures(Mounted<?> weapon) {
        WeaponType wt = (WeaponType) weapon.getType();
        float minRange = wt.getMinimumRange();
        float shortRange = wt.getShortRange();
        float mediumRange = wt.getMediumRange();
        float longRange = wt.getLongRange();
        float damage = wt.getDamage();
        float isOperational = weapon.isOperable() ? 1f : 0f;
        float isCluster = (wt.getRackSize() > 1 || wt.hasFlag(WeaponType.F_MISSILE)) ? 1f : 0f;
        float numClusters = wt.getRackSize();
        
        float salvosRemaining = 99f;
        if (wt.getAmmoType() != null && wt.getAmmoType() != megamek.common.equipment.AmmoType.AmmoTypeEnum.NA) {
            Mounted<?> ammo = weapon.getLinked();
            if (ammo != null) {
                salvosRemaining = ammo.getUsableShotsLeft();
            } else {
                salvosRemaining = 0f;
            }
        }
        
        float heat = wt.getHeat();
        
        return new float[]{minRange, shortRange, mediumRange, longRange, damage, isOperational, isCluster, numClusters, salvosRemaining, heat};
    }

    public Map<String, Object> serializeGameState() {
        Game game = (Game) baseClient.getGame();
        Player localPlayer = baseClient.getLocalPlayer();
        
        Map<String, Object> state = new HashMap<>();
        
        state.put("phase_main", game.getPhase().name());
        state.put("turn_number", game.getTurnIndex());
        state.put("round_number", game.getRoundCount());
        
        int myActivations = 0;
        if (localPlayer != null) {
            state.put("current_player_id", localPlayer.getId());
            for (Entity e : baseClient.getEntitiesOwned()) {
                if (!e.isDone()) myActivations++;
            }
        }
        state.put("my_activations_left", myActivations);

        List<float[]> entityArray = new ArrayList<>();
        List<Integer> entityIds = new ArrayList<>();
        List<float[]> weaponArray = new ArrayList<>();
        List<int[]> equipsEdges = new ArrayList<>();
        List<int[]> losTargetEdges = new ArrayList<>();
        List<int[]> losThreatEdges = new ArrayList<>();
        List<int[]> partialCoverEdges = new ArrayList<>();
        List<int[]> movementThreatEdges = new ArrayList<>();
        
        int weaponNodeId = 0;
        int boardWidth = game.getBoard().getWidth();
        int boardHeight = game.getBoard().getHeight();
        
        for (int i = 0; i < game.getEntitiesVector().size(); i++) {
            Entity e1 = game.getEntitiesVector().get(i);
            entityIds.add(e1.getId());
            entityArray.add(extractEntityFeatures(e1));
            
            for (Mounted<?> m : e1.getEquipment()) {
                if (m.getType() instanceof WeaponType) {
                    weaponArray.add(extractWeaponFeatures(m));
                    equipsEdges.add(new int[]{weaponNodeId, i});
                    weaponNodeId++;
                }
            }
            
            // LOS Target Edges
            for (int j = 0; j < game.getEntitiesVector().size(); j++) {
                if (i == j) continue;
                Entity e2 = game.getEntitiesVector().get(j);
                megamek.common.LosEffects los = megamek.common.LosEffects.calculateLOS(game, e1, e2);
                if (los.canSee()) {
                    losTargetEdges.add(new int[]{i, j});
                }
            }
            
            // LOS Threat and Partial Cover Edges
            for (int hexY = 0; hexY < boardHeight; hexY++) {
                for (int hexX = 0; hexX < boardWidth; hexX++) {
                    megamek.common.board.Coords targetCoords = new megamek.common.board.Coords(hexX, hexY);
                    megamek.common.HexTarget target = new megamek.common.HexTarget(targetCoords, game.getBoard(), megamek.common.units.Targetable.TYPE_HEX_CLEAR);
                    megamek.common.LosEffects los = megamek.common.LosEffects.calculateLOS(game, e1, target);
                    if (los.canSee()) {
                        int hexIdx = hexY * boardWidth + hexX;
                        losThreatEdges.add(new int[]{i, hexIdx});
                        if (los.isTargetCover()) {
                            partialCoverEdges.add(new int[]{i, hexIdx});
                        }
                    }
                }
            }
            
            // Movement Threat Edges
            java.util.Set<Integer> reachableHexes = new java.util.HashSet<>();
            try {
                int maxMove = Math.max(e1.getWalkMP(), e1.getRunMP());
                if (maxMove > 0 && !e1.isImmobile()) {
                    megamek.common.pathfinder.ShortestPathFinder spfGround = megamek.common.pathfinder.ShortestPathFinder.newInstanceOfOneToAll(maxMove, megamek.common.enums.MoveStepType.FORWARDS, game);
                    spfGround.run(new megamek.common.moves.MovePath(game, e1, null));
                    for (megamek.common.moves.MovePath p : spfGround.getAllComputedPathsUncategorized()) {
                        if (p.getFinalCoords() != null) {
                            reachableHexes.add(p.getFinalCoords().getY() * boardWidth + p.getFinalCoords().getX());
                        }
                    }
                }
            } catch (Exception ex) {
                // Ignore if pathfinding fails for some entities
            }
            
            for (int hexIdx : reachableHexes) {
                movementThreatEdges.add(new int[]{i, hexIdx});
            }
        }
        
        state.put("entities", entityArray);
        state.put("entity_id_map", entityIds);
        state.put("weapons", weaponArray);
        state.put("equips_edges", equipsEdges);
        state.put("los_target_edges", losTargetEdges);
        state.put("los_threat_edges", losThreatEdges);
        state.put("partial_cover_edges", partialCoverEdges);
        state.put("movement_threat_edges", movementThreatEdges);

        return state;
    }

    public List<MovePath> buildMovementMask(Entity mover, List<RLActionMask.RLPathMask> serializedMaskOut) {
        Game game = (Game) baseClient.getGame();
        List<MovePath> calculatedPaths = new ArrayList<>();
        serializedMaskOut.clear();
        
        // Ground path generation
        int maxMove = Math.max(mover.getWalkMP(), Math.max(mover.getRunMPWithoutMASC(), mover.getJumpMP()));
        if (maxMove > 0) {
            megamek.common.pathfinder.ShortestPathFinder spfGround = megamek.common.pathfinder.ShortestPathFinder.newInstanceOfOneToAll(maxMove, megamek.common.enums.MoveStepType.FORWARDS, game);
            spfGround.run(new megamek.common.moves.MovePath(game, mover, null));
            calculatedPaths.addAll(spfGround.getAllComputedPathsUncategorized());
        }
        
        // Add jump paths if applicable
        if (mover.getAnyTypeMaxJumpMP() > 0) {
            megamek.common.pathfinder.ShortestPathFinder spfJump = megamek.common.pathfinder.ShortestPathFinder.newInstanceOfOneToAll(mover.getAnyTypeMaxJumpMP(), megamek.common.enums.MoveStepType.FORWARDS, game);
            spfJump.run(new megamek.common.moves.MovePath(game, mover, null).addStep(megamek.common.enums.MoveStepType.START_JUMP));
            calculatedPaths.addAll(spfJump.getAllComputedPathsUncategorized());
        }

        for (int i = 0; i < calculatedPaths.size(); i++) {
            MovePath p = calculatedPaths.get(i);
            
            // Filter out illegal destination states (stacking violations)
            if (!p.isMoveLegal() || megamek.common.compute.Compute.stackingViolation(game, mover.getId(), p.getFinalCoords(), mover.climbMode()) != null) {
                continue;
            }

            RLActionMask.RLPathMask pm = new RLActionMask.RLPathMask();
            pm.path_index = i;
            if (p.getFinalCoords() != null && game.getBoard() != null) {
                int destIndex = p.getFinalCoords().getY() * game.getBoard().getWidth() + p.getFinalCoords().getX();
                pm.dest_index = destIndex;
            }
            pm.dest_facing = p.getFinalFacing();
            pm.mp_used = p.getMpUsed();
            pm.is_jump = p.isJumping();
            serializedMaskOut.add(pm);
        }
        return calculatedPaths;
    }
    
    public void sendBehavioralCloningTrajectory(Entity mover, MovePath princessChosenPath) {
        if (!isConnected()) return;
        ensureTopologySent();
        
        try {
            List<RLActionMask.RLPathMask> serializedMask = new ArrayList<>();
            List<MovePath> calculatedPaths = buildMovementMask(mover, serializedMask);
            
            // Find target action index
            int chosenIndex = -1;
            boolean chosenIsJump = princessChosenPath.isJumping();
            int chosenFacing = princessChosenPath.getFinalFacing() != -1 ? princessChosenPath.getFinalFacing() : mover.getFacing();
            
            for (int i = 0; i < calculatedPaths.size(); i++) {
                MovePath mp = calculatedPaths.get(i);
                if (mp.getFinalCoords() != null && princessChosenPath.getFinalCoords() != null &&
                    mp.getFinalCoords().equals(princessChosenPath.getFinalCoords()) &&
                    mp.isJumping() == chosenIsJump &&
                    (mp.getFinalFacing() == chosenFacing || mp.getFinalFacing() == -1 && chosenFacing == mover.getFacing())) {
                    chosenIndex = i;
                    break;
                }
            }
            
            if (chosenIndex == -1) {
                // If somehow the engine allowed a move that mathematically our ShortestPathFinder didn't flag, 
                // we can't reliably use it for supervised training since the target pointer doesn't exist in the input sequence.
                return; 
            }
            
            // Translate the absolute calculated path index into the continuous subset index
            int subsetChoiceIndex = -1;
            for (int k = 0; k < serializedMask.size(); k++) {
                int absIdx = serializedMask.get(k).path_index;
                if (absIdx == chosenIndex) {
                    subsetChoiceIndex = k;
                    break;
                }
            }
            
            // If the maneuver was filtered out (e.g. stacking limit hit), discard this payload
            if (subsetChoiceIndex == -1) {
                return;
            }
            
            RLActionMask maskData = new RLActionMask();
            int activeIndex = baseClient.getGame().getEntitiesVector().indexOf(mover);
            maskData.active_entity_index = activeIndex;
            maskData.valid_paths = serializedMask;

            Map<String, Object> payload = new HashMap<>();
            payload.put("context", "MOVEMENT_BC");
            payload.put("state", serializeGameState());
            payload.put("mask", maskData);
            
            Map<String, Object> targetAction = new HashMap<>();
            targetAction.put("selected_path_index", subsetChoiceIndex);
            payload.put("target_action", targetAction);
            
            sendPayload(payload);
            System.err.println("RLDataPipeline: BC Trajectory Sent (Action " + chosenIndex + " / " + calculatedPaths.size() + " options)");
            
        } catch (Exception e) {
            logger.error("RLDataPipeline: Failed to send BC trajectory", e);
        }
    }
}
