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
                pythonSocket.setSoTimeout(60000); // 60 seconds read timeout on actual communications
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

    public void close() {
        try {
            if (pythonIn != null)
                pythonIn.close();
            if (pythonOut != null)
                pythonOut.close();
            if (pythonSocket != null)
                pythonSocket.close();
            if (pythonServerSocket != null)
                pythonServerSocket.close();
        } catch (Exception e) {
            logger.error(e, "Error closing RLDataPipeline sockets");
        }
    }

    public void ensureTopologySent() {
        System.err.println("RL_SYNC_DEBUG [ensureTopologySent]: Check. hasSent: " + hasSentTopology + ", connected: "
                + isConnected());
        if (!hasSentTopology && baseClient.getGame() != null && baseClient.getGame().getBoard() != null
                && isConnected()) {
            System.err.println("RL_SYNC_DEBUG [ensureTopologySent]: Sending now!");
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
                        hexes.add(new float[] { 0f, 0f, 0f, 0f, 0f });
                        continue;
                    }

                    float elev = hex.getLevel();
                    float woods = Math.max(hex.terrainLevel(Terrains.WOODS), hex.terrainLevel(Terrains.JUNGLE));
                    if (woods < 0)
                        woods = 0;
                    float water = hex.terrainLevel(Terrains.WATER);
                    if (water < 0)
                        water = 0;
                    float isPavement = hex.hasPavement() ? 1f : 0f;
                    float isBuilding = hex.containsTerrain(Terrains.BUILDING) ? 1f : 0f;
                    float rough = hex.terrainLevel(Terrains.ROUGH) > 0 ? hex.terrainLevel(Terrains.ROUGH) : 0f;
                    float rubble = hex.terrainLevel(Terrains.RUBBLE) > 0 ? hex.terrainLevel(Terrains.RUBBLE) : 0f;
                    float swamp = hex.terrainLevel(Terrains.SWAMP) > 0 ? hex.terrainLevel(Terrains.SWAMP) : 0f;
                    float mud = hex.terrainLevel(Terrains.MUD) > 0 ? hex.terrainLevel(Terrains.MUD) : 0f;
                    float ice = hex.terrainLevel(Terrains.ICE) > 0 ? hex.terrainLevel(Terrains.ICE) : 0f;
                    float snow = hex.terrainLevel(Terrains.SNOW) > 0 ? hex.terrainLevel(Terrains.SNOW) : 0f;
                    float fire = hex.terrainLevel(Terrains.FIRE) > 0 ? hex.terrainLevel(Terrains.FIRE) : 0f;
                    float smoke = hex.terrainLevel(Terrains.SMOKE) > 0 ? hex.terrainLevel(Terrains.SMOKE) : 0f;
                    float isImpassable = hex.containsTerrain(Terrains.IMPASSABLE) ? 1f : 0f;

                    hexes.add(new float[] { elev, woods, water, isPavement, isBuilding, rough, rubble, swamp, mud, ice,
                            snow, fire, smoke, isImpassable });

                    int currentIndex = y * width + x;
                    for (int dir = 0; dir < 6; dir++) {
                        Hex adj = board.getHexInDir(x, y, dir);
                        if (adj != null) {
                            int adjIndex = adj.getCoords().getY() * width + adj.getCoords().getX();
                            edges.add(new int[] { currentIndex, adjIndex, dir });
                        }
                    }
                }
            }

            payload.put("hex_nodes", hexes);
            payload.put("hex_edges", edges);

            java.util.Map<String, Integer> featureDims = new java.util.HashMap<>();
            featureDims.put("hex", 14);
            featureDims.put("unit", 45);
            featureDims.put("weapon", 10);
            payload.put("feature_dims", featureDims);

            sendPayload(payload);
        } catch (Exception e) {
            System.err.println("RLDataPipeline: Failed to send topology: " + e.getMessage());
            e.printStackTrace(System.err);
            logger.error(e, "RLDataPipeline: Failed to send topology");
        }
    }

    public void broadcastStartGame() {
        if (!isConnected()) {
            return;
        }
        try {
            Map<String, Object> payload = new HashMap<>();
            payload.put("context", "START_GAME");
            sendPayload(payload);
        } catch (Exception e) {
            System.err.println("RLDataPipeline: Failed to broadcast START_GAME: " + e.getMessage());
            e.printStackTrace(System.err);
            logger.error(e, "RLDataPipeline: Failed to broadcast START_GAME");
        }
    }

    public synchronized void sendPayload(Map<String, Object> payload) throws Exception {
        byte[] bytes = msgpackMapper.writeValueAsBytes(payload);
        pythonOut.write(java.nio.ByteBuffer.allocate(4).putInt(bytes.length).array());
        pythonOut.write(bytes);
        pythonOut.flush();
    }

    public static boolean DEBUG_RL_SYNC = true;

    public <T> T queryPython(String actionContext, Object mask, Class<T> responseType) {
        if (!isConnected()) {
            return null;
        }

        ensureTopologySent();

        long startTime = System.currentTimeMillis();
        try {
            Map<String, Object> payload = new HashMap<>();
            payload.put("context", actionContext);
            payload.put("state", serializeGameState());
            payload.put("mask", mask);
            payload.put("rewards", calculateRewards());
            
            // We no longer send step-by-step reports.
            // Game logs are natively extracted at the end of the episode by the Python worker.

            byte[] bytes = msgpackMapper.writeValueAsBytes(payload);

            if (DEBUG_RL_SYNC) {
                System.err.println("RL_SYNC_DEBUG [queryPython]: Sending payload for context '" + actionContext
                        + "' | Size: " + bytes.length + " bytes");
            }

            pythonOut.write(java.nio.ByteBuffer.allocate(4).putInt(bytes.length).array());
            pythonOut.write(bytes);
            pythonOut.flush();

            // Read response
            byte[] lenBytes = new byte[4];
            java.io.DataInputStream dis = new java.io.DataInputStream(pythonIn);
            dis.readFully(lenBytes);
            int len = java.nio.ByteBuffer.wrap(lenBytes).getInt();

            byte[] dataBytes = new byte[len];
            dis.readFully(dataBytes);

            long endTime = System.currentTimeMillis();
            if (DEBUG_RL_SYNC) {
                System.err.println("RL_SYNC_DEBUG [queryPython]: Received response for '" + actionContext + "' | Size: "
                        + len + " bytes | Time: " + (endTime - startTime) + "ms");
            }

            return msgpackMapper.readValue(dataBytes, responseType);
        } catch (java.net.SocketTimeoutException e) {
            System.err.println("RL_SYNC_DEBUG [queryPython]: Socket timeout after "
                    + (System.currentTimeMillis() - startTime) + "ms! Context: " + actionContext);
            e.printStackTrace(System.err);
            logger.error(e, "RLDataPipeline: Python query failed due to timeout");
            return null;
        } catch (java.io.IOException e) {
            System.err.println(
                    "RL_SYNC_DEBUG [queryPython]: IOException (Broken Pipe / EOF). Disconnecting python client.");
            this.close();
            pythonSocket = null;
            hasSentTopology = false;
            System.err.println("RL_SYNC_DEBUG [queryPython]: Terminating MegaMek Server due to Python disconnection.");
            System.exit(0);
            return null;
        } catch (Exception e) {
            System.err.println("RL_SYNC_DEBUG [queryPython]: Failed! Exception: " + e.getClass().getName() + " - "
                    + e.getMessage());
            e.printStackTrace(System.err);
            logger.error(e, "RLDataPipeline: Python query failed");
            return null;
        }
    }

    public Map<String, Object> calculateRewards() {
        Game game = (Game) baseClient.getGame();
        Map<String, Object> rewards = new HashMap<>();
        if (game == null) return rewards;

        // Compute current BV for Player 1 and Player 2
        double bv1 = 0;
        double bv2 = 0;
        double vp1 = 0;
        double vp2 = 0;
        double tp1 = 0;
        double tp2 = 0;

        megamek.server.victory.VictoryResult vr = null;
        java.lang.reflect.Method getPlayerScoreMethod = null;
        try {
            vr = game.getVictoryResult();
            getPlayerScoreMethod = vr.getClass().getDeclaredMethod("getPlayerScore", int.class);
            getPlayerScoreMethod.setAccessible(true);
        } catch (Exception e) {
            // Ignore
        }

        for (Player p : game.getPlayersList()) {
            double bv = 0;
            double tp = 0;
            double vp = 0;
            
            if (vr != null && getPlayerScoreMethod != null) {
                try {
                    vp = (Double) getPlayerScoreMethod.invoke(vr, p.getId());
                } catch (Exception ex) {}
                if (vr.isVictory()) {
                    if (vr.getWinningPlayer() == p.getId()) vp += 100.0;
                    else if (!vr.isDraw()) vp -= 100.0;
                }
            }
            
            for (Entity e : game.getEntitiesVector()) {
                if (e == null) continue;
                if (e.getOwnerId() == p.getId()) {
                    if (!e.isDestroyed()) {
                        bv += e.calculateBattleValue();
                    }
                    // TP calculation
                    if (e.getHeat() >= 5)
                        tp += 1; // movement penalty
                    if (e.getHeat() >= 8)
                        tp += 1; // accuracy penalty
                    if (e.getHeat() >= 14)
                        tp += 1; // shutdown check
                    if (e.getHeat() >= 19 && e.getAmmo().size() > 0)
                        tp += 1; // ammo explosion
                }
            }
            if (baseClient.getLocalPlayer() != null && p.getTeam() == baseClient.getLocalPlayer().getTeam()) {
                bv1 += bv;
                vp1 += vp;
                tp1 += tp;
            } else {
                bv2 += bv;
                vp2 += vp;
                tp2 += tp;
            }
        }

        rewards.put("bv1", bv1);
        rewards.put("bv2", bv2);
        rewards.put("vp1", vp1 - vp2); // Zero sum
        rewards.put("vp2", vp2 - vp1); // Zero sum
        rewards.put("tp1", tp1);
        rewards.put("tp2", tp2);

        return rewards;
    }

    public float[] extractEntityFeatures(Entity e) {
        Game game = (Game) baseClient.getGame();
        Player localPlayer = baseClient.getLocalPlayer();

        float isFriendly = (localPlayer != null && e.getOwnerId() == localPlayer.getId()) ? 1f : 0f;
        float x = e.getPosition() != null ? e.getPosition().getX() : -1f;
        float y = e.getPosition() != null ? e.getPosition().getY() : -1f;
        float sinFacing = (float) Math.sin(e.getFacing() * Math.PI / 3.0);
        float cosFacing = (float) Math.cos(e.getFacing() * Math.PI / 3.0);
        float heat = e.getHeat();
        float maxHeat = e.getHeatCapacity();
        float armor = e.getTotalArmor();
        float structure = e.getTotalInternal();
        float tmm = megamek.common.compute.Compute.getTargetMovementModifier(game, e.getId()).getValue();

        float speedMode = 0f;
        if (e.moved != megamek.common.units.EntityMovementType.MOVE_NONE) {
            if (e.moved == megamek.common.units.EntityMovementType.MOVE_WALK)
                speedMode = 1f;
            else if (e.moved == megamek.common.units.EntityMovementType.MOVE_RUN)
                speedMode = 2f;
            else if (e.moved == megamek.common.units.EntityMovementType.MOVE_JUMP)
                speedMode = 3f;
        }

        float gunnery = e.getCrew() != null ? e.getCrew().getGunnery() : 4f;
        float piloting = e.getCrew() != null ? e.getCrew().getPiloting() : 5f;

        float walkMP = e.getWalkMP();
        float runMP = e.getRunMP();
        float jumpMP = e.getAnyTypeMaxJumpMP();
        float weight = (float) e.getWeight();
        float isProne = e.isProne() ? 1f : 0f;
        float isDestroyed = e.isDestroyed() ? 1f : 0f;
        float isImmobile = e.isImmobile() ? 1f : 0f;
        float height = (float) e.getHeight();

        float[] features = new float[45];
        features[0] = isFriendly;
        features[1] = x;
        features[2] = y;
        features[3] = sinFacing;
        features[4] = cosFacing;
        features[5] = heat;
        features[6] = maxHeat;
        features[7] = armor;
        features[8] = structure;
        features[9] = tmm;
        features[10] = speedMode;
        features[11] = gunnery;
        features[12] = piloting;
        features[13] = walkMP;
        features[14] = runMP;
        features[15] = jumpMP;
        features[16] = weight;
        features[17] = isProne;
        features[18] = isDestroyed;
        features[19] = isImmobile;
        features[20] = height;

        for (int i = 0; i < 8; i++) {
            if (i < e.locations()) {
                float maxArmor = Math.max(1f, e.getOArmor(i));
                features[21 + i] = e.getArmor(i) / maxArmor;
                
                float maxRearArmor = e.hasRearArmor(i) ? Math.max(1f, e.getOArmor(i, true)) : 1f;
                features[29 + i] = e.hasRearArmor(i) ? (e.getArmor(i, true) / maxRearArmor) : 0f;
                
                float maxInternal = Math.max(1f, e.getOInternal(i));
                features[37 + i] = e.getInternal(i) / maxInternal;
            } else {
                features[21 + i] = 0f;
                features[29 + i] = 0f;
                features[37 + i] = 0f;
            }
        }

        return features;
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

        return new float[] { minRange, shortRange, mediumRange, longRange, damage, isOperational, isCluster,
                numClusters, salvosRemaining, heat };
    }

    public Map<String, Object> serializeGameState() {
        Game game = (Game) baseClient.getGame();
        Player localPlayer = baseClient.getLocalPlayer();

        Map<String, Object> state = new HashMap<>();

        state.put("phase_main", game.getPhase().name());
        state.put("turn_number", game.getTurnIndex());
        state.put("round_number", game.getRoundCount());

        int myActivations = 0;
        int enemyActivations = 0;
        if (localPlayer != null) {
            state.put("current_player_id", localPlayer.getId());
            for (Entity e : game.getEntitiesVector()) {
                if (e == null) continue;
                if (!e.isDone() && !e.isDestroyed()) {
                    if (e.getOwnerId() == localPlayer.getId()) {
                        myActivations++;
                    } else {
                        enemyActivations++;
                    }
                }
            }
        }
        state.put("my_activations_left", myActivations);
        state.put("enemy_activations_left", enemyActivations);

        List<float[]> entityArray = new ArrayList<>();
        List<Integer> entityIds = new ArrayList<>();
        List<float[]> weaponArray = new ArrayList<>();
        List<int[]> equipsEdges = new ArrayList<>();
        List<int[]> losTargetEdges = new ArrayList<>();
        List<int[]> losThreatEdges = new ArrayList<>();
        List<int[]> partialCoverEdges = new ArrayList<>();
        List<int[]> movementThreatEdges = new ArrayList<>();
        List<int[]> moveTypeTMM0Edges = new ArrayList<>();
        List<int[]> moveTypeTMM1Edges = new ArrayList<>();
        List<int[]> moveTypeTMM2Edges = new ArrayList<>();
        List<int[]> moveTypeTMM3Edges = new ArrayList<>();
        List<int[]> moveTypeTMM4Edges = new ArrayList<>();

        int weaponNodeId = 0;
        int boardWidth = game.getBoard().getWidth();

        // Pass 1: Compute reachable hexes for all entities
        java.util.Map<Integer, java.util.Map<Integer, Integer>> entityReachableHexes = new java.util.HashMap<>();
        java.util.Set<Integer> globalReachableHexes = new java.util.HashSet<>();

        for (int i = 0; i < game.getEntitiesVector().size(); i++) {
            Entity e1 = game.getEntitiesVector().get(i);
            if (e1 == null) continue;
            java.util.Map<Integer, Integer> reachableHexes = new java.util.HashMap<>();

            // A unit can always "reach" its own hex (stationary TMM = 0)
            if (e1.getPosition() != null) {
                reachableHexes.put(e1.getPosition().getY() * boardWidth + e1.getPosition().getX(), 0);

                try {
                    int groundMove = Math.max(e1.getWalkMP(), e1.getRunMPWithoutMASC());
                    int jumpMove = e1.getAnyTypeMaxJumpMP();
                    int maxMove = Math.max(groundMove, jumpMove);

                    if (maxMove > 0 && !e1.isImmobile()) {
                        boolean airborneNonAerospace = e1.isAirborneVTOLorWIGE();
                        // Ground paths
                        if (groundMove > 0) {
                            megamek.common.pathfinder.ShortestPathFinder spfGround = megamek.common.pathfinder.ShortestPathFinder
                                    .newInstanceOfOneToAll(groundMove, megamek.common.enums.MoveStepType.FORWARDS,
                                            game);
                            spfGround.run(new megamek.common.moves.MovePath(game, e1, null));
                            for (megamek.common.moves.MovePath p : spfGround.getAllComputedPathsUncategorized()) {
                                if (p.getFinalCoords() != null) {
                                    int hexIdx = p.getFinalCoords().getY() * boardWidth + p.getFinalCoords().getX();
                                    int tmm = megamek.common.compute.Compute
                                            .getTargetMovementModifier(p.getHexesMoved(),
                                                    p.isJumping(), airborneNonAerospace, game)
                                            .getValue();
                                    if (!reachableHexes.containsKey(hexIdx) || tmm > reachableHexes.get(hexIdx)) {
                                        reachableHexes.put(hexIdx, tmm);
                                    }
                                }
                            }
                        }
                        // Jump paths
                        if (jumpMove > 0) {
                            megamek.common.pathfinder.ShortestPathFinder spfJump = megamek.common.pathfinder.ShortestPathFinder
                                    .newInstanceOfOneToAll(jumpMove, megamek.common.enums.MoveStepType.FORWARDS, game);
                            spfJump.run(new megamek.common.moves.MovePath(game, e1, null)
                                    .addStep(megamek.common.enums.MoveStepType.START_JUMP));
                            for (megamek.common.moves.MovePath p : spfJump.getAllComputedPathsUncategorized()) {
                                if (p.getFinalCoords() != null) {
                                    int hexIdx = p.getFinalCoords().getY() * boardWidth + p.getFinalCoords().getX();
                                    int tmm = megamek.common.compute.Compute
                                            .getTargetMovementModifier(p.getHexesMoved(),
                                                    p.isJumping(), airborneNonAerospace, game)
                                            .getValue();
                                    if (!reachableHexes.containsKey(hexIdx) || tmm > reachableHexes.get(hexIdx)) {
                                        reachableHexes.put(hexIdx, tmm);
                                    }
                                }
                            }
                        }
                    }
                } catch (Exception ex) {
                    // Ignore if pathfinding fails for some entities
                }
            }

            entityReachableHexes.put(i, reachableHexes);
            globalReachableHexes.addAll(reachableHexes.keySet());
        }

        // Pass 2: Extract features and build edges
        List<java.util.Map<String, Object>> entitiesMeta = new java.util.ArrayList<>();
        for (int i = 0; i < game.getEntitiesVector().size(); i++) {
            Entity e1 = game.getEntitiesVector().get(i);
            if (e1 == null) continue;
            entityIds.add(e1.getId());
            entityArray.add(extractEntityFeatures(e1));

            java.util.Map<String, Object> stateMeta = new java.util.HashMap<>();
            stateMeta.put("id", e1.getId());
            stateMeta.put("name", e1.getDisplayName());
            stateMeta.put("owner", e1.getOwner() != null ? e1.getOwner().getName() : "Unknown");
            List<java.util.Map<String, Object>> locs = new java.util.ArrayList<>();
            for (int loc = 0; loc < e1.locations(); loc++) {
                java.util.Map<String, Object> locData = new java.util.HashMap<>();
                locData.put("name", e1.getLocationName(loc));
                locData.put("armor", e1.getArmor(loc));
                locData.put("o_armor", e1.getOArmor(loc));
                locData.put("internal", e1.getInternal(loc));
                locData.put("o_internal", e1.getOInternal(loc));
                if (e1.hasRearArmor(loc)) {
                    locData.put("rear_armor", e1.getArmor(loc, true));
                    locData.put("o_rear_armor", e1.getOArmor(loc, true));
                }
                locs.add(locData);
            }
            stateMeta.put("locations", locs);
            
            List<String> weapons = new java.util.ArrayList<>();
            for (Mounted<?> m : e1.getEquipment()) {
                if (m.getType() instanceof WeaponType) {
                    WeaponType wt = (WeaponType) m.getType();
                    String wName = wt.getName();
                    if (m instanceof megamek.common.equipment.WeaponMounted) {
                        megamek.common.equipment.WeaponMounted wm = (megamek.common.equipment.WeaponMounted) m;
                        if (wm.getLinkedAmmo() != null) {
                            wName += " (" + wm.getLinkedAmmo().getUsableShotsLeft() + ")";
                        }
                    }
                    weapons.add(wName);
                }
            }
            stateMeta.put("weapons", weapons);
            entitiesMeta.add(stateMeta);

            for (Mounted<?> m : e1.getEquipment()) {
                if (m.getType() instanceof WeaponType) {
                    weaponArray.add(extractWeaponFeatures(m));
                    equipsEdges.add(new int[] { weaponNodeId, i });
                    weaponNodeId++;
                }
            }

            // LOS Target Edges
            for (int j = 0; j < game.getEntitiesVector().size(); j++) {
                if (i == j)
                    continue;
                Entity e2 = game.getEntitiesVector().get(j);
            if (e2 == null || !e1.isEnemyOf(e2)) continue;
                megamek.common.LosEffects los = megamek.common.LosEffects.calculateLOS(game, e1, e2);
                if (los.canSee()) {
                    losTargetEdges.add(new int[] { i, j });
                }
            }

            boolean isFriendly = (localPlayer != null && e1.getOwnerId() == localPlayer.getId());

            // LOS Threat and Partial Cover Edges (Restricted to reachable hexes and enemy units)
            if (!isFriendly) {
                for (int hexIdx : globalReachableHexes) {
                    int hexX = hexIdx % boardWidth;
                    int hexY = hexIdx / boardWidth;
                    megamek.common.board.Coords targetCoords = new megamek.common.board.Coords(hexX, hexY);
                    megamek.common.HexTarget target = new megamek.common.HexTarget(targetCoords, game.getBoard(),
                            megamek.common.units.Targetable.TYPE_HEX_CLEAR) {
                        @Override
                        public int getHeight() {
                            return 2; // Assume standard Mek height for evaluating hex visibility
                        }
                    };
                    megamek.common.LosEffects los = megamek.common.LosEffects.calculateLOS(game, e1, target);
                    if (los.canSee()) {
                        losThreatEdges.add(new int[] { i, hexIdx });
                        if (los.getTargetCover() > megamek.common.LosEffects.COVER_NONE) {
                            partialCoverEdges.add(new int[] { i, hexIdx });
                        }
                    }
                }
            }

            // Movement Threat and TMM Edges
            for (java.util.Map.Entry<Integer, Integer> entry : entityReachableHexes.get(i).entrySet()) {
                int hexIdx = entry.getKey();
                int tmm = entry.getValue();

                if (isFriendly) {
                    int[] edge = new int[] { i, hexIdx };
                    if (tmm <= 0)
                        moveTypeTMM0Edges.add(edge);
                    else if (tmm == 1)
                        moveTypeTMM1Edges.add(edge);
                    else if (tmm == 2)
                        moveTypeTMM2Edges.add(edge);
                    else if (tmm == 3)
                        moveTypeTMM3Edges.add(edge);
                    else
                        moveTypeTMM4Edges.add(edge);
                } else {
                    movementThreatEdges.add(new int[] { i, hexIdx });
                }
            }
        }

        state.put("entities", entityArray);
        state.put("entities_meta", entitiesMeta);
        state.put("entity_id_map", entityIds);
        state.put("weapons", weaponArray);
        state.put("equips_edges", equipsEdges);
        state.put("los_target_edges", losTargetEdges);
        state.put("los_threat_edges", losThreatEdges);
        state.put("partial_cover_edges", partialCoverEdges);
        state.put("movement_threat_edges", movementThreatEdges);
        state.put("move_type_tmm_0_edges", moveTypeTMM0Edges);
        state.put("move_type_tmm_1_edges", moveTypeTMM1Edges);
        state.put("move_type_tmm_2_edges", moveTypeTMM2Edges);
        state.put("move_type_tmm_3_edges", moveTypeTMM3Edges);
        state.put("move_type_tmm_4_edges", moveTypeTMM4Edges);

        return state;
    }

    public List<MovePath> buildMovementMask(Entity mover, List<RLActionMask.RLPathMask> serializedMaskOut) {
        Game game = (Game) baseClient.getGame();
        List<MovePath> calculatedPaths = new ArrayList<>();
        serializedMaskOut.clear();

        // Ground path generation
        int maxMove = Math.max(mover.getWalkMP(), Math.max(mover.getRunMPWithoutMASC(), mover.getJumpMP()));
        if (maxMove > 0) {
            megamek.common.pathfinder.ShortestPathFinder spfGround = megamek.common.pathfinder.ShortestPathFinder
                    .newInstanceOfOneToAll(maxMove, megamek.common.enums.MoveStepType.FORWARDS, game);
            spfGround.run(new megamek.common.moves.MovePath(game, mover, null));
            calculatedPaths.addAll(spfGround.getAllComputedPathsUncategorized());
        }

        // Add jump paths if applicable
        if (mover.getAnyTypeMaxJumpMP() > 0) {
            megamek.common.pathfinder.ShortestPathFinder spfJump = megamek.common.pathfinder.ShortestPathFinder
                    .newInstanceOfOneToAll(mover.getAnyTypeMaxJumpMP(), megamek.common.enums.MoveStepType.FORWARDS,
                            game);
            spfJump.run(new megamek.common.moves.MovePath(game, mover, null)
                    .addStep(megamek.common.enums.MoveStepType.START_JUMP));
            calculatedPaths.addAll(spfJump.getAllComputedPathsUncategorized());
        }

        for (int i = 0; i < calculatedPaths.size(); i++) {
            MovePath p = calculatedPaths.get(i);

            // Filter out illegal destination states (stacking violations)
            if (!p.isMoveLegal() || megamek.common.compute.Compute.stackingViolation(game, mover.getId(),
                    p.getFinalCoords(), mover.climbMode()) != null) {
                continue;
            }

            // Prevent NPEs by ensuring no hex in the path is null (e.g. stepping off board)
            boolean hasNullHex = false;
            if (p.getFinalCoords() != null && game.getBoard().getHex(p.getFinalCoords()) == null) {
                hasNullHex = true;
            } else {
                for (megamek.common.moves.MoveStep step : p.getStepVector()) {
                    if (game.getBoard().getHex(step.getPosition()) == null) {
                        hasNullHex = true;
                        break;
                    }
                }
            }
            if (hasNullHex) {
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
            pm.is_walk = (!pm.is_jump && pm.mp_used <= mover.getWalkMP());
            pm.is_run = (!pm.is_jump && pm.mp_used > mover.getWalkMP());
            serializedMaskOut.add(pm);
        }
        return calculatedPaths;
    }

    public void sendBehavioralCloningTrajectory(Entity mover, MovePath princessChosenPath) {
        if (!isConnected())
            return;
        ensureTopologySent();

        try {
            List<RLActionMask.RLPathMask> serializedMask = new ArrayList<>();
            List<MovePath> calculatedPaths = buildMovementMask(mover, serializedMask);

            // Find target action index
            int chosenIndex = -1;
            boolean chosenIsJump = princessChosenPath.isJumping();
            int chosenFacing = princessChosenPath.getFinalFacing() != -1 ? princessChosenPath.getFinalFacing()
                    : mover.getFacing();

            for (int i = 0; i < calculatedPaths.size(); i++) {
                MovePath mp = calculatedPaths.get(i);
                if (mp.getFinalCoords() != null && princessChosenPath.getFinalCoords() != null &&
                        mp.getFinalCoords().equals(princessChosenPath.getFinalCoords()) &&
                        mp.isJumping() == chosenIsJump &&
                        (mp.getFinalFacing() == chosenFacing
                                || mp.getFinalFacing() == -1 && chosenFacing == mover.getFacing())) {
                    chosenIndex = i;
                    break;
                }
            }

            if (chosenIndex == -1) {
                // If somehow the engine allowed a move that mathematically our
                // ShortestPathFinder didn't flag,
                // we can't reliably use it for supervised training since the target pointer
                // doesn't exist in the input sequence.
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

            // If the maneuver was filtered out (e.g. stacking limit hit), discard this
            // payload
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
            payload.put("rewards", calculateRewards());

            Map<String, Object> targetAction = new HashMap<>();
            targetAction.put("selected_path_index", subsetChoiceIndex);
            payload.put("target_action", targetAction);

            sendPayload(payload);
            System.err.println("RLDataPipeline: BC Trajectory Sent (Action " + chosenIndex + " / "
                    + calculatedPaths.size() + " options)");

        } catch (Exception e) {
            logger.error("RLDataPipeline: Failed to send BC trajectory", e);
        }
    }

    public List<RLActionMask.RLTwistMask> buildFiringMask(Entity shooter) {
        List<RLActionMask.RLTwistMask> validTwists = new ArrayList<>();
        megamek.common.game.Game game = baseClient.getGame();

        int originalFacing = shooter.getSecondaryFacing();
        List<Integer> facingChanges = new ArrayList<>(
                megamek.client.bot.princess.FireControl.getValidFacingChanges(shooter));
        facingChanges.add(0); // "no facing change"

        for (int twist : facingChanges) {
            // Apply the twist temporarily
            int newFacing = megamek.client.bot.princess.FireControl.correctFacing(originalFacing + twist);
            shooter.setSecondaryFacing(newFacing, false);

            List<RLActionMask.RLTargetMask> targetsMask = new ArrayList<>();
            for (Entity target : game.getEntitiesVector()) {
                if (target == null) continue;
                if (!target.isTargetable() || target.isDestroyed() || !target.isEnemyOf(shooter)) {
                    continue;
                }

                boolean inFrontArc = megamek.common.compute.ComputeArc.isInArc(shooter.getPosition(),
                        shooter.getSecondaryFacing(), target, shooter.getForwardArc());
                int secondaryPenalty = 2;
                if (inFrontArc || shooter instanceof megamek.common.battleArmor.BattleArmor) {
                    secondaryPenalty = 1;
                }
                if (shooter.hasAbility(megamek.common.options.OptionsConstants.GUNNERY_MULTI_TASKER)) {
                    secondaryPenalty--;
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
                        wData.secondary_to_hit = wData.to_hit + secondaryPenalty;
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

            if (!targetsMask.isEmpty()) {
                RLActionMask.RLTwistMask twistMask = new RLActionMask.RLTwistMask();
                twistMask.twist = twist;
                twistMask.valid_targets = targetsMask;
                validTwists.add(twistMask);
            }

            // Revert the twist
            shooter.setSecondaryFacing(originalFacing, false);
        }

        return validTwists;
    }

    public void sendWeaponBehavioralCloningTrajectory(Entity shooter,
            java.util.Vector<megamek.common.actions.EntityAction> attacks, int chosenTwist) {
        if (!isConnected())
            return;

        ensureTopologySent();

        try {
            RLActionMask maskData = new RLActionMask();
            int activeIndex = baseClient.getGame().getEntitiesVector().indexOf(shooter);
            maskData.active_entity_index = activeIndex;
            maskData.valid_twists = buildFiringMask(shooter);

            RLActionMask.RLTargetAction targetAction = new RLActionMask.RLTargetAction();
            targetAction.torso_twist = chosenTwist;
            targetAction.attacks = new ArrayList<>();

            for (megamek.common.actions.EntityAction ea : attacks) {
                if (ea instanceof megamek.common.actions.WeaponAttackAction) {
                    megamek.common.actions.WeaponAttackAction waa = (megamek.common.actions.WeaponAttackAction) ea;
                    Entity target = baseClient.getGame().getEntity(waa.getTargetId());

                    if (target != null) {
                        RLActionMask.RLAttack att = new RLActionMask.RLAttack();
                        att.target_entity_index = baseClient.getGame().getEntitiesVector().indexOf(target);
                        att.weapon_id = waa.getWeaponId();
                        targetAction.attacks.add(att);
                    }
                }
            }

            maskData.target_action = targetAction;

            Map<String, Object> payload = new HashMap<>();
            payload.put("context", "WEAPON_BC");
            payload.put("state", serializeGameState());
            payload.put("mask", maskData);
            payload.put("rewards", calculateRewards());
            payload.put("target_action", targetAction);

            sendPayload(payload);

        } catch (Exception e) {
            logger.error("RLDataPipeline: Failed to send Weapon BC trajectory", e);
        }
    }

    public List<RLActionMask.RLTargetMask> buildPhysicalMask(Entity shooter) {
        List<RLActionMask.RLTargetMask> targetsMask = new ArrayList<>();
        megamek.common.game.Game game = baseClient.getGame();

        for (Entity target : game.getEntitiesVector()) {
                if (target == null) continue;
            if (!target.isTargetable() || target.isDestroyed() || !target.isEnemyOf(shooter))
                continue;

            List<RLActionMask.RLPhysicalMask> validAttacks = new ArrayList<>();

            // Try PUNCH
            for (int i = 0; i < 2; i++) {
                int arm = (i == 0) ? megamek.common.actions.PunchAttackAction.LEFT
                        : megamek.common.actions.PunchAttackAction.RIGHT;
                int poType = (i == 0) ? megamek.client.bot.PhysicalOption.PUNCH_LEFT
                        : megamek.client.bot.PhysicalOption.PUNCH_RIGHT;

                megamek.common.ToHitData th = megamek.common.actions.PunchAttackAction.toHit(
                        game, shooter.getId(), target, arm, false);
                if (th.getValue() != megamek.common.rolls.TargetRoll.IMPOSSIBLE
                        && th.getValue() != megamek.common.rolls.TargetRoll.AUTOMATIC_FAIL) {
                    RLActionMask.RLPhysicalMask att = new RLActionMask.RLPhysicalMask();
                    att.action_type = poType;
                    att.name = (i == 0) ? "PUNCH_LEFT" : "PUNCH_RIGHT";
                    att.to_hit = th.getValue();
                    validAttacks.add(att);
                }
            }

            // Try KICK
            for (int i = 0; i < 2; i++) {
                int leg = (i == 0) ? megamek.common.actions.KickAttackAction.LEFT
                        : megamek.common.actions.KickAttackAction.RIGHT;
                int poType = (i == 0) ? megamek.client.bot.PhysicalOption.KICK_LEFT
                        : megamek.client.bot.PhysicalOption.KICK_RIGHT;

                megamek.common.ToHitData th = megamek.common.actions.KickAttackAction.toHit(
                        game, shooter.getId(), target, leg);
                if (th.getValue() != megamek.common.rolls.TargetRoll.IMPOSSIBLE
                        && th.getValue() != megamek.common.rolls.TargetRoll.AUTOMATIC_FAIL) {
                    RLActionMask.RLPhysicalMask att = new RLActionMask.RLPhysicalMask();
                    att.action_type = poType;
                    att.name = (i == 0) ? "KICK_LEFT" : "KICK_RIGHT";
                    att.to_hit = th.getValue();
                    validAttacks.add(att);
                }
            }

            // Try PUSH
            megamek.common.ToHitData pushTh = megamek.common.actions.PushAttackAction.toHit(game, shooter.getId(),
                    target);
            if (pushTh.getValue() != megamek.common.rolls.TargetRoll.IMPOSSIBLE
                    && pushTh.getValue() != megamek.common.rolls.TargetRoll.AUTOMATIC_FAIL) {
                RLActionMask.RLPhysicalMask att = new RLActionMask.RLPhysicalMask();
                att.action_type = megamek.client.bot.PhysicalOption.PUSH_ATTACK;
                att.name = "PUSH";
                att.to_hit = pushTh.getValue();
                validAttacks.add(att);
            }

            if (!validAttacks.isEmpty()) {
                RLActionMask.RLTargetMask tm = new RLActionMask.RLTargetMask();
                tm.target_entity_index = game.getEntitiesVector().indexOf(target);
                tm.target_entity_id = target.getId();
                tm.valid_attacks = validAttacks;
                targetsMask.add(tm);
            }
        }

        return targetsMask;
    }

    public void sendPhysicalBehavioralCloningTrajectory(Entity shooter,
            java.util.Vector<megamek.common.actions.EntityAction> attacks) {
        if (!isConnected())
            return;

        ensureTopologySent();

        try {
            RLActionMask maskData = new RLActionMask();
            int activeIndex = baseClient.getGame().getEntitiesVector().indexOf(shooter);
            maskData.active_entity_index = activeIndex;
            maskData.valid_targets = buildPhysicalMask(shooter);

            RLActionMask.RLTargetAction targetAction = new RLActionMask.RLTargetAction();
            targetAction.attacks = new ArrayList<>();

            for (megamek.common.actions.EntityAction ea : attacks) {
                if (ea instanceof megamek.common.actions.PhysicalAttackAction
                        || ea instanceof megamek.common.actions.PushAttackAction) {
                    megamek.common.actions.AbstractAttackAction paa = (megamek.common.actions.AbstractAttackAction) ea;
                    Entity target = baseClient.getGame().getEntity(paa.getTargetId());

                    if (target != null) {
                        RLActionMask.RLAttack att = new RLActionMask.RLAttack();
                        att.target_entity_index = baseClient.getGame().getEntitiesVector().indexOf(target);
                        // Map physical actions back to the constants used in the mask
                        if (paa instanceof megamek.common.actions.PunchAttackAction) {
                            att.physical_action_type = (((megamek.common.actions.PunchAttackAction) paa)
                                    .getArm() == megamek.common.actions.PunchAttackAction.LEFT)
                                            ? megamek.client.bot.PhysicalOption.PUNCH_LEFT
                                            : megamek.client.bot.PhysicalOption.PUNCH_RIGHT;
                        } else if (paa instanceof megamek.common.actions.KickAttackAction) {
                            att.physical_action_type = (((megamek.common.actions.KickAttackAction) paa)
                                    .getLeg() == megamek.common.actions.KickAttackAction.LEFT)
                                            ? megamek.client.bot.PhysicalOption.KICK_LEFT
                                            : megamek.client.bot.PhysicalOption.KICK_RIGHT;
                        } else if (paa instanceof megamek.common.actions.PushAttackAction) {
                            att.physical_action_type = megamek.client.bot.PhysicalOption.PUSH_ATTACK;
                        } else if (paa instanceof megamek.common.actions.ClubAttackAction) {
                            att.physical_action_type = megamek.client.bot.PhysicalOption.USE_CLUB;
                        }

                        if (att.physical_action_type != null) {
                            targetAction.attacks.add(att);
                        }
                    }
                }
            }

            maskData.target_action = targetAction;

            Map<String, Object> payload = new HashMap<>();
            payload.put("context", "PHYSICAL_BC");
            payload.put("state", serializeGameState());
            payload.put("mask", maskData);
            payload.put("rewards", calculateRewards());
            payload.put("target_action", targetAction);

            sendPayload(payload);

        } catch (Exception e) {
            logger.error("RLDataPipeline: Failed to send Physical BC trajectory", e);
        }
    }
}
