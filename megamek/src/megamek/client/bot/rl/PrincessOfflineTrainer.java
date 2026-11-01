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

import java.io.File;

import megamek.common.commandLine.AbstractCommandLineParser;
import megamek.common.commandLine.ClientServerCommandLineParser;
import megamek.client.HeadlessClient;
import megamek.client.bot.BotClient;

import megamek.logging.MMLogger;
import megamek.server.Server;
import megamek.server.totalWarfare.TWGameManager;
import megamek.client.bot.princess.Princess;

public class PrincessOfflineTrainer {
    private static final MMLogger logger = MMLogger.create(PrincessOfflineTrainer.class);

    public static void start(String[] args) {
        // Prevent MegaMek's default uncaught exception handler from popping up a Swing dialog on headless servers
        Thread.setDefaultUncaughtExceptionHandler(new Thread.UncaughtExceptionHandler() {
            @Override
            public void uncaughtException(Thread t, Throwable e) {
                System.err.println("CRITICAL: Uncaught exception in thread " + t.getName());
                e.printStackTrace();
            }
        });

        System.out.println("PrincessOfflineTrainer start() called! autoGen check incoming...");
        boolean autoGen = false;
        boolean selfPlay = false;
        boolean bcDataGen = false;
        boolean randomMap = false;
        String p1Meks = "";
        String p2Meks = "";
        int randomBVTarget = 0;
        int maxMeks = 4;
        boolean autoSaveMidGame = false;
        String midGameSaveDir = "data/midgame_saves";
        if (System.getenv("AUTOSAVE_MIDGAME") != null && System.getenv("AUTOSAVE_MIDGAME").equalsIgnoreCase("true")) {
            autoSaveMidGame = true;
        }
        if (System.getenv("AUTOSAVE_MIDGAME_DIR") != null) {
            midGameSaveDir = System.getenv("AUTOSAVE_MIDGAME_DIR");
        }
        
        java.util.List<String> cleanArgs = new java.util.ArrayList<>();
        System.out.println("Args passed to PrincessOfflineTrainer:");
        for (int i = 0; i < args.length; i++) {
            System.out.println("  " + args[i]);
            if (args[i].equalsIgnoreCase("-autogen")) {
                autoGen = true;
            } else if (args[i].equalsIgnoreCase("-bcdatagen")) {
                bcDataGen = true;
                autoGen = true;
            } else if (args[i].equalsIgnoreCase("-selfplay")) {
                selfPlay = true;
                autoGen = true; // selfPlay implies we auto-generate the match
            } else if (args[i].equalsIgnoreCase("-randomMap")) {
                randomMap = true;
            } else if (args[i].equalsIgnoreCase("-p1meks")) {
                p1Meks = args[++i];
            } else if (args[i].equalsIgnoreCase("-p2meks")) {
                p2Meks = args[++i];
            } else if (args[i].equalsIgnoreCase("-randomBV")) {
                randomBVTarget = Integer.parseInt(args[++i]);
            } else if (args[i].equalsIgnoreCase("-maxMeks")) {
                maxMeks = Integer.parseInt(args[++i]);
            } else if (args[i].equalsIgnoreCase("-autoSaveMidGame")) {
                autoSaveMidGame = true;
                if (i + 1 < args.length && !args[i + 1].startsWith("-")) {
                    midGameSaveDir = args[++i];
                }
            } else if (args[i].equalsIgnoreCase("-load") || args[i].equalsIgnoreCase("-savegame")) {
                cleanArgs.add("-savegame");
                cleanArgs.add(args[++i]);
            } else {
                cleanArgs.add(args[i]);
            }
        }

        ClientServerCommandLineParser parser = new ClientServerCommandLineParser(cleanArgs.toArray(new String[0]),
                "RLEXPORT",
                true, false, false);
        try {
            parser.parse();
        } catch (AbstractCommandLineParser.ParseException e) {
            logger.error("Incorrect arguments:" + e.getMessage() + '\n' + parser.help());
        }

        int port = megamek.common.preference.PreferenceManager.getClientPreferences().getLastServerPort();
        System.out.println("DEBUG: RL_SERVER_PORT env var is: " + System.getenv("RL_SERVER_PORT"));
        if (System.getenv("RL_SERVER_PORT") != null) {
            port = Integer.parseInt(System.getenv("RL_SERVER_PORT"));
        }
        System.out.println("DEBUG: Chosen server port is: " + port);

        ClientServerCommandLineParser.Resolver resolver = parser.getResolver(
                null,
                port,
                null, null);

        // kick off a RNG check
        megamek.common.compute.Compute.d6();

        File gameFile = resolver.getSaveGameFile();
        Server server;

        try {
            final boolean doSaveMidGame = autoSaveMidGame && (gameFile == null);
            final int targetSaveRound = doSaveMidGame ? (4 + megamek.common.compute.Compute.randomInt(9)) : -1;
            final java.util.concurrent.atomic.AtomicBoolean saved = new java.util.concurrent.atomic.AtomicBoolean(false);
            final String finalSaveDir = midGameSaveDir;
            final int serverPort = port;

            server = new Server(resolver.password, resolver.port, new megamek.server.totalWarfare.TWGameManager() {
                @Override
                public void autoSave() {
                    if (doSaveMidGame && !saved.get() && getGame() != null && getGame().getCurrentRound() >= targetSaveRound) {
                        saved.set(true);
                        try {
                            File dir = new File(finalSaveDir);
                            if (!dir.exists()) {
                                dir.mkdirs();
                            }
                            File[] existing = dir.listFiles((d, name) -> name.endsWith(".sav") || name.endsWith(".sav.gz"));
                            if (existing != null && existing.length >= 500) {
                                java.util.Arrays.sort(existing, java.util.Comparator.comparingLong(File::lastModified));
                                int toDelete = existing.length - 499;
                                for (int k = 0; k < toDelete; k++) {
                                    existing[k].delete();
                                }
                            }
                            String filename = String.format("snapshot_r%d_%d_%d.sav.gz", getGame().getCurrentRound(), System.currentTimeMillis(), serverPort);
                            File targetFile = new File(dir, filename);
                            try (java.io.OutputStream os = new java.io.FileOutputStream(targetFile);
                                 java.io.OutputStream gzo = new java.util.zip.GZIPOutputStream(os);
                                 java.io.Writer writer = new java.io.OutputStreamWriter(gzo, java.nio.charset.StandardCharsets.UTF_8)) {
                                megamek.common.util.SerializationHelper.getSaveGameXStream().toXML(getGame(), writer);
                            }
                            System.out.println("PSI: Saved mid-game snapshot at round " + getGame().getCurrentRound() + " to " + targetFile.getAbsolutePath());
                            logger.info("PSI: Saved mid-game snapshot at round " + getGame().getCurrentRound() + " to " + targetFile.getAbsolutePath());
                        } catch (Exception e) {
                            System.err.println("PSI: Failed to save mid-game snapshot: " + e.getMessage());
                            logger.error("PSI: Failed to save mid-game snapshot", e);
                        }
                    }
                }
            }, resolver.registerServer, resolver.announceUrl, null, true);
        } catch (Throwable ex) {
            System.err.println("FATAL THROWABLE IN SERVER START: " + ex);
            ex.printStackTrace(System.err);
            logger.error("Error: could not start server at localhost:" + resolver.port, ex);
            return;
        }

        String targetP1Name;
        String targetP2Name;
        if (selfPlay) {
            targetP1Name = "RL_Agent_Alpha";
            targetP2Name = "RL_Agent_Beta";
        } else if (autoGen && !bcDataGen) {
            targetP1Name = "RL_Agent";
            targetP2Name = "Princess_Beta";
        } else {
            targetP1Name = "Princess_Alpha";
            targetP2Name = "Princess_Beta";
        }

        if (null != gameFile) {
            boolean loaded = server.loadGame(gameFile);
            if (!loaded) {
                System.err.println("FATAL: Failed to load savegame: " + gameFile.getAbsolutePath());
                logger.error("FATAL: Failed to load savegame: " + gameFile.getAbsolutePath());
                System.exit(1);
            }
            // Remap saved ghost player names to match the connecting bot names for PSI restoration
            java.util.List<megamek.common.Player> combatantGhosts = new java.util.ArrayList<>();
            for (megamek.common.Player p : server.getGame().getPlayersList()) {
                if (p != null && !"RL_Host".equals(p.getName()) && !p.isObserver()) {
                    combatantGhosts.add(p);
                }
            }
            combatantGhosts.sort(java.util.Comparator.comparingInt(megamek.common.Player::getId));
            if (combatantGhosts.size() >= 1) {
                System.out.println("PSI: Remapping saved Player 1 '" + combatantGhosts.get(0).getName() + "' -> '" + targetP1Name + "'");
                combatantGhosts.get(0).setName(targetP1Name);
                combatantGhosts.get(0).setGhost(true);
            }
            if (combatantGhosts.size() >= 2) {
                System.out.println("PSI: Remapping saved Player 2 '" + combatantGhosts.get(1).getName() + "' -> '" + targetP2Name + "'");
                combatantGhosts.get(1).setName(targetP2Name);
                combatantGhosts.get(1).setGhost(true);
            }
        }

        logger.info("MegaMek Offline Princess Trainer started on port " + resolver.port);

        // Connect a HeadlessClient first so it gets Host privileges
        HeadlessClient watcher = new HeadlessClient("RL_Host", "localhost", resolver.port);
        watcher.setSendDoneOnVictoryAutomatically(true); // Matches QuickGameRunner config
        watcher.connect();

        // Connect the Princess bots to the local server
        BotClient p1 = null;
        BotClient p2 = null;
        try {
            if (selfPlay) {
                p1 = new RLBotClient(targetP1Name, "localhost", resolver.port, resolver.port + 1000);
                p2 = new RLBotClient(targetP2Name, "localhost", resolver.port, resolver.port + 1001);
            } else if (autoGen && !bcDataGen) {
                p1 = new RLBotClient(targetP1Name, "localhost", resolver.port, resolver.port + 1000);
                RLDataCollectionPrincess p2Bot = new RLDataCollectionPrincess(targetP2Name, "localhost", resolver.port, resolver.port + 1001);
                megamek.client.bot.princess.BehaviorSettings bs = p2Bot.getBehaviorSettings().getCopy();
                bs.setAutoFlee(false);
                bs.setForcedWithdrawal(false);
                p2Bot.setBehaviorSettings(bs);
                p2 = p2Bot;
            } else {
                // Used for bcDataGen or default manual operation
                p1 = new RLDataCollectionPrincess(targetP1Name, "localhost", resolver.port, resolver.port + 1000);
                p2 = new RLDataCollectionPrincess(targetP2Name, "localhost", resolver.port, resolver.port + 1001);
            }
            
            p1.connect();
            p2.connect();
            
            Thread.sleep(1000);
            watcher.sendPlayerInfo();
            p1.sendPlayerInfo();
            p2.sendPlayerInfo();
        } catch (Exception e) {
            logger.error(e, "Error launching Princess offline bots");
        }
        
        // Wait and configure autogen properties
        if (autoGen) {
            try {
                Thread.sleep(2500); // Give all clients time to establish Lobby connection and receive packet syncs
                
                megamek.common.game.Game game = (megamek.common.game.Game) server.getGame();
                
                // Map setup
                if (gameFile == null && randomMap) {
                    System.out.println("Generating random map...");
                    logger.info("Generating random map...");
                    megamek.common.loaders.MapSettings ms = megamek.common.loaders.MapSettings.getInstance();
                    ms.setBoardSize(16, 16);
                    System.out.println("Calling generateRandom...");
                    megamek.common.board.Board board = megamek.common.util.BoardUtilities.generateRandom(ms);
                    System.out.println("Board generated!");
                    game.setBoard(board);
                    logger.info("Random map generated.");
                }
                
                // Enforce a hard Round/Turn limit so offline bots don't stalemate infinitely
                game.getOptions().getOption(megamek.common.options.OptionsConstants.VICTORY_USE_GAME_TURN_LIMIT).setValue(true);
                game.getOptions().getOption(megamek.common.options.OptionsConstants.VICTORY_GAME_TURN_LIMIT).setValue(30);
                game.getOptions().getOption(megamek.common.options.OptionsConstants.BASE_MAX_NUMBER_ROUND_SAVES).setValue(0);
                
                // Get Players
                System.out.println("Waiting for Princess players to appear in lobby...");
                logger.info("Waiting for Princess players to appear in lobby...");
                megamek.common.Player player1 = null;
                megamek.common.Player player2 = null;
                int attempts = 0;
                while ((player1 == null || player2 == null) && attempts < 120) {
                    for (megamek.common.Player p : game.getPlayersList()) {
                        System.out.println("Found player: " + p.getName());
                        if (p.getName().equals("RL_Agent") || p.getName().equals("Princess_Alpha") || p.getName().equals("RL_Agent_Alpha")) player1 = p;
                        if (p.getName().equals("Princess_Beta") || p.getName().equals("RL_Agent_Beta")) player2 = p;
                    }
                    if (player1 == null || player2 == null) {
                        System.out.println("Missing a Princess, sleeping...");
                        Thread.sleep(500);
                        attempts++;
                    }
                }
                
                System.out.println("Attempt loop finished.");
                
                if (player1 != null && player2 != null) {
                    if (gameFile == null) {
                        System.out.println("Found both Princess players in Lobby, assigning Entities...");
                        logger.info("Found both Princess players in Lobby, assigning Entities...");
                        
                        // Generate forces so that the players have entities recognised by the Turn Generator
                        megamek.common.force.Force f1 = megamek.common.force.Force.createToplevelForce("Force Alpha", player1);
                        megamek.common.force.Force f2 = megamek.common.force.Force.createToplevelForce("Force Beta", player2);
                        int f1Id = game.getForces().addTopLevelForce(f1, player1);
                        int f2Id = game.getForces().addTopLevelForce(f2, player2);
                        
                        player1.setTeam(1);
                        player2.setTeam(2);

                        System.out.println("Forces created.");

                    // Assign Forces
                    if (!p1Meks.isEmpty()) {
                        for (String mekFile : p1Meks.split(",")) {
                            System.out.println("Parsing P1 mek: " + mekFile);
                            // Extract just the Mek.
                            megamek.common.units.Entity ent = null;
                            try {
                                ent = new megamek.common.loaders.MekFileParser(new File(mekFile.trim().replace("\"", ""))).getEntity();
                            } catch (Exception e) {
                                System.out.println("Exception parsing Mek: " + e.getMessage());
                            }
                            if (ent != null) {
                                ent.setOwner(player1);
                                ent.setDeployed(true);
                                // Set arbitrary starting coords on opposite sides
                                int x = megamek.common.compute.Compute.randomInt(4);
                                int y = megamek.common.compute.Compute.randomInt(game.getBoard().getHeight());
                                ent.setPosition(new megamek.common.board.Coords(x, y));
                                ent.setId(game.getNextEntityId());
                                game.addEntity(ent);
                                game.getForces().addEntity(ent, f1Id);
                            } else {
                                System.out.println("WARNING: P1 Mek parsed as NULL: " + mekFile);
                            }
                        }
                    }
                    if (!p2Meks.isEmpty()) {
                        for (String mekFile : p2Meks.split(",")) {
                            System.out.println("Parsing P2 mek: " + mekFile);
                            megamek.common.units.Entity ent = null;
                            try {
                                ent = new megamek.common.loaders.MekFileParser(new File(mekFile.trim().replace("\"", ""))).getEntity();
                            } catch (Exception e) {
                                System.out.println("Exception parsing Mek: " + e.getMessage());
                            }
                            if (ent != null) {
                                ent.setOwner(player2);
                                ent.setDeployed(true);
                                // Set arbitrary starting coords on opposite sides
                                int x = game.getBoard().getWidth() - 1 - megamek.common.compute.Compute.randomInt(4);
                                int y = megamek.common.compute.Compute.randomInt(game.getBoard().getHeight());
                                ent.setPosition(new megamek.common.board.Coords(x, y));
                                ent.setId(game.getNextEntityId());
                                game.addEntity(ent);
                                game.getForces().addEntity(ent, f2Id);
                            } else {
                                System.out.println("WARNING: P2 Mek parsed as NULL: " + mekFile);
                            }
                        }
                    }
                    
                    if (p1Meks.isEmpty() && p2Meks.isEmpty() && randomBVTarget > 0) {
                        System.out.println("Generating random forces with target BV: " + randomBVTarget);
                        logger.info("Generating random forces with target BV: " + randomBVTarget);
                        
                        megamek.common.loaders.MekSummary[] allMeks = megamek.common.loaders.MekSummaryCache.getInstance().getAllMeks();
                        java.util.List<megamek.common.loaders.MekSummary> validMeks = new java.util.ArrayList<>();
                        for (megamek.common.loaders.MekSummary ms : allMeks) {
                            if ("Mek".equals(ms.getUnitType())) {
                                validMeks.add(ms);
                            }
                        }
                        
                        if (!validMeks.isEmpty()) {
                            // Generate force 1
                            java.util.List<megamek.common.units.Entity> force1 = generateForceByBV(randomBVTarget, validMeks, maxMeks);
                            for (megamek.common.units.Entity ent : force1) {
                                ent.setOwner(player1);
                                ent.setDeployed(true);
                                int x = megamek.common.compute.Compute.randomInt(4);
                                int y = megamek.common.compute.Compute.randomInt(game.getBoard().getHeight());
                                ent.setPosition(new megamek.common.board.Coords(x, y));
                                ent.setId(game.getNextEntityId());
                                game.addEntity(ent);
                                game.getForces().addEntity(ent, f1Id);
                            }
                            
                            // Generate force 2
                            java.util.List<megamek.common.units.Entity> force2 = generateForceByBV(randomBVTarget, validMeks, maxMeks);
                            for (megamek.common.units.Entity ent : force2) {
                                ent.setOwner(player2);
                                ent.setDeployed(true);
                                int x = game.getBoard().getWidth() - 1 - megamek.common.compute.Compute.randomInt(4);
                                int y = megamek.common.compute.Compute.randomInt(game.getBoard().getHeight());
                                ent.setPosition(new megamek.common.board.Coords(x, y));
                                ent.setId(game.getNextEntityId());
                                game.addEntity(ent);
                                game.getForces().addEntity(ent, f2Id);
                            }
                        }
                    }
                } else {
                    System.out.println("PSI: Resuming match from loaded save game: " + gameFile.getAbsolutePath());
                    logger.info("PSI: Resuming match from loaded save game: " + gameFile.getAbsolutePath());
                }
            
            System.out.println("Broadcasting entities...");
                    // Broadcast the newly added entities to the clients.
                    try {
                        megamek.server.totalWarfare.TWGameManager twm = (megamek.server.totalWarfare.TWGameManager) server.getGameManager();
                        twm.send(twm.createFullEntitiesPacket());
                        
                        twm.send(new megamek.common.net.packets.Packet(megamek.common.net.enums.PacketCommand.SENDING_BOARD, new java.util.HashMap<>(game.getBoards())));
                    } catch (Exception ex) {
                    }
                    
                    System.out.println("Sending done packets to push through phases...");
                    
                    int lobbyWaitSeconds = 0;
                    while (!server.getGame().getPhase().isInitiative() && !server.getGame().getPhase().isMovement()) {
                        System.out.println("CURRENT PHASE: " + server.getGame().getPhase().name());
                        System.out.println("isEmptyLobby? (Phase isLounge: " + server.getGame().getPhase().isLounge() + ", Objects: " + server.getGame().getInGameObjects().size() + ")");
                        for (megamek.common.Player p : server.getGame().getPlayersList()) {
                            System.out.println("Player " + p.getName() + " (Ghost: " + p.isGhost() + ", Observer: " + p.isObserver() + ") isDone: " + p.isDone());
                        }

                        if (p1 != null) p1.sendDone(true);
                        if (p2 != null) p2.sendDone(true);
                        watcher.sendDone(true);
                        
                        Thread.sleep(1000);
                        lobbyWaitSeconds++;
                        if (lobbyWaitSeconds >= 60) {
                            System.err.println("FATAL: Lobby phase transition timed out after 60s without reaching INITIATIVE/MOVEMENT. Exiting.");
                            logger.error("Lobby phase transition timed out after 60s. Exiting.");
                            System.exit(1);
                        }
                    }
                    
                    System.out.println("Sent host and bot Done packets. Reached INITIATIVE phase!");
                    
                    long lastPhaseChangeTime = System.currentTimeMillis();
                    megamek.common.enums.GamePhase lastObservedPhase = null;
                    while (true) {
                        try {
                            megamek.common.game.IGame g = server.getGame();
                            if (g == null) break;
                            megamek.common.enums.GamePhase phase = g.getPhase();
                            if (phase == null || phase.isVictory() || phase.isLounge() || !watcher.isConnected()) {
                                System.out.println("Game Loop Breaking! Reason:");
                                if (phase == null) System.out.println("- phase is null");
                                else {
                                    System.out.println("- phase.isVictory(): " + phase.isVictory());
                                    System.out.println("- phase.isLounge(): " + phase.isLounge());
                                }
                                System.out.println("- watcher.isConnected(): " + watcher.isConnected());
                                break;
                            }
                            if (phase != lastObservedPhase) {
                                lastObservedPhase = phase;
                                lastPhaseChangeTime = System.currentTimeMillis();
                            } else if (System.currentTimeMillis() - lastPhaseChangeTime > 300000) {
                                System.err.println("FATAL: Game loop stalled in phase " + phase + " for >300s. Exiting.");
                                logger.error("Game loop stalled in phase " + phase + " for >300s. Exiting.");
                                System.exit(1);
                            }
                        } catch (Exception ex) {
                            System.out.println("Loop exception: " + ex.getMessage());
                            break;
                        }
                        Thread.sleep(1000);
                    }
                    
                    System.out.println("Game Ended! Shutting down MegaMek server.");
                    
                    try {
                        megamek.common.game.IGame g = server.getGame();
                        if (g != null && p1 instanceof RLBotClient) {
                            if (player1 != null) {
                                boolean p1Won = false;
                                if (player1.getTeam() == megamek.common.Player.TEAM_NONE) {
                                    p1Won = player1.getId() == ((megamek.common.game.Game)g).getVictoryPlayerId();
                                } else {
                                    p1Won = player1.getTeam() == ((megamek.common.game.Game)g).getVictoryTeam();
                                }
                                System.out.println("P1 (RLBotClient) inferred won? " + p1Won);
                                ((RLBotClient)p1).sendGameOver(p1Won);
                            } else {
                                System.out.println("WARNING: player1 was null. Defaulting to false.");
                                ((RLBotClient)p1).sendGameOver(false);
                            }
                        }
                        if (g != null && p2 instanceof RLBotClient) {
                            if (player2 != null) {
                                boolean p2Won = false;
                                if (player2.getTeam() == megamek.common.Player.TEAM_NONE) {
                                    p2Won = player2.getId() == ((megamek.common.game.Game)g).getVictoryPlayerId();
                                } else {
                                    p2Won = player2.getTeam() == ((megamek.common.game.Game)g).getVictoryTeam();
                                }
                                System.out.println("P2 (RLBotClient) inferred won? " + p2Won);
                                ((RLBotClient)p2).sendGameOver(p2Won);
                            } else {
                                System.out.println("WARNING: player2 was null. Defaulting to false.");
                                ((RLBotClient)p2).sendGameOver(false);
                            }
                        }
                    } catch (Exception e) {
                        System.err.println("Error sending final game state: " + e.getMessage());
                    }

                    // Sleep briefly to ensure socket flush before hard close
                    Thread.sleep(500);

                    server.die();
                } else {
                    System.out.println("Could not find players!");
                }
                
            } catch (Throwable ex) {
                System.out.println("FATAL EXCEPTION: " + ex.getMessage());
                ex.printStackTrace();
            } finally {
                System.out.println("PrincessOfflineTrainer start() exiting, invoking System.exit(0)");
                System.exit(0);
            }
        }
    }

    private static java.util.List<megamek.common.units.Entity> generateForceByBV(int targetBV, java.util.List<megamek.common.loaders.MekSummary> validMeks, int maxMeksPerSide) {
        java.util.List<megamek.common.units.Entity> force = new java.util.ArrayList<>();
        int maxMeks = maxMeksPerSide > 0 ? maxMeksPerSide : 4;
        
        // Choose number of units randomly between maxMeks/2 and maxMeks to ensure we can reach targetBV
        int minUnits = Math.max(1, maxMeks / 2);
        int numUnits = minUnits;
        if (maxMeks > minUnits) {
            if (maxMeks == 2) {
                // Bias towards 1v1 (70% chance for 1, 30% chance for 2)
                numUnits = (new java.util.Random().nextDouble() < 0.7) ? 1 : 2;
            } else {
                numUnits = minUnits + megamek.common.compute.Compute.randomInt(maxMeks - minUnits + 1);
            }
        }
        
        int currentBV = 0;
        int attempts = 0;
        int maxAttempts = 500;
        
        while (force.size() < numUnits && currentBV < targetBV && attempts < maxAttempts) {
            attempts++;
            int remainingUnits = numUnits - force.size();
            int remainingBV = targetBV - currentBV;
            
            int unitTargetBV;
            if (remainingUnits == 1) {
                unitTargetBV = remainingBV;
            } else {
                double factor = 0.7 + new java.util.Random().nextDouble() * 0.6; // 0.7 to 1.3
                unitTargetBV = (int)((remainingBV / (double)remainingUnits) * factor);
            }
            
            megamek.common.loaders.MekSummary candidate = validMeks.get(megamek.common.compute.Compute.randomInt(validMeks.size()));
            int candidateBV = candidate.getBV();
            if (candidateBV <= 0) {
                continue;
            }
            
            boolean acceptable;
            if (remainingUnits == 1) {
                acceptable = Math.abs(candidateBV - remainingBV) <= (targetBV * 0.15) || (candidateBV <= remainingBV + (targetBV * 0.15));
            } else {
                acceptable = candidateBV >= (unitTargetBV * 0.5) && candidateBV <= (unitTargetBV * 1.5);
            }
            
            if (acceptable && (currentBV + candidateBV <= targetBV + (targetBV * 0.15))) {
                try {
                    megamek.common.units.Entity ent = new megamek.common.loaders.MekFileParser(candidate.getSourceFile(), candidate.getEntryName()).getEntity();
                    if (ent != null) {
                        force.add(ent);
                        currentBV += candidateBV;
                    }
                } catch (Exception e) {
                    // Ignore parsing issues
                }
            }
            
            // Fallback if we make too many attempts and fail to match the criteria
            if (attempts > 300) {
                if (candidateBV > 0 && currentBV + candidateBV <= targetBV + (targetBV * 0.15)) {
                    try {
                        megamek.common.units.Entity ent = new megamek.common.loaders.MekFileParser(candidate.getSourceFile(), candidate.getEntryName()).getEntity();
                        if (ent != null) {
                            force.add(ent);
                            currentBV += candidateBV;
                            if (currentBV >= targetBV - (targetBV * 0.15)) {
                                break;
                            }
                        }
                    } catch (Exception e) {
                    }
                }
            }
        }
        
        return force;
    }
}
