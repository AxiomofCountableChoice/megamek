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

        Server server;

        try {
            server = new Server(resolver.password, resolver.port, new megamek.server.totalWarfare.TWGameManager() {
                @Override
                public void autoSave() {
                    // Disable autosave to avoid disk IO bottlenecks and server deadlocks during RL matches
                }
            }, resolver.registerServer, resolver.announceUrl, null, true);
        } catch (Throwable ex) {
            System.err.println("FATAL THROWABLE IN SERVER START: " + ex);
            ex.printStackTrace(System.err);
            logger.error("Error: could not start server at localhost:" + resolver.port, ex);
            return;
        }

        File gameFile = resolver.getSaveGameFile();
        if (null != gameFile) {
            server.loadGame(gameFile);
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
                p1 = new RLBotClient("RL_Agent_Alpha", "localhost", resolver.port, resolver.port + 1000);
                p2 = new RLBotClient("RL_Agent_Beta", "localhost", resolver.port, resolver.port + 1001);
            } else if (autoGen && !bcDataGen) {
                p1 = new RLBotClient("RL_Agent", "localhost", resolver.port, resolver.port + 1000);
                RLDataCollectionPrincess p2Bot = new RLDataCollectionPrincess("Princess_Beta", "localhost", resolver.port, resolver.port + 1001);
                megamek.client.bot.princess.BehaviorSettings bs = p2Bot.getBehaviorSettings().getCopy();
                bs.setAutoFlee(false);
                bs.setForcedWithdrawal(false);
                p2Bot.setBehaviorSettings(bs);
                p2 = p2Bot;
            } else {
                // Used for bcDataGen or default manual operation
                p1 = new RLDataCollectionPrincess("Princess_Alpha", "localhost", resolver.port, resolver.port + 1000);
                p2 = new RLDataCollectionPrincess("Princess_Beta", "localhost", resolver.port, resolver.port + 1001);
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
                if (randomMap) {
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
                            java.util.List<megamek.common.units.Entity> force1 = generateForceByBV(randomBVTarget, validMeks);
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
                            java.util.List<megamek.common.units.Entity> force2 = generateForceByBV(randomBVTarget, validMeks);
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
                    
                    System.out.println("Broadcasting entities...");
                    // Broadcast the newly added entities to the clients.
                    try {
                        megamek.server.totalWarfare.TWGameManager twm = (megamek.server.totalWarfare.TWGameManager) server.getGameManager();
                        twm.send(twm.createFullEntitiesPacket());
                        
                        twm.send(new megamek.common.net.packets.Packet(megamek.common.net.enums.PacketCommand.SENDING_BOARD, new java.util.HashMap<>(game.getBoards())));
                    } catch (Exception ex) {
                    }
                    
                    System.out.println("Sending done packets to push through phases...");
                    
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
                    }
                    
                    System.out.println("Sent host and bot Done packets. Reached INITIATIVE phase!");
                    
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

    private static java.util.List<megamek.common.units.Entity> generateForceByBV(int targetBV, java.util.List<megamek.common.loaders.MekSummary> validMeks) {
        java.util.List<megamek.common.units.Entity> force = new java.util.ArrayList<>();
        int currentBV = 0;
        int maxAttempts = 200;
        
        while (currentBV < targetBV && maxAttempts > 0) {
            megamek.common.loaders.MekSummary candidate = validMeks.get(megamek.common.compute.Compute.randomInt(validMeks.size()));
            if (candidate.getBV() > 0 && currentBV + candidate.getBV() <= targetBV + (targetBV * 0.15)) {
                try {
                    megamek.common.units.Entity ent = new megamek.common.loaders.MekFileParser(candidate.getSourceFile(), candidate.getEntryName()).getEntity();
                    if (ent != null) {
                        force.add(ent);
                        currentBV += candidate.getBV();
                        if (currentBV >= targetBV - (targetBV * 0.15)) {
                            break;
                        }
                    }
                } catch (Exception e) {
                    System.out.println("Exception parsing Mek: " + e.getMessage());
                }
            }
            maxAttempts--;
        }
        
        return force;
    }
}
