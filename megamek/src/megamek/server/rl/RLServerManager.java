package megamek.server.rl;

import java.io.File;

import megamek.common.commandline.AbstractCommandLineParser;
import megamek.common.commandline.ClientServerCommandLineParser;
import megamek.common.commandline.MegaMekCommandLineFlag;
import megamek.common.preference.PreferenceManager;
import megamek.logging.MMLogger;
import megamek.server.Server;
import megamek.server.totalwarfare.TWGameManager;
import megamek.client.bot.rl.RLBotClient;
import megamek.common.MULParser;
import megamek.common.Board;
import megamek.common.Coords;
import megamek.common.Entity;
import megamek.common.Player;

public class RLServerManager {
    private static final MMLogger logger = MMLogger.create(RLServerManager.class);

    public static void start(String[] args) {
        ClientServerCommandLineParser parser = new ClientServerCommandLineParser(args,
                MegaMekCommandLineFlag.RLSERVER.toString(),
                true, false, false);
        try {
            parser.parse();
        } catch (AbstractCommandLineParser.ParseException e) {
            logger.error("Incorrect arguments:" + e.getMessage() + '\n' + parser.help());
        }

        ClientServerCommandLineParser.Resolver resolver = parser.getResolver(
                null,
                PreferenceManager.getClientPreferences().getLastServerPort(),
                null, null);

        // kick off a RNG check
        megamek.common.Compute.d6();

        Server server;

        try {
            server = new Server(resolver.password, resolver.port, new TWGameManager(), resolver.registerServer,
                    resolver.announceUrl, null, true);
        } catch (Exception ex) {
            logger.error("Error: could not start server at localhost:" + resolver.port, ex);
            return;
        }

        boolean dualMulMode = args.length >= 2;
        File gameFile = resolver.getSaveGameFile();
        if (!dualMulMode && gameFile != null) {
            server.loadGame(gameFile);
        } else if (dualMulMode) {
            String boardPath = "Map Set 2/16x17 BattleTech.board";
            if (args.length >= 3) {
                boardPath = args[2];
            }
            try {
                Board board = new Board(16, 17);
                File bFile = new megamek.common.util.fileUtils.MegaMekFile(megamek.common.Configuration.boardsDir(), boardPath).getFile();
                try (java.io.InputStream is = new java.io.FileInputStream(bFile)) {
                    board.load(is, null, false);
                }
                server.getGame().setBoard(0, board);
            } catch (Exception ex) {
                logger.error("Failed to load map: " + boardPath, ex);
                return;
            }
        }

        logger.info("MegaMek Server started on port " + resolver.port);
        
        megamek.common.Game game = (megamek.common.Game) server.getGame();
        if (game == null) {
            logger.error("No scenario/game loaded! Please pass a save file or MUL file.");
            return;
        }

        int pythonSocketPort = resolver.port + 10000;
        java.util.List<megamek.client.bot.BotClient> connectedBots = new java.util.ArrayList<>();

        if (dualMulMode) {
            logger.info("Orchestrating RLBotClient for: RLAgent");
            megamek.client.bot.BotClient rlBot = new RLBotClient("RLAgent", "localhost", resolver.port, pythonSocketPort);
            ((RLBotClient)rlBot).connect();
            connectedBots.add(rlBot);

            logger.info("Orchestrating Princess for: Princess");
            megamek.client.bot.princess.BehaviorSettings bs = new megamek.client.bot.princess.BehaviorSettings();
            megamek.client.bot.BotClient princess = megamek.client.bot.princess.Princess.createPrincess("Princess", "localhost", resolver.port, bs);
            princess.connect();
            connectedBots.add(princess);
            
            int retries = 0;
            while (server.getGame().getPlayersList().size() < 2 && retries++ < 100) {
                try { Thread.sleep(50); } catch (Exception e) {}
            }

            for (megamek.client.bot.BotClient bot : connectedBots) {
                retries = 0;
                while (bot.getLocalPlayer() == null && retries++ < 100) {
                    try { Thread.sleep(50); } catch (Exception e) {}
                }
                bot.sendPlayerInfo();
            }

            String[] mulPaths = {args[0], args[1]};
            int teamIter = 1;
            for (Player p : server.getGame().getPlayersList()) {
                p.setTeam(teamIter++);
                int index = p.getName().contains("RLAgent") ? 0 : 1;
                try {
                    MULParser parserMul = new MULParser(new File(mulPaths[index]), null);
                    int xStart = (index == 0) ? 2 : 14;
                    int yStart = 5;
                    for (Entity e : parserMul.getEntities()) {
                        e.setOwner(p);
                        e.setPosition(new Coords(xStart, yStart++));
                        e.setDeployed(true);
                        game.addEntity(e);
                    }
                } catch (Exception ex) {
                    logger.error("Failed to parse MUL for player: " + p.getName(), ex);
                }
            }
            
            // Broadcast the newly added entities to the clients.
            // Since they connected before the entities were parsed, they have empty unit lists!
            try {
                megamek.server.totalwarfare.TWGameManager twm = (megamek.server.totalwarfare.TWGameManager) server.getGameManager();
                twm.send(twm.createFullEntitiesPacket());
            } catch (Exception ex) {
                logger.error("Failed to broadcast entities to clients", ex);
            }

        } else {
            // Loop through the players loaded by the scenario
            for (Player player : game.getPlayersList()) {
                if (!player.isBot()) continue;
                
                megamek.client.bot.BotClient botClient;
                if (player.getName().toLowerCase().contains("rlagent")) {
                    logger.info("Orchestrating RLBotClient for: " + player.getName());
                    botClient = new RLBotClient(player.getName(), "localhost", resolver.port, pythonSocketPort);
                    ((RLBotClient)botClient).connect();
                } else {
                    logger.info("Orchestrating Princess for: " + player.getName());
                    megamek.client.bot.princess.BehaviorSettings bs = game.getBotSettings().get(player.getName());
                    botClient = megamek.client.bot.princess.Princess.createPrincess(player.getName(), "localhost", resolver.port, bs);
                    botClient.connect();
                }
                
                connectedBots.add(botClient);
                
                // Wait for player synchronization
                int retries = 0;
                while (botClient.getLocalPlayer() == null && retries++ < 100) {
                    try { Thread.sleep(50); } catch (Exception e) {}
                }
                
                // Send player info
                botClient.sendPlayerInfo();
            }
        }

        // Ready up all bots to trigger game start!
        for (megamek.client.bot.BotClient bot : connectedBots) {
            bot.sendDone(true);
        }
        
        logger.info("All orchestrator bots ready! The match has begun entirely autonomously.");
    }
}
