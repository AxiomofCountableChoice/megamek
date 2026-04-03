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

        File gameFile = resolver.getSaveGameFile();
        if (null != gameFile) {
            server.loadGame(gameFile);
        }

        logger.info("MegaMek Server started on port " + resolver.port);
        
        megamek.common.Game game = (megamek.common.Game) server.getGame();
        if (game == null) {
            logger.error("No scenario/game loaded! Please pass a save file or MUL file.");
            return;
        }

        int pythonSocketPort = resolver.port + 10000;
        java.util.List<megamek.client.bot.BotClient> connectedBots = new java.util.ArrayList<>();

        // Loop through the players loaded by the scenario
        for (megamek.common.Player player : game.getPlayersList()) {
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

        // Ready up all bots to trigger game start!
        for (megamek.client.bot.BotClient bot : connectedBots) {
            bot.sendDone(true);
        }
        
        logger.info("All orchestrator bots ready! The match has begun entirely autonomously.");
    }
}
