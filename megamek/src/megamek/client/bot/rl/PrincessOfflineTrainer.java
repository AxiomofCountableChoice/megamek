package megamek.client.bot.rl;

import java.io.File;

import megamek.common.commandline.AbstractCommandLineParser;
import megamek.common.commandline.ClientServerCommandLineParser;
import megamek.common.commandline.MegaMekCommandLineFlag;
import megamek.common.preference.PreferenceManager;
import megamek.logging.MMLogger;
import megamek.server.Server;
import megamek.server.totalwarfare.TWGameManager;
import megamek.client.bot.princess.Princess;

public class PrincessOfflineTrainer {
    private static final MMLogger logger = MMLogger.create(PrincessOfflineTrainer.class);

    public static void start(String[] args) {
        ClientServerCommandLineParser parser = new ClientServerCommandLineParser(args,
                MegaMekCommandLineFlag.RLEXPORT.toString(),
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

        logger.info("MegaMek Offline Princess Trainer started on port " + resolver.port);

        TrajectoryLogger tLogger = new TrajectoryLogger("princess_offline_trajectories.jsonl.gz");
        
        // In a complete implementation we would attach the logger:
        // server.getGame().addGameListener(tLogger);

        // Connect the Princess bots to the local server
        try {
            Princess p1 = new Princess("Princess_Alpha", "localhost", resolver.port);
            Princess p2 = new Princess("Princess_Beta", "localhost", resolver.port);
        } catch (Exception e) {
            logger.error(e, "Error launching Princess offline bots");
        }
    }
}
