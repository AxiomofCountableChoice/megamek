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
        boolean autoGen = false;
        boolean randomMap = false;
        String p1Meks = "";
        String p2Meks = "";
        
        java.util.List<String> cleanArgs = new java.util.ArrayList<>();
        for (int i = 0; i < args.length; i++) {
            if (args[i].equalsIgnoreCase("-autogen")) {
                autoGen = true;
            } else if (args[i].equalsIgnoreCase("-randomMap")) {
                randomMap = true;
            } else if (args[i].equalsIgnoreCase("-p1meks")) {
                p1Meks = args[++i];
            } else if (args[i].equalsIgnoreCase("-p2meks")) {
                p2Meks = args[++i];
            } else {
                cleanArgs.add(args[i]);
            }
        }

        ClientServerCommandLineParser parser = new ClientServerCommandLineParser(cleanArgs.toArray(new String[0]),
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

        // Connect the Princess bots to the local server
        Princess p1 = null;
        Princess p2 = null;
        try {
            p1 = new RLDataCollectionPrincess("Princess_Alpha", "localhost", resolver.port, 8001);
            p2 = new RLDataCollectionPrincess("Princess_Beta", "localhost", resolver.port, 8002);
            
            p1.connect();
            p2.connect();
        } catch (Exception e) {
            logger.error(e, "Error launching Princess offline bots");
        }
        
        // Wait and configure autogen properties
        if (autoGen) {
            try {
                Thread.sleep(1500); // Give Princess instances time to establish Lobby connection
                
                megamek.common.Game game = (megamek.common.Game) server.getGame();
                
                // Map setup
                if (randomMap) {
                    megamek.common.MapSettings ms = megamek.common.MapSettings.getInstance();
                    ms.setBoardSize(16, 16);
                    megamek.common.Board board = megamek.common.util.BoardUtilities.generateRandom(ms);
                    game.setBoard(board);
                }
                
                // Get Players
                megamek.common.Player player1 = null;
                megamek.common.Player player2 = null;
                for (megamek.common.Player p : game.getPlayersList()) {
                    if (p.getName().equals("Princess_Alpha")) player1 = p;
                    if (p.getName().equals("Princess_Beta")) player2 = p;
                }
                
                if (player1 != null && player2 != null) {
                    // Assign Forces
                    if (!p1Meks.isEmpty()) {
                        for (String mekFile : p1Meks.split(",")) {
                            // Extract just the Mek.
                            megamek.common.Entity ent = new megamek.common.MekFileParser(new File("data/mekfiles/meks/" + mekFile.trim().replace("\"", ""))).getEntity();
                            if (ent != null) {
                                ent.setOwner(player1);
                                game.addEntity(ent);
                            }
                        }
                    }
                    if (!p2Meks.isEmpty()) {
                        for (String mekFile : p2Meks.split(",")) {
                            megamek.common.Entity ent = new megamek.common.MekFileParser(new File("data/mekfiles/meks/" + mekFile.trim().replace("\"", ""))).getEntity();
                            if (ent != null) {
                                ent.setOwner(player2);
                                game.addEntity(ent);
                            }
                        }
                    }
                    
                    // Automatically trigger deploy actions by setting ready
                    // Send PlayerCommand to change "done" state
                    if (p1 != null) p1.sendDone(true);
                    if (p2 != null) p2.sendDone(true);
                    
                    // In QuickGameRunner, that handles automatic transition to deployment and then game
                }
                
            } catch (Exception ex) {
                logger.error(ex, "Failed to completely configure autogen game");
            }
        }
    }
}
