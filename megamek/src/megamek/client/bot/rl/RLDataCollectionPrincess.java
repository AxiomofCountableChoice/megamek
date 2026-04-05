package megamek.client.bot.rl;

import megamek.client.bot.princess.Princess;
import megamek.common.Entity;
import megamek.common.moves.MovePath;
import megamek.logging.MMLogger;

public class RLDataCollectionPrincess extends Princess {
    private static final MMLogger logger = MMLogger.create(RLDataCollectionPrincess.class);
    
    private RLDataPipeline dataPipeline;

    public RLDataCollectionPrincess(String playerName, String host, int port, int listenPort) throws Exception {
        super(playerName, host, port);
        
        // Disable enhanced targeting and set behavior specifics if required here
        // ...
        
        this.dataPipeline = new RLDataPipeline(this);
        this.dataPipeline.listenForPython(listenPort);
    }
    
    @Override
    protected MovePath continueMovementFor(final Entity entity) {
        // Princess natively calculates paths, ranks them, and returns the chosen MovePath
        MovePath chosenPath = super.continueMovementFor(entity);
        
        // If a valid path is chosen, stream it down for supervised learning extraction
        if (chosenPath != null && dataPipeline.isConnected()) {
            dataPipeline.sendBehavioralCloningTrajectory(entity, chosenPath);
        }
        
        return chosenPath;
    }
}
