package megamek.client.bot.rl;

import java.io.FileOutputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.zip.GZIPOutputStream;

import com.fasterxml.jackson.databind.ObjectMapper;

import megamek.common.event.GameEvent;
import megamek.common.event.GameListenerAdapter;
import megamek.common.event.GameTurnChangeEvent;
import megamek.logging.MMLogger;

public class TrajectoryLogger extends GameListenerAdapter {
    private static final MMLogger logger = MMLogger.create(TrajectoryLogger.class);

    private ObjectMapper mapper;
    private GZIPOutputStream gzipOut;
    private Map<String, Object> lastState;
    private Object lastAction;
    private double lastReward;

    public TrajectoryLogger(String outputFile) {
        this.mapper = new ObjectMapper(); // JSON Lines
        try {
            this.gzipOut = new GZIPOutputStream(new FileOutputStream(outputFile));
            logger.info("Trajectory Logger initialized: " + outputFile);
        } catch (IOException e) {
            logger.error(e, "Failed to initialize Trajectory Logger");
        }
    }

    @Override
    public void gameTurnChange(GameTurnChangeEvent e) {
        // Here we hook into the turn changes to capture (s, a, r, s') sequence.
        logTrajectoryStep();
    }

    private synchronized void logTrajectoryStep() {
        if (gzipOut == null) return;
        
        try {
            Map<String, Object> step = new HashMap<>();
            step.put("state", lastState);
            step.put("action", lastAction);
            step.put("reward", lastReward);
            // step.put("next_state", currentState); // TODO: populate full state
            
            byte[] bytes = mapper.writeValueAsBytes(step);
            gzipOut.write(bytes);
            gzipOut.write('\n');
            gzipOut.flush();
        } catch (IOException e) {
            logger.error(e, "Error writing trajectory tuple");
        }
    }

    public void close() {
        if (gzipOut != null) {
            try {
                gzipOut.close();
            } catch (IOException e) {
                logger.error(e, "Error closing trajectory logger");
            }
        }
    }
}
