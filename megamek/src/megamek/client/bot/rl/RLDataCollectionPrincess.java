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

import megamek.client.bot.princess.Princess;
import megamek.common.units.Entity;
import megamek.common.moves.MovePath;
import megamek.logging.MMLogger;

public class RLDataCollectionPrincess extends Princess {
    private static final MMLogger logger = MMLogger.create(RLDataCollectionPrincess.class);
    
    private RLDataPipeline dataPipeline;

    public RLDataCollectionPrincess(String playerName, String host, int port, int listenPort) throws Exception {
        super(playerName, host, port);
        
        // Disable enhanced targeting and set behavior specifics if required here
        megamek.client.bot.princess.BehaviorSettings bs = getBehaviorSettings().getCopy();
        bs.setAutoFlee(false);
        bs.setForcedWithdrawal(false);
        setBehaviorSettings(bs);
        
        this.dataPipeline = new RLDataPipeline(this);
        this.dataPipeline.listenForPython(listenPort);
    }
    
    @Override
    public void die() {
        if (dataPipeline != null) {
            dataPipeline.close();
        }
        super.die();
    }
    
    @Override
    protected MovePath continueMovementFor(final Entity entity) {
        MovePath chosenPath = null;
        try {
            // Princess natively calculates paths, ranks them, and returns the chosen MovePath
            chosenPath = super.continueMovementFor(entity);
            
            // Validate the path against off-board hexes to prevent headless Server NullPointerExceptions
            if (chosenPath != null) {
                java.util.ListIterator<megamek.common.moves.MoveStep> it = chosenPath.getSteps();
                while (it.hasNext()) {
                    megamek.common.moves.MoveStep step = it.next();
                    megamek.common.Hex hex = getGame().getBoard(chosenPath.getFinalBoardId()).getHex(step.getPosition());
                    if (hex == null) {
                        logger.error("RLDataCollectionPrincess intercepted a MovePath with an off-board hex. Scrubbing path to prevent server crash.");
                        MovePath safePath = new MovePath(getGame(), entity);
                        chosenPath = safePath;
                        break;
                    }
                }
            }
        } catch (Exception e) {
            logger.error(e, "RLDataCollectionPrincess encountered an exception during Movement calculation. Returning empty path to prevent server hang.");
            MovePath safePath = new MovePath(getGame(), entity);
            return safePath;
        }
        
        // If a valid path is chosen, stream it down for supervised learning extraction
        if (chosenPath != null && dataPipeline.isConnected()) {
            dataPipeline.sendBehavioralCloningTrajectory(entity, chosenPath);
        }
        
        return chosenPath;
    }
    
    @Override
    protected megamek.client.bot.PhysicalOption calculatePhysicalTurn() {
        try {
            return super.calculatePhysicalTurn();
        } catch (Exception e) {
            logger.error(e, "RLDataCollectionPrincess encountered an exception during Physical calculation. Returning null to prevent server hang.");
            // Returning null here safely clears the physical turn and allows BotClient to send an empty attack vector, preventing deadlocks!
            return null;
        }
    }

    @Override
    public void sendAttackData(int aen, java.util.Vector<megamek.common.actions.EntityAction> attacks) {
        if (getGame().getPhase() == megamek.common.enums.GamePhase.FIRING) {
             Entity shooter = getGame().getEntity(aen);
             if (shooter != null && dataPipeline.isConnected()) {
                 int chosenTwist = 0;
                 for (megamek.common.actions.EntityAction ea : attacks) {
                     if (ea instanceof megamek.common.actions.TorsoTwistAction) {
                         megamek.common.actions.TorsoTwistAction tta = (megamek.common.actions.TorsoTwistAction) ea;
                         int diff = tta.getFacing() - shooter.getFacing();
                         if (diff > 3) diff -= 6;
                         if (diff < -3) diff += 6;
                         chosenTwist = diff;
                     }
                 }
                 dataPipeline.sendWeaponBehavioralCloningTrajectory(shooter, attacks, chosenTwist);
             }
        } else if (getGame().getPhase() == megamek.common.enums.GamePhase.PHYSICAL) {
             Entity shooter = getGame().getEntity(aen);
             if (shooter != null && dataPipeline.isConnected()) {
                 dataPipeline.sendPhysicalBehavioralCloningTrajectory(shooter, attacks);
             }
        }
        super.sendAttackData(aen, attacks);
    }
}
