#!/bin/bash
# Helper script to launch MegaMek in autogen mode connected to the Interactive Debugger

# Navigate to the root gradle directory
cd $(dirname $0)/../..

export RL_DEBUG_TIMEOUT=900000
export RL_SERVER_PORT=4001

echo "Starting Python Interactive Debugger..."
cd python-rl
nohup ../.venv/bin/python validations/interactive_debugger.py > debugger_output.log 2>&1 &
FLASK_PID=$!
cd ..

echo "Starting MegaMek Java Client in Offline Trainer mode..."
cleanup() {
    echo "Stopping MegaMek Client and Python Debugger..."
    kill $FLASK_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

./gradlew run --args="-rlexport -randomMap -randomBV 5000 -autogen"
