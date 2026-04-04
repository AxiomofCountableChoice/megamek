import time
from env import MegaMekEnvironment

def main():
    env = MegaMekEnvironment(host='localhost', port=12346)
    
    print("Initializing environment...")
    state, mask = env.reset()
    
    print("\n--- Initial State (HeteroData) parsed successfully! ---")
    if getattr(state, "node_types", None):
        for nt in state.node_types:
            print(f"Node Type '{nt}': {state[nt].x.shape}")
        for et in state.edge_types:
            print(f"Edge Type '{et}': {state[et].edge_index.shape}")
    else:
        print(state)
        
    print("\nStarting Actor Loop...")
    
    try:
        while True:
            # Dummy response logic
            response = {"selected_path_index": 0}
            
            state, mask, done = env.step(response)
            
            if done:
                print("Environment episode finished.")
                break
                
            if state is not None and getattr(state, "node_types", None):
                print(f"Update -> Mechs: {state['mech'].x.shape[0]} nodes | Active Hex Occupancies: {state['mech', 'occupies', 'hex'].edge_index.shape[1]}")
            
    except KeyboardInterrupt:
        print("\nDummy Actor terminated by user.")
    except Exception as e:
        print(f"Error during runtime: {e}")
    finally:
        env.close()

if __name__ == "__main__":
    main()
