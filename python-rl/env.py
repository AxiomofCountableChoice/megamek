import time
from network import MegaMekNetwork
from rewards import RewardCalculator
from state_parser import StateParser

class MegaMekEnvironment:
    """
    A Gym-like environment wrapper that manages headless MegaMek via TCP.
    It synchronously maintains a connection and parses MessagePack state payloads
    into PyTorch Geometric HeteroData objects.
    """
    def __init__(self, host='localhost', port=12346, device=None, connect_on_init=True, logger=None):
        self.device = device
        self.network = MegaMekNetwork(host=host, port=port, logger=logger)
        self.reward_calculator = RewardCalculator()
        self.state_parser = StateParser()
        self.done = False
        
        # Connect immediately if requested
        if connect_on_init:
            self.connect()

    def connect(self):
        self.network.connect()

    @property
    def sock(self):
        return self.network.sock
        
    @property
    def port(self):
        return self.network.port
        
    @property
    def _connected(self):
        return self.network.connected
        
    @_connected.setter
    def _connected(self, value):
        self.network.connected = value

    # Expose state_parser properties that other scripts might access
    @property
    def static_hex_features(self):
        return self.state_parser.static_hex_features
        
    @static_hex_features.setter
    def static_hex_features(self, value):
        self.state_parser.static_hex_features = value
        
    @property
    def static_hex_adjacency_edges(self):
        return self.state_parser.static_hex_adjacency_edges
        
    @static_hex_adjacency_edges.setter
    def static_hex_adjacency_edges(self, value):
        self.state_parser.static_hex_adjacency_edges = value

    @property
    def board_width(self):
        return self.state_parser.board_width

    @property
    def feature_dims(self):
        return self.state_parser.feature_dims

    def reset(self):
        self.done = False
        if self.network.logger: 
            self.network.logger.info("Awaiting initial state payload...")
        else: 
            print("Awaiting initial state payload...")
            
        while True:
            payload = self.network.receive_payload()
            if payload is None:
                raise ConnectionError("Server closed connection during reset.")
                
            ctx = payload.get("context")
            if ctx == "START_GAME":
                print("Received START_GAME broadcast. Awaiting actual state...")
                continue
            elif ctx == "TOPOLOGY":
                self.topology_payload = payload
                print(f"Received TOPOLOGY payload. Parsing Board shape ({payload.get('width', 0)}x{payload.get('height')})...")
                self.state_parser.handle_topology(payload)
                continue
            else:
                # Actual actionable state
                break

        data_graph, mask = self.state_parser.parse_to_heterodata(payload)
        if self.device is not None:
            data_graph = data_graph.to(self.device)
            if self.state_parser.static_hex_features is not None and self.state_parser.static_hex_features.size(0) > 0:
                self.state_parser.static_hex_features = self.state_parser.static_hex_features.to(self.device)
            if self.state_parser.static_hex_adjacency_edges is not None and self.state_parser.static_hex_adjacency_edges.size(1) > 0:
                self.state_parser.static_hex_adjacency_edges = self.state_parser.static_hex_adjacency_edges.to(self.device)
            
        # Initialize the dense reward base metrics silently
        self.reward_calculator.reset()
        self.reward_calculator.compute_reward(payload)
            
        return data_graph, mask, payload

    def step(self, action_dict):
        """
        Submit an action dict (e.g. {"selected_path_index": 0}) and await the next state.
        Dynamically shifts the next payload onto self.device natively if set.
        """
        self.network.send_action(action_dict)
        
        # Await next state
        payload = self.network.receive_payload()
        if payload is None:
            self.done = True
            return None, None, True, None 
            
        if payload.get("game_over"):
            self.done = True
            return None, None, True, payload
            
        state, mask = self.state_parser.parse_to_heterodata(payload)
        if (self.device is not None) and (state is not None):
            state = state.to(self.device)
            
        # Compute dynamic reward inside the environment abstraction
        reward = self.reward_calculator.compute_reward(payload)
        payload["reward"] = reward
            
        return state, mask, False, payload

    def _receive_payload(self):
        """Backwards compatibility for direct payload reading."""
        return self.network.receive_payload()

    def _parse_to_heterodata(self, payload):
        """Backwards compatibility for parsing."""
        return self.state_parser.parse_to_heterodata(payload)

    def close(self):
        self.network.close()
