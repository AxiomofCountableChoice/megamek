import socket
import struct
import msgpack
import time
import torch
import numpy as np

# We conditionally import torch_geometric so it doesn't crash if not installed yet.
try:
    from torch_geometric.data import HeteroData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("Warning: torch_geometric not found. Please install PyTorch Geometric for full functionality.")

class MegaMekEnvironment:
    """
    A Gym-like environment wrapper that manages headless MegaMek via TCP.
    It synchronously maintains a connection and parses MessagePack state payloads
    into PyTorch Geometric HeteroData objects.
    """
    def __init__(self, host='localhost', port=12346):
        self.host = host
        self.port = port
        self.sock = None
        self._connected = False
        
        # Static Topology Cache
        # Populated once per game match to prevent redundant IPC overhead
        self.static_hex_features = None 
        self.static_hex_adjacency_edges = None

    def connect(self):
        print(f"Connecting to MegaMek RLServer at {self.host}:{self.port}...")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                self.sock.connect((self.host, self.port))
                self._connected = True
                print("Connected successfully!")
                break
            except ConnectionRefusedError:
                print("Waiting for MegaMek server to start...")
                time.sleep(2)
                
    def reset(self):
        """
        Wait for the next match state payload and return the initial state.
        """
        if not self._connected:
            self.connect()
            
        print("Awaiting initial state payload...")
        payload = self._receive_payload()
        if payload is None:
            raise ConnectionError("Server disconnected during reset.")
            
        return self._parse_to_heterodata(payload)

    def step(self, action_dict):
        """
        Submit an action dict (e.g. {"selected_path_index": 0}) and await the next state.
        """
        res_bytes = msgpack.packb(action_dict, use_bin_type=True)
        self.sock.sendall(struct.pack('>I', len(res_bytes)))
        self.sock.sendall(res_bytes)
        
        # Await next state
        payload = self._receive_payload()
        if payload is None:
            # Done True
            return None, None, True 
            
        state, mask = self._parse_to_heterodata(payload)
        # Using dummy reward/done for now
        return state, mask, False

    def _receive_payload(self):
        # Read 4-byte length prefix
        length_buf = self.sock.recv(4)
        if not length_buf:
            return None
        
        payload_len = struct.unpack('>I', length_buf)[0]
        
        # Read exact payload length
        data = b''
        while len(data) < payload_len:
            packet = self.sock.recv(payload_len - len(data))
            if not packet:
                return None
            data += packet
            
        return msgpack.unpackb(data, raw=False)

    def _parse_to_heterodata(self, payload):
        """
        Converts the dynamic dictionary state into a PyTorch Geometric Graph.
        Utilizes Delta Caching for static terrain geometries.
        """
        context = payload.get("context", "UNKNOWN")
        raw_state = payload.get("state", {})
        mask = payload.get("mask", {})
        
        if not HAS_PYG:
            # Fallback for compilation testing before library installs
            return {"raw_state": raw_state, "context": context}, mask
            
        data = HeteroData()
        
        # 1. Global Phase Features ($p \rightarrow z_{context}$)
        phase_str = raw_state.get('phase_main', "UNKNOWN")
        turn = raw_state.get('turn_number', 0)
        
        # Dummy embedding for phase (Usually an MLP projection in encoder)
        data.global_context = torch.tensor([turn, len(phase_str)], dtype=torch.float32)
        
        # 2. Dynamic Entity Nodes ($V_U$)
        # In a real payload, we'd iterate over raw_state["entities"]
        # For scaffolding, we mock a single Mech node:
        mech_features = [
            [0.0, 1.0, 0.0]  # e.g., [is_friendly, heat_ratio, armor_ratio]
        ]
        data['mech'].x = torch.tensor(mech_features, dtype=torch.float32)
        
        # 3. Static Hex/Topology Cache ($V_H$ and $E_{adj}$)
        if self.static_hex_features is None:
            # Placeholder: Initialize static cache from Java's initial boot sequence
            self.static_hex_features = torch.tensor([
                [0.0, 0.0, 0.0],  # Hex 0: [elevation, woods, is_objective]
                [1.0, 0.0, 0.0]   # Hex 1: ...
            ], dtype=torch.float32)
            
            # Edges between adjacent hexes
            self.static_hex_adjacency_edges = torch.tensor([
                [0, 1],
                [1, 0]
            ], dtype=torch.long)
            
        data['hex'].x = self.static_hex_features
        data['hex', 'adjacent_to', 'hex'].edge_index = self.static_hex_adjacency_edges
        
        # 4. Ephemeral Edges ($E_{occ}$)
        # Where are the mechs standing right now?
        # Placeholder: Mech 0 occupies Hex 1
        data['mech', 'occupies', 'hex'].edge_index = torch.tensor([
            [0],
            [1]
        ], dtype=torch.long)

        # In production this parses real MessagePack arrays dynamically
        
        return data, mask

    def close(self):
        if self.sock:
            self.sock.close()
            self._connected = False
            print("Environment socket closed.")
