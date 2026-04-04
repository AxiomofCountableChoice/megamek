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
                
    def reset(self, device=None):
        """
        Wait for the next match state payload and return the initial state.
        If device is provided (e.g. 'cuda:0' or a torch.device object), moves the parsed HeteroData graph and its tensors to this device.
        """
        if not self._connected:
            self.connect()
            
        print("Awaiting initial state payload...")
        payload = self._receive_payload()
        if payload is None:
            raise ConnectionError("Server disconnected during reset.")
            
        if payload.get("context") == "TOPOLOGY":
            print(f"Received TOPOLOGY payload. Parsing Board shape ({payload.get('width')}x{payload.get('height')})...")
            if HAS_PYG:
                nodes = payload.get("hex_nodes", [])
                edges = payload.get("hex_edges", [])
                
                if nodes:
                    self.static_hex_features = torch.tensor(nodes, dtype=torch.float32)
                else:
                    self.static_hex_features = torch.empty((0, 5), dtype=torch.float32)
                    
                if edges:
                    edge_array = np.array(edges, dtype=np.int64).T
                    self.static_hex_adjacency_edges = torch.tensor(edge_array, dtype=torch.long)
                else:
                    self.static_hex_adjacency_edges = torch.empty((2, 0), dtype=torch.long)
                    
            print("Topology cached. Awaiting actual initial state...")
            payload = self._receive_payload()
            if payload is None:
                 raise ConnectionError("Server disconnected while waiting for STATE.")
                    
        data_graph, mask = self._parse_to_heterodata(payload)
        if device is not None and HAS_PYG:
            data_graph = data_graph.to(device)
            self.static_hex_features = self.static_hex_features.to(device)
            self.static_hex_adjacency_edges = self.static_hex_adjacency_edges.to(device)
            
        return data_graph, mask

    def step(self, action_dict, device=None):
        """
        Submit an action dict (e.g. {"selected_path_index": 0}) and await the next state.
        Optional device parameter dynamically shifts the next payload onto GPU natively.
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
        if device is not None and HAS_PYG and state is not None:
            state = state.to(device)
            
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
        
        # Dummy embedding for phase
        data.global_context = torch.tensor([turn, len(phase_str)], dtype=torch.float32)
        
        # 2. Dynamic Entity Nodes ($V_U$)
        raw_entities = raw_state.get("entities", [])
        if raw_entities:
            data['mech'].x = torch.tensor(raw_entities, dtype=torch.float32)
        else:
            # 12 covariates: isMine, x, y, facing, heat, maxHeat, armor, structure, tmm, speedMode, gunnery, piloting
            data['mech'].x = torch.empty((0, 12), dtype=torch.float32)
        
        # 3. Static Hex/Topology Cache ($V_H$ and $E_{adj}$)
        if self.static_hex_features is not None:
            data['hex'].x = self.static_hex_features
        
        if self.static_hex_adjacency_edges is not None:
            data['hex', 'adjacent_to', 'hex'].edge_index = self.static_hex_adjacency_edges
        
        # 4. Ephemeral Edges ($E_{occ}$)
        width = raw_state.get("global_state", {}).get("board_width", 0)
        mech_indices = []
        hex_indices = []
        
        for i, ent in enumerate(raw_entities):
            ex, ey = ent[1], ent[2]
            if ex >= 0 and ey >= 0 and width > 0:
                hex_idx = int(ey * width + ex)
                mech_indices.append(i)
                hex_indices.append(hex_idx)
                
        if mech_indices:
            data['mech', 'occupies', 'hex'].edge_index = torch.tensor([mech_indices, hex_indices], dtype=torch.long)
        else:
            data['mech', 'occupies', 'hex'].edge_index = torch.empty((2, 0), dtype=torch.long)

        return data, mask

    def close(self):
        if self.sock:
            self.sock.close()
            self._connected = False
            print("Environment socket closed.")
