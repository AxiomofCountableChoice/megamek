import socket
import struct
import msgpack
import time

class MegaMekNetwork:
    """
    Manages TCP connectivity and MessagePack data serialization 
    for the MegaMek headless RLServer.
    """
    def __init__(self, host='localhost', port=12346, logger=None, max_attempts=50):
        self.host = host
        self.port = port
        self.logger = logger
        self.max_attempts = max_attempts
        self.sock = None
        self.connected = False

    def connect(self):
        if self.connected:
            return

        if self.logger:
            self.logger.info(f"Connecting to MegaMek RLServer at {self.host}:{self.port}...")
        else:
            print(f"Connecting to MegaMek RLServer at {self.host}:{self.port}...")
            
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        attempt = 0
        while attempt < self.max_attempts:
            try:
                self.sock.connect((self.host, self.port))
                self.connected = True
                if self.logger:
                    self.logger.info("Connected successfully!")
                else:
                    print("Connected successfully!")
                break
            except ConnectionRefusedError:
                if self.logger:
                    self.logger.info("Waiting for MegaMek server to start...")
                else:
                    print("Waiting for MegaMek server to start...")
                time.sleep(2)
            attempt += 1
            
        if attempt == self.max_attempts:
            raise ConnectionError("Failed to connect to MegaMek server.")

    def send_action(self, action_dict):
        """Serializes and sends an action dictionary to the server."""
        if not self.connected:
            raise ConnectionError("Not connected to server.")
        res_bytes = msgpack.packb(action_dict, use_bin_type=True)
        self.sock.sendall(struct.pack('>I', len(res_bytes)))
        self.sock.sendall(res_bytes)

    def receive_payload(self):
        """Blocks and waits to receive a complete MessagePack payload from the server."""
        if not self.connected:
            raise ConnectionError("Not connected to server.")
            
        # Read 4-byte length prefix
        length_buf = b''
        while len(length_buf) < 4:
            packet = self.sock.recv(4 - len(length_buf))
            if not packet:
                return None
            length_buf += packet
        
        payload_len = struct.unpack('>I', length_buf)[0]
        
        # Read exact payload length
        data = b''
        while len(data) < payload_len:
            packet = self.sock.recv(payload_len - len(data))
            if not packet:
                return None
            data += packet
            
        payload = msgpack.unpackb(data, raw=False)
        return payload

    def close(self):
        if self.sock:
            self.sock.close()
            self.connected = False
            if self.logger:
                self.logger.info("Environment socket closed.")
            else:
                print("Environment socket closed.")
