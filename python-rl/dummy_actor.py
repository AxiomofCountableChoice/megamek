import socket
import struct
import msgpack
import time
import sys

def main():
    host = 'localhost'
    port = 12346  # 2346 (MegaMek default) + 10000

    print(f"Connecting to MegaMek RLServer at {host}:{port}...")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((host, port))
        print("Connected successfully!")
    except ConnectionRefusedError:
        print("Connection refused. Make sure MegaMek is running with the -rlserver flag.")
        sys.exit(1)

    try:
        while True:
            # Read length prefixed (4 bytes big-endian)
            length_buf = sock.recv(4)
            if not length_buf:
                print("Server disconnected.")
                break
            
            payload_len = struct.unpack('>I', length_buf)[0]
            
            # Read exact payload length bytes
            data = b''
            while len(data) < payload_len:
                packet = sock.recv(payload_len - len(data))
                if not packet:
                    print("Connection dropped during payload receive.")
                    return
                data += packet
                
            payload = msgpack.unpackb(data, raw=False)
            
            context = payload.get("context")
            state = payload.get("state", {})
            mask = payload.get("mask", {})
            
            print(f"--- Received Action Context: {context} ---")
            print(f"State Phase: {state.get('phase_main')} (Turn: {state.get('turn_number')})")
            
            # Form dummy response
            response = {}
            
            # Send MessagePack response
            res_bytes = msgpack.packb(response, use_bin_type=True)
            sock.sendall(struct.pack('>I', len(res_bytes)))
            sock.sendall(res_bytes)
            
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nDummy Actor terminated by user.")
    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        sock.close()
        print("Connection closed.")

if __name__ == "__main__":
    main()
