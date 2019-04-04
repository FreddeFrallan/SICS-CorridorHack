import socket
import threading
import signal
import sys

def handle(c, storage):
    ## read one utf-8 encoded line and some binary data
    ## "name" "space" "number_of_bytes" "newline" "bytes"
    data = b''
    while True:
        d = c.recv(10**6)
        if not d: break   # EOF

        data = data + d
        if not b'\n' in data:
            continue

        i = data.index(b'\n')
        x, data = data[:i].decode('utf-8'), data[i+1:]
        name, num_bytes = x.split(' ')
            
        try:
            num_bytes = int(num_bytes)
            while len(data) < num_bytes:
                d = c.recv(num_bytes - len(data))
                if not d: break # too little data
                data = data + d

            if len(data) == num_bytes:
                storage[name] = data
                c.send(b'thanks\n')
                data = data[num_bytes:]
            else:
                break
        except:
            break

    print('closing connection')
    c.close()

def listener(host, port, storage):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port)) 
    s.listen(5) # number of pending connections before refusing = 5

    while True:
        # establish connection with client
        print('listening')
        try:
            c, addr = s.accept()
            print(f'got connection from {addr}')
            threading.Thread(target=handle, args=(c, storage), daemon=True).start()
        except KeyboardInterrupt:
            print(f'goodbye {addr}')
            s.close()
            break

class Sender:
    def __init__(self, host, port):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((host, port))
        
    def send(self, name, data):
        header = bytearray(f'{name} {len(data)}\n', 'utf-8')
        self.s.send(header + data)
        assert (self.s.recv(1000) == b'thanks\n')

    def __enter__(self):
        return self
    def __exit__(self, type, value, tb):
        self.s.close()

def start_listener(host, port, storage):
    threading.Thread(target=listener, args=(host, port, storage)).start()

if __name__ == '__main__':
    d = {}
    start_listener('localhost', port=int(sys.argv[1]), storage=d)
    while True:
        import time
        time.sleep(1)
        for k in d:
            print(f'{k}:{len(d[k])} ', end='')
        print()
