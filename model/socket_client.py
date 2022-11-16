import socket
import sys

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    path = sys.argv[1]
    send_data = bytes(path, encoding='raw_unicode_escape')
    s.connect(('127.0.0.1', 5007))
    s.sendall(send_data)
    data = s.recv(1024)
    print(repr(data))
