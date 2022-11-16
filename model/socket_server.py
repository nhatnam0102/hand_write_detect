import socket
from model import model

final_model = model.Model.load_model()

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('127.0.0.1', 5007))
    s.listen(1)
    while True:
        conn, adr = s.accept()
        with conn:
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                print('data: {}, address: {}'.format(data, adr))
                result = model.Model.recognize(data.decode("utf-8"), final_model)
                send_data = bytes(result, encoding='raw_unicode_escape')
                conn.sendall(b'Receive:' + send_data)
