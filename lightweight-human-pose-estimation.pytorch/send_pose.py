import socket
import json
import time

pose_data = {
    "0": [467, 290], "1": [450, 377], "2": [308, 373], "3": [250, 250], "4": [300, 350],
    "5": [586, 375], "6": [320, 400], "7": [250, 300], "8": [200, 350], "9": [250, 300],
    "10": [350, 400], "11": [250, 300], "12": [350, 400], "13": [400, 450],
    "14": [435, 247], "15": [506, 261], "16": [394, 238], "17": [542, 261]
}

# ✅ Convert to JSON format that Unity expects
pose_json = {
    "items": [
        {"index": str(k), "position": [v[0] / 100, v[1] / 100]}  # scaled to Unity units
        for k, v in pose_data.items()
    ]
}

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("127.0.0.1", 9999))
sock.sendall((json.dumps(pose_json) + "\n").encode())
time.sleep(1)
sock.close()
