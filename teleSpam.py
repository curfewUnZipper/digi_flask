import requests
import random
import time

URL = "http://127.0.0.1:5000/telemetry"

while True:

    payload = {
        "cpu_usage": random.randint(40, 90),
        "temperature": random.randint(60, 95),
        "power": random.randint(20, 60),
        "frequency": random.randint(3000, 4500),
        "fan_rpm": random.randint(2500, 4500)
    }

    r = requests.post(URL, json=payload)

    print(r.json())

    time.sleep(2)