import requests
import pandas as pd
import numpy as np

URL = "http://127.0.0.1:5000/predict_series"

# =========================
# GENERATE DUMMY DATA
# =========================
rows = 100

data = []

for i in range(rows):
    data.append({
        "fan1": 3000 - i*2,
        "fan2": 2900 - i*2,
        "cpu_temp": 50 + i*0.05,
        "gpu_temp": 55 + i*0.04,
        "nvidia_temp": 54 + i*0.03,
        "cpu_usage": 30 + (i % 50),
        "current": 1.5 + i*0.002,
        "power": 50 + i*0.1
    })

# =========================
# CALL API
# =========================
response = requests.post(URL, json=data)

result = response.json()

# =========================
# PRINT RESULT
# =========================
rul_series = result["rul_series"]

print("\nLast 5 RUL values:")
for r in rul_series[-5:]:
    print(r)