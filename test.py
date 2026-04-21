import requests
import json

url = "http://127.0.0.1:5000/predict"

test_cases = [
    {
        "name": "Normal Operation",
        "data": {
            "cpu_usage": 35,
            "cpu_temp": 55,
            "gpu_temp": 45,
            "power": 25,
            "cpu_freq": 2800,
            "fan1": 50
        }
    },
    {
        "name": "Early Degradation",
        "data": {
            "cpu_usage": 60,
            "cpu_temp": 75,
            "gpu_temp": 60,
            "power": 45,
            "cpu_freq": 3200,
            "fan1": 60
        }
    },
    {
        "name": "Maintenance Required",
        "data": {
            "cpu_usage": 70,
            "cpu_temp": 85,
            "gpu_temp": 70,
            "power": 55,
            "cpu_freq": 3500,
            "fan1": 65
        }
    },
    {
        "name": "Failure Imminent",
        "data": {
            "cpu_usage": 80,
            "cpu_temp": 95,
            "gpu_temp": 80,
            "power": 65,
            "cpu_freq": 3700,
            "fan1": 60
        }
    }
]

print("\n🔥 Running Test Cases...\n")

for i, test in enumerate(test_cases, 1):
    print(f"===== Test {i}: {test['name']} =====")

    try:
        res = requests.post(url, json=test["data"])
        output = res.json()

        print("Input:")
        print(json.dumps(test["data"], indent=2))

        print("\nOutput:")
        print(json.dumps(output, indent=2))

    except Exception as e:
        print("Error:", e)

    print("\n" + "="*40 + "\n")