STEPS:
1. Train regression model
2. Compute residual
3. Group by load
4. Compute μ, σ
5. Compute z-score


Classify: 

Class	          z
Normal	         x-x
Early	         1–2
Maintenance	     2–3
Failure	          >3


running backend:
python app.py

example jsons:
POST http://127.0.0.1:5000/predict

normal:
{
  "cpu_usage": 35,
  "cpu_temp": 55,
  "gpu_temp": 45,
  "power": 25,
  "cpu_freq": 2800,
  "fan1": 50
}

early deg:
{
  "cpu_usage": 60,
  "cpu_temp": 75,
  "gpu_temp": 60,
  "power": 45,
  "cpu_freq": 3200,
  "fan1": 60
}

maintenance:
{
  "cpu_usage": 70,
  "cpu_temp": 85,
  "gpu_temp": 70,
  "power": 55,
  "cpu_freq": 3500,
  "fan1": 65
}



fail:
{
  "cpu_usage": 80,
  "cpu_temp": 95,
  "gpu_temp": 80,
  "power": 65,
  "cpu_freq": 3700,
  "fan1": 60
}