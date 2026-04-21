import math

# =========================
# TEST CASES
# =========================
test_cases = [
    ("Normal", -0.74),
    ("Early", -4.03),
    ("Maintenance", -4.70),
    ("Failure", -5.0)
]

# =========================
# SCORING FUNCTIONS
# =========================
def linear(z):
    return min((abs(z) / 5) * 100, 100)

def exponential(z):
    return min(100 * (1 - math.exp(-abs(z) / 2)), 100)

def log_scale(z):
    return min(math.log1p(abs(z)) * 40, 100)

def piecewise(z):
    z = abs(z)
    if z < 1:
        return z * 20
    elif z < 3:
        return 20 + (z - 1) * 25
    else:
        return 70 + min((z - 3) * 15, 30)

scoring_methods = {
    "linear": linear,
    "exponential": exponential,
    "log": log_scale,
    "piecewise": piecewise
}

# =========================
# HEALTH MAPPINGS
# =========================
def mapping_v1(score):
    if score < 30:
        return "Normal"
    elif score < 55:
        return "Early"
    elif score < 80:
        return "Maintenance"
    else:
        return "Failure"

def mapping_v2(score):
    if score < 35:
        return "Normal"
    elif score < 65:
        return "Early"
    elif score < 85:
        return "Maintenance"
    else:
        return "Failure"

def mapping_v3(score):
    if score < 25:
        return "Normal"
    elif score < 50:
        return "Early"
    elif score < 75:
        return "Maintenance"
    else:
        return "Failure"

mappings = {
    "mapping_v1": mapping_v1,
    "mapping_v2": mapping_v2,
    "mapping_v3": mapping_v3
}

# =========================
# RUN TESTS
# =========================
print("\n🔥 SCORING EXPERIMENTS\n")

for s_name, scorer in scoring_methods.items():
    print(f"\n=== SCORING: {s_name} ===")

    for m_name, mapper in mappings.items():
        print(f"\n--- {m_name} ---")

        for name, z in test_cases:
            score = scorer(z)
            health = mapper(score)

            print(f"{name:12} | z={z:.2f} | score={score:.2f} | → {health}")

        print()