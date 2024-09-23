print("beginning")

try:
    import porespy as py
    print("porespy imported successfully")
except Exception as e:
    print(f"Failed to import porespy: {e}")

print("end")
