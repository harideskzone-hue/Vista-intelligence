with open("run_phase7f_endurance.py", "r") as f:
    content = f.read()

content = content.replace('env["INFERENCE_BATCH_SIZE"] = "4"', 'env["INFERENCE_BATCH_SIZE"] = "1"')
content = content.replace('mode="legacy"', 'mode="mps_batch"')

with open("run_phase7f_endurance.py", "w") as f:
    f.write(content)
print("Patched batch size and mode")
