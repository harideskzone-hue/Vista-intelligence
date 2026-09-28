with open("plot_telemetry.py", "r") as f:
    content = f.read()

content = content.replace(
    "df = pd.read_csv(csv_path)",
    "df = pd.read_csv(csv_path)\n    for col in df.columns:\n        if df[col].dtype == 'object':\n            df[col] = df[col].astype(str).str.strip().astype(float)"
)

with open("plot_telemetry.py", "w") as f:
    f.write(content)
