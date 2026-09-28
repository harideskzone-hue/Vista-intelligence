import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_telemetry(csv_path="phase7f_data.csv", output_dir="."):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip().astype(float)
    if df.empty:
        print("Error: CSV is empty.")
        return
        
    # Time axis
    t = df['Min']
    
    # Create subplots
    fig, axes = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    fig.suptitle('Phase 7F Endurance Test Telemetry (Batch=1)', fontsize=16)
    
    # 1. CPU
    axes[0].plot(t, df['CPU'], color='red')
    axes[0].set_ylabel('CPU Usage (%)')
    axes[0].grid(True)
    
    # 2. RSS Memory
    axes[1].plot(t, df['RSS'], color='blue')
    axes[1].set_ylabel('RSS Memory (MB)')
    axes[1].grid(True)
    
    # 3. FPS
    axes[2].plot(t, df['FPS'], color='green')
    axes[2].set_ylabel('Avg FPS/cam')
    axes[2].grid(True)
    
    # 4. Queue Depth
    axes[3].plot(t, df['Queue'], color='purple')
    axes[3].set_ylabel('Queue Depth')
    axes[3].grid(True)
    
    # 5. Reconnect Events
    axes[4].plot(t, df['Reconnects'], color='orange')
    axes[4].set_ylabel('Total Reconnects')
    axes[4].set_xlabel('Time (Minutes)')
    axes[4].grid(True)
    
    plt.tight_layout()
    # Save directly to artifact directory!
    artifact_path = "/Users/hariharans/.gemini/antigravity-ide/brain/f3daadea-3376-439d-a7d4-77d9950cc54b/phase7f_telemetry_plot.png"
    plt.savefig(artifact_path)
    print(f"Plot saved to {artifact_path}")
    
    # Print statistics
    print("\n--- Telemetry Statistics ---")
    print(f"Duration: {t.max():.1f} minutes")
    print(f"Average CPU: {df['CPU'].mean():.1f}%")
    print(f"Peak CPU: {df['CPU'].max():.1f}%")
    if len(df['CPU']) > 5:
        print(f"P95 CPU: {df['CPU'].quantile(0.95):.1f}%")
    print(f"Average RSS: {df['RSS'].mean():.1f} MB")
    print(f"Peak RSS: {df['RSS'].max():.1f} MB")
    print(f"Average FPS: {df['FPS'].mean():.1f}")
    print(f"Max Queue Depth: {df['Queue'].max()}")
    print(f"Total Reconnects: {df['Reconnects'].max()}")

if __name__ == "__main__":
    plot_telemetry()
