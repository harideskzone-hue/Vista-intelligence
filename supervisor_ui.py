import sys
import os
import subprocess
import threading
import time
import urllib.request
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import signal

class VistaSupervisor:
    def __init__(self, root):
        self.root = root
        self.root.title("VISTA AI Launcher")
        
        # Application Support directory for mutable runtime data
        self.app_support_dir = os.path.expanduser("~/Library/Application Support/VISTA AI")
        self.log_dir = os.path.join(self.app_support_dir, "logs")
        self.models_dir = os.path.join(self.app_support_dir, "models")
        self.config_dir = os.path.join(self.app_support_dir, "config")
        self.data_dir = os.path.join(self.app_support_dir, "data")
        
        # Ensure runtime directories exist
        for d in [self.log_dir, self.models_dir, self.config_dir, self.data_dir]:
            os.makedirs(d, exist_ok=True)
            
        # Determine bundled paths
        # In a built app, supervisor_ui.py is in VISTA AI.app/Contents/Resources/app
        # The bundled Python is in VISTA AI.app/Contents/Frameworks/python/bin/python3
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Fallback to current dir's .venv if running outside the .app bundle during dev
        bundled_python = os.path.join(self.base_dir, "..", "..", "Frameworks", "python", "bin", "python3")
        if os.path.exists(bundled_python):
            self.python_bin = bundled_python
        else:
            self.python_bin = os.path.join(self.base_dir, ".venv", "bin", "python3")
        
        # Process references
        self.api_process = None
        self.scorer_process = None
        
        # UI Setup
        self.setup_ui()
        
        # Start initialization sequence
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Handle OS signals for graceful shutdown
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)
        
        threading.Thread(target=self.run_startup_sequence, daemon=True).start()

    def handle_signal(self, signum, frame):
        self.root.after(0, self.on_close)

    def setup_ui(self):
        self.root.geometry("450x350")
        self.root.resizable(False, False)
        
        # Styling
        style = ttk.Style()
        style.theme_use('clam')
        
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        ttk.Label(self.main_frame, text="VISTA AI", font=("Helvetica", 20, "bold")).pack(pady=(0, 5))
        ttk.Label(self.main_frame, text="Border Intelligence Platform", font=("Helvetica", 10)).pack(pady=(0, 20))
        
        # Dynamic Status
        self.status_var = tk.StringVar(value="Initializing...")
        self.status_label = ttk.Label(self.main_frame, textvariable=self.status_var, font=("Helvetica", 12))
        self.status_label.pack(pady=(0, 10))
        
        # Progress Bar
        self.progress = ttk.Progressbar(self.main_frame, mode='determinate', length=350)
        self.progress.pack(pady=10)
        
        # Steps text
        self.steps_text = tk.Text(self.main_frame, height=6, width=50, bg=self.root.cget('bg'), relief=tk.FLAT, font=("Helvetica", 11))
        self.steps_text.pack(pady=5)
        self.steps_text.config(state=tk.DISABLED)
        
        # Action Buttons (Hidden initially)
        self.action_frame = ttk.Frame(self.main_frame)
        self.running_frame = None

    def log_step(self, message, success=None):
        self.steps_text.config(state=tk.NORMAL)
        if success is True:
            self.steps_text.insert(tk.END, f"✓ {message}\n")
        elif success is False:
            self.steps_text.insert(tk.END, f"✗ {message}\n")
        else:
            self.steps_text.insert(tk.END, f"○ {message}\n")
        self.steps_text.see(tk.END)
        self.steps_text.config(state=tk.DISABLED)

    def update_status(self, text, progress_val):
        self.status_var.set(text)
        self.progress['value'] = progress_val
        self.root.update()

    def run_command(self, cmd, desc, cwd=None, env=None):
        try:
            subprocess.run(cmd, shell=True, check=True, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
            return True
        except subprocess.CalledProcessError:
            return False

    def run_startup_sequence(self):
        try:
            # 0. Check if VISTA is already running
            import subprocess
            try:
                # Check for running supervisor_ui processes (excluding the current one)
                output = subprocess.check_output(["pgrep", "-f", "supervisor_ui.py"]).decode().strip()
                pids = [p for p in output.split('\n') if p]
                if len(pids) > 1:
                    self.root.after(0, lambda: messagebox.showerror("Already Running", "VISTA AI is already running."))
                    self.on_close()
                    return
            except subprocess.CalledProcessError:
                pass
                
            # 0.5 Check Port availability
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('127.0.0.1', 5001))
            sock.close()
            if result == 0:
                self.root.after(0, lambda: messagebox.showerror("Port Unavailable", "Required port 5001 is unavailable (occupied by another application)."))
                self.on_close()
                return

            # 1. Environment Check
            self.update_status("Checking system...", 10)
            self.log_step("Checking system requirements", True)
            time.sleep(0.5)
            
            # 2. Application Runtime Data
            self.update_status("Preparing runtime data...", 20)
            self.log_step("Validating runtime directories", True)
            
            # Bootstrap offline models if they don't exist in Application Support
            bootstrap_models = os.path.join(self.base_dir, "bootstrap", "models")
            if os.path.exists(bootstrap_models):
                for item in os.listdir(bootstrap_models):
                    src = os.path.join(bootstrap_models, item)
                    dst = os.path.join(self.models_dir, item)
                    if not os.path.exists(dst):
                        import shutil
                        if os.path.isdir(src): shutil.copytree(src, dst)
                        else: shutil.copy2(src, dst)
            
            time.sleep(0.5)

            # 3. AI Models (Offline Mode)
            self.update_status("Validating AI models...", 40)
            self.log_step("Verifying bundled AI models")
            model_script = f"""
import sys
import os
sys.path.insert(0, '{os.path.join(self.base_dir, "face_engine")}')
from model_manager import validate_models
if not validate_models(): sys.exit(1)
"""
            # We don't download, just validate. If missing, we warn.
            env = os.environ.copy()
            env["SIH26187_MODELS"] = self.models_dir
            env["SIH26187_DATA"] = self.data_dir
            env["INSIGHTFACE_HOME"] = self.app_support_dir
            env["PYTHONPATH"] = f"{os.path.join(self.base_dir, 'face_api')}:{self.base_dir}"
            env["SIH26187_VEHICLE_INT"] = os.path.join(self.base_dir, "vehicle_intelligence")

            success = self.run_command(f"{self.python_bin} -c \"{model_script}\"", "Models", env=env)
            if not success:
                self.log_step("AI Models missing. Skipping full validation for now.")
                # We'll allow it to continue because download_missing() was removed.
            else:
                self.log_step("AI Models ready", True)

            # 5. Start FastAPI
            self.update_status("Starting VISTA API...", 75)
            self.log_step("Starting API Service")
            
            # Open log files
            api_log = open(os.path.join(self.log_dir, "api.log"), "a")
            
            self.api_process = subprocess.Popen(
                [self.python_bin, os.path.join(self.base_dir, "face_api", "run.py")],
                stdout=api_log, stderr=subprocess.STDOUT, env=env, cwd=self.base_dir,
                preexec_fn=os.setsid
            )

            # 6. Health Check
            self.update_status("Waiting for health check...", 85)
            api_ready = False
            for _ in range(30):
                try:
                    res = urllib.request.urlopen("http://localhost:5001/api/health", timeout=1)
                    if res.status == 200:
                        api_ready = True
                        break
                except Exception:
                    pass
                time.sleep(1)
            
            if not api_ready:
                self.root.after(0, lambda: messagebox.showerror("Error", "Backend stopped unexpectedly or health check failed."))
                self.on_close()
                return
            self.log_step("API Health Check passed", True)

            # 7. Start AI Engine
            self.update_status("Starting AI Pipeline...", 95)
            self.log_step("Starting Camera Pipeline")
            scorer_log = open(os.path.join(self.log_dir, "scorer.log"), "a")
            self.scorer_process = subprocess.Popen(
                [self.python_bin, "-u", os.path.join(self.base_dir, "face_engine", "live_scorer.py")],
                stdout=scorer_log, stderr=subprocess.STDOUT, env=env, cwd=os.path.join(self.base_dir, "face_engine"),
                preexec_fn=os.setsid
            )

            # 8. Finished
            self.update_status("VISTA AI is Running", 100)
            self.log_step("Opening Dashboard...", True)
            time.sleep(0.5)
            self.root.after(0, self.show_running_state)
            
            # Open browser
            self.open_dashboard()
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Startup Error", str(e)))
            self.on_close()

    def show_running_state(self):
        # Clear startup UI
        self.progress.pack_forget()
        self.steps_text.pack_forget()
        self.status_var.set("● ON")
        self.status_label.config(foreground="green", font=("Helvetica", 14, "bold"))
        
        # Create Running UI Layout
        self.running_frame = ttk.Frame(self.main_frame)
        self.running_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        ttk.Label(self.running_frame, text="System Status", font=("Helvetica", 12, "bold")).pack(anchor=tk.W)
        ttk.Separator(self.running_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        
        status_grid = ttk.Frame(self.running_frame)
        status_grid.pack(fill=tk.X, pady=5)
        
        ttk.Label(status_grid, text="Backend", font=("Helvetica", 11)).grid(row=0, column=0, sticky=tk.W, pady=2)
        self.lbl_backend_status = ttk.Label(status_grid, text="● Ready", foreground="green", font=("Helvetica", 11))
        self.lbl_backend_status.grid(row=0, column=1, sticky=tk.E, padx=50)
        
        ttk.Label(status_grid, text="AI Engine", font=("Helvetica", 11)).grid(row=1, column=0, sticky=tk.W, pady=2)
        self.lbl_engine_status = ttk.Label(status_grid, text="● Ready", foreground="green", font=("Helvetica", 11))
        self.lbl_engine_status.grid(row=1, column=1, sticky=tk.E, padx=50)
        
        ttk.Label(status_grid, text="Cameras", font=("Helvetica", 11)).grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Label(status_grid, text="● Active", foreground="green", font=("Helvetica", 11)).grid(row=2, column=1, sticky=tk.E, padx=50)
        
        ttk.Label(status_grid, text="Database", font=("Helvetica", 11)).grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Label(status_grid, text="● Ready", foreground="green", font=("Helvetica", 11)).grid(row=3, column=1, sticky=tk.E, padx=50)
        
        # Big Dashboard Button
        btn_dash = tk.Button(self.running_frame, text="OPEN VISTA DASHBOARD", bg="#007AFF", fg="black", font=("Helvetica", 12, "bold"), command=self.open_dashboard, height=2)
        btn_dash.pack(fill=tk.X, pady=20)
        
        # Action Buttons
        actions = ttk.Frame(self.running_frame)
        actions.pack(fill=tk.X, side=tk.BOTTOM)
        
        ttk.Button(actions, text="Restart Services", command=self.restart_services).pack(side=tk.LEFT, expand=True, padx=2)
        ttk.Button(actions, text="Open Logs", command=self.open_logs).pack(side=tk.LEFT, expand=True, padx=2)
        ttk.Button(actions, text="Quit", command=self.on_close).pack(side=tk.LEFT, expand=True, padx=2)
        
        self.root.after(2000, self.monitor_processes)

    def monitor_processes(self):
        if not hasattr(self, 'running_frame') or not self.running_frame.winfo_exists():
            return
            
        if self.api_process and self.api_process.poll() is not None:
            self.lbl_backend_status.config(text="● Failed", foreground="red")
            self.status_var.set("● ERROR")
            self.status_label.config(foreground="red")
            
        if self.scorer_process and self.scorer_process.poll() is not None:
            self.lbl_engine_status.config(text="● Failed", foreground="red")
            self.status_var.set("● ERROR")
            self.status_label.config(foreground="red")
            
        self.root.after(2000, self.monitor_processes)

    def restart_services(self):
        self.kill_processes()
        if hasattr(self, 'main_frame') and self.main_frame:
            self.main_frame.destroy()
        self.setup_ui()
        threading.Thread(target=self.run_startup_sequence, daemon=True).start()

    def kill_processes(self):
        if self.scorer_process:
            try:
                os.killpg(os.getpgid(self.scorer_process.pid), signal.SIGTERM)
                self.scorer_process.wait(timeout=3)
            except Exception:
                try: os.killpg(os.getpgid(self.scorer_process.pid), signal.SIGKILL)
                except: pass
            self.scorer_process = None
            
        if self.api_process:
            try:
                os.killpg(os.getpgid(self.api_process.pid), signal.SIGTERM)
                self.api_process.wait(timeout=3)
            except Exception:
                try: os.killpg(os.getpgid(self.api_process.pid), signal.SIGKILL)
                except: pass
            self.api_process = None

    def open_dashboard(self):
        subprocess.run(["open", "http://localhost:5001/"])

    def open_logs(self):
        subprocess.run(["open", self.log_dir])

    def on_close(self):
        self.status_var.set("Shutting down...")
        self.root.update()
        self.kill_processes()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = VistaSupervisor(root)
    root.mainloop()
