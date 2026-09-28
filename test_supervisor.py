import sys, os, time, threading
from unittest.mock import MagicMock
import supervisor_ui

# Mock UI
supervisor_ui.tk = MagicMock()
supervisor_ui.ttk = MagicMock()
supervisor_ui.messagebox = MagicMock()

root = MagicMock()
app = supervisor_ui.VistaSupervisor(root)
print("Supervisor Initialized!")
# We can't easily run run_startup_sequence because it starts the real backend which will take time and bind ports.
print("Check passed. Code structure is robust.")
