import tkinter as tk
from tkinter import ttk
import subprocess
import sys
import os

def start_camera():
    """
    Launch main.py in a new console window.
    """
    python = sys.executable
    creation_flags = 0
    if os.name == "nt":
        creation_flags = subprocess.CREATE_NEW_CONSOLE
    
    subprocess.Popen(
        [python, "main.py"],
        creationflags = creation_flags
    )

def start_profiles():
    """
    Launch profiles_gui.py in a new console window.
    """
    python = sys.executable
    creation_flags = 0
    if os.name == "nt":
        creation_flags = subprocess.CREATE_NEW_CONSOLE
    
    subprocess.Popen(
        [python, "profiles_gui.py",],
        creationflags = creation_flags
    )

def main():
    root = tk.Tk()
    root.title("GestureX Launcher")
    root.geometry("360x200")
    root.resizable(False, False)

    title = ttk.Label(root, text = "GestureX Launcher", font = ("Segoe UI", 14, "bold"))
    title.pack(pady = (15, 5))

    subtitle = ttk.Label(root, text = "Start the camera loop or profile manager", font = ("Segoe UI", 9))
    subtitle.pack(pady = (0, 15))

    button_frame = ttk.Frame(root)
    button_frame.pack(fill = "x", padx = 20)

    camera_button = ttk.Button(button_frame, text = "Start Camera", command = start_camera)
    profiles_button = ttk.Button(button_frame, text = "Open Profile Manager", command = start_profiles)

    button_frame.columnconfigure(0, weight = 1)

    camera_button.grid(row = 0, column = 0, sticky = "ew", padx = 5, pady = 5)
    profiles_button.grid(row = 1, column = 0, sticky = "ew", padx = 5, pady = 5)

    quit_frame = ttk.Frame(root)
    quit_frame.pack(fill = "x", padx = 20, pady = (10, 10))

    quit_button = ttk.Button(quit_frame, text = "Exit Launcher", command = root.destroy)
    quit_button.pack(side = "right")

    root.mainloop()

if __name__ == "__main__":
    main()