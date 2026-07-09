import tkinter as tk
from tkinter import messagebox, simpledialog
import json
import subprocess
import sys
import os

class GestureAppUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Pipeline Controller")
        self.root.geometry("800x750")
        self.root.configure(padx=20, pady=20)

        # Internal Data Structure
        self.config_data = {
            "labels": {
                "OPEN_BROWSER_READY": ["OPEN_BROWSER"],
                "CLOSE_WINDOW_READY": ["CLOSE_WINDOW"]
            },
            "actions": {
                "OPEN_BROWSER": "OPEN chrome",
                "CLOSE_WINDOW": "CLOSE WINDOW"
            },
            "gesture_dur": 0.5
        }

        self.python_exe = sys.executable

        self.build_top_bar()
        self.build_editor_ui()
        self.build_pipeline_controls()
        
        # Populate initial data
        self.refresh_static_list()

    # --- 1. TOP BAR (LOAD/SAVE) ---
    def build_top_bar(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(fill="x", pady=(0, 15))

        tk.Label(top_frame, text="Config Name (no .json):", font=("Arial", 10, "bold")).pack(side="left")
        self.config_name_var = tk.StringVar(value="myconfig")
        tk.Entry(top_frame, textvariable=self.config_name_var, width=20).pack(side="left", padx=10)

        tk.Button(top_frame, text="Load", command=self.load_config, bg="#2196F3", fg="white", width=10).pack(side="left", padx=5)
        tk.Button(top_frame, text="Save", command=self.save_config, bg="#4CAF50", fg="white", width=10).pack(side="left")

    # --- 2. CONFIGURATION EDITOR UI ---
    def build_editor_ui(self):
        editor_frame = tk.LabelFrame(self.root, text="Visual Configuration Editor", font=("Arial", 10, "bold"), padx=10, pady=10)
        editor_frame.pack(fill="both", expand=True, pady=(0, 15))

        # 3-Column Layout
        editor_frame.columnconfigure(0, weight=1)
        editor_frame.columnconfigure(1, weight=1)
        editor_frame.columnconfigure(2, weight=1)

        # -- Column 1: Static Gestures --
        tk.Label(editor_frame, text="1. Static Gestures").grid(row=0, column=0, sticky="w")
        self.list_static = tk.Listbox(editor_frame, exportselection=False, height=12)
        self.list_static.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.list_static.bind("<<ListboxSelect>>", self.on_static_select)

        btn_frame1 = tk.Frame(editor_frame)
        btn_frame1.grid(row=2, column=0, sticky="ew", padx=5)
        tk.Button(btn_frame1, text="+ Add", command=self.add_static).pack(side="left", expand=True, fill="x")
        tk.Button(btn_frame1, text="- Del", command=self.del_static).pack(side="left", expand=True, fill="x")

        # -- Column 2: Dynamic Gestures --
        tk.Label(editor_frame, text="2. Dynamic Gestures").grid(row=0, column=1, sticky="w")
        self.list_dynamic = tk.Listbox(editor_frame, exportselection=False, height=12)
        self.list_dynamic.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        self.list_dynamic.bind("<<ListboxSelect>>", self.on_dynamic_select)

        btn_frame2 = tk.Frame(editor_frame)
        btn_frame2.grid(row=2, column=1, sticky="ew", padx=5)
        tk.Button(btn_frame2, text="+ Add", command=self.add_dynamic).pack(side="left", expand=True, fill="x")
        tk.Button(btn_frame2, text="- Del", command=self.del_dynamic).pack(side="left", expand=True, fill="x")

        # -- Column 3: Actions --
        tk.Label(editor_frame, text="3. Action Command").grid(row=0, column=2, sticky="w")
        action_frame = tk.Frame(editor_frame)
        action_frame.grid(row=1, column=2, sticky="nsew", padx=5, pady=5)
        
        self.action_var = tk.StringVar()
        self.action_entry = tk.Entry(action_frame, textvariable=self.action_var)
        self.action_entry.pack(fill="x", pady=5)
        tk.Button(action_frame, text="Save Action", command=self.save_action_string, bg="#FFC107").pack(fill="x")

        tk.Label(editor_frame, text="4. Gesture Duration").grid(row=2, column=2, sticky="w")
        gesture_dur_frame = tk.Frame(editor_frame)
        gesture_dur_frame.grid(row=3, column=2, sticky="nsew", padx=5, pady=5)

        self.gesture_dur_scale = tk.Scale(gesture_dur_frame, from_=0.2, to=2, resolution=0.1, orient="horizontal", command=self.gesture_dur_changed)
        self.gesture_dur_scale.pack(fill="x")
        self.gesture_dur_scale.set(self.config_data.get("gesture_dur"))


    # --- 3. PIPELINE CONTROLS ---
    def build_pipeline_controls(self):
        pipeline_frame = tk.LabelFrame(self.root, text="Pipeline Execution", font=("Arial", 10, "bold"), padx=10, pady=10)
        pipeline_frame.pack(fill="x", pady=10)

        buttons = [
            ("1. Record Static Gestures", ["main.py", "{config}", "sample", "static"], "#2196F3"),
            ("2. Train Static Model", ["training_static_model.py", "{config}", "static"], "#FF9800"),
            ("3. Record Dynamic Gestures", ["main.py", "{config}", "sample", "dynamic"], "#2196F3"),
            ("4. Train Dynamic Model", ["training_static_model.py", "{config}", "dynamic"], "#FF9800"),
            ("5. Test Pipeline (Live)", ["main.py", "{config}", "test"], "#9C27B0")
        ]

        for text, cmd_template, color in buttons:
            tk.Button(
                pipeline_frame, 
                text=text, bg=color, fg="white", font=("Arial", 10, "bold"),
                command=lambda c=cmd_template: self.run_script(c)
            ).pack(fill="x", pady=4)

    # --- LOGIC: UI UPDATES ---
    def refresh_static_list(self):
        self.list_static.delete(0, tk.END)
        for static_name in self.config_data["labels"]:
            self.list_static.insert(tk.END, static_name)
        self.list_dynamic.delete(0, tk.END)
        self.action_var.set("")
        self.gesture_dur_scale.set(self.config_data.get("gesture_dur"))

    def on_static_select(self, event):
        selection = self.list_static.curselection()
        if not selection: return
        
        static_name = self.list_static.get(selection[0])
        self.list_dynamic.delete(0, tk.END)
        self.action_var.set("")
        
        for dyn_name in self.config_data["labels"].get(static_name, []):
            self.list_dynamic.insert(tk.END, dyn_name)

    def on_dynamic_select(self, event):
        selection = self.list_dynamic.curselection()
        if not selection: return
        
        dyn_name = self.list_dynamic.get(selection[0])
        action_str = self.config_data["actions"].get(dyn_name, "")
        self.action_var.set(action_str)


    # --- LOGIC: ADD / DELETE ---
    def add_static(self):
        name = simpledialog.askstring("Add Static", "Enter Static Gesture Name (e.g., VOLUME_READY):")
        if name and name not in self.config_data["labels"]:
            self.config_data["labels"][name] = []
            self.refresh_static_list()

    def del_static(self):
        selection = self.list_static.curselection()
        if not selection: return
        name = self.list_static.get(selection[0])
        
        # Clean up orphaned actions tied to this static gesture
        for dyn in self.config_data["labels"][name]:
            if dyn in self.config_data["actions"]:
                del self.config_data["actions"][dyn]
                
        del self.config_data["labels"][name]
        self.refresh_static_list()

    def add_dynamic(self):
        static_sel = self.list_static.curselection()
        if not static_sel:
            messagebox.showwarning("Warning", "Select a Static Gesture first.")
            return
            
        static_name = self.list_static.get(static_sel[0])
        name = simpledialog.askstring("Add Dynamic", f"Add dynamic gesture for {static_name}:")
        
        if name and name not in self.config_data["labels"][static_name]:
            self.config_data["labels"][static_name].append(name)
            self.config_data["actions"][name] = "ENTER_ACTION_HERE"
            self.on_static_select(None) # Refresh dynamic list

    def del_dynamic(self):
        static_sel = self.list_static.curselection()
        dyn_sel = self.list_dynamic.curselection()
        if not static_sel or not dyn_sel: return
        
        static_name = self.list_static.get(static_sel[0])
        dyn_name = self.list_dynamic.get(dyn_sel[0])
        
        self.config_data["labels"][static_name].remove(dyn_name)
        if dyn_name in self.config_data["actions"]:
            del self.config_data["actions"][dyn_name]
            
        self.on_static_select(None) # Refresh dynamic list

    def save_action_string(self):
        dyn_sel = self.list_dynamic.curselection()
        if not dyn_sel:
            messagebox.showwarning("Warning", "Select a Dynamic Gesture first.")
            return
            
        dyn_name = self.list_dynamic.get(dyn_sel[0])
        new_action = self.action_var.get()
        self.config_data["actions"][dyn_name] = new_action
        messagebox.showinfo("Saved", f"Action updated for {dyn_name}")

    def gesture_dur_changed(self, value):
        self.config_data["gesture_dur"] = float(value)

    # --- LOGIC: FILE IO & SCRIPT EXECUTION ---
    def save_config(self):
        config_name = self.config_name_var.get().strip()
        if not config_name:
            messagebox.showwarning("Warning", "Please enter a configuration name.")
            return
            
        filename = f"{config_name}.json"
        with open(f"Configs/{filename}", 'w') as f:
            json.dump(self.config_data, f, indent=4)
        messagebox.showinfo("Success", f"Saved to 'Configs/{filename}'!")

    def load_config(self):
        config_name = self.config_name_var.get().strip()
        filename = f"{config_name}.json"
        
        if not os.path.exists(f"Configs/{filename}"):
            messagebox.showerror("Error", f"File 'Configs/{filename}' not found.")
            return
            
        with open(f"Configs/{filename}", 'r') as f:
            self.config_data = json.load(f)
            
        self.refresh_static_list()
        messagebox.showinfo("Loaded", f"Successfully loaded 'Configs/{filename}'")

    def run_script(self, cmd_template):
        config_name = self.config_name_var.get().strip()
        actual_cmd = [self.python_exe] + [arg.replace("{config}", config_name) for arg in cmd_template]
        
        # Save before running to ensure scripts use the latest UI state
        self.save_config()
        
        try:
            subprocess.Popen(actual_cmd)
        except Exception as e:
            messagebox.showerror("Execution Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = GestureAppUI(root)
    root.mainloop()
