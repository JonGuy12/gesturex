import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

CONFIG_PATH = Path(__file__).with_name("profiles.json")

def load_profiles():
    if not CONFIG_PATH.exists():
        return {
            "current_profile": "Default",
            "profiles": {
                "Default": {
                    "description": "Default empty profile",
                    "gestures": {}
                }
            }
        }
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
    

def save_profiles(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent = 2)

class ProfilesGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GestureX Profiles Editor")
        self.geometry("600x450")

        self.allowed_key_values = {"space", "esc", "enter",
            "left", "right", "up", "down",
            "tab", "backspace",
            *[chr(c) for c in range(ord('a'), ord('z') + 1)],
            *[str(d) for d in range(10)]
        }

        self.cfg = load_profiles()
        self.key_entry_original = ""

        top_frame = tk.Frame(self)
        top_frame.pack(fill = "x", padx = 10, pady = 10)

        tk.Label(top_frame, text = "Profile:").pack(side = "left")

        self.profile_var = tk.StringVar()
        profiles = list(self.cfg.get("profiles", {}).keys())
        if not profiles:
            profiles = ["Default"]
        self.profile_var.set(self.cfg.get("current_profile", profiles[0]))

        self.profile_menu = ttk.OptionMenu(
            top_frame, self.profile_var, self.profile_var.get(), *profiles,
            command = self.on_profile_change)

        self.profile_menu.pack(side = "left", padx = 5)

        tk.Button(top_frame, text = "Set as Current", command = self.set_as_current).pack(side = "left", padx = 5)
        tk.Button(top_frame, text="New Profile", command=self.create_new_profile).pack(side="left", padx=5)
        tk.Button(top_frame, text="Delete Profile", command=self.delete_profile).pack(side="left", padx=5)

        mid_frame = tk.Frame(self)
        mid_frame.pack(fill = "both", expand = True, padx = 10, pady = 5)

        columns = ("gesture", "type", "vaule")
        self.tree = ttk.Treeview(mid_frame, columns = columns, show = "headings", height = 10)
        for col in columns:
            self.tree.heading(col, text = col.title())
            self.tree.column(col, width = 150, anchor = "w")
        self.tree.pack(side = "left", fill = "both", expand = True)

        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

        scrollbar = ttk.Scrollbar(mid_frame, orient = "vertical", command = self.tree.yview)
        scrollbar.pack(side = "right", fill = "y")
        self.tree.config(yscrollcommand = scrollbar.set)

        bottom_frame = tk.LabelFrame(self, text = "Add / Update Mapping")
        bottom_frame.pack(fill = "x", padx = 10, pady = 10)

        tk.Label(bottom_frame, text = "Gesture Label:").grid (row = 0, column = 0, sticky = "w", padx = 5, pady = 3)
        self.gesture_var = tk.StringVar(value="")
        self.gesture_box = ttk.Combobox(bottom_frame, textvariable = self.gesture_var, values = ["", "Open Palm", "Closed Fist", "Point (Index)", "Thumbs Up/Side"], width = 20, state = "readonly")
        self.gesture_box.grid(row = 0, column = 1, sticky = "w", padx = 5, pady = 3)

        tk.Label(bottom_frame, text = "Type:").grid(row = 0, column = 2, sticky = "w", padx = 5, pady = 3)
        self.type_var = tk.StringVar(value = "key")
        type_box = ttk.Combobox(bottom_frame, textvariable = self.type_var, values = ["key", "volume"], width = 10, state = "readonly")
        type_box.grid(row = 0, column = 3, sticky = "w", padx = 5, pady = 3)

        tk.Label(bottom_frame, text = "Move Up/Down:").grid(row = 1, column = 4, sticky = "w", padx = 5, pady = 3)
        tk.Button(bottom_frame, text = "   ↑   ", command = lambda: self.move_selected("up")).grid(row = 1, column = 5, sticky = "w", padx = 5, pady = 3)
        tk.Button(bottom_frame, text = "   ↓   ", command = lambda: self.move_selected("down")).grid(row = 2, column = 5, sticky = "w", padx = 5, pady = 3)

        tk.Label(bottom_frame, text = "Key Value:").grid(row = 1, column = 0, sticky = "w", padx = 5, pady = 3)
        self.key_entry = tk.Entry(bottom_frame, width = 23)
        self.key_entry.grid(row = 1, column = 1, sticky = "w", padx = 5, pady = 3)

        button_frame = tk.Frame(bottom_frame)
        button_frame.grid(row = 2, column = 0, columnspan = 4, sticky = "ew", padx = 5, pady = 5)

        button_frame.columnconfigure(0, weight = 1)
        button_frame.columnconfigure(1, weight = 0)
        
        left_group = tk.Frame(button_frame)
        left_group.grid(row = 0, column = 0, sticky = "w")
        tk.Button(left_group, text = "Save Mapping", command = self.save_mapping).pack(side = "left", padx = 5)
        tk.Button(left_group, text = "Delete Mapping", command = self.delete_mapping).pack(side = "left", padx = 5)

        self.refresh_tree()

    def current_profile_name(self):
        return self.profile_var.get()
    
    def on_profile_change(self, *_):
        self.refresh_tree()
    
    def set_as_current(self):
        name = self.current_profile_name()
        if name not in self.cfg.get("profiles", {}):
            messagebox.showerror("Error", f"Profile '{name}' not found.")
            return
        self.cfg["current_profile"] = name
        save_profiles(self.cfg)
        messagebox.showinfo("Profile", f"Current profile set to: {name}")
    
    def refresh_tree(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        profs = self.cfg.get("profiles", {})
        prof = profs.get(self.current_profile_name(), {})
        gestures = prof.get("gestures", {})

        for gesture, action in gestures.items():
            a_type = action.get("type", "key")
            val = action.get("value", {})
            self.tree.insert("", "end", iid = gesture, values = (gesture, a_type, val))

    def on_tree_select(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        item_id = sel[0]
        values = self.tree.item(item_id, "values")

        self.gesture_var.set(values[0])
        self.type_var.set(values[1])
        self.key_entry.delete(0, tk.END)
        self.key_entry.insert(0, values[2])
        
        self.key_entry_original = values[2]

    def save_mapping(self):
        gesture = self.gesture_var.get().strip()
        key_val = self.key_entry.get().strip()
        a_type = self.type_var.get().strip()

        if not gesture or not key_val:
            messagebox.showerror("Error", "Gesture and key value cannot be empty.")
            self.key_entry.delete(0, tk.END)
            self.key_entry.insert(0, self.key_entry_original)
            return
        
        if gesture == "Pinch":
            messagebox.showerror("Error", "Cannot alter pinch gesture.")
            self.key_entry.delete(0, tk.END)
            self.key_entry.insert(0, self.key_entry_original)
            return
        
        if a_type == "volume" and gesture != "Pinch":
            messagebox.showerror("Error", "Cannot use volume type with this gesture.")
            self.key_entry.delete(0, tk.END)
            self.key_entry.insert(0, self.key_entry_original)
            return
        
        if a_type == "key":
            lower_key_val = key_val.lower()
            if lower_key_val not in self.allowed_key_values:
                messagebox.showerror("Invalid key",
                                     f"'{key_val}' is not a supported keyboard key.\n\n"
                                     f"Examples: space, esc, left, right, a-z, 0-9"
                                     )
                self.key_entry.delete(0, tk.END)
                self.key_entry.insert(0, self.key_entry_original)
                return

        profs = self.cfg.setdefault("profiles", {})
        prof = profs.setdefault(self.current_profile_name(), {"description": "", "gestures": {}})
        gestures = prof.setdefault("gestures", {})

        gestures[gesture] = {"type": a_type, "value": key_val}
        save_profiles(self.cfg)
        self.refresh_tree()
        self.key_entry_original = key_val

    def delete_mapping(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showerror("Error", "No gesture selected.")
            return

        gesture = sel[0]
        if gesture == "Pinch":
            messagebox.showerror("Error", "Cannot delete pinch gesture.")
            return
        
        profs = self.cfg.get("profiles", {})
        prof = profs.get(self.current_profile_name(), {})
        gestures = prof.get("gestures", {})

        if gesture in gestures:
            if messagebox.askyesno("Delete", f"Delete mapping for '{gesture}'?"):
                del gestures[gesture]
                save_profiles(self.cfg)
                self.refresh_tree()
                self.gesture_var.set("")
                self.key_entry.delete(0, tk.END)
        else:
            messagebox.showerror("Error", f"No mapping found for '{gesture}'.")

    def move_selected(self, direction):
        sel = self.tree.selection()
        if not sel:
            return
        
        item = sel[0]
        parent = self.tree.parent(item)
        index = self.tree.index(item)

        if direction == "up":
            new_index = index - 1
        else: # direction == "down"
            new_index = index + 1

        children = self.tree.get_children(parent)
        if new_index < 0 or new_index >= len(children):
            return
        
        self.tree.move(item, parent, new_index)
        self.tree.selection_set(item)
        self.save_order_from_tree()
    
    def save_order_from_tree(self):
        profs = self.cfg.get("profiles", {})
        prof = profs.get(self.current_profile_name(), {})
        gestures_dict = prof.get("gestures", {})

        new_gestures = {}
        for item in self.tree.get_children(""):
            gesture, a_type, val = self.tree.item(item, "values")
            new_gestures[gesture] = {
                "type": a_type,
                "value": val
            }
        
        prof["gestures"] = new_gestures
        save_profiles(self.cfg)

    def _rebuild_profile_menu(self):
        menu = self.profile_menu["menu"]
        menu.delete(0, "end")
        profiles = list(self.cfg.get("profiles", {}).keys())
        for name in profiles:
            menu.add_command(label=name, command=lambda n=name: self.profile_var.set(n))

    def create_new_profile(self):
        name = tk.simpledialog.askstring("New Profile", "Enter a new profile name:")
        if not name:
            return
        name = name.strip()
        if not name:
            return

        profs = self.cfg.setdefault("profiles", {})
        if name in profs:
            messagebox.showerror("Error", f"Profile '{name}' already exists.")
            return

        profs[name] = {"description": "", "gestures": {}}
        self.cfg["current_profile"] = name

        save_profiles(self.cfg)

        self._rebuild_profile_menu()
        self.profile_var.set(name)
        self.refresh_tree()
        messagebox.showinfo("Profile", f"Created new profile: {name}")

    def delete_profile(self):
        name = self.current_profile_name()
        profs = self.cfg.get("profiles", {})

        if name == "Default":
            messagebox.showerror("Error", "The Default profile cannot be deleted.")
            return

        if name not in profs:
            messagebox.showerror("Error", f"Profile '{name}' not found.")
            return

        if not messagebox.askyesno("Delete Profile", f"Delete profile '{name}'?\n\nThis action cannot be undone."):
            return

        del profs[name]

        self.cfg["current_profile"] = "Default"
        self.profile_var.set("Default")

        save_profiles(self.cfg)

        self._rebuild_profile_menu()
        self.refresh_tree()

        messagebox.showinfo("Profile Deleted", f"Profile '{name}' was deleted.\nSwitched to Default.")


if __name__ == "__main__":
    app = ProfilesGUI()
    app.mainloop()