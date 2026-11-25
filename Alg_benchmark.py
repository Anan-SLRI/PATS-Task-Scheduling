import os
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import scipy.io as sio
import sys
import re
sys.setrecursionlimit(10000) 

from algorithms import pats, heft, pets, lookahead, ceft, peft


class SchedulerBenchmarkGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Heterogeneous Task Scheduling Benchmark (PATS vs State-of-the-Art)")
        self.root.geometry("820x540")
        self.root.resizable(True, True)

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.datasets_dir = os.path.join(self.base_dir, "datasets")
        self.current_graph = None
        self.current_size = None

        self.algorithms = {
            "PATS": pats.solve,
            "HEFT": heft.solve,
            "PETS": pets.solve,
            "Lookahead": lookahead.solve,
            "CEFT": ceft.solve,
            "PEFT": peft.solve,
        }

        self._setup_ui()
        self._scan_datasets()

    def _setup_ui(self):
        # --- 1. Frame Dataset ---
        frame_top = ttk.LabelFrame(self.root, text="1. Select Dataset", padding=10)
        frame_top.pack(fill="x", padx=15, pady=10)

        ttk.Label(frame_top, text="Dataset Folder:").grid(row=0, column=0, sticky="w", padx=5)
        self.dataset_var = tk.StringVar()
        self.combo_dataset = ttk.Combobox(frame_top, textvariable=self.dataset_var, width=35, state="readonly")
        self.combo_dataset.grid(row=0, column=1, padx=5)
        self.combo_dataset.bind("<<ComboboxSelected>>", self._on_dataset_selected)

        ttk.Button(frame_top, text="Browse Folder...", command=self._browse_folder).grid(row=0, column=2, padx=5)

        self.lbl_info = ttk.Label(frame_top, text="Graph Info: Not loaded", foreground="blue")
        self.lbl_info.grid(row=1, column=0, columnspan=3, sticky="w", padx=5, pady=(8, 0))

        # --- 2. Frame algorithms ---
        frame_alg = ttk.LabelFrame(self.root, text="2. Run Scheduling Algorithms", padding=10)
        frame_alg.pack(fill="x", padx=15, pady=5)

        self.alg_buttons = []
        col_idx = 0
        for alg_name in self.algorithms.keys():
            btn = ttk.Button(
                frame_alg,
                text=f"Run {alg_name}",
                command=lambda name=alg_name: self._run_single_thread(name),
            )
            btn.grid(row=0, column=col_idx, padx=4, pady=5, sticky="ew")
            self.alg_buttons.append(btn)
            col_idx += 1

        btn_run_all = ttk.Button(
            frame_alg,
            text="▶ Run ALL Algorithms",
            command=self._run_all_thread,
        )
        btn_run_all.grid(row=1, column=0, columnspan=4, padx=4, pady=5, sticky="ew")
        self.alg_buttons.append(btn_run_all)

        btn_clear = ttk.Button(frame_alg, text="Clear Table", command=self._clear_table)
        btn_clear.grid(row=1, column=4, columnspan=2, padx=4, pady=5, sticky="ew")

        # --- 3. Frame result ---
        frame_res = ttk.LabelFrame(self.root, text="3. Benchmark Results", padding=10)
        frame_res.pack(fill="both", expand=True, padx=15, pady=10)

        columns = ("algorithm", "dataset", "tasks_pus", "makespan", "time_sec", "verify")
        self.tree = ttk.Treeview(frame_res, columns=columns, show="headings", height=12)

        self.tree.heading("algorithm", text="Algorithm")
        self.tree.heading("dataset", text="Dataset")
        self.tree.heading("tasks_pus", text="Tasks / PUs")
        self.tree.heading("makespan", text="Schedule Length")
        self.tree.heading("time_sec", text="Running Time (s)")
        self.tree.heading("verify", text="Verification")

        self.tree.column("algorithm", width=150, anchor="w")
        self.tree.column("dataset", width=120, anchor="center")
        self.tree.column("tasks_pus", width=110, anchor="center")
        self.tree.column("makespan", width=130, anchor="center")
        self.tree.column("time_sec", width=130, anchor="center")
        self.tree.column("verify", width=110, anchor="center")

        scrollbar = ttk.Scrollbar(frame_res, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # --- Status Bar ---
        self.status_var = tk.StringVar(value="Ready.")
        lbl_status = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w", padding=(10, 3))
        lbl_status.pack(fill="x", side="bottom")

    def _scan_datasets(self):
        if os.path.exists(self.datasets_dir):
            folders = [
                f for f in os.listdir(self.datasets_dir)
                if os.path.isdir(os.path.join(self.datasets_dir, f))
            ]

            # Natural Sort
            def natural_sort_key(text):
                return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", text)]

            folders.sort(key=natural_sort_key)
            self.combo_dataset["values"] = folders
            if folders:
                self.combo_dataset.current(0)
                self._load_dataset_from_path(os.path.join(self.datasets_dir, folders[0]))


    def _browse_folder(self):
        folder_selected = filedialog.askdirectory(initialdir=self.base_dir, title="Select Dataset Folder (.mat)")
        if folder_selected:
            self.dataset_var.set(os.path.basename(folder_selected))
            self._load_dataset_from_path(folder_selected)

    def _on_dataset_selected(self, event):
        folder_name = self.dataset_var.get()
        self._load_dataset_from_path(os.path.join(self.datasets_dir, folder_name))

    def _load_dataset_from_path(self, folder_path):
        try:
            mat_files = [
                f for f in os.listdir(folder_path)
                if f.endswith(".mat") and f != "size_a.mat"
            ]
            size_path = os.path.join(folder_path, "size_a.mat")

            if not mat_files or not os.path.exists(size_path):
                messagebox.showerror(
                    "Error"
                )
                return

            target_graph_file = "graph_x.mat" if "graph_x.mat" in mat_files else mat_files[0]

            graph_path = os.path.join(folder_path, target_graph_file)
            mat_graph = sio.loadmat(graph_path)
            self.current_graph = mat_graph["graph_x"].astype(float)

            mat_size = sio.loadmat(size_path)
            self.current_size = mat_size["size_a"].flatten()

            a1, a2 = int(self.current_size.item(0)), int(self.current_size.item(1))
            self.lbl_info.config(
                text=f"Loaded: {target_graph_file}  |  Tasks (v) = {a1}  |  Processing Units (p) = {a2}"
            )
            self.status_var.set(
                f"Loaded dataset '{os.path.basename(folder_path)}' successfully."
            )
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def _set_buttons_state(self, state):
        for btn in self.alg_buttons:
            btn.config(state=state)

    def _run_single_thread(self, alg_name):
        if self.current_graph is None:
            messagebox.showwarning("Warning", "Please select a dataset before running.")
            return
        threading.Thread(target=self._execute_algorithm, args=(alg_name,), daemon=True).start()

    def _run_all_thread(self):
        if self.current_graph is None:
            messagebox.showwarning("Warning", "Please select a dataset before running.")
            return
        threading.Thread(target=self._execute_all, daemon=True).start()

    def _execute_all(self):
        for alg_name in self.algorithms.keys():
            self._execute_algorithm(alg_name)
    
    def _verify_schedule(self, graph_x, schedule_mat, a1, a2):
        b1 = graph_x.shape[0]  
        a = graph_x[:, b1 : b1 + a2]

 
        max_sum_a = int(np.max(np.sum(a, axis=0)) + np.sum(graph_x[:, :b1])) + 500
        pu_idle_state = np.ones((max_sum_a, a2), dtype=int)
        chk = np.zeros(a2)

        for i in range(a1):
            pu_idx = int(schedule_mat[i, 1]) - 1
            st_idx = int(schedule_mat[i, 2]) - 1
            ft_idx = int(schedule_mat[i, 3])
            if pu_idx < 0 or st_idx < 0 or ft_idx > pu_idle_state.shape[0]:  

                return False
            pu_idle_state[st_idx:ft_idx, pu_idx] = 0
            chk[pu_idx] += graph_x[i, b1 + pu_idx]

        idle_diff = pu_idle_state.shape[0] - np.sum(pu_idle_state, axis=0)  
        if not (np.all(chk == idle_diff) and np.sum(chk) == np.sum(idle_diff)):

            return False


        edges_i, edges_j = np.where(graph_x[:, :b1] > 0)
        for u_idx, v_idx in zip(edges_i, edges_j):
            pu_u, ft_u = schedule_mat[u_idx, 1], schedule_mat[u_idx, 3]
            pu_v, st_v = schedule_mat[v_idx, 1], schedule_mat[v_idx, 2]
            comm_cost = 0 if pu_u == pu_v else graph_x[u_idx, v_idx]
            if st_v < ft_u + comm_cost + 1:
                print(f"\nVIOLATION: Task {u_idx+1} (PU {int(pu_u)}, FT={int(ft_u)}) -> Task {v_idx+1} (PU {int(pu_v)}, ST={int(st_v)}), Comm={int(comm_cost)}")
                return False

        return True




    def _execute_algorithm(self, alg_name):
        os.system('cls' if os.name == 'nt' else 'clear')
        self.root.after(0, lambda: self._set_buttons_state("disabled"))
        self.root.after(0, lambda: self.status_var.set(f"Running {alg_name}... Please wait."))

        try:
            solve_func = self.algorithms[alg_name]
            t0 = time.perf_counter()

            makespan, is_valid_internal, schedule_mat = solve_func(self.current_graph.copy(), self.current_size.copy())
            elapsed = time.perf_counter() - t0

            dataset_name = self.dataset_var.get()
            a1, a2 = int(self.current_size.item(0)), int(self.current_size.item(1))
            is_valid = is_valid_internal and self._verify_schedule(self.current_graph, schedule_mat, a1, a2)
            verify_text = "PASS ✅" if is_valid else "FAIL ❌"


            self.root.after(
                0,
                lambda: self.tree.insert(
                    "",
                    "end",
                    values=(alg_name, dataset_name, f"{a1} / {a2}", int(makespan), f"{elapsed:.4f}", verify_text),
                ),
            )
            self.root.after(0, lambda: self.status_var.set(f"Finished {alg_name} in {elapsed:.4f}s (Makespan = {int(makespan)})"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Execution Error", f"{alg_name} failed:\n{str(e)}"))
            self.root.after(0, lambda: self.status_var.set(f"Error running {alg_name}."))
        finally:
            self.root.after(0, lambda: self._set_buttons_state("normal"))

    def _clear_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)


if __name__ == "__main__":
    root = tk.Tk()
    app = SchedulerBenchmarkGUI(root)
    root.mainloop()
