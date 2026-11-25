import os
import time
import numpy as np
import scipy.io as sio


# ==============================================================================
# ==============================================================================
def insert_h(xv, pu_idle_state, assigne_task, graph_x, b2):
    pu_val, st_val, ft_val, task_val = xv

    pu_idx = int(pu_val) - 1
    st_idx = int(st_val) - 1
    ft_idx = int(ft_val)
    task_idx = int(task_val) - 1

    xpu1, xpu2 = pu_idle_state.shape

    slot_x = np.sum(pu_idle_state[st_idx:ft_idx, pu_idx])
    duration = int(ft_val - st_val + 1)
    diff_slot_x = duration - slot_x

    if diff_slot_x != 0:
        if pu_idle_state[st_idx, pu_idx] == 0:
            (emx_arr,) = np.where(pu_idle_state[st_idx:xpu1, pu_idx] != 0)
            emx_first = int(emx_arr.item(0))
            next_start = st_val + emx_first
            xv2 = np.array(
                [pu_val, next_start, next_start + (ft_val - st_val), task_val]
            )
            pu_idle_state, assigne_task, graph_x = insert_h(
                xv2, pu_idle_state, assigne_task, graph_x, b2
            )
        else:
            ft_start_idx = int(ft_val) - 1
            (emx_arr,) = np.where(pu_idle_state[ft_start_idx:xpu1, pu_idx] != 0)
            emx_first = int(emx_arr.item(0))
            next_start = ft_val + emx_first
            xv2 = np.array(
                [pu_val, next_start, next_start + (ft_val - st_val), task_val]
            )
            pu_idle_state, assigne_task, graph_x = insert_h(
                xv2, pu_idle_state, assigne_task, graph_x, b2
            )
    else:
        assigne_task[task_idx, :] = [task_val, pu_val, st_val, ft_val]
        graph_x[task_idx, b2 - 4 : b2] = [task_val, pu_val, st_val, ft_val]

    return pu_idle_state, assigne_task, graph_x


# ==============================================================================
# ==============================================================================
def EFTh(task_ni, graph_x, cpux, FTx, pu_idle_state):
    b1, b2 = graph_x.shape

    (pre_task,) = np.where(graph_x[:b1, task_ni] != 0)

    if len(pre_task) == 0:
        FTx[task_ni, cpux] = graph_x[task_ni, b1 + cpux]
    else:
        for p_task in pre_task:
            if graph_x[p_task, b2 - 3] == (cpux + 1):
                dx = 0.0
            else:
                dx = graph_x[p_task, task_ni]

            FTx_new = graph_x[p_task, b2 - 1] + graph_x[task_ni, b1 + cpux] + dx

            st_temp = graph_x[p_task, b2 - 1] + dx + 1
            x_temp = np.array([cpux + 1, st_temp, FTx_new, task_ni + 1])

            graph_x_temp = graph_x.copy()
            graph_x_temp[task_ni, b2 - 3 : b2] = x_temp[:3]

            assigne_task = np.column_stack(
                (graph_x[:, b2 - 5], graph_x[:, b2 - 3 : b2])
            )

            _, _, graph_x_temp = insert_h(
                x_temp, pu_idle_state.copy(), assigne_task, graph_x_temp, b2
            )
            FTx_new = graph_x_temp[task_ni, b2 - 1]

            if FTx[task_ni, cpux] < FTx_new:
                FTx[task_ni, cpux] = FTx_new

    return FTx

def solve(graph_x, size_a):
    a1, a2 = int(size_a.item(0)), int(size_a.item(1))
    b1, b2 = graph_x.shape
    a = graph_x[:, b1 : b1 + a2]

    max_sum_a = int(np.max(np.sum(a, axis=0)))
    pu_idle_state = np.ones((max_sum_a, a2), dtype=int)
    l = graph_x[:, b2 - 4]

    ACC = np.mean(graph_x[:, b1 : b1 + a2], axis=1)
    DTC = np.zeros(b1, dtype=float)
    RPT = np.zeros(b1, dtype=float)
    rank_v = np.zeros(b1, dtype=float)
    Priority = np.zeros(b1, dtype=int)
    FTx = np.zeros((b1, a2), dtype=float)

    count = 0 
    for level_x in range(1, int(np.max(l)) + 1):
        (task_list_l_arr,) = np.where(graph_x[:, b2 - 4] == level_x)
        task_list_l = list(task_list_l_arr)
        num_task_l = len(task_list_l)

        for j in range(num_task_l):
            curr_t = task_list_l[j]
            DTC[curr_t] = np.sum(graph_x[curr_t, :b1])
            (pre_task,) = np.where(graph_x[:b1, curr_t] != 0)
            max_rank_v = 0.0
            for p_t in pre_task:
                if max_rank_v < rank_v[p_t]:
                    max_rank_v = rank_v[p_t]
            RPT[curr_t] = max_rank_v
            rank_v[curr_t] = np.floor(ACC[curr_t] + DTC[curr_t] + RPT[curr_t] + 0.5)

        rank_v_temp = rank_v.copy()
        for j in range(1, num_task_l + 1):
            priority_temp = -1.0
            max_priority_temp = task_list_l[0]
            for jj in range(num_task_l):
                cand_t = task_list_l[jj]
                if priority_temp < rank_v_temp[cand_t]:
                    priority_temp = rank_v_temp[cand_t]
                    max_priority_temp = cand_t
                elif priority_temp == rank_v_temp[cand_t]:
                    if ACC[max_priority_temp] > ACC[cand_t]:
                        priority_temp = rank_v_temp[cand_t]
                        max_priority_temp = cand_t
            rank_v_temp[max_priority_temp] = -1.0
            Priority[max_priority_temp] = j

        Priority_temp = Priority.copy()
        while len(task_list_l) > 0:
            min_priority_task_l = b1 + 1
            task_vk_index = 0
            for j in range(len(task_list_l)):
                cand_t = task_list_l[j]
                if min_priority_task_l > Priority_temp[cand_t]:
                    min_priority_task_l = Priority_temp[cand_t]
                    task_vk_index = j

            task_vk = task_list_l[task_vk_index]
            Priority_temp[task_vk] = b1 + 1

            for pu in range(a2):
                FTx = EFTh(task_vk, graph_x, pu, FTx, pu_idle_state)

            xx2 = int(np.argmin(FTx[task_vk, :]))
            st_val = FTx[task_vk, xx2] - graph_x[task_vk, b1 + xx2] + 1
            ft_val = FTx[task_vk, xx2]
            graph_x[task_vk, b2 - 3 : b2] = [xx2 + 1, st_val, ft_val]

            st_idx, ft_idx, pu_idx = int(st_val) - 1, int(ft_val), xx2
            if st_idx >= 0 and ft_idx <= pu_idle_state.shape[0]:
                pu_idle_state[st_idx:ft_idx, pu_idx] = 0

            task_list_l.pop(task_vk_index)
            count += 1
            print(f"\rPETS Assign : {count} / {a1}", end="", flush=True)

    assign_task = np.column_stack((graph_x[:, b2 - 5], graph_x[:, b2 - 3 : b2]))
    final_makespan = int(np.max(graph_x[:, b2 - 3 : b2]))

    chk = np.zeros(a2, dtype=float)
    for i in range(a2):
        for j in range(a1):
            if assign_task[j, 1] == (i + 1):
                chk[i] += assign_task[j, 3] - assign_task[j, 2] + 1
    chk2 = pu_idle_state.shape[0] - np.sum(pu_idle_state, axis=0)
    is_valid = np.array_equal(chk, chk2)

    return final_makespan, is_valid, assign_task
    
# ==============================================================================
# ==============================================================================
def main():
    """Standalone test runner using a sample .mat dataset."""
    start_time = time.perf_counter()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(current_dir, "graph_x.mat")):
        graph_path = os.path.join(current_dir, "graph_x.mat")
        size_path = os.path.join(current_dir, "size_a.mat")
    elif os.path.exists(os.path.join(current_dir, "FFT_graph.mat")):
        graph_path = os.path.join(current_dir, "FFT_graph.mat")
        size_path = os.path.join(current_dir, "size_a.mat")
    else:
        sample_dir = os.path.join(os.path.dirname(current_dir), "datasets", "FFT_64")
        graph_path = os.path.join(sample_dir, "graph_x.mat")
        size_path = os.path.join(sample_dir, "size_a.mat")

    graph_x = sio.loadmat(graph_path)["graph_x"].astype(float)
    size_a = sio.loadmat(size_path)["size_a"]

    final_makespan, is_valid, assign_task = solve(graph_x.copy(), size_a.copy())
    elapsed = time.perf_counter() - start_time

    a1, a2 = int(size_a.item(0)), int(size_a.item(1))
    b1 = len(graph_x)
    a = graph_x[:, b1 : b1 + a2]

    max_sum_a = int(np.max(np.sum(a, axis=0)) + np.sum(graph_x[:, :b1])) + 500
    pu_idle_final = np.ones((max_sum_a, a2), dtype=int)
    for row in assign_task:
        pu_idx = int(row.item(1)) - 1
        st_idx = int(row.item(2)) - 1
        ft_idx = int(row.item(3))
        if st_idx >= 0 and ft_idx <= len(pu_idle_final):
            pu_idle_final[st_idx:ft_idx, pu_idx] = 0

    chk = np.zeros(a2, dtype=float)
    for i in range(a2):
        for j in range(a1):
            if assign_task[j, 1] == (i + 1):
                chk[i] += assign_task[j, 3] - assign_task[j, 2] + 1

    chk2 = len(pu_idle_final) - np.sum(pu_idle_final, axis=0)
    sumx = sum(graph_x[i, b1 + int(assign_task[i, 1]) - 1] for i in range(b1))

    print(f"\nFinal schedule length is equal : {final_makespan}")
    print(f"Elapsed time: {elapsed:.4f} seconds")
    print("\n--- Verification ---")
    print("Check array (chk)  :", chk)
    print("Idle state diff    :", chk2)
    print("Total exec time    :", sumx)
    print("Validation Status  :", "PASS" if is_valid else "FAIL")
    print("\nFinal Schedule (Task, PU, ST, FT):")
    print(assign_task.astype(int))


if __name__ == "__main__":
    main()

