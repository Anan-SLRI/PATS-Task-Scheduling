import os
import time
import numpy as np
import scipy.io as sio

def EFTh(task_ni, graph_x, cpux, FTx, pu_idle_state):
    b1, b2 = graph_x.shape

    pre_task_tuple = np.where(graph_x[:b1, task_ni] != 0)
    pre_task = pre_task_tuple.item(0) if False else pre_task_tuple[0]

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

            _, _, graph_x_temp = insert_c(
                x_temp, pu_idle_state.copy(), assigne_task, graph_x_temp, b2
            )
            FTx_new = graph_x_temp[task_ni, b2 - 1]

            if FTx[task_ni, cpux] < FTx_new:
                FTx[task_ni, cpux] = FTx_new

    return FTx

    
# ==============================================================================
# ==============================================================================
def insert_c(xv, pu_idle_state, assigne_task, graph_x, b2):
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
            pu_idle_state, assigne_task, graph_x = insert_c(
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
            pu_idle_state, assigne_task, graph_x = insert_c(
                xv2, pu_idle_state, assigne_task, graph_x, b2
            )
    else:
        assigne_task[task_idx, :] = [task_val, pu_val, st_val, ft_val]
        graph_x[task_idx, b2 - 4 : b2] = [task_val, pu_val, st_val, ft_val]

    return pu_idle_state, assigne_task, graph_x


# ==============================================================================
# ==============================================================================
def EFTc(task_ni, graph_x, cpux, FTx, pu_idle_state):
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

            _, _, graph_x_temp = insert_c(
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
    l = graph_x[:, b2 - 4].copy()

    G_tmp = [int(x) - 1 for x in graph_x[:, b2 - 5]]
    AEC = np.zeros(b1, dtype=float)
    PredC = np.full(b1, -1, dtype=int)
    CP_count = []
    l_temp = l.copy()
    count = 0
    CP_set = np.zeros((b1, b1 + 2), dtype=int)

    while len(G_tmp) > 0:
        L_vi = np.zeros(b1, dtype=float)
        num_task_G_temp = len(G_tmp)

        for i in range(num_task_G_temp):
            curr_u = G_tmp[i]
            (succ,) = np.where(graph_x[curr_u, :b1] != 0)
            for s_node in succ:
                AEC[s_node] = np.mean(graph_x[s_node, b1 : b1 + a2])
                if L_vi[s_node] <= L_vi[curr_u] + AEC[s_node] + graph_x[curr_u, s_node]:
                    L_vi[s_node] = L_vi[curr_u] + AEC[s_node] + graph_x[curr_u, s_node]
                    PredC[s_node] = curr_u

        node = int(graph_x[G_tmp[-1], b2 - 5]) - 1
        CP_count.append(node)
        node_x = [node]

        while l_temp[node] != 1:
            node = PredC[node]
            CP_count.append(node)
            node_x.append(node)

        count += 1
        CP_node = node_x
        CP_node_1based = [x + 1 for x in CP_node]
        CP_set[count - 1, : len(CP_node) + 2] = [len(CP_node), -count] + CP_node_1based

        for nx in node_x:
            if nx in G_tmp:
                G_tmp.remove(nx)

        for g_node in G_tmp:
            (pre_task,) = np.where(graph_x[:b1, g_node] != 0)
            num_pre_task = len(pre_task)
            for p_task in pre_task:
                for cp_c in CP_count:
                    if p_task == cp_c:
                        num_pre_task -= 1
                if num_pre_task == 0:
                    l_temp[g_node] = 1

    num_active_rows = len(np.where(CP_set[:, 0] != 0)[0])
    num_active_cols = len(np.where(CP_set[0, :] != 0)[0])
    CP_set2 = CP_set[:num_active_rows, 2:num_active_cols]
    xv1, xv2 = CP_set2.shape
    CP_set3 = np.zeros((xv1, xv2), dtype=int)

    for i in range(xv1):
        nonzero_len = len(np.where(CP_set2[i, :] != 0)[0])
        k = 0
        for j in range(nonzero_len - 1, -1, -1):
            CP_set3[i, k] = CP_set2[i, j]
            k += 1

    Queue = [int(CP_set3[0, 0])]
    CP_set3[0, 0] = 0
    Queue_x = list(Queue)
    CCPs_count = 0

    while len(Queue) != b1:
        for i in range(xv1):
            CPX = []
            for j in range(xv2):
                if CP_set3[i, j] > 0:
                    target_task_idx = int(CP_set3[i, j]) - 1
                    (pre_task,) = np.where(graph_x[:b1, target_task_idx] != 0)
                    CT = len(pre_task)
                    for p_idx in pre_task:
                        p_val_1based = p_idx + 1
                        for q_val in Queue:
                            if p_val_1based == q_val:
                                CT -= 1
                    if CT == 0:
                        Queue.append(int(CP_set3[i, j]))
                        CPX.append(int(CP_set3[i, j]))
                        CP_set3[i, j] = 0

            if len(CPX) != 0:
                Queue_x.extend(CPX + [-1])
                CCPs_count += 1

    Queue_x_temp = list(Queue_x)
    FTx = np.zeros((a1, a2), dtype=float)
    FTx_temp = FTx.copy()

    count_assign = 0
    for _ in range(CCPs_count):
        CP_temp = []
        while Queue_x_temp[0] > 0:
            CP_temp.append(Queue_x_temp.pop(0) - 1)
        Queue_x_temp.pop(0)

        for j in range(a2):
            cpux = j
            pu_idle_state_temp = pu_idle_state.copy()
            graph_x_temp = graph_x.copy()

            for task_i in CP_temp:
                FTx_temp = EFTh(task_i, graph_x_temp, cpux, FTx_temp, pu_idle_state_temp)
                st_temp = FTx_temp[task_i, cpux] - graph_x_temp[task_i, b1 + cpux] + 1
                ft_temp = FTx_temp[task_i, cpux]
                graph_x_temp[task_i, b2 - 3 : b2] = [cpux + 1, st_temp, ft_temp]

                st_idx, ft_idx = int(st_temp) - 1, int(ft_temp)
                if st_idx >= 0 and ft_idx <= pu_idle_state_temp.shape[0]:
                    pu_idle_state_temp[st_idx:ft_idx, cpux] = 0
                
                
                

        last_task_in_cp = CP_temp[-1]
        min2 = int(np.argmin(FTx_temp[last_task_in_cp, :]))

        for task_i in CP_temp:
            st_final = FTx_temp[task_i, min2] - graph_x[task_i, b1 + min2] + 1
            ft_final = FTx_temp[task_i, min2]
            graph_x[task_i, b2 - 3 : b2] = [min2 + 1, st_final, ft_final]

            st_f_idx, ft_f_idx = int(st_final) - 1, int(ft_final)
            if st_f_idx >= 0 and ft_f_idx <= pu_idle_state.shape[0]:
                pu_idle_state[st_f_idx:ft_f_idx, min2] = 0
                
            count_assign += 1
            print(f"\rCEFT Assign : {count_assign} / {a1}", end="", flush=True)        

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


