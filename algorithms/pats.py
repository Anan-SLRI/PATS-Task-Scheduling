import os
import time
import numpy as np
import scipy.io as sio



def run_EST(
    level_x,
    graph_x,
    EST_table,
    EST_table_u_temp,
    u_temp,
    pu_idle_state,
    a1,
    a2,
    b1,
    b2,
    l,
):
    b1, b2 = graph_x.shape


    (x,) = np.where(graph_x[:, b2 - 4] == level_x)
    x_min = int(np.min(x))
    x_max = int(np.max(x))


    pre_task = []
    for i in range(x_min, x_max + 1):
        task_1based = i + 1
        pre_task.append(-task_1based)
        (preds,) = np.where(graph_x[:b1, i] != 0)
        for p in preds:
            pre_task.append(int(p + 1))

    size_pre2 = len(pre_task)
    pux1, pux2 = pu_idle_state.shape

    for i in range(size_pre2):
        if size_pre2 == 1:
            t_idx = (-1 * pre_task.item(i) if hasattr(pre_task, 'item') else -1 * pre_task[i]) - 1
            EST_table[t_idx, :] = 1.0
            EST_table_u_temp[t_idx, :] = 1.0
        else:
            curr_val = pre_task[i]
            if curr_val < 0:
                curr_task_1based = -1 * curr_val
                curr_task_idx = curr_task_1based - 1

                for pu in range(a2):
                    pu_1based = pu + 1
                    j = i + 1
                    if j >= size_pre2:
                        j = 0

                    while (pre_task[j] > 0) and (j < size_pre2):
                        pred_task_1based = pre_task[j]
                        pred_task_idx = pred_task_1based - 1


                        if graph_x[pred_task_idx, b2 - 3] == pu_1based:
                            delay_x = graph_x[pred_task_idx, b2 - 1] + 1.0
                        else:
                            delay_x = (
                                graph_x[pred_task_idx, b2 - 1]
                                + graph_x[pred_task_idx, curr_task_idx]
                                + 1.0
                            )

                        delay_idx = int(delay_x) - 1
                        if pu_idle_state[delay_idx, pu] != 1:
                            (next_slot,) = np.where(
                                pu_idle_state[delay_idx:pux1, pu] != 0
                            )
                            delay_x = delay_x + int(next_slot.item(0))

                        if EST_table[curr_task_idx, pu] < delay_x:
                            EST_table[curr_task_idx, pu] = delay_x
                            EST_table_u_temp[curr_task_idx, pu] = delay_x

                            u_temp[curr_task_idx, pu] = pred_task_1based

                        j += 1
                        if j >= size_pre2:
                            j = 0

    return EST_table, EST_table_u_temp, u_temp



def run_EFT(
    level_x,
    graph_x,
    EST_table,
    pu_idle_state,
    assigne_task,
    k,
    a1,
    a2,
    b1,
    b2,
    l,
):
    b1, b2 = graph_x.shape
    cpu = a2

    (task_l,) = np.where(graph_x[:, b2 - 4] == level_x)
    tsl1 = len(task_l)

    EFT_table = -1.0 * np.ones((a1, a2))


    for t_idx in task_l:
        EFT_table[t_idx, :] = EST_table[t_idx, :] + graph_x[t_idx, b1 : b1 + cpu] - 1.0

    for i in range(tsl1):
        max_val = np.max(EFT_table)
        max_rows, _ = np.where(EFT_table == max_val)
        max1_idx = int(max_rows.item(0))

        min_val = np.min(EFT_table[max1_idx, :])
        (min_cols,) = np.where(EFT_table[max1_idx, :] == min_val)
        min2_idx = int(min_cols.item(0))

        st_val = EST_table[max1_idx, min2_idx]
        ft_val = st_val + graph_x[max1_idx, b1 + min2_idx] - 1.0
        x_temp = np.array([min2_idx + 1, st_val, ft_val, max1_idx + 1])

        st_idx = int(st_val) - 1
        ft_idx = int(ft_val)
        slot_x = np.sum(pu_idle_state[st_idx:ft_idx, min2_idx])
        diff_slot_x = int(ft_val - st_val + 1) - slot_x

        if diff_slot_x != 0:
            task_l_1based = task_l + 1
            (
                pu_idle_state,
                assigne_task,
                graph_x,
                k,
                task_l_1based,
            ) = insert_pats(
                x_temp,
                pu_idle_state,
                assigne_task,
                graph_x,
                b2,
                slot_x,
                diff_slot_x,
                k,
                task_l_1based,
            )
            task_l = task_l_1based - 1
        else:
            graph_x[max1_idx, b2 - 3 : b2] = [min2_idx + 1, st_val, ft_val]
            print(f"\rPATS Assign : {k} / {b1}", end="", flush=True)
            assigne_task = np.column_stack(
                (graph_x[:, b2 - 5], graph_x[:, b2 - 3 : b2])
            )
            k += 1
            pu_idle_state[st_idx:ft_idx, min2_idx] = 0
            task_l = task_l[task_l != max1_idx]


        if i != (tsl1 - 1):
            for t_rem_idx in task_l:
                dur = int(graph_x[t_rem_idx, b1 + min2_idx])
                curr_st = int(EST_table[t_rem_idx, min2_idx])

                search_idx = max(0, curr_st - 1)
                while np.sum(pu_idle_state[search_idx : search_idx + dur, min2_idx]) < dur:
                    search_idx += 1
                new_st = float(search_idx + 1)
                EST_table[t_rem_idx, min2_idx] = new_st
                EFT_table[t_rem_idx, min2_idx] = new_st + dur - 1.0

        EFT_table[max1_idx, :] = -1.0
        EST_table[max1_idx, :] = -1.0

    return graph_x, pu_idle_state, assigne_task, k, EST_table


def insert_pats(
    xv,
    pu_idle_state,
    assigne_task,
    graph_x,
    b2,
    slot_x,
    diff_slot_x,
    k,
    task_l,
):
    """Find the earliest contiguous idle slot on the target PU without shifting already-scheduled tasks."""
    b1, b2 = graph_x.shape

    pu_1based, st_1based, ft_1based, task_1based = xv
    pu_idx = int(pu_1based) - 1
    task_idx = int(task_1based) - 1
    dur = int(graph_x[task_idx, b1 + pu_idx])


    search_idx = max(0, int(st_1based) - 1)
    max_len = len(pu_idle_state)
    while search_idx + dur <= max_len:
        if np.sum(pu_idle_state[search_idx : search_idx + dur, pu_idx]) == dur:
            break
        search_idx += 1

    new_st = float(search_idx + 1)
    new_ft = new_st + dur - 1.0

    graph_x[task_idx, b2 - 3 : b2] = [pu_1based, new_st, new_ft]
    print(f"\rPATS Assign : {k} / {b1}", end="", flush=True)
    assigne_task = np.column_stack((graph_x[:, b2 - 5], graph_x[:, b2 - 3 : b2]))
    k += 1
    pu_idle_state[search_idx : search_idx + dur, pu_idx] = 0
    task_l = task_l[task_l != task_1based]

    return pu_idle_state, assigne_task, graph_x, k, task_l



def run_EST2(
    ss_idx,
    graph_x,
    assign_table_temp_2,
    EST_table_u_temp,
    max_pu,
    a1,
    a2,
    b1,
    b2,
):
    (pre_task,) = np.where(graph_x[:b1, ss_idx] != 0)
    size_pre2 = len(pre_task)

    ss_pu_idx = int(assign_table_temp_2[ss_idx, 1]) - 1
    delay_max = 0.0

    if size_pre2 == 0:
        EST_table_u_temp[ss_idx, :] = 1.0
        delay_max = 1.0
    else:
        for p_idx in pre_task:
            if assign_table_temp_2[p_idx, 1] == assign_table_temp_2[ss_idx, 1]:
                delay_x = assign_table_temp_2[p_idx, 3] + 1.0
            else:
                delay_x = (
                    assign_table_temp_2[p_idx, 3]
                    + graph_x[p_idx, ss_idx]
                    + 1.0
                )

            if delay_x <= max_pu[ss_pu_idx]:
                delay_x = max_pu[ss_pu_idx] + 1.0

            if delay_max < delay_x:
                delay_max = delay_x

    EST_table_u_temp[ss_idx, ss_pu_idx] = delay_max
    return EST_table_u_temp

def _check_no_overlap(schedule_mat, num_pus):

    for pu in range(1, num_pus + 1):
        tasks_on_pu = schedule_mat[schedule_mat[:, 1] == pu]
        if len(tasks_on_pu) > 1:

            sorted_tasks = tasks_on_pu[np.argsort(tasks_on_pu[:, 2])]

            if np.any(sorted_tasks[1:, 2] <= sorted_tasks[:-1, 3]):
                return False
    return True

def _check_valid_schedule(schedule_mat, graph_x, a2, b1):


    for pu in range(1, a2 + 1):
        tasks_on_pu = schedule_mat[schedule_mat[:, 1] == pu]
        if len(tasks_on_pu) > 1:
            sorted_tasks = tasks_on_pu[np.argsort(tasks_on_pu[:, 2])]
            if np.any(sorted_tasks[1:, 2] <= sorted_tasks[:-1, 3]):
                return False


    edges_i, edges_j = np.where(graph_x[:, :b1] > 0)
    for u_idx, v_idx in zip(edges_i, edges_j):
        pu_u, ft_u = schedule_mat[u_idx, 1], schedule_mat[u_idx, 3]
        pu_v, st_v = schedule_mat[v_idx, 1], schedule_mat[v_idx, 2]
        comm_cost = 0 if pu_u == pu_v else graph_x[u_idx, v_idx]
        if st_v < ft_u + comm_cost + 1:
            return False

    return True

    
def solve(graph_x, size_a):
    """Run Predict and Arrange Task Scheduling (PATS): Level-based EFT scheduling + upward idle reduction."""
    a1, a2 = int(size_a.item(0)), int(size_a.item(1))
    b1, b2 = graph_x.shape
    a = graph_x[:, b1 : b1 + a2]

    max_sum_a = int(np.max(np.sum(a, axis=0)) + np.sum(graph_x[:, :b1])) + 500
    pu_idle_state = np.ones((max_sum_a, a2), dtype=int)

    l = graph_x[:, b2 - 4].astype(int)

    assigne_task = np.zeros((a1, 4))
    EST_table = -1.0 * np.ones((a1, a2))
    EST_table_u_temp = -1.0 * np.ones((a1, a2))
    u_temp = -1 * np.ones((a1, a2), dtype=int)
    k = 1

    # Phase 1: Level-based Predictive Scheduling
    max_level = int(np.max(l))
    for level_x in range(1, max_level + 1):
        EST_table, EST_table_u_temp, u_temp = run_EST(
            level_x, graph_x, EST_table, EST_table_u_temp, u_temp, pu_idle_state, a1, a2, b1, b2, l
        )
        graph_x, pu_idle_state, assigne_task, k, EST_table = run_EFT(
            level_x, graph_x, EST_table, pu_idle_state, assigne_task, k, a1, a2, b1, b2, l
        )

    assign_table = np.column_stack((graph_x[:, b2 - 5], graph_x[:, b2 - 3 : b2]))
    phase1_table = assign_table.copy()

    max_val = np.max(graph_x[:, b2 - 3 : b2])
    lt1_rows, _ = np.where(graph_x[:, b2 - 3 : b2] == max_val)
    LT1 = int(lt1_rows.item(0))

    u = np.zeros(b1, dtype=int)
    for uvi in range(b1):
        pu_assigned = int(graph_x[uvi, b2 - 3]) - 1
        u[uvi] = u_temp[uvi, pu_assigned]

    # Phase 2: Upward Idle Slot Reduction
    k_curr = LT1
    assign_table_temp = assign_table.copy()
    visited_tasks = set()

    while k_curr >= 0 and l[k_curr] != 1:
        if k_curr in visited_tasks:
            break
        visited_tasks.add(k_curr)

        schedule_length = np.max(assign_table_temp[:, 1:4])
        (pu_k,) = np.where(assign_table_temp[:, 1] == assign_table_temp[k_curr, 1])

        late_task = np.array([0.0, 0.0, 0.0, 0.0])
        for idx_p in pu_k:
            if (
                assign_table_temp[idx_p, 3] > late_task.item(3)
                and assign_table_temp[idx_p, 3] < assign_table_temp[k_curr, 3]
            ):
                late_task = assign_table_temp[idx_p, :].copy()

        late_task_id = int(late_task.item(0))
        if late_task_id != 0:
            late_idx = late_task_id - 1
            xd1 = graph_x[k_curr, b2 - 2] - graph_x[late_idx, b2 - 1] - 1
            possible_start = graph_x[late_idx, b2 - 1] + 1
        else:
            xd1 = graph_x[k_curr, b2 - 2] - 1
            possible_start = 1.0

        uk_val = int(u[k_curr])
        if xd1 <= 0 and uk_val > 0:
            if late_task_id == 0:
                break
            k_curr = late_task_id - 1
        elif uk_val > 0:
            uk_idx = uk_val - 1
            pu_k_idx = int(assign_table_temp[k_curr, 1]) - 1

            if EST_table_u_temp[uk_idx, pu_k_idx] <= possible_start:
                assign_table_temp_2 = assign_table_temp.copy()
                new_pu = assign_table_temp_2[k_curr, 1]
                new_ft = possible_start + graph_x[uk_idx, b1 + int(new_pu) - 1] - 1
                assign_table_temp_2[uk_idx, :] = [uk_val, new_pu, possible_start, new_ft]

                (shiftable_set,) = np.where(
                    (assign_table_temp_2[:, 2] >= possible_start)
                    & (assign_table_temp_2[:, 0] != uk_val)
                )
                assign_table_temp_shiftable_set = assign_table_temp[shiftable_set, :].copy()

                max_pu = np.zeros(a2)
                max_pu[int(new_pu) - 1] = new_ft

                for _ in range(len(shiftable_set)):
                    ss2 = int(np.argmin(assign_table_temp_shiftable_set[:, 2]))
                    ss_id = int(assign_table_temp_shiftable_set[ss2, 0])
                    ss_idx = ss_id - 1

                    EST_table_u_temp = run_EST2(
                        ss_idx, graph_x, assign_table_temp_2, EST_table_u_temp, max_pu, a1, a2, b1, b2
                    )

                    ss_pu_idx = int(assign_table_temp_2[ss_idx, 1]) - 1
                    assign_table_temp_2[ss_idx, 2] = EST_table_u_temp[ss_idx, ss_pu_idx]
                    assign_table_temp_2[ss_idx, 3] = (
                        assign_table_temp_2[ss_idx, 2] + graph_x[ss_idx, b1 + ss_pu_idx] - 1
                    )
                    max_pu[ss_pu_idx] = assign_table_temp_2[ss_idx, 3]

                    assign_table_temp_shiftable_set = np.delete(assign_table_temp_shiftable_set, ss2, axis=0)

                if (
                    schedule_length > np.max(assign_table_temp_2[:, 1:4])
                    and _check_valid_schedule(assign_table_temp_2, graph_x, a2, b1)
                ):
                    assign_table_temp = assign_table_temp_2.copy()
                    graph_x[:, b2 - 3 : b2] = assign_table_temp_2[:, 1:4]

            k_curr = uk_idx
        else:
            break

    # Final validation check before returning
    assign_task = np.column_stack((graph_x[:, b2 - 5], graph_x[:, b2 - 3 : b2]))
    if not _check_valid_schedule(assign_task, graph_x, a2, b1):
        assign_task = phase1_table
        graph_x[:, b2 - 3 : b2] = phase1_table[:, 1:4]

    final_makespan = int(np.max(assign_task[:, 1:4]))

    # Verification
    pu_idle_final = np.ones_like(pu_idle_state)
    max_rows = len(pu_idle_final)
    for row in assign_task:
        pu_idx = int(row.item(1)) - 1
        st_idx = int(row.item(2)) - 1
        ft_idx = int(row.item(3))
        if st_idx >= 0 and ft_idx <= max_rows:
            pu_idle_final[st_idx:ft_idx, pu_idx] = 0

    chk = np.zeros(a2, dtype=float)
    for i in range(a2):
        for j in range(a1):
            if assign_task[j, 1] == (i + 1):
                chk[i] += assign_task[j, 3] - assign_task[j, 2] + 1
    chk2 = max_rows - np.sum(pu_idle_final, axis=0)
    is_valid = np.array_equal(chk, chk2) and _check_valid_schedule(assign_task, graph_x, a2, b1)

    return final_makespan, is_valid, assign_task






    
# ==============================================================================
# PATS (Main Script)
# ==============================================================================

def main():
    """Standalone test runner for PATS using a sample .mat dataset."""
    start_time = time.perf_counter()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Default to datasets/FFT_64 if FFT_graph.mat is not in the same directory
    local_graph = os.path.join(current_dir, "graph_x.mat")
    if os.path.exists(local_graph):
        graph_path = local_graph
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
    b1 = graph_x.shape.item(0) if hasattr(graph_x.shape, "item") else graph_x.shape[0]
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

