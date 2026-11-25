import os
import math
import numpy as np
import scipy.io as sio


def generate_fft_dag(m, num_pus=3, ccr=1.0, comp_range=(6, 37), seed=42):
    """Generate a Fast Fourier Transformation (FFT) DAG with size m = 2^k."""

    rng = np.random.default_rng(seed)
    k = int(math.log2(m))
    assert 2**k == m

    num_rec_tasks = 2 * m - 1
    num_but_tasks = m * k
    v = num_rec_tasks + num_but_tasks

    b2 = v + num_pus + 5
    graph_x = np.zeros((v, b2), dtype=float)

    c_min, c_max = int(comp_range[0]), int(comp_range[1])
    comp_costs = rng.integers(c_min, c_max + 1, size=(v, num_pus))
    graph_x[:, v : v + num_pus] = comp_costs
    avg_comp = float(np.mean(comp_costs))

    mean_comm = max(1.0, avg_comp * ccr)
    comm_low = max(1, int(round(mean_comm * 0.4)))
    comm_high = max(comm_low + 1, int(round(mean_comm * 1.8)))

    graph_x[:, b2 - 5] = np.arange(1, v + 1, dtype=float)

    idx = 0
    for lv in range(1, k + 2):
        nodes_in_lv = 2 ** (lv - 1)
        for _ in range(nodes_in_lv):
            graph_x[idx, b2 - 4] = float(lv)
            if lv <= k:
                left_child = 2 * idx + 1
                right_child = 2 * idx + 2
                graph_x[idx, left_child] = float(rng.integers(comm_low, comm_high + 1))
                graph_x[idx, right_child] = float(rng.integers(comm_low, comm_high + 1))
            idx += 1

    for stage in range(k):
        lv = (k + 1) + (stage + 1)
        prev_start = (m - 1) + stage * m
        curr_start = (m - 1) + (stage + 1) * m
        stride = 2 ** (k - 1 - stage)

        for j in range(m):
            curr_node = curr_start + j
            graph_x[curr_node, b2 - 4] = float(lv)

            parent_1 = prev_start + j
            parent_2 = prev_start + (j ^ stride)

            graph_x[parent_1, curr_node] = float(rng.integers(comm_low, comm_high + 1))
            graph_x[parent_2, curr_node] = float(rng.integers(comm_low, comm_high + 1))

    size_a = np.array([[v, num_pus]], dtype=int)
    return graph_x, size_a


def generate_gaussian_dag(m, num_pus=3, ccr=0.5, calculation_weight=20, seed=42):
    """Generate a Gaussian Elimination DAG expanded recursively from the 14-node base matrix (Topcuoglu et al., 2002)."""
    
    assert m >= 5
    rng = np.random.default_rng(seed)

    b_rows = [
        "0 1 1 1 1 0 0 0 0 0 0 0 0 0",  # 1
        "0 0 0 0 0 1 0 0 0 0 0 0 0 0",  # 2
        "0 0 0 0 0 0 1 0 0 0 0 0 0 0",  # 3
        "0 0 0 0 0 0 0 1 0 0 0 0 0 0",  # 4
        "0 0 0 0 0 0 0 0 1 0 0 0 0 0",  # 5
        "0 0 0 0 0 0 1 1 1 0 0 0 0 0",  # 6
        "0 0 0 0 0 0 0 0 0 1 0 0 0 0",  # 7
        "0 0 0 0 0 0 0 0 0 0 1 0 0 0",  # 8
        "0 0 0 0 0 0 0 0 0 0 0 1 0 0",  # 9
        "0 0 0 0 0 0 0 0 0 0 1 1 0 0",  # 10
        "0 0 0 0 0 0 0 0 0 0 0 0 1 0",  # 11
        "0 0 0 0 0 0 0 0 0 0 0 0 0 1",  # 12
        "0 0 0 0 0 0 0 0 0 0 0 0 0 1",  # 13
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0",  # 14
    ]
    b = np.array([list(map(float, row.split())) for row in b_rows], dtype=float)

    v = (m * (m + 1)) // 2 - 1

    axr = np.maximum(1, rng.poisson(lam=calculation_weight, size=(v, num_pus))).astype(float)

    bxr = b.copy()
    for k in range(1, m - 5 + 1):
        mtr = k + 5
        new_size = bxr.shape[0] + mtr
        cxr = np.zeros((new_size, new_size), dtype=float)
        cxr[mtr:, mtr:] = bxr
        cxr[0, 1:mtr] = 1.0
        for r in range(1, mtr):
            cxr[r, r + mtr - 1] = 1.0
        bxr = cxr

    b1 = bxr.shape[0]

    lxr = -1 * np.ones(b1, dtype=int)
    for i in range(b1):
        (xp,) = np.where(bxr[:, i] != 0)
        if len(xp) == 0:
            lxr[i] = 1
        elif len(xp) == 1:
            lxr[i] = lxr[xp[0]] + 1
        else:
            max_lxr = int(np.max(lxr[xp]))
            lxr[i] = max_lxr + 1
            for p_idx in xp:
                lxr[p_idx] = lxr[i] - 1

    edges_i, edges_j = np.where(bxr == 1)
    for ei, ej in zip(edges_i, edges_j):
        w = round(ccr * rng.poisson(lam=calculation_weight))
        bxr[ei, ej] = float(max(1, w))

    sv = np.zeros((b1, 3), dtype=float)
    name = np.arange(1, b1 + 1, dtype=float).reshape(-1, 1)
    lxr_col = lxr.reshape(-1, 1).astype(float)

    graph_x = np.hstack([bxr, axr, name, lxr_col, sv])
    size_a = np.array([[v, num_pus]], dtype=int)
    return graph_x, size_a






def generate_montage_dag(num_images=20, num_pus=3, ccr=1.0, comp_range=(10, 50), seed=42):
    """Generate a Montage astronomical image mosaic workflow DAG with fork-join bottlenecks."""
    
    rng = np.random.default_rng(seed)
    c_min, c_max = int(comp_range[0]), int(comp_range[1])

    n0 = 1                           # Lv 1: Entry / Split Input
    n1 = num_images                  # Lv 2: mProjectPP
    n2 = max(2, num_images * 2)      # Lv 3: mDiffFit
    n3 = 1                           # Lv 4: mConcatFit (Bottleneck)
    n4 = 1                           # Lv 5: mBgModel (Bottleneck)
    n5 = num_images                  # Lv 6: mBackground
    n6 = max(2, num_images // 4)     # Lv 7: mImgTbl + mAdd
    n7 = 1                           # Lv 8: mShrink
    n8 = 1                           # Lv 9: mJPEG (Exit)

    level_counts = [n0, n1, n2, n3, n4, n5, n6, n7, n8]
    v = sum(level_counts)
    b2 = v + num_pus + 5
    graph_x = np.zeros((v, b2), dtype=float)

    comp_costs = rng.integers(c_min, c_max + 1, size=(v, num_pus))
    graph_x[:, v : v + num_pus] = comp_costs
    avg_comp = float(np.mean(comp_costs))

    mean_comm = max(1.0, avg_comp * ccr)
    comm_low = max(1, int(round(mean_comm * 0.5)))
    comm_high = max(comm_low + 1, int(round(mean_comm * 1.5)))

    graph_x[:, b2 - 5] = np.arange(1, v + 1, dtype=float)

    level_nodes = []
    curr = 0
    for lv_idx, cnt in enumerate(level_counts, start=1):
        nodes = list(range(curr, curr + cnt))
        for nd in nodes:
            graph_x[nd, b2 - 4] = float(lv_idx)
        level_nodes.append(nodes)
        curr += cnt

    entry_node = level_nodes[0][0]
    for nd1 in level_nodes[1]:
        graph_x[entry_node, nd1] = float(rng.integers(comm_low, comm_high + 1))

    # Lv 2 -> Lv 3 (mProjectPP -> mDiffFit)
    for idx_2, nd2 in enumerate(level_nodes[2]):
        p1 = level_nodes[1][idx_2 % n1]
        p2 = level_nodes[1][(idx_2 + 1) % n1]
        graph_x[p1, nd2] = float(rng.integers(comm_low, comm_high + 1))
        if p1 != p2:
            graph_x[p2, nd2] = float(rng.integers(comm_low, comm_high + 1))

    # Lv 3 -> Lv 4 (mDiffFit -> mConcatFit)
    concat_node = level_nodes[3][0]
    for nd2 in level_nodes[2]:
        graph_x[nd2, concat_node] = float(rng.integers(comm_low, comm_high + 1))

    # Lv 4 -> Lv 5 (mConcatFit -> mBgModel)
    bgmodel_node = level_nodes[4][0]
    graph_x[concat_node, bgmodel_node] = float(rng.integers(comm_low, comm_high + 1))

    for idx_5, nd5 in enumerate(level_nodes[5]):
        graph_x[bgmodel_node, nd5] = float(rng.integers(comm_low, comm_high + 1))
        graph_x[level_nodes[1][idx_5], nd5] = float(rng.integers(comm_low, comm_high + 1))

    # Lv 6 -> Lv 7 (mBackground -> mAdd)
    for idx_5, nd5 in enumerate(level_nodes[5]):
        target_6 = level_nodes[6][idx_5 % n6]
        graph_x[nd5, target_6] = float(rng.integers(comm_low, comm_high + 1))

    # Lv 7 -> Lv 8 -> Lv 9 (mAdd -> mShrink -> mJPEG)
    shrink_node = level_nodes[7][0]
    exit_node = level_nodes[8][0]
    for nd6 in level_nodes[6]:
        graph_x[nd6, shrink_node] = float(rng.integers(comm_low, comm_high + 1))
    graph_x[shrink_node, exit_node] = float(rng.integers(comm_low, comm_high + 1))

    size_a = np.array([[v, num_pus]], dtype=int)
    return graph_x, size_a



def generate_molecular_dag(num_stages=8, width=5, num_pus=3, ccr=1.0, comp_range=(10, 50), seed=42):
    """Generate a multi-stage Molecular Dynamics diamond-mesh DAG."""
    
    rng = np.random.default_rng(seed)
    c_min, c_max = comp_range

    level_counts = list((1,))
    for s in range(num_stages):
        level_counts.append(width if s % 2 == 0 else max(2, width - 1))
    level_counts.append(1)

    v = sum(level_counts)
    b2 = v + num_pus + 5
    graph_x = np.zeros((v, b2), dtype=float)

    comp_costs = rng.integers(int(c_min), int(c_max) + 1, size=(v, num_pus))
    graph_x[:, v : v + num_pus] = comp_costs
    avg_comp = float(np.mean(comp_costs))

    mean_comm = max(1.0, avg_comp * ccr)
    comm_low = max(1, int(round(mean_comm * 0.5)))
    comm_high = max(comm_low + 1, int(round(mean_comm * 1.5)))

    graph_x[:, b2 - 5] = np.arange(1, v + 1, dtype=float)

    level_nodes = []
    curr = 0
    for lv_idx, cnt in enumerate(level_counts, start=1):
        nodes = list(range(curr, curr + cnt))
        for nd in nodes:
            graph_x[nd, b2 - 4] = float(lv_idx)
        level_nodes.append(nodes)
        curr += cnt

    for lv_idx in range(len(level_nodes) - 1):
        curr_lv = level_nodes[lv_idx]
        next_lv = level_nodes[lv_idx + 1]
        if len(curr_lv) == 1:
            for u_node in curr_lv:
                for v_node in next_lv:
                    graph_x[u_node, v_node] = float(rng.integers(comm_low, comm_high + 1))
        elif len(next_lv) == 1:
            for u_node in curr_lv:
                for v_node in next_lv:
                    graph_x[u_node, v_node] = float(rng.integers(comm_low, comm_high + 1))
        else:
            for i, u_node in enumerate(curr_lv):
                targets = {next_lv[i % len(next_lv)], next_lv[(i + 1) % len(next_lv)]}
                for v_node in targets:
                    graph_x[u_node, v_node] = float(rng.integers(comm_low, comm_high + 1))

    size_a = np.array([[v, num_pus]], dtype=int)
    return graph_x, size_a




def generate_epigenomics_dag(num_branches=16, pipeline_len=5, num_pus=3, ccr=1.0, comp_range=(10, 50), seed=42):
    """Generate an Epigenomics bioinformatics workflow DAG with parallel pipelines."""
    
    rng = np.random.default_rng(seed)
    c_min, c_max = int(comp_range[0]), int(comp_range[1])

    v = 1 + (num_branches * pipeline_len) + 1
    b2 = v + num_pus + 5
    graph_x = np.zeros((v, b2), dtype=float)

    comp_costs = rng.integers(c_min, c_max + 1, size=(v, num_pus))
    graph_x[:, v : v + num_pus] = comp_costs
    avg_comp = float(np.mean(comp_costs))

    mean_comm = max(1.0, avg_comp * ccr)
    comm_low = max(1, int(round(mean_comm * 0.5)))
    comm_high = max(comm_low + 1, int(round(mean_comm * 1.5)))

    graph_x[:, b2 - 5] = np.arange(1, v + 1, dtype=float)

    graph_x[0, b2 - 4] = 1.0
    exit_node = v - 1
    graph_x[exit_node, b2 - 4] = float(pipeline_len + 2)

    idx = 1
    for _ in range(num_branches):
        prev_node = 0
        for step in range(pipeline_len):
            curr_node = idx
            graph_x[curr_node, b2 - 4] = float(step + 2)
            graph_x[prev_node, curr_node] = float(rng.integers(comm_low, comm_high + 1))
            prev_node = curr_node
            idx += 1
        graph_x[prev_node, exit_node] = float(rng.integers(comm_low, comm_high + 1))

    size_a = np.array([[v, num_pus]], dtype=int)
    return graph_x, size_a


def generate_laplace_dag(m=10, num_pus=3, ccr=1.0, comp_range=(10, 50), seed=42):
    """Generate a symmetric 2D wavefront Laplace equation solver DAG (v = m^2)."""
    
    rng = np.random.default_rng(seed)
    c_min, c_max = comp_range

    v = m * m
    b2 = v + num_pus + 5
    graph_x = np.zeros((v, b2), dtype=float)

    comp_costs = rng.integers(int(c_min), int(c_max) + 1, size=(v, num_pus))
    graph_x[:, v : v + num_pus] = comp_costs
    avg_comp = float(np.mean(comp_costs))

    mean_comm = max(1.0, avg_comp * ccr)
    comm_low = max(1, int(round(mean_comm * 0.5)))
    comm_high = max(comm_low + 1, int(round(mean_comm * 1.5)))

    graph_x[:, b2 - 5] = np.arange(1, v + 1, dtype=float)

    grid_to_id = {}
    idx = 0
    for diag in range(2 * m - 1):
        lv = diag + 1
        for r in range(m):
            c = diag - r
            if 0 <= c < m:
                grid_to_id[(r, c)] = idx
                graph_x[idx, b2 - 4] = float(lv)
                idx += 1

    for r in range(m):
        for c in range(m):
            u = grid_to_id[(r, c)]
            if r + 1 < m:
                v_down = grid_to_id[(r + 1, c)]
                graph_x[u, v_down] = float(rng.integers(comm_low, comm_high + 1))
            if c + 1 < m:
                v_right = grid_to_id[(r, c + 1)]
                graph_x[u, v_right] = float(rng.integers(comm_low, comm_high + 1))

    size_a = np.array([[v, num_pus]], dtype=int)
    return graph_x, size_a


def generate_bitonic_sort_dag(n=16, num_pus=3, ccr=1.0, comp_range=(10, 50), seed=42):
    """Generate a symmetric Bitonic sorting network DAG for n = 2^k inputs."""
    
    rng = np.random.default_rng(seed)
    c_min, c_max = comp_range
    k = int(math.log2(n))
    assert 2**k == n

    num_stages = (k * (k + 1)) // 2
    v = 1 + (num_stages * n) + 1
    b2 = v + num_pus + 5
    graph_x = np.zeros((v, b2), dtype=float)

    comp_costs = rng.integers(int(c_min), int(c_max) + 1, size=(v, num_pus))
    graph_x[:, v : v + num_pus] = comp_costs
    avg_comp = float(np.mean(comp_costs))

    mean_comm = max(1.0, avg_comp * ccr)
    comm_low = max(1, int(round(mean_comm * 0.5)))
    comm_high = max(comm_low + 1, int(round(mean_comm * 1.5)))

    graph_x[:, b2 - 5] = np.arange(1, v + 1, dtype=float)
    graph_x[0, b2 - 4] = 1.0
    exit_node = v - 1
    graph_x[exit_node, b2 - 4] = float(num_stages + 2)

    stage_idx = 0
    for step in range(1, k + 1):
        for stage in range(step, 0, -1):
            lv = stage_idx + 2
            curr_start = 1 + stage_idx * n
            stride = 2 ** (stage - 1)

            for i in range(n):
                curr_node = curr_start + i
                graph_x[curr_node, b2 - 4] = float(lv)

                if stage_idx == 0:
                    graph_x[0, curr_node] = float(rng.integers(comm_low, comm_high + 1))
                else:
                    prev_start = 1 + (stage_idx - 1) * n
                    p1 = prev_start + i
                    p2 = prev_start + (i ^ stride)
                    graph_x[p1, curr_node] = float(rng.integers(comm_low, comm_high + 1))
                    graph_x[p2, curr_node] = float(rng.integers(comm_low, comm_high + 1))

                if stage_idx == num_stages - 1:
                    graph_x[curr_node, exit_node] = float(rng.integers(comm_low, comm_high + 1))

            stage_idx += 1

    size_a = np.array([[v, num_pus]], dtype=int)
    return graph_x, size_a

def generate_lu_dag(m=8, num_pus=3, ccr=1.5, comp_range=(10, 50), seed=42):
    """Generate a Block LU Decomposition DAG with diagonal, panel, and trailing submatrix updates."""
    
    rng = np.random.default_rng(seed)
    c_min, c_max = comp_range

    v = sum(1 + 2 * (m - k) + (m - k) ** 2 for k in range(1, m)) + 1
    b2 = v + num_pus + 5
    graph_x = np.zeros((v, b2), dtype=float)

    comp_costs = rng.integers(int(c_min), int(c_max) + 1, size=(v, num_pus))
    graph_x[:, v : v + num_pus] = comp_costs
    avg_comp = float(np.mean(comp_costs))

    mean_comm = max(1.0, avg_comp * ccr)
    comm_low = max(1, int(round(mean_comm * 0.5)))
    comm_high = max(comm_low + 1, int(round(mean_comm * 1.5)))

    graph_x[:, b2 - 5] = np.arange(1, v + 1, dtype=float)

    idx = 0
    latest_block_node = {}

    for k in range(1, m):
        lv_diag = 3 * k - 2
        lv_panel = 3 * k - 1
        lv_update = 3 * k

        # 1. Diagonal factorization node D(k, k)
        diag_node = idx
        graph_x[diag_node, b2 - 4] = float(lv_diag)
        idx += 1

        if (k, k) in latest_block_node:
            prev_node = latest_block_node[(k, k)]
            graph_x[prev_node, diag_node] = float(rng.integers(comm_low, comm_high + 1))

        l_nodes = {}
        u_nodes = {}
        for i in range(k + 1, m + 1):
            ln = idx
            l_nodes[i] = ln
            graph_x[ln, b2 - 4] = float(lv_panel)
            idx += 1

            graph_x[diag_node, ln] = float(rng.integers(comm_low, comm_high + 1))
            if (i, k) in latest_block_node:
                graph_x[latest_block_node[(i, k)], ln] = float(rng.integers(comm_low, comm_high + 1))

        for j in range(k + 1, m + 1):
            un = idx
            u_nodes[j] = un
            graph_x[un, b2 - 4] = float(lv_panel)
            idx += 1

            graph_x[diag_node, un] = float(rng.integers(comm_low, comm_high + 1))
            if (k, j) in latest_block_node:
                graph_x[latest_block_node[(k, j)], un] = float(rng.integers(comm_low, comm_high + 1))

        for i in range(k + 1, m + 1):
            for j in range(k + 1, m + 1):
                an = idx
                graph_x[an, b2 - 4] = float(lv_update)
                idx += 1

                graph_x[l_nodes[i], an] = float(rng.integers(comm_low, comm_high + 1))
                graph_x[u_nodes[j], an] = float(rng.integers(comm_low, comm_high + 1))

                if (i, j) in latest_block_node:
                    graph_x[latest_block_node[(i, j)], an] = float(rng.integers(comm_low, comm_high + 1))

                latest_block_node[(i, j)] = an

    exit_node = idx
    graph_x[exit_node, b2 - 4] = float(3 * m - 2)
    graph_x[latest_block_node[(m, m)], exit_node] = float(rng.integers(comm_low, comm_high + 1))

    size_a = np.array([[v, num_pus]], dtype=int)
    return graph_x, size_a


def _save_dataset(datasets_dir, folder_name, graph_x, size_a):
    
    folder_path = os.path.join(datasets_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    size_b = np.array([[graph_x.shape[0], graph_x.shape[1]]], dtype=int)

    sio.savemat(
        os.path.join(folder_path, "graph_x.mat"),
        {"graph_x": graph_x},
        do_compression=True,
    )
    sio.savemat(os.path.join(folder_path, "size_a.mat"), {"size_a": size_a})
    sio.savemat(os.path.join(folder_path, "size_b.mat"), {"size_b": size_b})

    print(f"Created {folder_name:18s} -> Tasks: {size_a[0, 0]:5d}, PUs: {size_a[0, 1]}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(base_dir, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)

    NUM_PUS = 3 # number of processing units
    CCR = 0.5

    fft_sizes = [16, 32, 64, 128]
    print("--- Generating FFT Datasets ---")
    for m in fft_sizes:
        graph_x, size_a = generate_fft_dag(m=m, num_pus=NUM_PUS, ccr=CCR, seed=100 + m)
        _save_dataset(datasets_dir, f"FFT_{m}", graph_x, size_a)

    gaussian_sizes = [10, 20, 30, 40, 50, 60, 70]
    print("\n--- Generating Gaussian Datasets ---")
    for m in gaussian_sizes:
        graph_x, size_a = generate_gaussian_dag(m=m, num_pus=NUM_PUS, ccr=0.5, seed=200 + m)
        _save_dataset(datasets_dir, f"Gaussian_{m}", graph_x, size_a)


    montage_sizes = [20, 50, 100]
    print("\n--- Generating Montage Datasets ---")
    for imgs in montage_sizes:
        graph_x, size_a = generate_montage_dag(num_images=imgs, num_pus=NUM_PUS, ccr=2.0, seed=300 + imgs)
        _save_dataset(datasets_dir, f"Montage_{imgs}", graph_x, size_a)

    molecular_configs = [(8, 5), (12, 7)]
    print("\n--- Generating Molecular Datasets ---")
    for idx, (stages, width) in enumerate(molecular_configs, start=1):
        graph_x, size_a = generate_molecular_dag(
            num_stages=stages, width=width, num_pus=NUM_PUS, ccr=2.0, seed=400 + idx
        )
        _save_dataset(datasets_dir, f"Molecular_{stages}x{width}", graph_x, size_a)

    epigenomics_branches = [16, 32]
    print("\n--- Generating Epigenomics Datasets ---")
    for branches in epigenomics_branches:
        graph_x, size_a = generate_epigenomics_dag(
            num_branches=branches, pipeline_len=5, num_pus=NUM_PUS, ccr=2.0, seed=500 + branches
        )
        _save_dataset(datasets_dir, f"Epigenomics_{branches}", graph_x, size_a)

    

    Laplace_Grid = [10,15,30,50]    
    print("\n--- Generating Laplace Equation Solver Datasets ---")
    for m in Laplace_Grid:
        gx, sa = generate_laplace_dag(m=m, num_pus=NUM_PUS, ccr=1.5, seed=600 + m)
        _save_dataset(datasets_dir, f"Laplace_{m}x{m}", gx, sa)

    Bitonic_Sort = [16, 32 , 64 , 128]
    print("\n--- Generating Bitonic Sorting Network Datasets ---")
    for n in Bitonic_Sort:
        gx, sa = generate_bitonic_sort_dag(n=n, num_pus=NUM_PUS, ccr=1.5, seed=700 + n)
        _save_dataset(datasets_dir, f"Bitonic_{n}", gx, sa)

    print("\n--- Generating LU Decomposition Datasets ---")
    LU_Decomposition = [6,8,10,16]
    for m in LU_Decomposition:
        gx, sa = generate_lu_dag(m=m, num_pus=NUM_PUS, ccr=1.5, seed=800 + m)
        _save_dataset(datasets_dir, f"LU_{m}x{m}", gx, sa)



    print("\nAll datasets generated successfully in 'datasets/' folder!")
    
if __name__ == "__main__":
    main()
