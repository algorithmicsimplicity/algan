from algan.taichi_compat import ti


@ti.kernel
def _query_state_from_edits(
        times: ti.types.ndarray(),  # [T], times at which to materialize the state vector
        head: ti.types.ndarray(),  # [N + 1]
        sorted_edit_ids: ti.types.ndarray(),  # [U] (int32)
        edit_timestamps: ti.types.ndarray(),  # [E]
        sorted_values: ti.types.ndarray(),  # [U, D]
        out: ti.types.ndarray()  # [T, N, D]
):
    for t, j in ti.ndrange(times.shape[0], head.shape[0] - 1):
        query_time = times[t]
        start = head[j]
        end = head[j + 1]

        # Binary search for the smallest timestamp > query_time
        low = start
        high = end
        while low < high:
            mid = (low + high) // 2
            edit_id = sorted_edit_ids[mid]
            edit_time = edit_timestamps[edit_id]
            if edit_time > query_time:
                high = mid
            else:
                low = mid + 1

        D = sorted_values.shape[1]
        if low < end:
            for d in range(D):
                out[t, j, d] = sorted_values[low, d]
        else:
            for d in range(D):
                out[t, j, d] = 0.0


@ti.kernel
def _query_selected_state_from_edits(
        times: ti.types.ndarray(),  # [T]
        rows: ti.types.ndarray(),  # [R], global attribute-buffer row ids
        head: ti.types.ndarray(),  # [N + 1]
        sorted_edit_ids: ti.types.ndarray(),  # [U] (int32)
        edit_timestamps: ti.types.ndarray(),  # [E]
        sorted_values: ti.types.ndarray(),  # [U, D]
        out: ti.types.ndarray()  # [T, N, D]
):
    """Query only ``rows`` while preserving the full global-row output layout.

    Animated functions address attributes through stable global row ids, so a
    compact ``[T, R, D]`` result would require remapping every replayed
    operation.  Writing selected rows into the ordinary ``[T, N, D]`` buffer
    retains that API while avoiding the edit-history search for mobs that
    cannot contribute to the current render window.
    """
    for t, r in ti.ndrange(times.shape[0], rows.shape[0]):
        j = rows[r]
        query_time = times[t]
        start = head[j]
        end = head[j + 1]

        low = start
        high = end
        while low < high:
            mid = (low + high) // 2
            edit_id = sorted_edit_ids[mid]
            edit_time = edit_timestamps[edit_id]
            if edit_time > query_time:
                high = mid
            else:
                low = mid + 1

        D = sorted_values.shape[1]
        if low < end:
            for d in range(D):
                out[t, j, d] = sorted_values[low, d]
        else:
            for d in range(D):
                out[t, j, d] = 0.0

