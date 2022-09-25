import numpy as np

lib = np.ctypeslib.load_library('Supervoxel_utils/main.so', './')
c_test = lib.main
c_test.restype = None
c_test.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=2),
]

def compute_sv(input_pos, n_sp):
    """
    Generate supervoxel.

    ----------
    Input:
        input_pos (float): point positions [N, 3]
        n_sp (int): desired supervoxel number
    -------
    Returns:
        output_label (int): supervoxel label for each point
    """
    input_pos = input_pos.astype(np.float32)
    num_input_points = input_pos.shape[0]

    output_label = np.random.rand(num_input_points)
    output_label = output_label.astype(np.int32)

    num_points = np.array([num_input_points, n_sp])
    num_points = num_points.astype(np.int32)

    output_color = np.random.rand(num_input_points, 3)
    output_color = output_color.astype(np.int32)

    c_test(input_pos, num_points, output_label, output_color)
    return output_label
