import numpy as np

def mat_distance(mat1, mat2, mode="Euclidean"):
    """
    mode: "Manhattan", "Euclidean"
    """
    if mode=="Euclidean":
        return np.sum((mat1-mat2)**2, axis=-1)**0.5
    elif mode=="Manhattan":
        return np.sum(np.abs(mat1-mat2), axis=-1)
    else:
        raise ValueError("Unrecognized distance mode: "+mode)
