import numpy as np

def equation(Ef: np.ndarray, epsilon: np.ndarray, theta: np.ndarray, r: np.ndarray, params: np.ndarray) -> np.ndarray:
    """ Mathematical function for the dipole moment

    Args:
        Ef: A numpy array representing observations of the electric field.
        epsilon: A numpy array representing observations of the electric constant or permittivity of the medium.
        theta: A numpy array representing observations of the angle between the dipole axis and the position vector.
        r: A numpy array representing observations of the distance from the dipole to the point where the electric field is being measured.
        params: Array of numeric constants or parameters to be optimized

    Return:
        A numpy array representing the dipole moment as the result of applying the mathematical function to the inputs.
    """
    output = params[0] * Ef * (2 * np.pi * epsilon * r**params[1]) / (np.cos(theta * params[2]) + 1e-6)
    return output