import numpy as np
import cvxpy as cp

def compute_vertex_centroid(vertices):
    """
    Computes the centroid of a set of vertices.
    
    Args:
        vertices (list of np.ndarray): List of vertices, each vertex is a numpy array.
        
    Returns:
        np.ndarray: Centroid of the vertices.
    """
    if not vertices:
        raise ValueError("The list of vertices is empty.")
    
    centroid = np.zeros(vertices[0].shape)
    for vertex in vertices:
        centroid += vertex
    centroid /= len(vertices)
    return centroid


def solve_constrained_QP(regressor_matrix, observation_vector, Hp, hp):
    """
    Solves a Quadratic Programming (QP) problem with linear constraints.
    
    Args:
        regressor_matrix (np.ndarray): Regressor matrix.
        observation_vector (np.ndarray): Observation vector.
        Hp (np.ndarray): Linear constraint matrix.
        hp (np.ndarray): Constraint vector.
        initial_guess (np.ndarray): Initial estimate.
        
    Returns:
        np.ndarray: Optimal solution of the QP problem.
    """

    regressor_matrix = np.array(regressor_matrix, dtype=np.float64)
    observation_vector = np.array(observation_vector, dtype=np.float64).flatten()
    Hp = np.array(Hp, dtype=np.float64)
    hp = np.array(hp, dtype=np.float64).flatten()
    n_params = regressor_matrix.shape[1]
    
    # Decision variable
    theta = cp.Variable(n_params)
    
    # Objective function
    objective = cp.Minimize(cp.norm2(regressor_matrix @ theta - observation_vector) ** 2)
    
    # Constraints
    constraints = [Hp @ theta <= hp]
    
    # Solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve(warm_start=True)
    
    if problem.status != cp.OPTIMAL:
        raise ValueError("The QP problem does not have an optimal solution.")
    
    return theta.value


# def solve_regularized_constrained_QP(regressor_matrix, observation_vector, Hp, hp, initial_guess, regularization_parameter, decay_rate, expected_value):
#     """
#     Solves a Quadratic Programming (QP) problem with regularization.
    
#     Args:
#         regressor_matrix (np.ndarray): Regressor matrix.
#         observation_vector (np.ndarray): Observation vector.
#         Hp (np.ndarray): Linear constraint matrix.
#         hp (np.ndarray): Constraint vector.
#         initial_guess (np.ndarray): Initial estimate.
#         regularization_parameter (float): Regularization parameter.
#         decay_rate (float): Regularization decay rate.
#         expected_value (np.ndarray): Expected value for regularization.
        
#     Returns:
#         np.ndarray: Optimal solution of the regularized problem.
#     """
#     # Compute the smallest singular value
#     sigma_min = np.linalg.svd(regressor_matrix, compute_uv=False).min()
#     lambda_reg = regularization_parameter * np.exp(-decay_rate * sigma_min)
    
#     n_params = regressor_matrix.shape[1]
    
#     # Decision variable
#     theta = cp.Variable(n_params)
    
#     # Objective function with regularization
#     objective = cp.Minimize(
#         cp.norm2(regressor_matrix @ theta - observation_vector) ** 2 +
#         lambda_reg * cp.norm2(theta - expected_value) ** 2
#     )
    
#     # Constraints
#     constraints = [Hp @ theta <= hp]
    
#     # Solve the problem
#     problem = cp.Problem(objective, constraints)
#     problem.solve(warm_start=True)
    
#     if problem.status != cp.OPTIMAL:
#         raise ValueError("The regularized problem does not have an optimal solution.")
    
#     return theta.value

