import numpy as np

def compute_discrete_function_terms_single_step_euler(
    previous_state_measurement, 
    previous_control_input, 
    autonomous_function, 
    input_function
):
    """
    Single step Euler integration for input map g(u) and autonomous map f(x).
    Computes: x_{k+1} = x_k + dt*f(x_k) + dt*g(u_k)
    
    Args:
        previous_state_measurement (np.ndarray): Previous state (x_k-1).
        previous_control_input (np.ndarray): Previous control input (u_k-1).
        autonomous_function: Function f(x) for the autonomous map.
        input_function: Function g(u) for the input map.

    Returns:
        tuple: (f_discrete, g_discrete, x_discrete)
            - f_discrete (np.ndarray): Discrete autonomous map f(x_k).
            - g_discrete (np.ndarray): Discrete input map g(u_k).
            - x_discrete (np.ndarray): Updated state x_{k+1}.
    """
    controller_timestep = 0.2
    integration_timestep = 0.01

    # Initialize variables
    x_discrete = np.zeros_like(previous_state_measurement)
    f_discrete = np.zeros_like(previous_state_measurement)
    g_discrete = np.zeros_like(previous_state_measurement)

    # Compute f_discrete
    f_discrete = previous_state_measurement + integration_timestep * autonomous_function
    
    # Compute g_discrete
    g_discrete = integration_timestep * input_function

    # Compute x_discrete
    x_discrete = f_discrete + g_discrete

    return f_discrete, g_discrete, x_discrete