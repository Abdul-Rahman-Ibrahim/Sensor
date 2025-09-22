import numpy as np

#-----------------------------------------------------------------------------------
# Core Memristor Model (from the original script)
#-----------------------------------------------------------------------------------
class Memristor:
    """
    A class to represent a well-posed memristor model based on the hys_example from the paper.
    """
    def __init__(self, R=1.0, tau=1.0, initial_s=0.0):
        self.R = R
        self.tau = tau
        self.s = initial_s
        self.v_history = []
        self.s_history = []
        self.i_history = []
        
    def f1(self, v, s):
        """
        Calculates the current (i) based on voltage (v) and state (s).
        This is the algebraic current-voltage relationship.
        """
        return (v / self.R) * (np.tanh(s) + 1)

    def f2(self, v, s):
        """
        Calculates the rate of change of the state variable (ds/dt) based on voltage (v) and state (s).
        This is the differential equation for the internal state.
        """
        return (1 / self.tau) * (v - s**3 + s)

    def simulate_step(self, v, dt):
        """
        Simulates one time step of the memristor's behavior. This is crucial for
        "programming" the memristor over time to a desired state.
        """
        self.v_history.append(v)
        self.s_history.append(self.s)
        
        ds_dt = self.f2(v, self.s)
        
        self.s += ds_dt * dt
        
        current = self.f1(v, self.s)
        
        self.i_history.append(current)
        
        return current

    def get_conductance(self):
        """
        Returns the current effective conductance of the memristor.
        G = 1/R_eff = (tanh(s) + 1) / R
        """
        return (np.tanh(self.s) + 1) / self.R

#-----------------------------------------------------------------------------------
# Memristor Programming Simulation
#-----------------------------------------------------------------------------------
def program_memristor(memristor_obj, target_conductance, programming_voltage, dt=1e-9, tolerance=1e-3, max_steps=1000):
    """
    Simulates applying voltage pulses to program a memristor to a target conductance.

    Args:
        memristor_obj (Memristor): The memristor object to program.
        target_conductance (float): The desired final conductance.
        programming_voltage (float): The voltage of the programming pulses.
        dt (float): The time step for each pulse.
        tolerance (float): The acceptable error from the target conductance.
        max_steps (int): Maximum number of simulation steps to prevent infinite loops.
    """
    current_conductance = memristor_obj.get_conductance()
    print(f"Programming started. Target G: {target_conductance:.4f}")
    
    steps = 0
    while abs(current_conductance - target_conductance) > tolerance and steps < max_steps:
        # Apply the programming voltage to change the state
        memristor_obj.simulate_step(programming_voltage, dt)
        
        # Update current conductance and step count
        current_conductance = memristor_obj.get_conductance()
        steps += 1
        
    print(f"Programming finished in {steps} steps. Final G: {current_conductance:.4f}")

#-----------------------------------------------------------------------------------
# Neural Network Layer Implementation
#-----------------------------------------------------------------------------------
class NeuralNetwork:
    """
    Simulates a multi-layer neural network using memristor arrays.
    Each layer uses a differential pair of memristor arrays to represent
    both positive and negative weights.
    """
    def __init__(self, weights_list, R=1.0, s_max=2.0):
        """
        Initializes the neural network with a list of weight matrices.

        Args:
            weights_list (list): A list of 2D numpy arrays, one for each layer's weights.
            R (float): The resistance parameter for each memristor.
            s_max (float): Scaling factor for the initial 's' values.
        """
        self.layers = []
        for weights in weights_list:
            # Split the weight matrix into positive and negative components
            W_plus = np.maximum(0, weights)
            W_minus = np.maximum(0, -weights)
            
            M, N = weights.shape
            
            # Create a pair of memristor arrays for the layer
            mem_array_plus = np.empty((M, N), dtype=object)
            mem_array_minus = np.empty((M, N), dtype=object)
            
            # Program each memristor in the positive array
            if np.max(W_plus) > 0:
                scaled_s_plus = s_max * (W_plus / np.max(W_plus))
            else:
                scaled_s_plus = np.zeros_like(W_plus)
                
            for i in range(M):
                for j in range(N):
                    mem_array_plus[i, j] = Memristor(R=R, initial_s=scaled_s_plus[i, j])

            # Program each memristor in the negative array
            if np.max(W_minus) > 0:
                scaled_s_minus = s_max * (W_minus / np.max(W_minus))
            else:
                scaled_s_minus = np.zeros_like(W_minus)
                
            for i in range(M):
                for j in range(N):
                    mem_array_minus[i, j] = Memristor(R=R, initial_s=scaled_s_minus[i, j])
            
            self.layers.append({
                "plus": mem_array_plus,
                "minus": mem_array_minus
            })

    def predict(self, input_vector, dt=1e-9):
        """
        Performs a full forward pass through the neural network.

        Args:
            input_vector (np.array): The initial input vector.
            dt (float): Simulation time step for each matrix-vector multiplication.

        Returns:
            np.array: The final output vector of the network.
        """
        current_input = input_vector
        
        for i, layer in enumerate(self.layers):
            print(f"\nProcessing Layer {i+1}...")
            M, N = layer['plus'].shape
            
            # Perform matrix-vector multiplication for the positive array
            plus_output = np.zeros(M)
            for m in range(M):
                for n in range(N):
                    v_in = current_input[n]
                    current = layer['plus'][m, n].simulate_step(v_in, dt)
                    plus_output[m] += current
                    
            # Perform matrix-vector multiplication for the negative array
            minus_output = np.zeros(M)
            for m in range(M):
                for n in range(N):
                    v_in = current_input[n]
                    current = layer['minus'][m, n].simulate_step(v_in, dt)
                    minus_output[m] += current
            
            # The final layer output is the difference between the two arrays' outputs
            # This is the essence of the differential pair.
            raw_output = plus_output - minus_output
            
            # Apply a non-linear activation function (e.g., ReLU)
            # This is typically done with external CMOS circuitry.
            current_input = np.maximum(0, raw_output) 
            
            print(f"Layer {i+1} Raw Output: {raw_output}")
            print(f"Layer {i+1} Activated Output: {current_input}")
            
        return current_input

#-----------------------------------------------------------------------------------
# Example usage:
#-----------------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Part 1: Simulating Programming an Individual Memristor ---
    print("--- Part 1: Simulating Memristor Programming ---")
    my_memristor = Memristor(R=1.0, initial_s=0.0)
    target_G = 1.8 
    program_voltage = 1.0 
    program_memristor(my_memristor, target_G, program_voltage, dt=0.01)
    print("--- End of Part 1 ---")
    
    # -----------------------------------------------------------
    # --- Part 2: Simulating a Multi-Layer Neural Network ---
    print("\n--- Part 2: Simulating Multi-Layer Neural Network ---")

    # Define sample weights for a two-layer neural network
    # Layer 1: 4 inputs, 3 hidden neurons
    # Layer 2: 3 hidden neurons, 2 output neurons
    weights1 = np.array([
        [0.8, -1.2, 0.5, 2.0],
        [1.5, 0.9, -1.1, 0.4],
        [0.2, 0.7, 1.8, -1.3]
    ])
    
    weights2 = np.array([
        [1.0, -0.5, 2.0],
        [-0.8, 1.5, 0.2]
    ])
    
    # Create the neural network from the list of weight matrices
    print("Initializing neural network with two layers...")
    nn = NeuralNetwork(weights_list=[weights1, weights2])
    print("Neural network successfully created.")
    
    # Define an arbitrary input vector for the network (4x1)
    input_voltages = np.array([0.5, 1.0, 0.2, 0.8])
    print("\nInput Voltages:\n", input_voltages)
    
    # Perform a prediction on the entire network
    final_output = nn.predict(input_voltages)
    
    print("\n--- Final Prediction Result ---")
    print(final_output)
    print("-------------------------------")