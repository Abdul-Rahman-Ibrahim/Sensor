import numpy as np
import matplotlib.pyplot as plt

# The model is based on the following two equations:
# 1. i(t) = f1(v(t), s(t))
# 2. ds(t)/dt = f2(v(t), s(t))

# f1(v, s) = (v/R) * (tanh(s) + 1)
# f2(v, s) = (1/tau) * (v - s**3 + s)


class Memristor:

    def __init__(self, R=1.0, tau=1.0, initial_s=0.0):
        """
        Initializes the Memristor model with its parameters.

        Args:
            R (float): The resistance parameter from the f1 equation. Defaults to 1.0.
            tau (float): The time constant from the f2 equation. Defaults to 1.0.
            initial_s (float): The initial value of the internal state variable s. Defaults to 0.0.
        """
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
    
    def get_conductance(self):
        return (np.tanh(self.s) + 1)/self.R

    def simulate_step(self, v, dt):
        """
        Simulates one time step of the memristor's behavior.

        Args:
            v (float): The voltage across the device at this time step.
            dt (float): The duration of the time step.

        Returns:
            float: The current through the device at this time step.
        """

        self.v_history.append(v)
        self.s_history.append(self.s)
        

        ds_dt = self.f2(v, self.s)
        

        self.s += ds_dt * dt
        

        current = self.f1(v, self.s)
        

        self.i_history.append(current)
        
        return current

    def get_history(self):
        """
        Returns the stored history of voltage, state, and current.
        """
        return {
            "voltage": np.array(self.v_history),
            "state": np.array(self.s_history),
            "current": np.array(self.i_history),
        }



class NeuralNetwork:
    def __init__(self, weight_lists, R=1.0, tau=1.0):
        """
        Initialize a memristor-based neural network.

        Args:
            weight_lists (list of np.ndarray): List of weight matrices.
            R (float): Memristor resistance parameter.
            tau (float): Memristor time constant.
        """
        self.R = R
        self.tau = tau
        self.layers = []
        self.G_targets = []


        for weight in weight_lists:
            M, N = weight.shape
            layer_mems = [[Memristor(R=R, tau=tau, initial_s=0.0) for _ in range(N)] for _ in range(M)]
            self.layers.append(layer_mems)


            layer_G_targets = np.zeros_like(weight)
            for i in range(M):
                for j in range(N):
                    w = weight[i, j]
                    G_target = self.w_to_G_mapper(w)
                    layer_G_targets[i, j] = G_target
            self.G_targets.append(layer_G_targets)

    def w_to_G_mapper(self, w):

        return 2.0 / (1.0 + np.exp(-w))

    def program_weights(self, dt=0.001, max_steps=50000):

        for l, layer in enumerate(self.layers):
            M, N = len(layer), len(layer[0])
            for i in range(M):
                for j in range(N):
                    mem = layer[i][j]
                    G_target = self.G_targets[l][i, j]
                    program_to_conductance_ffpi(mem, G_target, dt=dt, max_steps=max_steps)

    def readout_weights(self):

        weights = []
        for layer in self.layers:
            M, N = len(layer), len(layer[0])
            W = np.zeros((M, N))
            for i in range(M):
                for j in range(N):
                    mem = layer[i][j]
                    G = mem.get_conductance()
                    # invert mapper: G → w
                    w = np.log((2.0 / G) - 1.0) * -1.0
                    W[i, j] = w
            weights.append(W)
        return weights


def s_from_G(G, R):

    arg = R * G - 1.0
    if arg <= -1.0 or arg >= 1.0:
        raise ValueError(f"G out of achievable range for R={R}. arg={arg}")
    return np.arctanh(arg)

def program_to_conductance_ffpi(mem, G_target,
                                dt=0.001,
                                max_steps=200000,
                                tol_G=1e-4,
                                Kp=5.0,
                                Ki=50.0,
                                v_min=-5.0,
                                v_max=5.0,
                                hold_after=0.05):

    R = mem.R


    try:
        s_target = s_from_G(G_target, R)
    except ValueError as e:
        raise


    v_ff = s_target**3 - s_target

    # PI state
    integ = 0.0
    last_time_in_tol = None
    steps = 0
    hold_steps = int(max(1, round(hold_after / dt)))

    for step in range(max_steps):
        steps += 1
        G = mem.get_conductance()
        err_G = G_target - G

        if abs(err_G) < tol_G:
            if last_time_in_tol is None:
                last_time_in_tol = step
            elif (step - last_time_in_tol) >= hold_steps:
                break
        else:
            last_time_in_tol = None

        eps = 1e-12
        arg = np.clip(R*G - 1.0, -1.0 + 1e-9, 1.0 - 1e-9)
        s_meas = np.arctanh(arg)

        err_s = s_target - s_meas
        integ += err_s * dt

        integ = np.clip(integ, -10.0, 10.0)

        v_cmd = v_ff + Kp * err_s + Ki * integ

        v_cmd = float(np.clip(v_cmd, v_min, v_max))

        mem.simulate_step(v_cmd, dt)

    return mem.get_conductance(), steps

def test_memristor_equations():
    mem = Memristor(R=1.0, tau=1.0, initial_s=0.0)

    v = 1.0
    s = 0.5

    i_expected = (v / mem.R) * (np.tanh(s) + 1)
    dsdt_expected = (1 / mem.tau) * (v - s**3 + s)

    assert np.isclose(mem.f1(v, s), i_expected)
    assert np.isclose(mem.f2(v, s), dsdt_expected)
    print("✅ Memristor f1/f2 tests passed.")


def test_programming():
    for G_target in [0.1, 0.5, 1.0, 1.5, 1.9]:
        mem = Memristor(R=1.0, tau=1.0, initial_s=0.0)
        G_final, steps = program_to_conductance_ffpi(mem, G_target, tol_G=1e-4)
        print(f"Target={G_target}, Final={G_final}, Steps={steps}")
        assert abs(G_final - G_target) < 1e-3
    print("✅ Programming test passed.")

def test_mapping():
    nn = NeuralNetwork([np.array([[0.0, 1.0], [-1.0, 2.0]])])
    W_orig = nn.G_targets[0]

    for w, G in zip(nn.layers[0][0], W_orig[0]):
        s = s_from_G(G, nn.R)
        G_back = (np.tanh(s) + 1)/nn.R
        assert np.isclose(G, G_back, atol=1e-8)

    print("✅ Mapping test passed.")

def test_nn_cycle():
    w1 = np.array([[0.0, 0.5], [-0.5, 1.0]])
    w2 = np.array([[1.5, -1.0]])
    nn = NeuralNetwork([w1, w2])

    nn.program_weights()
    W_read = nn.readout_weights()

    print("Original weights:")
    print(w1)
    print(w2)
    print("Readout weights:")
    print(W_read[0])
    print(W_read[1])

    assert np.allclose(w1, W_read[0], atol=0.2)
    assert np.allclose(w2, W_read[1], atol=0.2)
    print("✅ Full NN cycle test passed.")





# Example usage to demonstrate the hysteresis behavior:
if __name__ == "__main__":
    # --- Run one programming example and log conductance history ---
    mem = Memristor(R=1.0, tau=1.0, initial_s=0.0)
    G_target = 1.5

    # storage for plotting
    conductance_trace = []
    steps = 20000

    for _ in range(steps):
        G = mem.get_conductance()
        conductance_trace.append(G)
        program_to_conductance_ffpi(
            mem, G_target,
            dt=0.001,
            max_steps=1,   # run one step at a time
            tol_G=1e-6,
            Kp=8.0,
            Ki=80.0,
            v_min=-3.0,
            v_max=3.0,
            hold_after=0.0
        )

    # --- Plot ---
    plt.figure(figsize=(6,4))
    plt.plot(conductance_trace, label="G(t)")
    plt.axhline(G_target, color='r', linestyle='--', label="Target $G^*$")
    plt.xlabel("Programming steps")
    plt.ylabel("Conductance G")
    plt.title("Programming Dynamics of Memristor")
    plt.legend()
    plt.grid(True)
    plt.show()
