import numpy as np

class HysExample:
    """
    Implements:
       f1(v, s) = (v / R) * (tanh(s) + 1)
       f2(v, s) = (1/tau) * (v - s^3 + s)
    State variable: s (scalar)
    """

    def __init__(self, R=1.0, tau=1.0, s0=0.0):
        self.R = float(R)
        self.tau = float(tau)
        self.s = float(s0)
        # history containers (optional)
        self.vhist = []
        self.ihist = []
        self.shist = []

    def f1(self, v, s):
        return (v / self.R) * (np.tanh(s) + 1.0)

    def f2(self, v, s):
        return (1.0 / self.tau) * (v - s**3 + s)

    def step(self, v, dt):
        """
        Integrate one timestep (explicit Euler).
        Returns current i(t) after update.
        """
        self.vhist.append(v)
        self.shist.append(self.s)
        dsdt = self.f2(v, self.s)
        self.s += dsdt * dt
        i = self.f1(v, self.s)
        self.ihist.append(i)
        return i

    def conductance(self, v_read=0.1):
        """
        Return small-signal conductance G = I/V at v_read
        """
        i = self.f1(v_read, self.s)
        return i / v_read if v_read != 0 else 0.0

def program_memristor(memristor_obj, target_conductance, programming_voltage, dt=1e-9, tolerance=1e-3, max_steps=1000):
    current_conductance = memristor_obj.conductance()
    print(f"Programming started. Target G: {target_conductance:.4f}")
    
    steps = 0
    while abs(current_conductance - target_conductance) > tolerance and steps < max_steps:
        # Apply the programming voltage to change the state
        memristor_obj.step(programming_voltage, dt)
        
        # Update current conductance and step count
        current_conductance = memristor_obj.conductance()
        steps += 1
        
    print(f"Programming finished in {steps} steps. Final G: {current_conductance:.4f}")


if __name__ == "__main__":
    m = HysExample()
    G = 0.5
    program_memristor(m, G, 1)