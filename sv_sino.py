import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from caas_jupyter_tools import display_dataframe_to_user

# Inverse mapping s(v) for s^3 + s = v (unique real root)
def s_from_v(v):
    return np.cbrt(v/2 + np.sqrt((v**2)/4 + 1/27)) + np.cbrt(v/2 - np.sqrt((v**2)/4 + 1/27))

# Time and input parameters
fs = 5000               # sampling frequency (Hz)
T = 1.0                 # duration (seconds)
t = np.linspace(0, T, int(fs*T), endpoint=False)
f = 5.0                 # sinusoid frequency (Hz)
omega = 2*np.pi*f

# Two amplitudes to illustrate small and large amplitude behavior
A_small = 0.5
A_large = 2.0

v_small = A_small * np.sin(omega * t)
v_large = A_large * np.sin(omega * t)

s_small = s_from_v(v_small)
s_large = s_from_v(v_large)

# Save a sample of the data for inspection and provide as CSV
df = pd.DataFrame({
    "t": t,
    "v_small": v_small,
    "s_small": s_small,
    "v_large": v_large,
    "s_large": s_large
})
csv_path = "/mnt/data/sine_s_vs_v.csv"
df.to_csv(csv_path, index=False)

# Plot 1: input vs time (small amp)
plt.figure(figsize=(8,3))
plt.plot(t, v_small, label="v(t) = 0.5*sin(2π·5t)")
plt.plot(t, s_small, label="s(t) from cubic inverse", linestyle='--')
plt.xlabel("time (s)")
plt.ylabel("signal")
plt.title("Small amplitude: input v(t) and output s(t)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 2: input vs time (large amp)
plt.figure(figsize=(8,3))
plt.plot(t, v_large, label="v(t) = 2.0*sin(2π·5t)")
plt.plot(t, s_large, label="s(t) from cubic inverse", linestyle='--')
plt.xlabel("time (s)")
plt.ylabel("signal")
plt.title("Large amplitude: input v(t) and output s(t)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Function to compute and plot single-sided amplitude spectrum
def plot_spectrum(x, fs, title):
    N = len(x)
    X = np.fft.rfft(x * np.hanning(N))
    freqs = np.fft.rfftfreq(N, 1/fs)
    amp = np.abs(X) / (N/2)
    plt.figure(figsize=(8,3))
    plt.semilogy(freqs, amp)  # log scale to show harmonics clearly
    plt.xlim(0, 200)
    plt.xlabel("frequency (Hz)")
    plt.ylabel("amplitude (linear, log-plot)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot spectra
plot_spectrum(v_small, fs, "Spectrum of v(t) (A=0.5)")
plot_spectrum(s_small, fs, "Spectrum of s(t) (A=0.5)")

plot_spectrum(v_large, fs, "Spectrum of v(t) (A=2.0)")
plot_spectrum(s_large, fs, "Spectrum of s(t) (A=2.0)")

# Display a small table of first few samples
display_dataframe_to_user("Sample of time-domain signals", df.head(40))

csv_path

