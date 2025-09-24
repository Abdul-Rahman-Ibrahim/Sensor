import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

# Parameters from the paper
pulses = 200                  # number of light pulses
pulse_duration = 0.01         # 10 ms pulse (s)
rate = 2.5                    # Hz, repetition rate

# OLED optical power densities (in mW/mm^2)
blue_intensity = 0.25         # Blue OLED avg intensity
orange_intensity = 0.10       # Orange OLED avg intensity

# Opsin thresholds (mW/mm^2)
ChR2_threshold = 0.1          # ~10% firing probability
ChR2_saturation = 0.5         # ~100% firing probability
ChRmine_threshold = 0.005     # 5 uW/mm^2
ChRmine_saturation = 0.05     # 50 uW/mm^2

# Default values
opsin = 'ChRmine'   # change to 'ChR2' for blue
led_intensity = orange_intensity if opsin == 'ChRmine' else blue_intensity
initial_distance = 0.05   # 50 µm

# Function to calculate spike probability
def get_spike_probability(intensity, opsin):
    if opsin == 'ChR2':
        return np.clip((intensity - ChR2_threshold) / (ChR2_saturation - ChR2_threshold), 0, 1)
    else:
        return np.clip((intensity - ChRmine_threshold) / (ChRmine_saturation - ChRmine_threshold), 0, 1)

# Function to simulate spiking
def simulate_spikes(distance, led_intensity, opsin):
    neuron_intensity = led_intensity / (distance**2 / 0.05**2)
    p_spike = get_spike_probability(neuron_intensity, opsin)
    spikes = np.random.rand(pulses) < p_spike
    return spikes, p_spike

# Create figure with two subplots
fig, (ax_raster, ax_prob) = plt.subplots(1, 2, figsize=(14, 4))
plt.subplots_adjust(left=0.25, bottom=0.35)

# Initial simulation
spikes, p_spike = simulate_spikes(initial_distance, led_intensity, opsin)
eventplot = ax_raster.eventplot(np.where(spikes)[0] / rate, colors='k')
ax_raster.set_xlabel('Time (s)')
ax_raster.set_ylabel('Neuron spikes')
ax_raster.set_title(f"Raster: {opsin}, distance={initial_distance*1000:.0f} µm, p_spike={p_spike:.2f}")

# Initial probability vs. distance curve
distances = np.linspace(0.01, 0.2, 100)
probs = [get_spike_probability(led_intensity / (d**2 / 0.05**2), opsin) for d in distances]
prob_line, = ax_prob.plot(distances*1000, probs, 'b-')
ax_prob.set_xlabel('Distance (µm)')
ax_prob.set_ylabel('Spike Probability')
ax_prob.set_ylim(0, 1)
ax_prob.set_title('Spike Probability vs. Distance')

# Slider axes
ax_distance = plt.axes([0.25, 0.2, 0.65, 0.03])
ax_intensity = plt.axes([0.25, 0.15, 0.65, 0.03])

# Radio button axis
ax_opsin = plt.axes([0.05, 0.4, 0.15, 0.15])

# Sliders
distance_slider = Slider(ax_distance, 'Distance (mm)', 0.01, 0.2, valinit=initial_distance, valstep=0.005)
intensity_slider = Slider(ax_intensity, 'LED Intensity (mW/mm^2)', 0.01, 0.5, valinit=led_intensity, valstep=0.01)

# Radio buttons for opsin selection
opsin_selector = RadioButtons(ax_opsin, ('ChR2', 'ChRmine'), active=1)

# Update function
def update(val):
    distance = distance_slider.val
    intensity = intensity_slider.val
    selected_opsin = opsin_selector.value_selected
    
    # Update raster plot
    spikes, p_spike = simulate_spikes(distance, intensity, selected_opsin)
    ax_raster.clear()
    ax_raster.eventplot(np.where(spikes)[0] / rate, colors='k')
    ax_raster.set_xlabel('Time (s)')
    ax_raster.set_ylabel('Neuron spikes')
    ax_raster.set_title(f"Raster: {selected_opsin}, distance={distance*1000:.0f} µm, p_spike={p_spike:.2f}")
    
    # Update probability vs. distance plot
    probs = [get_spike_probability(intensity / (d**2 / 0.05**2), selected_opsin) for d in distances]
    prob_line.set_ydata(probs)
    prob_line.set_xdata(distances*1000)
    ax_prob.relim()
    ax_prob.autoscale_view()
    ax_prob.set_ylim(0, 1)
    ax_prob.set_title('Spike Probability vs. Distance')
    
    fig.canvas.draw_idle()

# Connect widgets to update function
distance_slider.on_changed(update)
intensity_slider.on_changed(update)
opsin_selector.on_clicked(update)

plt.show()