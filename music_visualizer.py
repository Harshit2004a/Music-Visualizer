import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sounddevice as sd
import random

# Constants
CHUNK = 1024 * 4  # Number of audio samples per frame
RATE = 44100  # Sampling rate in Hz
NUM_POINTS = 100  # Number of points in the electrified line
SMOOTHING_FACTOR = 0.02  # Increased smoothness
ANIMATION_INTERVAL = 10  # Faster updates
FREQ_BANDS = np.linspace(20, 20000, NUM_POINTS)  # Different frequency bands

# Matplotlib figure setup
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#000000')  # Black background
ax.set_facecolor('#000000')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

# Line setup
x_vals = np.linspace(-1, 1, NUM_POINTS)
electrified_line, = ax.plot(x_vals, np.zeros(NUM_POINTS), lw=2, color='#00ffff')
previous_magnitudes = np.zeros(NUM_POINTS)

# Audio callback function
def audio_callback(indata, frames, time, status):
    global audio_data
    audio_data = indata[:, 0]  # Take only one channel

# Initialize audio stream
audio_data = np.zeros(CHUNK)
stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=RATE, blocksize=CHUNK)
stream.start()

# Update function for animation
def update(frame):
    global audio_data, previous_magnitudes
    fft_data = np.fft.rfft(audio_data)
    freqs = np.fft.rfftfreq(len(audio_data), 1.0 / RATE)
    magnitude = np.abs(fft_data)
    
    # Interpolate magnitude for selected frequency bands
    interpolated_magnitude = np.interp(FREQ_BANDS, freqs, magnitude, left=0, right=0)
    scaled_magnitude = np.interp(interpolated_magnitude, (0, max(interpolated_magnitude)), (-1, 1))
    
    # Apply increased smoothing
    smoothed_magnitude = (SMOOTHING_FACTOR * scaled_magnitude) + ((1 - SMOOTHING_FACTOR) * previous_magnitudes)
    previous_magnitudes = smoothed_magnitude
    
    # Add randomness for an electrified effect
    jitter = np.random.uniform(-0.03, 0.03, NUM_POINTS)  # Reduced jitter for smoothness
    electrified_line.set_ydata(smoothed_magnitude + jitter)
    electrified_line.set_color(random.choice(['#00ffff', '#ff00ff', '#ffff00', '#ff0000']))  # Random color shifts
    
    return electrified_line,

# Run animation with faster updates
ani = animation.FuncAnimation(fig, update, interval=ANIMATION_INTERVAL, blit=True)
plt.show()

# Cleanup
stream.stop()
