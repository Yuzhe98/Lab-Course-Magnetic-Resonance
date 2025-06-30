import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Time-domain parameters
DW = 1 / 800  # dwell time
SI = 128  # size i.e. number of sampled points
timeStamp = DW * np.arange(SI)
s_t = np.sin(2 * np.pi * 50 * timeStamp + np.pi / 4)

# Frequency-domain computation
n = len(s_t)
frequencies = np.fft.fftfreq(n, d=DW)
fft_vals = np.fft.fft(s_t)
fft_vals_shifted = np.fft.fftshift(fft_vals)
frequencies_shifted = np.fft.fftshift(frequencies)

amplitude = np.abs(fft_vals_shifted)
phase = np.angle(fft_vals_shifted)

# Plotting setup
plt.rc("font", size=10)  # font size for all figures
plt.rc("figure", titlesize=10)  # Figure title
plt.rc("axes", titlesize=10)  # Axes title

fig = plt.figure(figsize=(8, 2), dpi=300)  # initialize a figure
gs = gridspec.GridSpec(nrows=1, ncols=3)  # create grid for multiple figures

ax00 = fig.add_subplot(gs[0, 0])
ax01 = fig.add_subplot(gs[0, 1])
ax02 = fig.add_subplot(gs[0, 2])
axs = [ax00, ax01, ax02]

# --- Left plot: Time-domain signal ---
axs[0].plot(timeStamp, s_t, color="tab:blue")
axs[0].set_xticks([0, 0.15])
axs[0].set_yticks([-1, 0, 1])
axs[0].set_title("Time Domain")
axs[0].set_xlabel("Time [s]")
axs[0].set_ylabel("Amplitude")

period = 1 / 50
arrow_left = (2 * np.pi + 1.25 * np.pi) / (2 * np.pi * 50)
arrow_right = (2 * np.pi + 3.25 * np.pi) / (2 * np.pi * 50)
# print((2 * np.pi * 50 * arrow_left + np.pi / 4) / np.pi)
# print((2 * np.pi * 50 * arrow_right + np.pi / 4) / np.pi)
print(np.sin(2 * np.pi * 50 * arrow_left + np.pi / 4))
print(np.sin(2 * np.pi * 50 * arrow_right + np.pi / 4))
# axs[0].annotate(
#     "",
#     xy=(arrow_right, -1),
#     xytext=(arrow_left, -1),
#     arrowprops=dict(arrowstyle="<->", lw=1),
# )
ax00.quiver(
    [period * 4, period * 4],
    [-1, -1],
    [0.5 * period, -0.5 * period],
    [0, 0],
    angles="xy",
    scale_units="xy",
    scale=1,
    color="black",
    width=0.02,
)


# --- Middle plot: Arrow with transform labels ---
axs[1].axis("off")
axs[1].annotate(
    "", xy=(0.9, 0.6), xytext=(-0.06, 0.6), arrowprops=dict(arrowstyle="->", lw=2)
)
axs[1].annotate(
    "", xy=(0.87, 0.5), xytext=(-0.11, 0.5), arrowprops=dict(arrowstyle="<-", lw=2)
)
axs[1].text(0.4, 0.65, "Fourier transform", ha="center", va="bottom")
axs[1].text(0.4, 0.45, "inverse Fourier transform", ha="center", va="top")

# --- Right plot: Frequency-domain representation ---
# axs[2].scatter(
#     frequencies_shifted,
#     amplitude,
#     marker="x",
#     color='tab:green',
#     label="Amplitude",
# )
axs[2].plot(
    frequencies_shifted,
    amplitude,
    # marker="x",
    color="tab:green",
    label="Amplitude",
)
# axs[2].set_xticks([50])
axs[2].set_yticks([0, 60])
axs[2].set_xlim(-40, 140)
axs[2].set_xlabel("Frequency [Hz]")
axs[2].set_ylabel("Amplitude")
axs[2].set_title("Frequency Domain")

# # Add phase on a twin y-axis
# ax2 = axs[2].twinx()
# ax2.plot(frequencies_shifted, phase, "r--", label="Phase")
# ax2.set_ylabel("Phase [rad]", color="red")
# ax2.tick_params(axis="y", labelcolor="red")

plt.show()
