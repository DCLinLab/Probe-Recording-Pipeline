import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import gaussian_kde


def plot_waveform(data):
    sns.lineplot(x="time", y="amp", data=data)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (mV)')
    df = pd.DataFrame(data)
    mean_waveform = df.groupby('time')['amp'].mean()
    max_time = mean_waveform.idxmax()
    max_amp = mean_waveform.max()
    plt.plot(max_time, max_amp, 'ro')  # 'ro' is a red dot marker
    plt.text(max_time, max_amp, f'Max: {max_amp:.2f} mV', color='red', ha='right')


def plot_quality_metrics(snr, fr, peak, imp, nclust):
    g = sns.JointGrid(x=snr, y=fr)
    scatter = g.ax_joint.scatter(snr, fr, c=peak, s=imp, cmap='viridis')
    g.plot_marginals(sns.kdeplot, fill=True, color='green')
    # sns.violinplot(x=snr, ax=g.ax_marg_x, color='green', fill=False)
    # sns.violinplot(y=fr, ax=g.ax_marg_y, color='green', fill=False)
    # sns.swarmplot(x=snr, ax=g.ax_marg_x, color='orange', alpha=0.7, size=3)
    # sns.swarmplot(y=fr, ax=g.ax_marg_y, color='orange', alpha=0.7, size=3)
    mean_x = np.mean(snr)
    mean_y = np.mean(fr)
    g.ax_marg_x.axvline(mean_x, color='blue', linestyle='--')
    g.ax_marg_y.axhline(mean_y, color='blue', linestyle='--')

    # Compute KDEs
    kde_x = gaussian_kde(snr, bw_method='scott')
    kde_y = gaussian_kde(fr, bw_method='scott')

    # Define the range of values to evaluate KDEs
    x_range = np.linspace(np.min(snr), np.max(snr), 1000)
    y_range = np.linspace(np.min(fr), np.max(fr), 1000)

    # Evaluate KDEs
    kde_x_values = kde_x(x_range)
    kde_y_values = kde_y(y_range)
    # Find maxima
    max_kde_x = np.max(kde_x_values)
    max_kde_y = np.max(kde_y_values)
    # Add text labels for mean values
    g.ax_marg_x.text(mean_x, max_kde_x, f'avg={mean_x:.2f}', color='blue', fontsize=12, ha='left', va='bottom')
    g.ax_marg_y.text(max_kde_y, mean_y, f'avg={mean_y:.2f}', color='blue', fontsize=12, ha='left', va='top',
                     rotation=270)

    axins = inset_axes(g.ax_joint,
                       width="30%",  # width of the colorbar
                       height="5%",  # height of the colorbar
                       loc='upper left',  # location of the colorbar
                       borderpad=1)
    plt.colorbar(scatter, label='Amplitude', cax=axins, orientation='horizontal')
    plt.colorbar(scatter, cax=axins, orientation='horizontal', label='Impedance',
                 ticks=np.linspace(min(peak), max(peak), 5).round(decimals=1))

    # Custom legend for point sizes
    sizes = [10, 50, 100]
    size_legend = ['100kΩ', '500kΩ', '1MΩ']  # Example sizes
    for s, size in zip(sizes, size_legend):
        g.ax_joint.scatter([], [], c='gray', alpha=0.6, s=s, label=size)

    # Add legend to the right bottom corner
    g.ax_joint.legend(scatterpoints=1, frameon=True, labelspacing=1, title="Impedance",
                      loc='lower right', bbox_to_anchor=(1, 0.02), borderpad=1)
    g.set_axis_labels("SNR", "Firing Rate")

    g.ax_joint.text(1, 1, f'#cluster: {sum(nclust)} ({round(sum(nclust) / len(nclust), 1)}/channel)', color='blue',
                    ha='right', va='top', transform=g.ax_joint.transAxes)

    plt.tight_layout()


def plot_auto_correlagrams(spikes, fs, bin_time=1, nbin_half=50):
    spikes = sorted(spikes)
    auto = [0] * (nbin_half + 1)
    bin_freq = fs * bin_time / 1000
    for i in spikes:
        for j in spikes:
            pos = round(abs(i - j) / bin_freq)
            if pos < len(auto):
                auto[pos] += 1
            else:
                break
    ymax = max(auto[1:])
    auto = np.array(auto[:0:-1] + auto) / 2
    sns.barplot(data=auto)
    plt.xlabel('Lag (ms)')
    plt.ylabel('Relevance (a.u.)')
    plt.ylim(0, ymax)
    plt.xticks(ticks=np.linspace(0, nbin_half * 2, 5),
               labels=np.linspace(-nbin_half * bin_time, nbin_half * bin_time, 5))
    plt.tight_layout()


def plot_phase_locking():
    pass