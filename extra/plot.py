import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import gaussian_kde
from scipy.signal import correlate, correlation_lags


def plot_waveform(data, i, max_pos, fr):
    sns.lineplot(x="time", y="amp", data=data, errorbar='sd')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (μV)')
    df = pd.DataFrame(data)
    mean_waveform = df.groupby('time')['amp'].mean()
    # max_time = mean_waveform.abs().idxmax()
    max_time = mean_waveform.index.tolist()[max_pos]
    max_amp = mean_waveform.tolist()[max_pos]
    plt.plot(max_time, max_amp, 'ro')  # 'ro' is a red dot marker
    plt.text(max_time, max_amp, f'cluster{i}: {max_amp:.1f} μV {fr:.1f}/s', color='red', ha='left')


def plot_quality_metrics(snr, amp, fr, imp, nchan):
    assert len(amp) == len(snr) == len(fr) == len(imp), "# of clusters must equal"
    # snr = np.log10(snr)
    # amp = np.log10(amp)
    g = sns.JointGrid(x=amp, y=snr)
    scatter = g.ax_joint.scatter(amp, snr, c=fr, s=imp, cmap='viridis', alpha=.7)
    ax = g.ax_joint
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    g.plot_marginals(sns.kdeplot, fill=True, color='green')
    # sns.violinplot(x=snr, ax=g.ax_marg_x, color='green', fill=False)
    # sns.violinplot(y=fr, ax=g.ax_marg_y, color='green', fill=False)
    # sns.swarmplot(x=snr, ax=g.ax_marg_x, color='orange', alpha=0.7, size=3)
    # sns.swarmplot(y=fr, ax=g.ax_marg_y, color='orange', alpha=0.7, size=3)
    mean_x = np.mean(amp)
    mean_y = np.mean(snr)
    g.ax_marg_x.axvline(mean_x, color='blue', linestyle='--')
    g.ax_marg_y.axhline(mean_y, color='blue', linestyle='--')

    # Compute KDEs
    a = np.log2(amp)
    b = np.log2(snr)
    kde_x = gaussian_kde(a, bw_method='scott')
    kde_y = gaussian_kde(b, bw_method='scott')

    # Define the range of values to evaluate KDEs
    x_range = np.linspace(np.min(a), np.max(a), 1000)
    y_range = np.linspace(np.min(b), np.max(b), 1000)

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
    plt.colorbar(scatter, label='Firing Rate', cax=axins, orientation='horizontal',
                 ticks=np.linspace(0, max(fr), 5).round().clip(0, max(fr)))

    # Custom legend for point sizes
    sizes = [10, 50, 100]
    size_legend = ['100kΩ', '500kΩ', '1MΩ']  # Example sizes
    for s, size in zip(sizes, size_legend):
        g.ax_joint.scatter([], [], c='gray', alpha=0.6, s=s, label=size)


    g.ax_marg_x.set_xscale('log', base=2)
    g.ax_marg_y.set_yscale('log', base=2)
    # Add legend to the right bottom corner
    g.ax_joint.legend(scatterpoints=1, frameon=True, labelspacing=1, title="Impedance",
                      loc='lower right', bbox_to_anchor=(1, 0.02), borderpad=1)
    g.set_axis_labels("Amplitude (µV)", "SNR")

    g.ax_joint.text(1, 1, f'#cluster: {len(snr)} ({round(len(snr) / nchan, 1)}/channel)', color='blue',
                    ha='right', va='top', transform=g.ax_joint.transAxes)


def plot_auto_correlagrams(spikes, fs, bin_time=1, max_lag=50):
    trace = np.zeros(round(max(spikes)))
    trace[np.nonzero(np.round(spikes))] = 1
    # bin_factor = int(bin_time * fs / 1000)
    plt.acorr(trace)
    plt.xlabel('Lag (ms)')
    plt.ylabel('Relevance (a.u.)')
    # plt.ylim(0, ymax)
    # plt.xticks(ticks=np.linspace(0, max_lag / bin_time * 2, 5),
    #            labels=np.linspace(-max_lag, max_lag, 5))
    plt.tight_layout()


def plot_phase_locking():
    pass