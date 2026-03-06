"""
This code plots the extended abstract figure using the npy values
"""
import os
import numpy as np
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'''
\usepackage{amsmath}
\usepackage{sfmath}
\renewcommand{\rmdefault}{cmss}
'''
params = {
    'text.usetex': True,
    'font.size': 9,
    'font.family': 'sans-serif',
    'axes.unicode_minus': False
}
plt.rcParams.update(params)

def p_to_stars(p):
    """helper: convert p-value to stars"""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'


def add_sig(ax, x1, x2, y, text, line_h=0.005):
    """Draw a bracket between x1 and x2 at height y and write text above it."""
    ax.plot([x1, x1, x2, x2], [y, y + line_h, y + line_h, y], lw=0.5, color='k')
    ax.text((x1 + x2) * 0.5, y+0.001, text, ha='center', va='bottom', fontsize=9)


# ################################################ Define the panel here ###############################################
textwidth_in_inches = 10  # Or fetch from LaTeX if you know the document class

# 4 Panels:
fig = plt.figure(figsize=(textwidth_in_inches, 6.5))

# 3 rows, 12 columns grid
gs = fig.add_gridspec(
    nrows=2,
    ncols=24,
    height_ratios=[1, 1],  # 5) Made second row shorter
    hspace=0.35,
    wspace=0.5,
    left=0.08,
    right=0.98,
    top=0.92,
    bottom=0.08
)

# ---------- Row 1 ----------
# Panel A
ax_1a = fig.add_subplot(gs[0, :6])
gs_a_right = gs[0, 8:].subgridspec(nrows=1, ncols=3, wspace=0.1)
ax_2a = []
for i in range(3):
    ax_2a.append(fig.add_subplot(gs_a_right[i]))

# ---------- Row 2 ----------
# Panel B (50% width)
ax_b = fig.add_subplot(gs[1, :17])

# Panel C
ax_c = fig.add_subplot(gs[1, 18:])

# gs_c_right = gs[2, 10:].subgridspec(nrows=1, ncols=3, wspace=0.1)
# ax_2c = []
# for i in range(3):
#     ax_2c.append(fig.add_subplot(gs_c_right[i]))


# ############################################## Panel A ###############################################################
y_tac_n0_c0_wocm = np.load(
    os.path.join('results', 'regression', 'wo-cm', 'aligned_set', 'global_error_tac_y_n0_c0.npy'))
y_tac_n0_c0_cm = np.load(os.path.join('results', 'regression', 'w-cm', 'late', 'aligned_set', 'global_error_tac_y_n0_c0.npy'))

results = {
    'wo-CM': np.mean(y_tac_n0_c0_wocm, axis=1),
    'w-CM': np.mean(y_tac_n0_c0_cm, axis=1)
}
alpha = 0.05
methods = list(results.keys())
ref_method = 'w-CM'
n_props = results[ref_method].shape[1]
n_methods = len(methods)

test_results = {}
for m in methods:
    if m == ref_method:
        continue
    test_results[m] = {'pvals': np.ones(n_props), 'symbol': ['ns'] * n_props}
    for prop_idx in range(n_props):
        x = results[ref_method][:, prop_idx]
        y = results[m][:, prop_idx]
        assert x.shape[0] == y.shape[0], "Paired tests require same number of runs."
        stat, p = stats.ttest_rel(x, y, nan_policy='omit')
        test_results[m]['pvals'][prop_idx] = p
        test_results[m]['symbol'][prop_idx] = p_to_stars(p)

# Print summary
print("Significance (w-CM vs others) per property:")
for m in methods:
    if m == ref_method: continue
    print(f"\nComparing w-CM vs {m}:")
    for prop_idx in range(n_props):
        p = test_results[m]['pvals'][prop_idx]
        star = test_results[m]['symbol'][prop_idx]
        mean_ref = results[ref_method][:, prop_idx].mean()
        mean_m = results[m][:, prop_idx].mean()
        print(f"  Property {prop_idx}: p={p:.4g}, {star}, means: {ref_method}={mean_ref:.4f}, {m}={mean_m:.4f}")

target_tac_outputs = [y_tac_n0_c0_wocm, y_tac_n0_c0_cm]

methods_label = [r'wo-CM', r'w-CM']
methods_color = [
    "#CC5308",  # Orange
    "#377EB8"  # Blue
]

tac_properties = ['Stiffness', 'Mass', 'Surf.\nFriction']  # 3) Two-line label

bar_width = 0.4
group_spacing = 0.80
num_subgroups, num_groups = 2, 3
group_labels = tac_properties
subgroup_labels = methods_label

object_error = np.zeros((len(methods_label), len(tac_properties)))
object_error_std = np.zeros((len(methods_label), len(tac_properties)))

# Intrinsic Properties
for i in range(len(target_tac_outputs)):
    method = target_tac_outputs[i]
    mean_error = np.mean(method, axis=0)
    for prop in range(len(tac_properties)):
        object_error[i, prop] = np.mean(mean_error[:, prop])
        object_error_std[i, prop] = np.std(mean_error[:, prop])

# Compute positions
indices = np.arange(num_groups) * (num_subgroups * bar_width + group_spacing)
bar_centers = np.zeros((num_subgroups, num_groups))
for i in range(num_subgroups):
    bar_centers[i, :] = indices + i * bar_width
ref_idx = methods_label.index(ref_method)

# Plot each subgroup
for i in range(num_subgroups):
    offset = i * bar_width
    ax_1a.bar(indices + offset, object_error[i],
           yerr=object_error_std[i],
           width=bar_width,
           label=subgroup_labels[i],
           color=methods_color[i],
           alpha=1.0,
           capsize=2,
           error_kw={
               'elinewidth': 0.75,
               'capthick': 0.75,
               'linestyle': 'dashed',
               'dash_capstyle': 'butt'
           })

# X-axis ticks centered on each group
ax_1a.set_xticks(indices + bar_width - 0.2)
ax_1a.set_xticklabels(group_labels)
ax_1a.spines['top'].set_visible(False)
ax_1a.spines['right'].set_visible(False)

ax_1a.set_ylabel('NMSE', fontsize=10)
ax_1a.set_xticklabels(group_labels, fontsize=10)

handles = [plt.Line2D([0], [0], color=methods_color[i], lw=4, alpha=1.0, linestyle=(0, ())) for i in
           range(len(methods_label))]
legend = ax_1a.legend(handles, methods_label, loc="upper center", bbox_to_anchor=(0.5, 1.15),
                    ncol=len(methods_label), frameon=False,
                    columnspacing=0.9, handlelength=1.25, handletextpad=0.5, fontsize=10)

ax_1a.set_ylim(0, 0.15)
ax_1a.yaxis.set_major_locator(plt.MaxNLocator(5))

target_tac_outputs = [y_tac_n0_c0_wocm, y_tac_n0_c0_cm]

methods_label = [r'wo-CM', r'w-CM']
methods_color = [
    "#CC5308",  # Orange
    "#377EB8"   # Blue
]

tac_properties = ['Stiffness', 'Mass', 'Surf. Friction']
prop_style = ['solid', 'solid', 'solid']
time_s = np.linspace(0, 30, 98)

# Intrinsic Properties
for i in range(len(target_tac_outputs)):
    method = target_tac_outputs[i]
    mean_error = np.mean(method, axis=0)
    std_error = np.std(method, axis=0)
    for prop in range(len(tac_properties)):
        ax_2a[prop].plot(time_s, mean_error[:, prop], color=methods_color[i], label=methods_label[i], linewidth=1.0, linestyle=prop_style[prop])
        ax_2a[prop].fill_between(time_s, mean_error[:, prop] - std_error[:, prop]/10, mean_error[:, prop] + std_error[:, prop]/10, color=methods_color[i], alpha=0.1)
        ax_2a[prop].set_title(tac_properties[prop], fontsize=10)
        ax_2a[prop].set_xlabel('Time (s)', fontsize=10)
        ax_2a[prop].grid(True, linestyle='--', linewidth=0.4, alpha=0.5)
        if prop != 0:
            ax_2a[prop].tick_params(labelleft=False)

# Compute global min and max
vymin, vymax = float('inf'), -float('inf')
tymin, tymax = float('inf'), -float('inf')

for i in range(len(target_tac_outputs)):
    method = target_tac_outputs[i]
    mean_error = np.mean(method, axis=0)
    std_error = np.std(method, axis=0)
    lower = np.min(mean_error - std_error/50)
    upper = np.max(mean_error + std_error/50)
    tymin = min(tymin, lower)
    tymax = max(tymax, upper)


vymin = 0
vymax = 0.22
tymin = 0
tymax = 0.22

tac_yticks = np.linspace(0, 0.22, 4)

two_decimal_formatter = ticker.FormatStrFormatter('%.2f')

for i in range(0, 3):
    ax_2a[i].set_yticks(tac_yticks)
    ax_2a[i].set_ylim(tymin, tymax)
    ax_2a[i].yaxis.set_major_formatter(two_decimal_formatter)

ax_2a[0].set_ylabel('NMSE', fontsize=10)
handles = [plt.Line2D([0], [0], color=methods_color[i], lw=1, alpha=1.0, linestyle=(0, ())) for i in range(len(methods_label))]
legend = ax_2a[1].legend(handles, methods_label, loc="upper center", bbox_to_anchor=(0.5, 1.2),
                    ncol=len(methods_label), frameon=False,
                    columnspacing=0.9, handlelength=1.25, handletextpad=0.5, fontsize=10)

# ##################################################### Panel B ########################################################
# ################################################# Load data wocm #################################################
y_vis_n0_c0_wocm = np.load(os.path.join('results', 'regression', 'wo-cm', 'aligned_set', 'global_error_vis_y_n0_c0.npy'))
y_vis_n2_c0_wocm = np.load(os.path.join('results', 'regression', 'wo-cm', 'aligned_set', 'global_error_vis_y_n2_c0.npy')) # noise condition
y_vis_n0_c2_wocm = np.load(os.path.join('results', 'regression', 'wo-cm', 'aligned_set', 'global_error_vis_y_n0_c2.npy')) # corruption condition

# ################################################# Load data wcm #################################################
y_vis_n0_c0_wcm = np.load(os.path.join('results', 'regression', 'w-cm', 'late', 'aligned_set', 'global_error_vis_y_n0_c0.npy'))
y_vis_n2_c0_wcm = np.load(os.path.join('results', 'regression', 'w-cm', 'late', 'aligned_set', 'global_error_vis_y_n2_c0.npy')) # noise condition
y_vis_n0_c2_wcm = np.load(os.path.join('results', 'regression', 'w-cm', 'late', 'aligned_set', 'global_error_vis_y_n0_c2.npy')) # corruption condition

target_vis_outputs = [y_vis_n0_c0_wocm, y_vis_n2_c0_wocm, y_vis_n0_c2_wocm, y_vis_n0_c0_wcm, y_vis_n2_c0_wcm, y_vis_n0_c2_wcm]

methods_label = [r'wo-CM', r'w-CM']
condition_label = ['nominal', 'noise', 'corruption']
vis_properties = ['Shape', 'Size', 'Vis. Texture']

methods_color = [
    "#CC5308",  # Orange
    "#377EB8"  # Blue
]
num_properties = 3
num_conditions = 3
num_methods = 2

# Compute mean and std for each combination
# Shape: [property, condition, method]
object_error = np.zeros((num_properties, num_conditions, num_methods))
object_error_std = np.zeros((num_properties, num_conditions, num_methods))

# Map target_vis_outputs to proper structure
# Order: [wocm_n0_c0, wocm_n2_c0, wocm_n0_c2, wcm_n0_c0, wcm_n2_c0, wcm_n0_c2]
data_map = [
    (0, 0, 0),  # y_vis_n0_c0_wocm -> property_all, condition_0 (nominal), method_0 (wo-CM)
    (0, 1, 0),  # y_vis_n2_c0_wocm -> property_all, condition_1 (noise), method_0 (wo-CM)
    (0, 2, 0),  # y_vis_n0_c2_wocm -> property_all, condition_2 (corruption), method_0 (wo-CM)
    (0, 0, 1),  # y_vis_n0_c0_wcm -> property_all, condition_0 (nominal), method_1 (w-CM)
    (0, 1, 1),  # y_vis_n2_c0_wcm -> property_all, condition_1 (noise), method_1 (w-CM)
    (0, 2, 1),  # y_vis_n0_c2_wcm -> property_all, condition_2 (corruption), method_1 (w-CM)
]

for idx, (prop_idx, cond_idx, meth_idx) in enumerate(data_map):
    method_data = target_vis_outputs[idx]
    mean_error = np.mean(method_data, axis=0)  # Average over runs

    # For each property
    for prop in range(num_properties):
        property_mean = np.mean(mean_error[:, prop])  # Average over time
        property_std = np.std(mean_error[:, prop])
        object_error[prop, cond_idx, meth_idx] = property_mean
        object_error_std[prop, cond_idx, meth_idx] = property_std

# Plotting parameters
# Plotting parameters
bar_width = 0.25
condition_spacing = 0.02  # Space between methods within a condition (reduced)
group_spacing = 0.3  # Space between condition groups (reduced)
property_spacing = 3.5  # Space between properties

# Calculate bar positions
x_positions = []
x_tick_positions = []
x_tick_labels = []

for prop in range(num_properties):
    property_offset = prop * property_spacing

    for cond in range(num_conditions):
        condition_offset = cond * (num_methods * bar_width + condition_spacing + group_spacing)

        for meth in range(num_methods):
            method_offset = meth * (bar_width + condition_spacing)
            x_pos = property_offset + condition_offset + method_offset
            x_positions.append(x_pos)

    # Calculate center position for property label
    prop_center = property_offset + (num_conditions * (num_methods * bar_width + condition_spacing + group_spacing) - group_spacing) / 2 - bar_width / 2
    x_tick_positions.append(prop_center)
    x_tick_labels.append(vis_properties[prop])

# Plot bars
bar_idx = 0
for prop in range(num_properties):
    for cond in range(num_conditions):
        for meth in range(num_methods):
            x_pos = x_positions[bar_idx]
            height = object_error[prop, cond, meth]
            yerr = object_error_std[prop, cond, meth]

            # Add hatching for different conditions
            if cond == 0:  # nominal
                hatch_pattern = None
            elif cond == 1:  # noise
                hatch_pattern = '///'
            else:  # corruption
                hatch_pattern = 'xx'

            ax_b.bar(x_pos, height,
                     yerr=yerr,
                     width=bar_width,
                     color=methods_color[meth],
                     alpha=1.0,
                     hatch=hatch_pattern,
                     edgecolor='black',
                     linewidth=0.5,
                     capsize=2,
                     error_kw={
                         'elinewidth': 0.75,
                         'capthick': 0.75,
                         'linestyle': 'dashed',
                         'dash_capstyle': 'butt'
                     })

            bar_idx += 1

# Set x-axis ticks to property names
ax_b.set_xticks(x_tick_positions)
ax_b.set_xticklabels(x_tick_labels, fontsize=10)

# Styling
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
ax_b.set_ylabel('NMSE', fontsize=10)
ax_b.set_ylim(0, 0.15)
ax_b.yaxis.set_major_locator(plt.MaxNLocator(5))

# Create legend
from matplotlib.patches import Patch

handles = []
for meth in range(num_methods):
    handles.append(Patch(facecolor=methods_color[meth],
                         edgecolor='black',
                         linewidth=0.5,
                         label=methods_label[meth]))

# Add condition hatching to legend
handles.append(Patch(facecolor='gray', edgecolor='black', linewidth=0.5,
                     label='Nominal', alpha=0.5))
handles.append(Patch(facecolor='gray', edgecolor='black', linewidth=0.5,
                     hatch='///', label='Noise', alpha=0.5))
handles.append(Patch(facecolor='gray', edgecolor='black', linewidth=0.5,
                     hatch='xxx', label='Corruption', alpha=0.5))

legend = ax_b.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.15),
                     ncol=5, frameon=False,
                     columnspacing=0.9, handlelength=1.25,
                     handletextpad=0.5, fontsize=9)

# ###################################################### Panel C #######################################################
y_tac_n0_c0_wocm = np.load(os.path.join('results', 'regression', 'wo-cm', 'surprise_set', 'global_error_tac_y.npy'))
y_tac_n0_c0_cm = np.load(os.path.join('results', 'regression', 'w-cm', 'late', 'surprise_set', 'global_error_tac_y.npy'))

results = {
    'wo-CM': np.mean(y_tac_n0_c0_wocm, axis=1),
    'w-CM': np.mean(y_tac_n0_c0_cm, axis=1)
}
alpha = 0.05
methods = list(results.keys())
ref_method = 'w-CM'
n_props = results[ref_method].shape[1]
n_methods = len(methods)

test_results = {}
for m in methods:
    if m == ref_method:
        continue
    test_results[m] = {'pvals': np.ones(n_props), 'symbol': ['ns'] * n_props}
    for prop_idx in range(n_props):
        x = results[ref_method][:, prop_idx]
        y = results[m][:, prop_idx]
        assert x.shape[0] == y.shape[0], "Paired tests require same number of runs."
        stat, p = stats.ttest_rel(x, y, nan_policy='omit')
        test_results[m]['pvals'][prop_idx] = p
        test_results[m]['symbol'][prop_idx] = p_to_stars(p)

# Print summary
print("Significance (w-CM vs others) per property:")
for m in methods:
    if m == ref_method: continue
    print(f"\nComparing w-CM vs {m}:")
    for prop_idx in range(n_props):
        p = test_results[m]['pvals'][prop_idx]
        star = test_results[m]['symbol'][prop_idx]
        mean_ref = results[ref_method][:, prop_idx].mean()
        mean_m = results[m][:, prop_idx].mean()
        print(f"  Property {prop_idx}: p={p:.4g}, {star}, means: {ref_method}={mean_ref:.4f}, {m}={mean_m:.4f}")

target_tac_outputs = [y_tac_n0_c0_wocm, y_tac_n0_c0_cm]

methods_label = [r'wo-CM', r'w-CM']
methods_color = [
    "#CC5308",  # Orange
    "#377EB8"  # Blue
]
tac_properties = ['Stiffness', 'Mass', 'Surf.\nFriction']  # 3) Two-line label

bar_width = 0.4
group_spacing = 0.80
num_subgroups, num_groups = 2, 3
group_labels = tac_properties
subgroup_labels = methods_label

object_error = np.zeros((len(methods_label), len(tac_properties)))
object_error_std = np.zeros((len(methods_label), len(tac_properties)))

# Intrinsic Properties
for i in range(len(target_tac_outputs)):
    method = target_tac_outputs[i]
    mean_error = np.mean(method, axis=0)
    for prop in range(len(tac_properties)):
        object_error[i, prop] = np.mean(mean_error[:, prop])
        object_error_std[i, prop] = np.std(mean_error[:, prop])

# Compute positions
indices = np.arange(num_groups) * (num_subgroups * bar_width + group_spacing)
bar_centers = np.zeros((num_subgroups, num_groups))
for i in range(num_subgroups):
    bar_centers[i, :] = indices + i * bar_width
ref_idx = methods_label.index(ref_method)

# Plot each subgroup
for i in range(num_subgroups):
    offset = i * bar_width
    ax_c.bar(indices + offset, object_error[i],
           yerr=object_error_std[i],
           width=bar_width,
           label=subgroup_labels[i],
           color=methods_color[i],
           alpha=1.0,
           hatch='-.-.',
           capsize=2,
           error_kw={
               'elinewidth': 0.75,
               'capthick': 0.75,
               'linestyle': 'dashed',
               'dash_capstyle': 'butt'
           })

# X-axis ticks centered on each group
ax_c.set_xticks(indices + bar_width - 0.2)
ax_c.set_xticklabels(group_labels)
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)

ax_c.set_ylabel('NMSE', fontsize=10)
ax_c.set_xticklabels(group_labels, fontsize=10)

# Create legend with hatching pattern
from matplotlib.patches import Patch

handles = [Patch(facecolor=methods_color[i],
                 edgecolor='black',
                 hatch='-.-.',
                 linewidth=0.5,
                 alpha=1.0)
           for i in range(len(methods_label))]

legend = ax_c.legend(handles, methods_label, loc="upper center", bbox_to_anchor=(0.5, 1.15),
                    ncol=len(methods_label), frameon=False,
                    columnspacing=0.9, handlelength=1.25, handletextpad=0.5, fontsize=10)

ax_c.set_ylim(0, 0.15)
ax_c.yaxis.set_major_locator(plt.MaxNLocator(5))

target_tac_outputs = [y_tac_n0_c0_wocm, y_tac_n0_c0_cm]

# methods_label = [r'wo-CM', r'w-CM']
# methods_color = [
#     "#CC5308",  # Orange
#     "#377EB8"   # Blue
# ]
#
# vis_properties = ['Shape', 'Size', 'Vis. Texture']
# tac_properties = ['Stiffness', 'Mass', 'Surf. Friction']
# prop_style = ['dashdot', 'dashdot', 'dashdot']
# time_s = np.linspace(0, 30, 98)
#
# # Intrinsic Properties
# for i in range(len(target_tac_outputs)):
#     method = target_tac_outputs[i]
#     mean_error = np.mean(method, axis=0)
#     std_error = np.std(method, axis=0)
#     for prop in range(len(tac_properties)):
#         ax_2c[prop].plot(time_s, mean_error[:, prop], color=methods_color[i], label=methods_label[i], linewidth=1.0, linestyle=prop_style[prop])
#         ax_2c[prop].fill_between(time_s, mean_error[:, prop] - std_error[:, prop]/10, mean_error[:, prop] + std_error[:, prop]/10, color=methods_color[i], alpha=0.1)
#         ax_2c[prop].set_title(tac_properties[prop], fontsize=10)
#         ax_2c[prop].set_xlabel('Time (s)', fontsize=10)
#         ax_2c[prop].grid(True, linestyle='--', linewidth=0.4, alpha=0.5)
#         if prop != 0:
#             ax_2c[prop].tick_params(labelleft=False)
#
# # Compute global min and max
# vymin, vymax = float('inf'), -float('inf')
# tymin, tymax = float('inf'), -float('inf')
#
# for i in range(len(target_tac_outputs)):
#     method = target_tac_outputs[i]
#     mean_error = np.mean(method, axis=0)
#     std_error = np.std(method, axis=0)
#     lower = np.min(mean_error - std_error/50)
#     upper = np.max(mean_error + std_error/50)
#     tymin = min(tymin, lower)
#     tymax = max(tymax, upper)
#
# tymin = 0
# tymax = 0.22
#
# tac_yticks = np.linspace(0, 0.22, 4)
#
# two_decimal_formatter = ticker.FormatStrFormatter('%.2f')
#
# for i in range(0, 3):
#     ax_2c[i].set_yticks(tac_yticks)
#     ax_2c[i].set_ylim(tymin, tymax)
#     ax_2c[i].yaxis.set_major_formatter(two_decimal_formatter)
#
# ax_2c[0].set_ylabel('NMSE', fontsize=10)
# handles = [plt.Line2D([0], [0], color=methods_color[i], lw=1, alpha=1.0, linestyle='dashdot') for i in range(len(methods_label))]
# legend = ax_2c[1].legend(handles, methods_label, loc="upper center", bbox_to_anchor=(0.5, 1.25),
#                     ncol=len(methods_label), frameon=False,
#                     columnspacing=0.9, handlelength=1.25, handletextpad=0.5, fontsize=10)

# Panel label (a), aligned with the legend row
fig.text(0.025, 0.95, r'{a}',
         ha='right', va='center', fontsize=12)

fig.text(0.325, 0.95, r'{b}',
         ha='right', va='center', fontsize=12)

fig.text(0.025, 0.475, r'{c}',
         ha='right', va='center', fontsize=12)

fig.text(0.725, 0.475, r'{d}',
         ha='right', va='center', fontsize=12)

# fig.text(0.025, 0.325, r'{d}',
#          ha='right', va='center', fontsize=12)
#
# fig.text(0.415, 0.325, r'{e}',
#          ha='right', va='center', fontsize=12)

# plt.show()
plt.savefig(os.path.join('results', 'Figure_2_abs.pdf'), dpi=500)
plt.savefig(os.path.join('results', 'Figure_2_abs.svg'), dpi=500)