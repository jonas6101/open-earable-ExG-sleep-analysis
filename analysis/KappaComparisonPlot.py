import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


# Example data
data = {
    "Group": ["Fold CV Binary", "Fold CV Binary", "Fold CV Binary",
              "Fold CV Multistage", "Fold CV Multistage", "Fold CV Multistage",
              "LOPO CV Binary", "LOPO CV Binary", "LOPO CV Binary",
              "LOPO CV Multistage", "LOPO CV Multistage", "LOPO CV Multistage"],
    "Device": ["OpenEarableExG", "Nakamura et al. 2020", "Mikkelsen et al. 2017",
               "OpenEarableExG", "Nakamura et al. 2020", "Mikkelsen et al. 2017",
               "OpenEarableExG", "Mikkelsen et al. 2017", "Mikkelsen et al. 2019",
               "OpenEarableExG", "Mikkelsen et al. 2017", "Mikkelsen et al. 2019"],
    "Cohen's Kappa": [0.701, 0.68, 0.73,
                      0.69, 0.61, 0.6,
                      0.512, 0.52, 0.85,
                      0.407, 0.45, 0.73],
}

df = pd.DataFrame(data)

# Define custom colors for each device
device_colors = {
    "OpenEarableExG": "#1f77b4",  # Blue
    "Nakamura et al. 2020": "#2ca02c",  # Green
    "Mikkelsen et al. 2017": "#ff7f0e",  # Orange
    "Mikkelsen et al. 2019": "#ff4823",  # Red
}

# Get unique groups
groups = df['Group'].unique()
devices_per_group = df.groupby('Group')['Device'].count().max()  # Max number of devices per group

# Generate positions for each bar
group_positions = np.arange(len(groups))  # Group positions on the x-axis
bar_width = 0.2  # Width of each bar
device_offsets = np.linspace(-bar_width, bar_width, devices_per_group)  # Device positions within a group

# Plot
plt.figure(figsize=(12, 6))

for i, group in enumerate(groups):
    group_data = df[df['Group'] == group]
    for j, (device, kappa) in enumerate(zip(group_data['Device'], group_data["Cohen's Kappa"])):
        # Calculate bar position
        bar_position = group_positions[i] + device_offsets[j]
        # Use the custom color for the device
        plt.bar(bar_position, kappa, bar_width, color=device_colors[device], label=device if device not in plt.gca().get_legend_handles_labels()[1] else "")

# Customize x-axis
plt.xticks(group_positions, groups, rotation=45, ha='right', fontsize=12)  # Set group names as x-tick labels
plt.ylabel("Cohen's Kappa", fontsize=14)
plt.ylim(0.3, 1)  # Set y-axis range
plt.legend(title="Device", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16, title_fontsize=16)
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()

