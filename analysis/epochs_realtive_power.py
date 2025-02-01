import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Load data
DATA_FILE = "../ml/features_V2/P004_241214_features_V2.csv"
df = pd.read_csv(DATA_FILE)

# Extract relative power columns, excluding CWT relative power
relative_power_cols = [col for col in df.columns if 'relative_power' in col and 'cwt' not in col]

# Set up plot
plt.figure(figsize=(15, 6))

# Plot stacked bar chart for each epoch
for i, col in enumerate(relative_power_cols):
    plt.bar(df['epoch'], df[col], width=1.0,label=col,
            bottom=df[relative_power_cols[:i]].sum(axis=1) if i > 0 else 0)

# Formatting
plt.xlabel('Epoch')
plt.ylabel('Relative Power')
plt.title('Relative Power of Frequency Bands')
plt.ylim(0, 1)
plt.legend()
plt.grid(True, axis='y')
plt.show()
