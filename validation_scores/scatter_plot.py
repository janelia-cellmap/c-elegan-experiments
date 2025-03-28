#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = accuracy_scores
#%%
# 2. Flatten the data into a Pandas DataFrame, ignoring NaN
rows = []
for dataset_name, crops in data.items():
    for crop_name, organelles in crops.items():
        for organelle, f1_score in organelles.items():
            # Skip if the score is None or NaN
            if f1_score is not None and not np.isnan(f1_score):
                rows.append({
                    'dataset': dataset_name,
                    'crop': crop_name,
                    'organelle': organelle,
                    'f1_score': f1_score
                })

df = pd.DataFrame(rows)
df
#%%

# 3. Create a custom boxplot for each dataset
#    - Whiskers = min & max
#    - Box = mean ± std
#    - Horizontal line in box = mean
#    - Show points (dots) for all data, ignoring NaN
datasets = df['dataset'].unique()
num_datasets = len(datasets)

fig, axes = plt.subplots(nrows=num_datasets, ncols=1, figsize=(10, 5 * num_datasets))
if num_datasets == 1:
    axes = [axes]  # so we can always iterate

for ax, ds in zip(axes, datasets):
    subset = df[df['dataset'] == ds]
    organelle_names = sorted(subset['organelle'].unique())

    # Build the stats for each organelle to pass to ax.bxp(...)
    stats = []
    for org in organelle_names:
        data_org = subset.loc[subset['organelle'] == org, 'f1_score'].dropna()  # drop NaNs explicitly
        if len(data_org) == 0:
            continue
        mean_val = data_org.mean()
        std_val = data_org.std(ddof=1)  # sample std
        min_val = data_org.min()
        max_val = data_org.max()

        q1 = mean_val - std_val
        q3 = mean_val + std_val

        stats_dict = {
            'mean': mean_val,
            # We place the mean in 'med' to draw it as the horizontal line in the box
            'med': mean_val,
            'q1': q1,
            'q3': q3,
            'whislo': min_val,
            'whishi': max_val,
            'label': org
        }
        stats.append(stats_dict)

    # Plot the custom boxplot
    ax.bxp(stats, showmeans=False, showfliers=False, meanline=True)
    ax.set_title(f'Accuracy : {ds}')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Organelle')
    ax.set_ylim(0.9, 1)  # Set y-axis limits from 0 to 1

    # Overlay the individual data points as scatter dots
    for i, org in enumerate(organelle_names):
        data_org = subset.loc[subset['organelle'] == org, 'f1_score'].dropna()
        xvals = np.random.normal(i+1, 0.04, size=len(data_org))  # jitter
        ax.scatter(xvals, data_org, alpha=0.6, color='black', s=30)

plt.tight_layout()
plt.savefig('/groups/cellmap/cellmap/zouinkhim/c-elegen/v2/validation_scores/accuracy_boxplot.png')
plt.show()


# %%
