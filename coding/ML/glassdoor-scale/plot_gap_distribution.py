# if there are ten uniformly distributed numbers from 0-1, 
# what would the distribution of the different between the fifth and sixth look like? 
# What distribution does that look like and why?

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

N_TRIALS = 10000
N_SAMPLES_PER_TRIAL = 10

rng = np.random.default_rng(seed=42)
samples = rng.uniform(0, 1, size=(N_TRIALS, N_SAMPLES_PER_TRIAL))
sorted_samples = np.sort(samples, axis=1)
gaps = sorted_samples[:, 5] - sorted_samples[:, 4]

emp_mean = gaps.mean()
emp_std = gaps.std(ddof=1)
theory_mean = 1 / 11
theory_var = (1 * 10) / ((1 + 10) ** 2 * (1 + 10 + 1))
theory_std = np.sqrt(theory_var)

print("Empirical mean:", emp_mean)
print("Empirical std:", emp_std)
print("Theoretical mean (Beta(1,10)):", theory_mean)
print("Theoretical std (Beta(1,10)):", theory_std)

x = np.linspace(0, 1, 400)
pdf = 10 * (1 - x) ** 9

plt.figure(figsize=(7, 5))
plt.hist(gaps, bins=50, density=True, alpha=0.6, edgecolor='black')
plt.plot(x, pdf, linewidth=2)
plt.xlabel("Gap D = X_(6) - X_(5)")
plt.ylabel("Density")
plt.title("Distribution of Gap Between 5th and 6th Order Stats (10 Uniform(0,1))")
plt.tight_layout()
plt.show()

# Save optional artifacts
pd.DataFrame({"gap": gaps}).to_csv("gap_samples.csv", index=False)
plt.figure(figsize=(7, 5))
plt.hist(gaps, bins=50, density=True, alpha=0.6, edgecolor='black')
plt.plot(x, pdf, linewidth=2)
plt.xlabel("Gap D = X_(6) - X_(5)")
plt.ylabel("Density")
plt.title("Distribution of Gap Between 5th and 6th Order Stats (10 Uniform(0,1))")
plt.tight_layout()
plt.savefig("gap_distribution.png")
