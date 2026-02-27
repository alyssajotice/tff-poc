import matplotlib
# This line MUST come before importing pyplot
# It tells Python to just generate the file and not open a window
matplotlib.use('Agg') 

import matplotlib.pyplot as plt

# 1. YOUR DATA
epsilons = [0.1, 1.0, 10.0, 100.0]
accuracies = [0.6209, 0.6974, 0.6860, 0.6218]

# 2. BENCHMARKS
centralized_baseline = 0.7335
federated_baseline = 0.6834

# 3. PLOTTING
print("[*] Generating the Privacy-Utility Curve...")
plt.figure(figsize=(10, 6))

plt.plot(epsilons, accuracies, marker='o', color='#1f77b4', linewidth=3, 
         markersize=8, label='DP-Federated Model')

plt.axhline(y=centralized_baseline, color='#d62728', linestyle='--', 
            alpha=0.8, label=f'Centralized Baseline ({centralized_baseline*100:.1f}%)')

plt.axhline(y=federated_baseline, color='#2ca02c', linestyle='-.', 
            alpha=0.8, label=f'Federated Baseline ({federated_baseline*100:.1f}%)')

# Formatting
plt.xscale('log') 
plt.xlabel('Privacy Budget ($\epsilon$)', fontsize=12)
plt.ylabel('Model Accuracy', fontsize=12)
plt.title('The Privacy-Utility Trade-off (Thesis Data)', fontsize=14)
plt.grid(True, which="both", linestyle='--', alpha=0.5)
plt.legend(loc='lower right')

# 4. SAVE AND EXIT
plt.tight_layout()
plt.savefig('privacy_utility_curve_final.png', dpi=300)

# Instead of plt.show(), we just confirm completion
print("✅ SUCCESS: Graph saved as 'privacy_utility_curve_final.png'")
print("[*] You can now exit the script.")