from matplotlib import pyplot as plt

x = [3, 6, 12, 15]
mean_rmse_walking = [0.013789465753950042, 0.23943243902490416, 0.011746272073953776, 0]
mean_kf_walking = [31, 40, 29, 0]

mean_rmse_sitting = [0.03485198780582431, 0.03972341105128272, 0.03678244093607894, 0.015855649131982844]
mean_kf_sitting = [18, 41, 45, 18]

# Plotgraph with x and y axis labels
plt.figure()
plt.plot(x, mean_rmse_walking, label='walking_xyz')
plt.plot(x, mean_rmse_sitting, label='sitting_rpy')
plt.xlabel('Number of Frames tracked')
plt.ylabel('ATE_RMSE')
plt.legend()
plt.tight_layout()
plt.savefig('temp_vs_rmse.pdf')

# Plot graph with x and y axis labels
plt.figure()
plt.plot(x, mean_kf_walking, label='walking_xyz')
plt.plot(x, mean_kf_sitting, label='sitting_rpy')
plt.xlabel('Number of Frames tracked')
plt.ylabel('Number of Keyframes')
plt.legend()
plt.tight_layout()
plt.savefig('temp_vs_kf.pdf')