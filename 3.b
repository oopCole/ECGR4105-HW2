print("\n--- Problem 3.b Results (11 Features with L2 Regularization) ---")

# Setup (using Standardization from 2.b - stored in BEST_X_TRAIN etc.)
initial_theta_reg_3b = np.zeros(BEST_X_TRAIN.shape[1])

# Run Regularized Gradient Descent
best_theta_reg_3b, train_loss_hist_reg_3b, val_loss_hist_reg_3b = gradient_descent_regularized(
    BEST_X_TRAIN, BEST_Y_TRAIN, initial_theta_reg_3b, ALPHA_SCALED, ITERATIONS_SCALED,
    LAMBDA_REG, BEST_X_TEST, BEST_Y_TEST
)

# Final Evaluation (Validation loss is UNREGULARIZED)
final_validation_loss_reg_3b = compute_loss(BEST_X_TEST, BEST_Y_TEST, best_theta_reg_3b)

print(f"L2 Regularization (lambda={LAMBDA_REG}) Val Loss: {final_validation_loss_reg_3b:.2f}")
print(f"Baseline Scaled Val Loss (2.b, Std): {BEST_VAL_LOSS_UNREG:.2f}")

# Plotting
plot_loss(train_loss_hist_reg_3b, val_loss_hist_reg_3b,
          f"3.b) Loss History (11 Features, {BEST_SCALING} + L2 Reg. $\lambda={LAMBDA_REG}$)",
          ALPHA_SCALED, ITERATIONS_SCALED)
