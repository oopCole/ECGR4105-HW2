features_1b = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 
               'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea']
X_1b_raw = df[features_1b].values

# Split data (80/20)
X_train_raw_1b, X_test_raw_1b, y_train_1b, y_test_1b = train_test_split(
    X_1b_raw, y_full, test_size=0.2, random_state=42
)

# Add Bias Term (NO scaling)
X_train_1b = np.hstack((np.ones((X_train_raw_1b.shape[0], 1)), X_train_raw_1b))
X_test_1b = np.hstack((np.ones((X_test_raw_1b.shape[0], 1)), X_test_raw_1b))

# Initialize Parameters
initial_theta_1b = np.zeros(X_train_1b.shape[1])

# Run Gradient Descent
best_theta_1b, train_loss_hist_1b, val_loss_hist_1b = gradient_descent(
    X_train_1b, y_train_1b, initial_theta_1b, ALPHA_BASE, ITERATIONS_BASE,
    X_test_1b, y_test_1b
)

# Final Evaluation
final_train_loss_1b = compute_loss(X_train_1b, y_train_1b, best_theta_1b)
final_validation_loss_1b = compute_loss(X_test_1b, y_test_1b, best_theta_1b)

print("\n--- Problem 1.b Results (Baseline, 11 Features) ---")
print(f"Features: {features_1b}")
print(f"Best Parameters (theta): {best_theta_1b}")
print(f"Final Train Loss (MSE): {final_train_loss_1b:.2f}")
print(f"Final Validation Loss (MSE): {final_validation_loss_1b:.2f}")
print("\n--- Comparison 1.a vs 1.b ---")
# Lower loss means better fitting.
print(f"1.a Validation Loss (5 features): {final_validation_loss_1a:.2f}")
print(f"1.b Validation Loss (11 features): {final_validation_loss_1b:.2f}")

# Plotting
plot_loss(train_loss_hist_1b, val_loss_hist_1b,
          "1.b) Training and Validation Loss (Baseline, 11 Features)",
          ALPHA_BASE, ITERATIONS_BASE)
