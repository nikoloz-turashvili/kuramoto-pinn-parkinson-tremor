"""
=========================================================
PHYSICS-INFORMED KURAMOTO SURROGATE MODEL (COMPLETE)
=========================================================
✔ Physics-Informed Loss + Early Stopping
✔ Complete visualizations
✔ Trajectory tracking for r(t) plots
✔ Initial vs Final phase distributions
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc

# Seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Using device: {device}")

# =========================================================
# PARAMETERS
# =========================================================
N = 10  # Number of oscillators
K = 2.0  # Coupling strength
SIGMA_OMEGA = 0.8  # Frequency spread
T = 5.0  # Time horizon
M_TRAIN = 5000
M_TEST = 500
BATCH_SIZE = 256
N_TIME_STEPS = 50  # For trajectory storage

# Early Stopping Parameters
MAX_EPOCHS = 150
PATIENCE = 20
MIN_DELTA = 1e-5

print("=" * 60)
print("🎯 KURAMOTO SURROGATE - Complete Version")
print("=" * 60)
print(f"Device: {device}")
print(f"N = {N} oscillators")
print(f"K = {K}, σω = {SIGMA_OMEGA}, T = {T}s")
print(f"Training samples: {M_TRAIN}")
print(f"Test samples: {M_TEST}")
print(f"Batch size: {BATCH_SIZE}")
print("=" * 60 + "\n")


# =========================================================
# 1. KURAMOTO ODE
# =========================================================
def kuramoto_rhs(t, theta, omega, K):
    """Kuramoto model differential equation"""
    N = len(theta)
    dtheta = omega.copy()
    for i in range(N):
        coupling = np.sum(np.sin(theta - theta[i]))
        dtheta[i] += (K / N) * coupling
    return dtheta


# =========================================================
# 2. CIRCULAR EMBEDDING
# =========================================================
def angles_to_cos_sin(theta):
    """θ → (cos θ, sin θ)"""
    return np.concatenate([np.cos(theta), np.sin(theta)])


def cos_sin_to_angles(x):
    """(cos θ, sin θ) → θ"""
    N = len(x) // 2
    return np.arctan2(x[N:], x[:N])


# =========================================================
# 3. ORDER PARAMETER
# =========================================================
def order_parameter(theta):
    """r = |⟨e^(iθ)⟩|"""
    return np.abs(np.mean(np.exp(1j * theta)))


def order_parameter_torch(x):
    """Torch version for batches"""
    N = x.shape[-1] // 2
    cos_part = x[:, :N]
    sin_part = x[:, N:]
    return torch.sqrt(
        torch.mean(cos_part, dim=1) ** 2 +
        torch.mean(sin_part, dim=1) ** 2
    )


# =========================================================
# 4. PHASE VARIANCE
# =========================================================
def phase_variance_torch(x):
    """Circular variance of phases"""
    N = x.shape[-1] // 2
    theta = torch.atan2(x[:, N:], x[:, :N])
    mean_angle = torch.atan2(
        torch.mean(torch.sin(theta), dim=1, keepdim=True),
        torch.mean(torch.cos(theta), dim=1, keepdim=True)
    )
    diff = torch.atan2(
        torch.sin(theta - mean_angle),
        torch.cos(theta - mean_angle)
    )
    return torch.mean(diff ** 2, dim=1)


# =========================================================
# 5. DATASET GENERATION WITH TRAJECTORY
# =========================================================
def generate_dataset_with_trajectory(M, omega, K, T, n_steps=50):
    """Generate M samples with full trajectory for r(t) plots"""
    X, Y, TRAJ = [], [], []
    t_eval = np.linspace(0, T, n_steps)

    print(f"\n📊 Generating {M} samples...")

    with tqdm(total=M, desc="Generating", ncols=80) as pbar:
        for i in range(M):
            theta0 = np.random.uniform(-np.pi, np.pi, N)

            sol = solve_ivp(
                lambda t, y: kuramoto_rhs(t, y, omega, K),
                (0, T),
                theta0,
                method='RK45',
                t_eval=t_eval,
                atol=1e-8,
                rtol=1e-8
            )

            theta_traj = sol.y.T  # shape (n_steps, N)

            X.append(angles_to_cos_sin(theta_traj[0]))
            Y.append(angles_to_cos_sin(theta_traj[-1]))
            TRAJ.append(theta_traj)

            pbar.update(1)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    TRAJ = np.array(TRAJ, dtype=np.float32)

    print(f"✓ Dataset created: X={X.shape}, TRAJ={TRAJ.shape}\n")
    return X, Y, TRAJ, t_eval


# =========================================================
# 6. PYTORCH DATASET
# =========================================================
class KuramotoDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# =========================================================
# 7. SURROGATE NETWORK
# =========================================================
class SurrogateNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, dim)
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# 8. PHYSICS-INFORMED LOSS
# =========================================================
class PhysicsInformedLoss(nn.Module):
    def __init__(self, alpha_mse=1.0, alpha_r=0.5, alpha_phase=0.3):
        super().__init__()
        self.mse = nn.MSELoss()
        self.alpha_mse = alpha_mse
        self.alpha_r = alpha_r
        self.alpha_phase = alpha_phase

    def forward(self, pred, target):
        L_mse = self.mse(pred, target)

        r_pred = order_parameter_torch(pred)
        r_true = order_parameter_torch(target)
        L_r = torch.mean((r_pred - r_true) ** 2)

        var_pred = phase_variance_torch(pred)
        var_true = phase_variance_torch(target)
        L_phase = torch.mean((var_pred - var_true) ** 2)

        total = (
                self.alpha_mse * L_mse +
                self.alpha_r * L_r +
                self.alpha_phase * L_phase
        )

        return total, L_mse, L_r, L_phase


# =========================================================
# 9. TRAINING
# =========================================================
def train(model, loader, max_epochs, patience=20, min_delta=1e-5):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = PhysicsInformedLoss()

    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    history = {
        'epoch': [],
        'loss': [],
        'mse': [],
        'r_loss': [],
        'phase_loss': []
    }

    print("\n🧠 Training started...\n")

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_r = 0.0
        epoch_phase = 0.0

        with tqdm(loader, desc=f"Epoch {epoch + 1:03d}", ncols=100, leave=False) as pbar:
            for xb, yb in pbar:
                xb, yb = xb.to(device), yb.to(device)

                optimizer.zero_grad()
                pred = model(xb)
                loss, mse, r, phase = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_mse += mse.item()
                epoch_r += r.item()
                epoch_phase += phase.item()

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        n_batches = len(loader)
        avg_loss = epoch_loss / n_batches

        history['epoch'].append(epoch + 1)
        history['loss'].append(avg_loss)
        history['mse'].append(epoch_mse / n_batches)
        history['r_loss'].append(epoch_r / n_batches)
        history['phase_loss'].append(epoch_phase / n_batches)

        # Print every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1:03d} | Loss: {avg_loss:.6f} | "
                  f"MSE: {epoch_mse / n_batches:.6f} | "
                  f"L_r: {epoch_r / n_batches:.6f}")

        # Early stopping
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\n🛑 Early stop at epoch {epoch + 1} | Best: {best_loss:.6f}")
            model.load_state_dict(best_model_state)
            break

    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

    print(f"\n✅ Training complete! Best loss: {best_loss:.6f}\n")
    return history


# =========================================================
# 10. EVALUATION
# =========================================================
def evaluate(model, X, Y):
    model = model.to(device)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(X_tensor).cpu()

    theta_true = np.array([cos_sin_to_angles(y) for y in Y])
    theta_pred = np.array([cos_sin_to_angles(p.numpy()) for p in pred])

    diff = np.arctan2(
        np.sin(theta_pred - theta_true),
        np.cos(theta_pred - theta_true)
    )

    MAE_theta = np.mean(np.abs(diff))
    RMSE_theta = np.sqrt(np.mean(diff ** 2))
    MAE_theta_deg = np.rad2deg(MAE_theta)

    r_true = np.array([order_parameter(t) for t in theta_true])
    r_pred = np.array([order_parameter(t) for t in theta_pred])
    MAE_r = np.mean(np.abs(r_pred - r_true))

    print("=" * 60)
    print("📊 EVALUATION METRICS")
    print("=" * 60)
    print(f"MAE_θ   = {MAE_theta:.6f} radians ({MAE_theta_deg:.2f}°)")
    print(f"RMSE_θ  = {RMSE_theta:.6f} radians")
    print(f"MAE_r   = {MAE_r:.6f}")
    print("=" * 60 + "\n")

    return theta_true, theta_pred, r_true, r_pred


# =========================================================
# 11. COMPLETE VISUALIZATIONS
# =========================================================

# 1. Training History
def plot_training_history(history, save_name='training_history.png'):
    """Loss evolution during training"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(history['epoch'], history['loss'], 'b-', linewidth=2)
    axes[0, 0].set_title('Total Loss', fontweight='bold', fontsize=12)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(history['epoch'], history['mse'], 'g-', linewidth=2)
    axes[0, 1].set_title('MSE Loss', fontweight='bold', fontsize=12)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(history['epoch'], history['r_loss'], 'r-', linewidth=2)
    axes[1, 0].set_title('Order Parameter Loss', fontweight='bold', fontsize=12)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('L_r')
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(history['epoch'], history['phase_loss'], 'purple', linewidth=2)
    axes[1, 1].set_title('Phase Variance Loss', fontweight='bold', fontsize=12)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('L_phase')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {save_name}")
    plt.close()


# 2. Phase Distribution (Initial vs Final ODE vs Final Surrogate)
def plot_phase_distribution(theta0, thetaT_true, thetaT_pred,
                           save_name="phase_distribution.png"):
    """Compare initial, ODE final, and surrogate final phases"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                             subplot_kw=dict(projection="polar"))

    axes[0].scatter(theta0, np.ones(len(theta0)), s=100, c='blue', alpha=0.7)
    axes[0].set_title("Initial Phases (t=0)", fontsize=14, fontweight='bold', pad=20)
    axes[0].set_ylim(0, 1.2)
    axes[0].grid(alpha=0.3)

    axes[1].scatter(thetaT_true, np.ones(len(thetaT_true)), s=100, c='green', alpha=0.7)
    axes[1].set_title(f"Final (ODE) t={T}s", fontsize=14, fontweight='bold', pad=20)
    axes[1].set_ylim(0, 1.2)
    axes[1].grid(alpha=0.3)

    axes[2].scatter(thetaT_pred, np.ones(len(thetaT_pred)), s=100, c='red', alpha=0.7)
    axes[2].set_title(f"Final (Surrogate) t={T}s", fontsize=14, fontweight='bold', pad=20)
    axes[2].set_ylim(0, 1.2)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_name}")


# 3. Order Parameter Evolution r(t)
def plot_order_parameter_vs_time(theta_traj, t_eval,
                                 save_name="order_parameter_time.png"):
    """Order parameter r(t) evolution over time"""
    r_t = [order_parameter(theta) for theta in theta_traj]

    plt.figure(figsize=(10, 6))
    plt.plot(t_eval, r_t, linewidth=2.5, color='blue', label='r(t)')
    plt.xlabel("Time (s)", fontsize=12, fontweight='bold')
    plt.ylabel("Order Parameter r(t)", fontsize=12, fontweight='bold')
    plt.title("Order Parameter Evolution Over Time", fontsize=14, fontweight='bold')
    plt.grid(alpha=0.4, linestyle='--')
    plt.legend(fontsize=11)
    plt.ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_name}")


# 4. Order Parameters Comparison (all test samples)
def plot_order_parameters(r_true, r_pred, save_name='order_parameters.png'):
    """Compare true vs predicted order parameters for all test samples"""
    fig, ax = plt.subplots(figsize=(10, 6))

    indices = np.arange(len(r_true))
    ax.plot(indices, r_true, 'b-', linewidth=2, alpha=0.7, label='TRUE (ODE)')
    ax.plot(indices, r_pred, 'r--', linewidth=2, alpha=0.7, label='SURROGATE (NN)')

    ax.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Order Parameter r', fontsize=12, fontweight='bold')
    ax.set_title('Order Parameter Comparison (Test Set)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.4)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {save_name}")
    plt.close()


# 5. Error Distribution
def plot_error_distribution(r_true, r_pred, save_name='error_distribution.png'):
    """Distribution of errors in order parameter prediction"""
    errors = r_pred - r_true
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(errors, bins=40, color='purple', alpha=0.7, edgecolor='black')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax1.set_xlabel('Error (r_pred - r_true)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Error Distribution', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.scatter(r_true, r_pred, alpha=0.5, s=20, color='green')
    ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Prediction')
    ax2.set_xlabel('r_true', fontsize=11, fontweight='bold')
    ax2.set_ylabel('r_pred', fontsize=11, fontweight='bold')
    ax2.set_title('True vs Predicted', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {save_name}")
    plt.close()


# 6. Final State Comparison (Polar)
def plot_final_state(thetaT_true, thetaT_pred, save_name='final_state.png'):
    """Side-by-side polar plots of final states"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                   subplot_kw=dict(projection='polar'))

    ax1.scatter(thetaT_true, np.ones(N), c='blue', s=100, alpha=0.7, label='TRUE')
    ax1.set_ylim(0, 1.2)
    ax1.set_title(f'Final State (t={T}s) - ODE', fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    ax2.scatter(thetaT_pred, np.ones(N), c='red', s=100, alpha=0.7, label='SURROGATE')
    ax2.set_ylim(0, 1.2)
    ax2.set_title(f'Final State (t={T}s) - NN', fontsize=14, fontweight='bold', pad=20)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {save_name}")
    plt.close()


# =========================================================
# 12. MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    omega = np.random.normal(0.0, SIGMA_OMEGA, N)

    # Generate datasets WITH TRAJECTORIES
    X_train, Y_train, TRAJ_train, t_eval = generate_dataset_with_trajectory(
        M_TRAIN, omega, K, T, n_steps=N_TIME_STEPS
    )
    X_test, Y_test, TRAJ_test, _ = generate_dataset_with_trajectory(
        M_TEST, omega, K, T, n_steps=N_TIME_STEPS
    )

    # Create DataLoader
    train_dataset = KuramotoDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, pin_memory=False)

    # Initialize model
    model = SurrogateNet(dim=2 * N)

    # Train
    history = train(model, train_loader, max_epochs=MAX_EPOCHS,
                    patience=PATIENCE, min_delta=MIN_DELTA)

    # Evaluate
    theta_true, theta_pred, r_true, r_pred = evaluate(model, X_test, Y_test)

    # =========================================================
    # CREATE ALL VISUALIZATIONS
    # =========================================================
    print("📊 Creating all visualizations...\n")

    # 1. Training history
    plot_training_history(history)

    # 2. Sample selection for detailed plots
    sample_idx = 0
    theta0 = cos_sin_to_angles(X_test[sample_idx])
    thetaT_sample_true = theta_true[sample_idx]
    thetaT_sample_pred = theta_pred[sample_idx]

    # 3. Phase distribution (3-panel)
    plot_phase_distribution(theta0, thetaT_sample_true, thetaT_sample_pred)

    # 4. Order parameter evolution r(t)
    plot_order_parameter_vs_time(TRAJ_test[sample_idx], t_eval)

    # 5. Final state polar comparison
    plot_final_state(thetaT_sample_true, thetaT_sample_pred)

    # 6. Order parameters for all test samples
    plot_order_parameters(r_true, r_pred)

    # 7. Error distribution
    plot_error_distribution(r_true, r_pred)

    print("\n" + "=" * 60)
    print("✅ ALL COMPLETED!")
    print("=" * 60)
    print("\n📊 Generated plots:")
    print("1. training_history.png - Loss evolution")
    print("2. phase_distribution.png - Initial vs Final (3 panels)")
    print("3. order_parameter_time.png - r(t) evolution")
    print("4. final_state.png - Polar comparison")
    print("5. order_parameters.png - Test set comparison")
    print("6. error_distribution.png - Error analysis")
    print("=" * 60)