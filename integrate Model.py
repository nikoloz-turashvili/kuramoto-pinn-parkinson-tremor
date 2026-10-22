import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from sklearn.decomposition import PCA


# ========================================
# 0. UTILITIES
# ========================================

def set_seed(seed=123):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def wrap_angles(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


# ========================================
# 1. PHYSICS CORE - Single Version
# ========================================

def kuramoto_rhs(theta, omega, K):
    """
    Physical Kuramoto RHS

    theta: (N,) - phases
    omega: (N,) - natural frequencies
    K: scalar - coupling strength

    Returns: (N,) - dθ/dt
    """
    N = len(theta)
    diff = theta[None, :] - theta[:, None]  # (N, N) matrix: θ_j - θ_i
    coupling = np.sum(np.sin(diff), axis=1)  # (N,)
    return omega + (K / N) * coupling


def kuramoto_rhs_torch(theta, omega, K):
    """
    PyTorch version for training

    theta: (batch, N) or (N,)
    omega: (N,)
    K: scalar tensor or (batch,) tensor
    """
    if theta.dim() == 1:
        theta = theta.unsqueeze(0)  # (1, N)

    N = theta.shape[1]
    batch_size = theta.shape[0]

    diff = theta.unsqueeze(2) - theta.unsqueeze(1)  # (batch, N, N)
    coupling = torch.sum(torch.sin(diff), dim=2)  # (batch, N)

    omega_expanded = omega.unsqueeze(0).expand(batch_size, -1)  # (batch, N)

    # Handle K as scalar or batch of scalars
    if K.dim() == 0:
        # Scalar K
        K_expanded = K
    else:
        # Batch of K values: (batch,) -> (batch, 1)
        K_expanded = K.unsqueeze(1)

    return omega_expanded + (K_expanded / N) * coupling


# ========================================
# 2. RESIDUAL NN (Hybrid Model)
# ========================================

class ResidualNN(nn.Module):
    """
    Learns: r_θ(θ) - everything that physics cannot explain
    Does NOT touch K
    Directly corrects θ̇
    """

    def __init__(self, N):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, N)
        )

    def forward(self, theta):
        return self.net(theta)


# ========================================
# 3. COUPLING NN (Advanced Model)
# ========================================

class CouplingNN(nn.Module):
    """
    Learns: ΔK(θ) - single scalar correction to K
    Cannot "override" residual role
    Outputs ONLY coupling correction
    """

    def __init__(self, N):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, theta):
        return self.net(theta).squeeze(-1)  # Returns scalar per batch


# ========================================
# 4. DATA LOADING FROM ALAMEDA CSV
# ========================================

def load_alameda_data(csv_path, N_osc=5, dt=0.01):
    """
    Load Parkinson's tremor data from ALAMEDA CSV
    Convert to phase trajectories using PCA + Hilbert transform

    Returns:
        theta_hist: (T, N_osc) - phase trajectories
        omega: (N_osc,) - estimated natural frequencies
        dt: time step
    """
    print("=" * 60)
    print("LOADING ALAMEDA PARKINSON'S DATA")
    print("=" * 60)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV. Shape: {df.shape}")
    print(f"Columns (first 30): {df.columns.tolist()[:30]}")

    # Detect if timeseries or features
    cols = df.columns.str.lower()
    is_timeseries = any(("accel" in c or "acceleration" in c or "gyro" in c or
                         "rotation" in c or "timeseries" in c) for c in cols)

    if not is_timeseries and df.shape[0] > 2000:
        is_timeseries = True

    print(f"Detected as timeseries: {is_timeseries}")

    # Extract numeric data
    numeric = df.select_dtypes(include=[np.number])
    data_matrix = numeric.values

    if is_timeseries:
        # Direct timeseries processing
        T = data_matrix.shape[0]
        print(f"Timeseries shape: {data_matrix.shape}")

        # PCA to reduce to N_osc components
        n_components = min(N_osc, data_matrix.shape[1])
        pca = PCA(n_components=n_components)
        comps = pca.fit_transform(data_matrix)  # (T, n_components)
        print(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")

        # Pad if needed
        if n_components < N_osc:
            pad = np.random.normal(scale=1e-3, size=(T, N_osc - n_components))
            comps = np.hstack([comps, pad])

        sigs = comps.T  # (N_osc, T)

    else:
        # Feature CSV - create pseudo-timeseries
        S = data_matrix.shape[0]
        print(f"Feature matrix shape: {data_matrix.shape}")

        n_components = min(N_osc, data_matrix.shape[1])
        pca = PCA(n_components=n_components)
        comps = pca.fit_transform(data_matrix)  # (S, n_components)
        print(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")

        # Interpolate to create longer timeseries
        T = max(1024, S * 4)
        sigs = np.zeros((N_osc, T))

        for i in range(n_components):
            xp = np.linspace(0, 1, num=S)
            x = np.linspace(0, 1, num=T)
            interp = np.interp(x, xp, comps[:, i])
            sigs[i] = interp - np.mean(interp)

        # Pad if needed
        if n_components < N_osc:
            sigs[n_components:] = np.random.normal(scale=1e-3, size=(N_osc - n_components, T))

    # Convert to phases using Hilbert transform
    theta_hist = np.zeros((sigs.shape[1], N_osc))

    for i in range(N_osc):
        analytic = hilbert(sigs[i])
        phase = np.angle(analytic)
        theta_hist[:, i] = wrap_angles(phase)

    print(f"Created theta_hist shape: {theta_hist.shape}")

    # Estimate natural frequencies
    unwrapped = np.unwrap(theta_hist, axis=0)
    dtheta = np.gradient(unwrapped, dt, axis=0)
    omega = np.median(dtheta, axis=0)

    print(f"Estimated omega: {omega}")
    print("✓ Data loading complete")

    return theta_hist, omega, dt


# ========================================
# 5. TRAINING FUNCTIONS
# ========================================

def train_residual_nn(residual_nn, theta_data, theta_dot_true, omega, K,
                      epochs=100, lr=1e-3, batch_size=64, device='cpu'):
    """
    STEP 2: Train Residual NN
    Target: θ̇_true - f_physics(θ, K)
    """
    print("\n" + "=" * 60)
    print("TRAINING RESIDUAL NN (Hybrid Model)")
    print("=" * 60)

    # Convert to torch
    theta_torch = torch.FloatTensor(theta_data).to(device)
    theta_dot_torch = torch.FloatTensor(theta_dot_true).to(device)
    omega_torch = torch.FloatTensor(omega).to(device)
    K_torch = torch.FloatTensor([K]).to(device)

    # Split train/val
    n_samples = len(theta_data)
    n_train = int(0.8 * n_samples)
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_dataset = TensorDataset(theta_torch[train_idx], theta_dot_torch[train_idx])
    val_dataset = TensorDataset(theta_torch[val_idx], theta_dot_torch[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(residual_nn.parameters(), lr=lr)
    best_val_loss = float('inf')
    patience = 20
    wait = 0

    for epoch in range(epochs):
        # Training
        residual_nn.train()
        train_loss = 0
        for theta_batch, theta_dot_batch in train_loader:
            theta_dot_phys = kuramoto_rhs_torch(theta_batch, omega_torch, K_torch)
            residual_pred = residual_nn(theta_batch)
            target = theta_dot_batch - theta_dot_phys

            loss = nn.MSELoss()(residual_pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        residual_nn.eval()
        val_loss = 0
        with torch.no_grad():
            for theta_batch, theta_dot_batch in val_loader:
                theta_dot_phys = kuramoto_rhs_torch(theta_batch, omega_torch, K_torch)
                residual_pred = residual_nn(theta_batch)
                target = theta_dot_batch - theta_dot_phys
                loss = nn.MSELoss()(residual_pred, target)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:3d}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print(f"✓ Residual NN training complete. Best val loss: {best_val_loss:.6f}")


def train_coupling_nn(coupling_nn, residual_nn, theta_data, theta_dot_true,
                      omega, K, epochs=100, lr=1e-3, batch_size=64, device='cpu'):
    """
    STEP 4: Train Coupling NN
    Target: θ̇_true - (f_physics(θ, K) + r_θ(θ))
    Residual NN is FROZEN
    """
    print("\n" + "=" * 60)
    print("TRAINING COUPLING NN (Advanced Model)")
    print("=" * 60)

    # CRITICAL: Freeze residual NN
    for p in residual_nn.parameters():
        p.requires_grad = False
    print("✓ Residual NN frozen")

    # Convert to torch
    theta_torch = torch.FloatTensor(theta_data).to(device)
    theta_dot_torch = torch.FloatTensor(theta_dot_true).to(device)
    omega_torch = torch.FloatTensor(omega).to(device)
    K_torch = torch.FloatTensor([K]).to(device)

    # Split train/val
    n_samples = len(theta_data)
    n_train = int(0.8 * n_samples)
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_dataset = TensorDataset(theta_torch[train_idx], theta_dot_torch[train_idx])
    val_dataset = TensorDataset(theta_torch[val_idx], theta_dot_torch[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(coupling_nn.parameters(), lr=lr)
    best_val_loss = float('inf')
    patience = 20
    wait = 0

    for epoch in range(epochs):
        # Training
        coupling_nn.train()
        train_loss = 0
        for theta_batch, theta_dot_batch in train_loader:
            phys = kuramoto_rhs_torch(theta_batch, omega_torch, K_torch)

            with torch.no_grad():
                residual = residual_nn(theta_batch)

            target = theta_dot_batch - (phys + residual)

            delta_K = coupling_nn(theta_batch)
            struct = kuramoto_rhs_torch(theta_batch, omega_torch, delta_K)

            loss = nn.MSELoss()(struct, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        coupling_nn.eval()
        val_loss = 0
        with torch.no_grad():
            for theta_batch, theta_dot_batch in val_loader:
                phys = kuramoto_rhs_torch(theta_batch, omega_torch, K_torch)
                residual = residual_nn(theta_batch)
                target = theta_dot_batch - (phys + residual)

                delta_K = coupling_nn(theta_batch)
                struct = kuramoto_rhs_torch(theta_batch, omega_torch, delta_K)

                loss = nn.MSELoss()(struct, target)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:3d}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print(f"✓ Coupling NN training complete. Best val loss: {best_val_loss:.6f}")


# ========================================
# 6. EVALUATION & VISUALIZATION
# ========================================

def evaluate_models(theta_data, theta_dot_true, residual_nn, coupling_nn, omega, K, device='cpu'):
    """
    Compare all three models
    """
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    theta_torch = torch.FloatTensor(theta_data).to(device)
    theta_dot_torch = torch.FloatTensor(theta_dot_true).to(device)
    K_torch = torch.FloatTensor([K]).to(device)
    omega_torch = torch.FloatTensor(omega).to(device)

    with torch.no_grad():
        # Physics only
        theta_dot_phys = kuramoto_rhs_torch(theta_torch, omega_torch, K_torch)
        mse_phys = nn.MSELoss()(theta_dot_phys, theta_dot_torch)

        # Hybrid (physics + residual)
        residual = residual_nn(theta_torch)
        theta_dot_hybrid = theta_dot_phys + residual
        mse_hybrid = nn.MSELoss()(theta_dot_hybrid, theta_dot_torch)

        # Advanced (physics + residual + coupling)
        delta_K = coupling_nn(theta_torch)
        theta_dot_coupled = kuramoto_rhs_torch(theta_torch, omega_torch, K_torch + delta_K)
        theta_dot_advanced = theta_dot_coupled + residual
        mse_advanced = nn.MSELoss()(theta_dot_advanced, theta_dot_torch)

    print(f"Physics-only MSE:  {mse_phys.item():.6f}")
    print(f"Hybrid MSE:        {mse_hybrid.item():.6f}")
    print(f"Advanced MSE:      {mse_advanced.item():.6f}")

    improvement_hybrid = (1 - mse_hybrid.item() / mse_phys.item()) * 100
    improvement_advanced = (1 - mse_advanced.item() / mse_hybrid.item()) * 100

    print(f"\nHybrid improvement:   {improvement_hybrid:.2f}%")
    print(f"Advanced improvement: {improvement_advanced:.2f}%")

    return mse_phys.item(), mse_hybrid.item(), mse_advanced.item()


def plot_phase_circles(theta_hist, theta_phys, theta_hybrid, theta_advanced, dt, save_dir=''):
    """
    Plot initial and final phase distributions on unit circles for each method
    """
    print("\n" + "=" * 60)
    print("CREATING PHASE CIRCLE VISUALIZATIONS")
    print("=" * 60)

    N = 5  # First 5 oscillators

    # Get initial and final states
    theta_initial = theta_hist[0, :N]
    theta_true_final = theta_hist[-1, :N]
    theta_phys_final = theta_phys[-1, :N]
    theta_hybrid_final = theta_hybrid[-1, :N]
    theta_advanced_final = theta_advanced[-1, :N]

    # Create figure with 5 subplots
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    colors = ['red', 'blue', 'green', 'orange', 'purple']

    # Function to plot on unit circle
    def plot_circle(ax, phases, title):
        # Draw unit circle
        circle = plt.Circle((0, 0), 1, fill=False, color='gray', linewidth=1.5, alpha=0.5)
        ax.add_patch(circle)

        # Plot phases
        for i, phase in enumerate(phases):
            x = np.cos(phase)
            y = np.sin(phase)
            ax.plot(x, y, 'o', color=colors[i], markersize=15, label=f'Osc {i + 1}')
            # Draw line from center
            ax.plot([0, x], [0, y], '-', color=colors[i], linewidth=2, alpha=0.6)

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)

    # Plot each method
    plot_circle(axes[0], theta_initial, 'Initial State')
    plot_circle(axes[1], theta_true_final, 'True Data (Final)')
    plot_circle(axes[2], theta_phys_final, 'Physics-Only (Final)')
    plot_circle(axes[3], theta_hybrid_final, 'Hybrid Model (Final)')
    plot_circle(axes[4], theta_advanced_final, 'Advanced Model (Final)')

    # Add legend to first subplot
    axes[0].legend(loc='upper right', fontsize=8, framealpha=0.9)

    plt.tight_layout()

    # Save to desktop
    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
    save_path = os.path.join(desktop_path, 'phase_circles_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_order_parameter_5sec(theta_hist, theta_phys, theta_hybrid, theta_advanced, dt, save_dir=''):
    """
    Plot order parameter for first 5 seconds for each method (4 separate plots)
    """
    print("\n" + "=" * 60)
    print("CREATING ORDER PARAMETER PLOTS (FIRST 5 SECONDS)")
    print("=" * 60)

    # Calculate how many timesteps in 5 seconds
    T_5sec = int(5.0 / dt)
    T_5sec = min(T_5sec, len(theta_hist))

    time = np.arange(T_5sec) * dt

    # Function to calculate order parameter
    def order_param(theta_traj):
        return np.abs(np.mean(np.exp(1j * theta_traj), axis=1))

    # Calculate order parameters
    r_true = order_param(theta_hist[:T_5sec])
    r_phys = order_param(theta_phys[:T_5sec])
    r_hybrid = order_param(theta_hybrid[:T_5sec])
    r_advanced = order_param(theta_advanced[:T_5sec])

    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')

    # Plot 1: True Data
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time, r_true, 'k-', linewidth=2.5)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Order Parameter r(t)', fontsize=12)
    ax.set_title('True Data - Order Parameter (First 5 seconds)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    save_path = os.path.join(desktop_path, 'order_param_true.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

    # Plot 2: Physics-Only
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time, r_phys, 'b-', linewidth=2.5)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Order Parameter r(t)', fontsize=12)
    ax.set_title('Physics-Only Model - Order Parameter (First 5 seconds)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    save_path = os.path.join(desktop_path, 'order_param_physics.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

    # Plot 3: Hybrid Model
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time, r_hybrid, 'g-', linewidth=2.5)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Order Parameter r(t)', fontsize=12)
    ax.set_title('Hybrid Model - Order Parameter (First 5 seconds)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    save_path = os.path.join(desktop_path, 'order_param_hybrid.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

    # Plot 4: Advanced Model
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time, r_advanced, 'r-', linewidth=2.5)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Order Parameter r(t)', fontsize=12)
    ax.set_title('Advanced Model - Order Parameter (First 5 seconds)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    save_path = os.path.join(desktop_path, 'order_param_advanced.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def rollout_and_visualize(theta_hist, residual_nn, coupling_nn, omega, K, dt, device='cpu'):
    """
    Rollout simulation and create visualizations
    """
    print("\n" + "=" * 60)
    print("ROLLOUT SIMULATION")
    print("=" * 60)

    T = len(theta_hist)
    N = theta_hist.shape[1]
    theta0 = theta_hist[0]

    # Storage
    theta_phys = np.zeros((T, N))
    theta_hybrid = np.zeros((T, N))
    theta_advanced = np.zeros((T, N))

    theta_phys[0] = theta0
    theta_hybrid[0] = theta0
    theta_advanced[0] = theta0

    # Define RHS functions
    def phys_rhs(theta):
        return kuramoto_rhs(theta, omega, K)

    def hybrid_rhs(theta):
        theta_torch = torch.FloatTensor(theta).to(device)
        with torch.no_grad():
            residual = residual_nn(theta_torch).cpu().numpy()
        return kuramoto_rhs(theta, omega, K) + residual

    def advanced_rhs(theta):
        theta_torch = torch.FloatTensor(theta).to(device)
        with torch.no_grad():
            delta_K = coupling_nn(theta_torch).item()
            residual = residual_nn(theta_torch).cpu().numpy()
        return kuramoto_rhs(theta, omega, K + delta_K) + residual

    # Integrate
    print("Running rollout...")
    for i in range(1, T):
        theta_phys[i] = wrap_angles(theta_phys[i - 1] + dt * phys_rhs(theta_phys[i - 1]))
        theta_hybrid[i] = wrap_angles(theta_hybrid[i - 1] + dt * hybrid_rhs(theta_hybrid[i - 1]))
        theta_advanced[i] = wrap_angles(theta_advanced[i - 1] + dt * advanced_rhs(theta_advanced[i - 1]))

    # Create phase circle visualizations
    plot_phase_circles(theta_hist, theta_phys, theta_hybrid, theta_advanced, dt)

    # Create order parameter plots (first 5 seconds)
    plot_order_parameter_5sec(theta_hist, theta_phys, theta_hybrid, theta_advanced, dt)

    return theta_phys, theta_hybrid, theta_advanced


# ========================================
# 7. MAIN EXECUTION
# ========================================

def main():
    print("=" * 60)
    print("KURAMOTO HYBRID-ADVANCED: PARKINSON'S TREMOR ANALYSIS")
    print("=" * 60)

    set_seed(123)

    # Configuration
    csv_path = "ALAMEDA_PD_tremor_dataset.csv"  # Change this to your path
    N_osc = 5
    K_assumed = 1.0
    dt = 0.01

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # STEP 1: Load and process data
    theta_hist, omega, dt = load_alameda_data(csv_path, N_osc=N_osc, dt=dt)

    # Compute derivatives
    print("\nComputing derivatives...")
    unwrapped = np.unwrap(theta_hist, axis=0)
    theta_dot_hist = np.gradient(unwrapped, dt, axis=0)
    print(f"theta_dot shape: {theta_dot_hist.shape}")

    # Use all but last point
    theta_data = theta_hist[:-1]
    theta_dot_data = theta_dot_hist[:-1]

    # STEP 2: Train Residual NN
    residual_nn = ResidualNN(N_osc).to(device)
    train_residual_nn(residual_nn, theta_data, theta_dot_data, omega, K_assumed,
                      epochs=200, lr=1e-3, batch_size=64, device=device)

    # STEP 3: Freeze Residual NN
    print("\n" + "=" * 60)
    print("STEP 3: Freezing Residual NN")
    print("=" * 60)
    for p in residual_nn.parameters():
        p.requires_grad = False
    print("✓ Residual NN parameters frozen")

    # STEP 4: Train Coupling NN
    coupling_nn = CouplingNN(N_osc).to(device)
    train_coupling_nn(coupling_nn, residual_nn, theta_data, theta_dot_data,
                      omega, K_assumed, epochs=200, lr=1e-3, batch_size=64, device=device)

    # Evaluation
    evaluate_models(theta_data[:2000], theta_dot_data[:2000],
                    residual_nn, coupling_nn, omega, K_assumed, device=device)

    # Rollout and visualize
    rollout_and_visualize(theta_hist, residual_nn, coupling_nn, omega, K_assumed, dt, device=device)

    # Save models
    torch.save(residual_nn.state_dict(), 'alameda_residual_nn.pt')
    torch.save(coupling_nn.state_dict(), 'alameda_coupling_nn.pt')
    print("\n✓ Models saved:")
    print("  - alameda_residual_nn.pt")
    print("  - alameda_coupling_nn.pt")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print("\nGenerated files on Desktop:")
    print("  1. phase_circles_comparison.png - Initial and final phase distributions")
    print("  2. order_param_true.png - True data order parameter (first 5 sec)")
    print("  3. order_param_physics.png - Physics-only order parameter (first 5 sec)")
    print("  4. order_param_hybrid.png - Hybrid model order parameter (first 5 sec)")
    print("  5. order_param_advanced.png - Advanced model order parameter (first 5 sec)")
    print("  6. alameda_residual_nn.pt - Trained residual NN weights")
    print("  7. alameda_coupling_nn.pt - Trained coupling NN weights")


if __name__ == "__main__":
    main() 