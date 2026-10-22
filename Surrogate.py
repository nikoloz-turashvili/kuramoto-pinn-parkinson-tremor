import os
import time
import json
import numpy as np
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==================== CONFIGURATION ====================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Problem parameters
N = 10
K_GLOBAL = 3.5
SIGMA_OMEGA = 0.25
T_FINAL = 3.0

# Dataset
NUM_TOTAL = 10000
NUM_TRAIN = 8000
NUM_VAL = 1000
NUM_TEST = 1000

# Solver
SOLVER_METHOD = 'RK45'
SOLVER_RTOL = 1e-9
SOLVER_ATOL = 1e-9

# Training
BATCH_SIZE = 256
LR = 3e-4
MAX_EPOCHS = 3000
PATIENCE = 150
WEIGHT_DECAY = 1e-5
HIDDEN_LAYERS = [768, 512, 512, 256, 128]
ACTIVATION = nn.ELU
DROPOUT = 0.05

# Physics loss weights
ALPHA_MSE = 1.0
ALPHA_ORDER = 0.5
ALPHA_PHASE = 0.3

# Paths
DATA_DIR = 'data'
MODEL_DIR = 'models'
RESULTS_DIR = 'results'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_FILE = os.path.join(DATA_DIR, f'kuramoto_ultra_N{N}_T{int(T_FINAL)}_K{K_GLOBAL}.npz')
MODEL_FILE = os.path.join(MODEL_DIR, 'surrogate_ultra.pth')
PLOT_FILE = os.path.join(RESULTS_DIR, 'predictions_ultra.png')
METRICS_FILE = os.path.join(RESULTS_DIR, 'metrics_ultra.json')


# ==================== UTILITIES ====================
def angles_to_complex(theta):
    return np.stack([np.cos(theta), np.sin(theta)], axis=-1)


def complex_to_angles(complex_repr):
    return np.arctan2(complex_repr[..., 1], complex_repr[..., 0])


def angular_wrap(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def angular_distance(a1, a2):
    return np.abs(angular_wrap(a1 - a2))


def order_parameter(theta):
    return np.abs(np.mean(np.exp(1j * theta), axis=-1))


def order_parameter_torch(complex_repr):
    if complex_repr.dim() == 2:
        complex_repr = complex_repr.reshape(-1, N, 2)
    cos_vals = complex_repr[..., 0]
    sin_vals = complex_repr[..., 1]
    mean_cos = cos_vals.mean(dim=-1)
    mean_sin = sin_vals.mean(dim=-1)
    r = torch.sqrt(mean_cos ** 2 + mean_sin ** 2)
    return r


def phase_coherence_torch(complex_repr):
    if complex_repr.dim() == 2:
        complex_repr = complex_repr.reshape(-1, N, 2)
    angles = torch.atan2(complex_repr[..., 1], complex_repr[..., 0])
    mean_angle = torch.atan2(
        complex_repr[..., 1].mean(dim=-1),
        complex_repr[..., 0].mean(dim=-1)
    )
    diffs = angles - mean_angle.unsqueeze(-1)
    diffs_wrapped = torch.atan2(torch.sin(diffs), torch.cos(diffs))
    variance = diffs_wrapped.var(dim=-1)
    return variance


# ==================== KURAMOTO DYNAMICS ====================
def kuramoto_rhs(t, theta, omega, K):
    sin_diffs = np.sin(theta[None, :] - theta[:, None])
    coupling = (K / len(theta)) * np.sum(sin_diffs, axis=1)
    return omega + coupling


def integrate_kuramoto(theta0, omega, K, T):
    sol = solve_ivp(
        fun=lambda t, y: kuramoto_rhs(t, y, omega, K),
        t_span=(0.0, T),
        y0=theta0,
        method=SOLVER_METHOD,
        atol=SOLVER_ATOL,
        rtol=SOLVER_RTOL,
        t_eval=[T]
    )
    return angular_wrap(sol.y[:, -1])


# ==================== DATA GENERATION ====================
def generate_dataset():
    print(f"\n{'=' * 70}")
    print("GENERATING ULTRA-OPTIMIZED DATASET")
    print(f"{'=' * 70}")
    print(f"Configuration:")
    print(f"  N = {N} oscillators")
    print(f"  K = {K_GLOBAL} (STRONG coupling)")
    print(f"  T = {T_FINAL} (SHORT time)")
    print(f"  sigma_omega = {SIGMA_OMEGA} (LOW spread)")
    print(f"  Total samples = {NUM_TOTAL}")
    print(f"{'=' * 70}\n")

    rng = np.random.RandomState(SEED)
    theta0_all = np.zeros((NUM_TOTAL, N))
    thetaT_all = np.zeros((NUM_TOTAL, N))
    omega_all = np.zeros((NUM_TOTAL, N))

    for i in tqdm(range(NUM_TOTAL), desc="Generating"):
        theta0 = rng.uniform(-np.pi, np.pi, N)
        omega = rng.normal(0.0, SIGMA_OMEGA, N)
        thetaT = integrate_kuramoto(theta0, omega, K_GLOBAL, T_FINAL)
        theta0_all[i] = theta0
        thetaT_all[i] = thetaT
        omega_all[i] = omega

    X_complex = angles_to_complex(theta0_all)
    Y_complex = angles_to_complex(thetaT_all)
    X = X_complex.reshape(NUM_TOTAL, -1)
    Y = Y_complex.reshape(NUM_TOTAL, -1)

    r_initial = order_parameter(theta0_all)
    r_final = order_parameter(thetaT_all)

    print(f"\nDataset Statistics:")
    print(f"  X shape: {X.shape}")
    print(f"  Y shape: {Y.shape}")
    print(f"  Initial r: {r_initial.mean():.3f} +/- {r_initial.std():.3f}")
    print(f"  Final r: {r_final.mean():.3f} +/- {r_final.std():.3f}")
    print(f"  Synchronization fraction: {(r_final > 0.9).mean() * 100:.1f}%")

    np.savez_compressed(
        DATA_FILE,
        X=X.astype(np.float32),
        Y=Y.astype(np.float32),
        theta0=theta0_all.astype(np.float32),
        thetaT=thetaT_all.astype(np.float32),
        omega=omega_all.astype(np.float32)
    )
    print(f"\nDataset saved: {DATA_FILE}\n")
    return X, Y, theta0_all, thetaT_all, omega_all


# ==================== PYTORCH COMPONENTS ====================
class KuramotoDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class ResidualBlock(nn.Module):
    def __init__(self, dim, activation=ACTIVATION, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.activation = activation()

    def forward(self, x):
        return self.activation(x + self.net(x))


class UltraSurrogate(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=HIDDEN_LAYERS, activation=ACTIVATION, dropout=DROPOUT, n_residual=2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.LayerNorm(hidden[0]),
            activation()
        )
        layers = []
        for i in range(len(hidden) - 1):
            layers.append(nn.Linear(hidden[i], hidden[i + 1]))
            layers.append(nn.LayerNorm(hidden[i + 1]))
            layers.append(activation())
            layers.append(nn.Dropout(dropout))
        self.hidden = nn.Sequential(*layers)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden[-1], activation, dropout) for _ in range(n_residual)
        ])
        self.output_proj = nn.Linear(hidden[-1], output_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.hidden(x)
        for block in self.residual_blocks:
            x = block(x)
        return self.output_proj(x)


class PhysicsInformedLoss(nn.Module):
    def __init__(self, alpha_mse=ALPHA_MSE, alpha_order=ALPHA_ORDER, alpha_phase=ALPHA_PHASE):
        super().__init__()
        self.alpha_mse = alpha_mse
        self.alpha_order = alpha_order
        self.alpha_phase = alpha_phase
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        loss_mse = self.mse(pred, target)
        r_pred = order_parameter_torch(pred)
        r_target = order_parameter_torch(target)
        loss_order = self.mse(r_pred, r_target)
        var_pred = phase_coherence_torch(pred)
        var_target = phase_coherence_torch(target)
        loss_phase = self.mse(var_pred, var_target)
        total_loss = (self.alpha_mse * loss_mse + self.alpha_order * loss_order + self.alpha_phase * loss_phase)
        return total_loss, loss_mse, loss_order, loss_phase


class EarlyStopping:
    def __init__(self, patience=PATIENCE, min_delta=1e-7):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_epoch = 0
        self.best_state = None
        self.should_stop = False

    def __call__(self, epoch, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                print(f"  Best: epoch {self.best_epoch}, loss {self.best_loss:.6e}")
                self.should_stop = True


# ==================== TRAINING ====================
def train_model(model, train_loader, val_loader, device):
    print(f"\n{'=' * 70}")
    print("ULTRA-OPTIMIZED TRAINING")
    print(f"{'=' * 70}")
    print(f"Device: {device}")
    print(f"Architecture: {HIDDEN_LAYERS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LR}")
    print(f"Physics-informed loss:")
    print(f"  alpha_mse={ALPHA_MSE}, alpha_order={ALPHA_ORDER}, alpha_phase={ALPHA_PHASE}")
    print(f"{'=' * 70}\n")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=40, min_lr=1e-6
    )
    criterion = PhysicsInformedLoss()
    early_stop = EarlyStopping(patience=PATIENCE)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        n_train = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss, loss_mse, loss_order, loss_phase = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
            n_train += len(xb)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        val_order = 0.0
        val_phase = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss, loss_mse, loss_order, loss_phase = criterion(pred, yb)
                batch_size = len(xb)
                val_loss += loss.item() * batch_size
                val_mse += loss_mse.item() * batch_size
                val_order += loss_order.item() * batch_size
                val_phase += loss_phase.item() * batch_size
                n_val += batch_size
        val_loss /= n_val
        val_mse /= n_val
        val_order /= n_val
        val_phase /= n_val

        scheduler.step(val_loss)

        if epoch == 1 or epoch % 20 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:4d} | Train: {train_loss:.6e} | Val: {val_loss:.6e} | LR: {lr:.2e}")
            print(f"  | MSE: {val_mse:.6e} | Order: {val_order:.6e} | Phase: {val_phase:.6e}")

        early_stop(epoch, val_loss, model)
        if early_stop.should_stop:
            model.load_state_dict(early_stop.best_state)
            break

    print(f"{'=' * 70}\n")
    return model, early_stop.best_epoch, early_stop.best_loss


# ==================== EVALUATION ====================
def evaluate(model, X_test, Y_test, theta0_test, thetaT_test, device):
    print(f"\n{'=' * 70}")
    print("EVALUATION")
    print(f"{'=' * 70}\n")

    model = model.to(device).eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X_test).float().to(device)
        Y_pred = model(X_t).cpu().numpy()

    Y_pred_complex = Y_pred.reshape(-1, N, 2)
    Y_true_complex = Y_test.reshape(-1, N, 2)
    theta_pred = complex_to_angles(Y_pred_complex)
    theta_true = thetaT_test

    ang_dists = angular_distance(theta_pred, theta_true)
    mae = ang_dists.mean()
    rmse = np.sqrt((ang_dists ** 2).mean())
    mae_deg = np.rad2deg(mae)
    rmse_deg = np.rad2deg(rmse)
    complex_mse = ((Y_pred_complex - Y_true_complex) ** 2).mean()

    r_true = order_parameter(theta_true)
    r_pred = order_parameter(theta_pred)
    r_mae = np.abs(r_true - r_pred).mean()
    r_rmse = np.sqrt(((r_true - r_pred) ** 2).mean())

    per_osc_error = ang_dists.mean(axis=0)

    print(f"Angular Errors:")
    print(f"  MAE: {mae:.4f} rad = {mae_deg:.2f} deg")
    print(f"  RMSE: {rmse:.4f} rad = {rmse_deg:.2f} deg")
    print(f"  Max: {ang_dists.max():.4f} rad = {np.rad2deg(ang_dists.max()):.2f} deg")
    print(f"\nPer-Oscillator MAE (degrees):")
    print(f"  {np.rad2deg(per_osc_error)}")
    print(f"\nComplex MSE: {complex_mse:.6e}")
    print(f"\nOrder Parameter:")
    print(f"  MAE: {r_mae:.5f}")
    print(f"  RMSE: {r_rmse:.5f}")
    print(f"  Correlation: {np.corrcoef(r_true, r_pred)[0, 1]:.5f}")

    if mae_deg < 10.0 and r_mae < 0.01:
        print(f"\n{'=' * 70}")
        print("SUCCESS! Both targets achieved!")
        print(f"  Angular MAE: {mae_deg:.2f} deg < 10 deg")
        print(f"  Order MAE: {r_mae:.5f} < 0.01")
        print(f"{'=' * 70}")
    else:
        print(f"\n{'=' * 70}")
        print("Targets not fully met:")
        print(f"  {'OK' if mae_deg < 10.0 else 'FAIL'} Angular MAE: {mae_deg:.2f} deg {'<' if mae_deg < 10.0 else '>='} 10 deg")
        print(f"  {'OK' if r_mae < 0.01 else 'FAIL'} Order MAE: {r_mae:.5f} {'<' if r_mae < 0.01 else '>='} 0.01")
        print(f"{'=' * 70}")

    print(f"\n{'=' * 70}\n")

    return {
        'mae_rad': float(mae),
        'mae_deg': float(mae_deg),
        'rmse_rad': float(rmse),
        'rmse_deg': float(rmse_deg),
        'complex_mse': float(complex_mse),
        'r_mae': float(r_mae),
        'r_rmse': float(r_rmse),
        'r_corr': float(np.corrcoef(r_true, r_pred)[0, 1]),
        'theta_pred': theta_pred,
        'theta_true': theta_true,
        'r_pred': r_pred,
        'r_true': r_true
    }


def measure_speedup(model, X_test, theta0_test, omega_test, device):
    print(f"{'=' * 70}")
    print("SPEEDUP MEASUREMENT")
    print(f"{'=' * 70}\n")

    n_samples = min(100, len(X_test))

    print(f"Timing ODE solver ({n_samples} samples)...")
    ode_times = []
    for i in tqdm(range(n_samples), desc="ODE"):
        t0 = time.perf_counter()
        _ = integrate_kuramoto(theta0_test[i], omega_test[i], K_GLOBAL, T_FINAL)
        ode_times.append(time.perf_counter() - t0)
    ode_mean = np.mean(ode_times)

    print(f"Timing NN ({n_samples} samples)...")
    model = model.to(device).eval()
    X_t = torch.from_numpy(X_test[:n_samples]).float().to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(X_t)

    n_repeats = 100
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_repeats):
            _ = model(X_t)
    nn_time = (time.perf_counter() - t0) / (n_repeats * n_samples)

    speedup = ode_mean / nn_time

    print(f"\nResults:")
    print(f"  ODE: {ode_mean * 1000:.2f} ms/sample")
    print(f"  NN: {nn_time * 1e6:.2f} us/sample")
    print(f"  Speedup: {speedup:.0f}x")
    print(f"\n{'=' * 70}\n")

    return {
        'ode_time_ms': float(ode_mean * 1000),
        'nn_time_us': float(nn_time * 1e6),
        'speedup': float(speedup)
    }


def plot_results(theta_true, theta_pred, r_true, r_pred):
    rng = np.random.RandomState(SEED)
    n_samples = 4
    sample_ids = rng.choice(len(theta_true), n_samples, replace=False)

    fig, axes = plt.subplots(2, n_samples, figsize=(4 * n_samples, 8))

    for i, sid in enumerate(sample_ids):
        ax = axes[0, i]
        x = np.arange(N)
        w = 0.35
        ax.bar(x - w / 2, theta_true[sid], w, label='True', alpha=0.8, color='#2E86AB')
        ax.bar(x + w / 2, theta_pred[sid], w, label='Pred', alpha=0.8, color='#A23B72')
        ax.set_xlabel('Oscillator', fontsize=10)
        ax.set_ylabel('Phase (rad)', fontsize=10)
        ax.set_title(f'Sample {sid}', fontsize=11, fontweight='bold')
        ax.set_ylim(-np.pi - 0.5, np.pi + 0.5)
        ax.axhline(0, color='k', linestyle='--', lw=0.8, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=9)
        ax.grid(alpha=0.2)

        error = angular_distance(theta_pred[sid], theta_true[sid])
        mae_sample = error.mean()
        ax.text(0.5, 0.95, f'MAE: {np.rad2deg(mae_sample):.1f} deg',
                transform=ax.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax = axes[1, i]
        bars = ax.bar(['True', 'Pred'], [r_true[sid], r_pred[sid]],
                      color=['#2E86AB', '#A23B72'], alpha=0.8, width=0.6)
        ax.set_ylabel('Order Parameter r', fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.set_title(f'r: {r_true[sid]:.3f} vs {r_pred[sid]:.3f}', fontsize=10)
        ax.grid(axis='y', alpha=0.2)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h, f'{h:.3f}',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=200, bbox_inches='tight')
    print(f"Plot saved: {PLOT_FILE}")
    plt.close()


# ==================== MAIN ====================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n" + "=" * 70)
    print("KURAMOTO SURROGATE - ULTRA-OPTIMIZED COMPLETE")
    print("=" * 70 + "\n")

    if os.path.exists(DATA_FILE):
        print(f"Loading: {DATA_FILE}\n")
        data = np.load(DATA_FILE)
        X, Y = data['X'], data['Y']
        theta0, thetaT, omega = data['theta0'], data['thetaT'], data['omega']
    else:
        X, Y, theta0, thetaT, omega = generate_dataset()

    X_train = X[:NUM_TRAIN]
    Y_train = Y[:NUM_TRAIN]
    X_val = X[NUM_TRAIN:NUM_TRAIN + NUM_VAL]
    Y_val = Y[NUM_TRAIN:NUM_TRAIN + NUM_VAL]
    X_test = X[NUM_TRAIN + NUM_VAL:NUM_TRAIN + NUM_VAL + NUM_TEST]
    Y_test = Y[NUM_TRAIN + NUM_VAL:NUM_TRAIN + NUM_VAL + NUM_TEST]
    theta0_test = theta0[NUM_TRAIN + NUM_VAL:NUM_TRAIN + NUM_VAL + NUM_TEST]
    thetaT_test = thetaT[NUM_TRAIN + NUM_VAL:NUM_TRAIN + NUM_VAL + NUM_TEST]
    omega_test = omega[NUM_TRAIN + NUM_VAL:NUM_TRAIN + NUM_VAL + NUM_TEST]

    print(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}\n")

    model = UltraSurrogate(input_dim=2 * N, output_dim=2 * N)

    if os.path.exists(MODEL_FILE):
        print(f"Loading model: {MODEL_FILE}\n")
        checkpoint = torch.load(MODEL_FILE, map_location='cpu')
        model.load_state_dict(checkpoint['model_state'])
        best_epoch = checkpoint.get('best_epoch', -1)
        best_loss = checkpoint.get('best_loss', -1)
    else:
        train_ds = KuramotoDataset(X_train, Y_train)
        val_ds = KuramotoDataset(X_val, Y_val)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        model, best_epoch, best_loss = train_model(model, train_loader, val_loader, device)

        torch.save({
            'model_state': model.state_dict(),
            'best_epoch': best_epoch,
            'best_loss': best_loss,
            'config': {
                'N': N,
                'K': K_GLOBAL,
                'T': T_FINAL,
                'hidden': HIDDEN_LAYERS,
                'batch_size': BATCH_SIZE,
                'lr': LR
            }
        }, MODEL_FILE)
        print(f"Model saved: {MODEL_FILE}\n")

    metrics = evaluate(model, X_test, Y_test, theta0_test, thetaT_test, device)
    speedup_metrics = measure_speedup(model, X_test, theta0_test, omega_test, device)
    plot_results(metrics['theta_true'], metrics['theta_pred'], metrics['r_true'], metrics['r_pred'])

    final_metrics = {
        'mae_deg': metrics['mae_deg'],
        'rmse_deg': metrics['rmse_deg'],
        'r_mae': metrics['r_mae'],
        'r_rmse': metrics['r_rmse'],
        'r_corr': metrics['r_corr'],
        'complex_mse': metrics['complex_mse'],
        'speedup': speedup_metrics['speedup'],
        'ode_time_ms': speedup_metrics['ode_time_ms'],
        'nn_time_us': speedup_metrics['nn_time_us'],
        'best_epoch': best_epoch,
        'best_val_loss': best_loss,
        'config': {
            'N': N,
            'K': K_GLOBAL,
            'T': T_FINAL,
            'num_train': NUM_TRAIN,
            'hidden_layers': HIDDEN_LAYERS,
            'batch_size': BATCH_SIZE,
            'lr': LR
        }
    }

    with open(METRICS_FILE, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Metrics saved: {METRICS_FILE}\n")

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"Order Error: {metrics['r_mae']:.5f} (target: < 0.01)")
    print(f"Speedup: {speedup_metrics['speedup']:.0f}x")
    print(f"Model: {MODEL_FILE}")
    print(f"Results: {RESULTS_DIR}/")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()


    def create_simple_plots(metrics, X_test, Y_test, theta0_test, thetaT_test):
        """სამი მთავარი ნახაზი"""

        theta_pred = metrics['theta_pred']
        theta_true = metrics['theta_true']
        r_pred = metrics['r_pred']
        r_true = metrics['r_true']

        # ========== 1. ORDER PARAMETER VS TIME ==========
        fig1 = plt.figure(figsize=(12, 6))

        sample_indices = np.arange(len(r_true))

        plt.plot(sample_indices, r_true, 'o-', label='True', alpha=0.6,
                 markersize=3, color='#2E86AB', linewidth=1.5)
        plt.plot(sample_indices, r_pred, 's-', label='Predicted', alpha=0.6,
                 markersize=3, color='#A23B72', linewidth=1.5)

        plt.xlabel('Sample Index (Time Sequence)', fontsize=13, fontweight='bold')
        plt.ylabel('Order Parameter r', fontsize=13, fontweight='bold')
        plt.title('Order Parameter Evolution', fontsize=15, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.ylim(0, 1.1)

        r_mae = metrics['r_mae']
        r_corr = metrics['r_corr']
        plt.text(0.02, 0.98, f'MAE: {r_mae:.5f}\nCorrelation: {r_corr:.4f}',
                 transform=plt.gca().transAxes, fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        order_file = os.path.join(RESULTS_DIR, 'order_parameter_time.png')
        plt.savefig(order_file, dpi=200, bbox_inches='tight')
        print(f"\nOrder parameter plot saved: {order_file}")
        plt.show()

        # ========== 2. PHASE DISTRIBUTION VS TIME ==========
        fig2 = plt.figure(figsize=(14, 7))

        n_samples_display = min(50, len(theta_true))
        samples_to_show = np.linspace(0, len(theta_true) - 1, n_samples_display, dtype=int)

        X_mesh, Y_mesh = np.meshgrid(samples_to_show, np.arange(N))

        plt.subplot(1, 2, 1)
        phases_true_subset = theta_true[samples_to_show].T
        im1 = plt.pcolormesh(X_mesh, Y_mesh, phases_true_subset, cmap='twilight',
                             shading='auto', vmin=-np.pi, vmax=np.pi)
        plt.colorbar(im1, label='Phase (radians)')
        plt.xlabel('Sample Index', fontsize=13, fontweight='bold')
        plt.ylabel('Oscillator Index', fontsize=13, fontweight='bold')
        plt.title('True Phase Distribution', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.2)

        plt.subplot(1, 2, 2)
        phases_pred_subset = theta_pred[samples_to_show].T
        im2 = plt.pcolormesh(X_mesh, Y_mesh, phases_pred_subset, cmap='twilight',
                             shading='auto', vmin=-np.pi, vmax=np.pi)
        plt.colorbar(im2, label='Phase (radians)')
        plt.xlabel('Sample Index', fontsize=13, fontweight='bold')
        plt.ylabel('Oscillator Index', fontsize=13, fontweight='bold')
        plt.title('Predicted Phase Distribution', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.2)

        plt.suptitle('Phase Evolution Over Time', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        phase_file = os.path.join(RESULTS_DIR, 'phase_distribution_time.png')
        plt.savefig(phase_file, dpi=200, bbox_inches='tight')
        print(f"Phase distribution plot saved: {phase_file}")
        plt.show()

        # ========== 3. ERROR EVOLUTION OVER TIME ==========
        fig3 = plt.figure(figsize=(12, 8))

        ang_dists = angular_distance(theta_pred, theta_true)

        plt.subplot(2, 1, 1)
        sample_errors = np.rad2deg(ang_dists.mean(axis=1))
        plt.plot(sample_indices, sample_errors, '-', alpha=0.7, color='#2E86AB', linewidth=1.5)
        plt.axhline(sample_errors.mean(), color='r', linestyle='--', lw=2,
                    label=f'Mean: {sample_errors.mean():.2f} deg')
        plt.fill_between(sample_indices, 0, sample_errors, alpha=0.3, color='#2E86AB')
        plt.xlabel('Sample Index', fontsize=13, fontweight='bold')
        plt.ylabel('Mean Angular Error (degrees)', fontsize=13, fontweight='bold')
        plt.title('Angular Error Evolution', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)

        plt.text(0.02, 0.98,
                 f'Min: {sample_errors.min():.2f} deg\nMax: {sample_errors.max():.2f} deg\nStd: {sample_errors.std():.2f} deg',
                 transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.subplot(2, 1, 2)
        order_errors = np.abs(r_pred - r_true)
        plt.plot(sample_indices, order_errors, '-', alpha=0.7, color='#A23B72', linewidth=1.5)
        plt.axhline(order_errors.mean(), color='r', linestyle='--', lw=2,
                    label=f'Mean: {order_errors.mean():.5f}')
        plt.fill_between(sample_indices, 0, order_errors, alpha=0.3, color='#A23B72')
        plt.xlabel('Sample Index', fontsize=13, fontweight='bold')
        plt.ylabel('Order Parameter Error', fontsize=13, fontweight='bold')
        plt.title('Order Parameter Error Evolution', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)

        plt.text(0.02, 0.98,
                 f'Min: {order_errors.min():.5f}\nMax: {order_errors.max():.5f}\nStd: {order_errors.std():.5f}',
                 transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        error_file = os.path.join(RESULTS_DIR, 'error_evolution_time.png')
        plt.savefig(error_file, dpi=200, bbox_inches='tight')
        print(f"Error evolution plot saved: {error_file}")
        plt.show()

        print("\n" + "=" * 70)
        print("ALL PLOTS COMPLETED!")
        print("=" * 70)


    if __name__ == '__main__':
        main()

        # ნახაზების შექმნა
        print("\nCreating additional plots...")
        data = np.load(DATA_FILE)
        X, Y = data['X'], data['Y']
        theta0, thetaT = data['theta0'], data['thetaT']

        X_test = X[NUM_TRAIN + NUM_VAL:NUM_TRAIN + NUM_VAL + NUM_TEST]
        Y_test = Y[NUM_TRAIN + NUM_VAL:NUM_TRAIN + NUM_VAL + NUM_TEST]
        theta0_test = theta0[NUM_TRAIN + NUM_VAL:NUM_TRAIN + NUM_VAL + NUM_TEST]
        thetaT_test = thetaT[NUM_TRAIN + NUM_VAL:NUM_TRAIN + NUM_VAL + NUM_TEST]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = UltraSurrogate(input_dim=2 * N, output_dim=2 * N)
        checkpoint = torch.load(MODEL_FILE, map_location='cpu')
        model.load_state_dict(checkpoint['model_state'])

        model = model.to(device).eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X_test).float().to(device)
            Y_pred = model(X_t).cpu().numpy()

        Y_pred_complex = Y_pred.reshape(-1, N, 2)
        Y_true_complex = Y_test.reshape(-1, N, 2)
        theta_pred = complex_to_angles(Y_pred_complex)
        theta_true = thetaT_test

        r_true = order_parameter(theta_true)
        r_pred = order_parameter(theta_pred)

        ang_dists = angular_distance(theta_pred, theta_true)
        mae = ang_dists.mean()
        rmse = np.sqrt((ang_dists ** 2).mean())
        mae_deg = np.rad2deg(mae)
        rmse_deg = np.rad2deg(rmse)
        r_mae = np.abs(r_true - r_pred).mean()
        r_rmse = np.sqrt(((r_true - r_pred) ** 2).mean())

        metrics = {
            'theta_pred': theta_pred,
            'theta_true': theta_true,
            'r_pred': r_pred,
            'r_true': r_true,
            'r_mae': r_mae,
            'r_corr': np.corrcoef(r_true, r_pred)[0, 1]
        }

        create_simple_plots(metrics, X_test, Y_test, theta0_test, thetaT_test)