"""Quick verification script for all model components."""
import sys
sys.path.insert(0, '.')
import torch
import numpy as np

# ── Test 1: TFT forward pass ──────────────────────────────────────────────────
from src.models.tft import TemporalFusionTransformer
model_tft = TemporalFusionTransformer(num_features=10, d_model=64, n_heads=4)
x = torch.randn(4, 63, 10)
h_t, weights = model_tft(x)
assert h_t.shape == (4, 64), f"TFT h_t shape wrong: {h_t.shape}"
assert weights.shape == (4, 63, 10), f"TFT weights shape wrong: {weights.shape}"
print(f"[OK] TFT: h_t={h_t.shape}, weights={weights.shape}")

# ── Test 2: MAFlow log_prob + sample ─────────────────────────────────────────
from src.models.maf import MAFlow
flow = MAFlow(dim=3, n_layers=4, hidden_dim=32, context_dim=64)
x_ret = torch.randn(4, 3)
log_p = flow.log_prob(x_ret, context=h_t)
assert log_p.shape == (4,), f"MAF log_prob shape wrong: {log_p.shape}"
samples = flow.sample(n_samples=50, context=h_t[:1])
assert samples.shape == (50, 3), f"MAF samples shape wrong: {samples.shape}"
mean_nll = (-log_p).mean().item()
print(f"[OK] MAFlow: log_prob={log_p.shape}, samples={samples.shape}, mean_NLL={mean_nll:.3f}")

# ── Test 3: Full ConditionalNormalizingFlow ───────────────────────────────────
from src.models.flow_model import ConditionalNormalizingFlow
full_model = ConditionalNormalizingFlow(
    num_macro_features=10, num_assets=3,
    tft_d_model=64, flow_n_layers=4, flow_hidden_dim=32, flow_n_hidden=2,
)
macro_seq = torch.randn(4, 63, 10)
returns = torch.randn(4, 3)
nll, wts = full_model(returns, macro_seq)
assert nll.shape == (), "NLL must be scalar"
samps = full_model.sample(macro_seq[:1], n_samples=100)
assert samps.shape == (100, 3)
params = full_model.count_parameters()
print(f"[OK] ConditionalNormalizingFlow: NLL={nll.item():.3f}, params={params['total']:,}")

# ── Test 4: Kupiec's POF test ─────────────────────────────────────────────────
from src.backtest.risk_metrics import kupiec_pof_test, compute_var, compute_es
r = kupiec_pof_test(breaches=5, n=500, alpha=0.01)
assert not r.reject_h0, "Good model should PASS Kupiec"
r2 = kupiec_pof_test(breaches=30, n=500, alpha=0.01)
assert r2.reject_h0, "Bad model should FAIL Kupiec"
print(f"[OK] Kupiec POF: good model p={r.p_value:.4f} PASS, bad model p={r2.p_value:.6f} FAIL")

# ── Test 5: VaR and ES ────────────────────────────────────────────────────────
np.random.seed(42)
samples_np = np.random.randn(10000, 3) * 0.01
var = compute_var(samples_np, alpha=0.01)
es  = compute_es(samples_np, alpha=0.01)
assert es <= var, "ES must be more extreme (lower) than VaR"
print(f"[OK] VaR={var*100:.3f}%, ES={es*100:.3f}% (ES <= VaR confirmed)")

print()
print("=== ALL TESTS PASSED ===")
