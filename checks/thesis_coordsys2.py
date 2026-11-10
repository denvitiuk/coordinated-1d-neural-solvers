"""
Воспроизведение coordsys2.py из диплома (numpy вместо scipy) и разбор шага.

Системы: A1 X1 = B1 - Q,  A2 X2 = B2 + Q  (DX = DY = I).
Проверяем:
  1. совпадает ли ход FUNC с распечаткой в дипломе;
  2. какой шаг реально используется (в коде ALFA = FUNC/‖GR‖², а не формула (30));
  3. формулу (30) как написано (φ − ψ) против правильной (φ + ψ);
  4. сколько итераций нужно точному шагу и CR на той же задаче.
"""
import numpy as np

NM = 10
A1 = np.zeros((NM, NM)); A2 = np.zeros((NM, NM))
B1 = np.full(NM, 75.0); B2 = np.full(NM, 59.0)
for i in range(NM):
    A1[i, i] = 25.0; A2[i, i] = 55.0
for i in range(2):
    A1[i, i] = A1[NM-i-1, NM-i-1] = 12.0
    A2[i, i] = A2[NM-i-1, NM-i-1] = 22.0
for i in range(NM - 1):
    A1[i, i+1] = A1[i+1, i] = 7.0
    A2[i, i+1] = A2[i+1, i] = 11.0

solve = np.linalg.solve  # A1, A2 симметричны: Aᵀ = A
func = lambda x1, x2: 0.5 * np.sum((x1 - x2) ** 2)
states = lambda q: (solve(A1, B1 - q), solve(A2, B2 + q))
grad = lambda r: solve(A1, -r) - solve(A2, r)          # GR = FIX - FIY (формула 21)
m_apply = lambda g: solve(A1, g) + solve(A2, g)          # r(q - αg) = r + α·M g

# --- 1-2. Алгоритм ровно как в coordsys2.py --------------------------------
Q = np.full(NM, 25.0)
X1, X2 = states(Q)
print("thesis algorithm (copy of coordsys2.py):")
print(f"  it 0  FUNC = {func(X1, X2)!r}")
GR = grad(X1 - X2)
amod = GR @ GR
ALFA = func(X1, X2) / amod
Q = Q - ALFA * GR
D1 = np.zeros(NM)            # в коде D1 нигде не обновляется
it = 0
for j in range(1, 300):
    GR1 = GR.copy()
    X1, X2 = states(Q)
    it = j
    f = func(X1, X2)
    if j <= 3 or f < 1e-20:
        print(f"  it {j:<2} FUNC = {f!r}  (ALFA used = {ALFA:.6f})")
    if f < 1e-20:
        break
    GR = grad(X1 - X2)
    amod, amod1 = GR @ GR, GR1 @ GR1
    D = -GR + (amod / amod1) * D1   # D1 == 0  ->  D = -GR: обычный спуск
    ALFA = f / amod                  # шаг Поляка J/‖g‖², не формула (30)
    Q = Q + ALFA * D

# --- 3. Формула (30) как в тексте против правильной -------------------------
Q = np.full(NM, 25.0)
X1, X2 = states(Q); r = X1 - X2; g = grad(r)
phi, psi = solve(A1, g), solve(A2, g)                    # (27): Aφ = p, Bψ = p
alpha_text = (r @ (phi - psi)) / ((phi - psi) @ (phi - psi))
alpha_true = -(r @ (phi + psi)) / ((phi + psi) @ (phi + psi))
alpha_simple = (g @ g) / (m_apply(g) @ m_apply(g))
print("\nstep at iteration 0:")
print(f"  formula (30) as written, (φ - ψ): α = {alpha_text:.6f}  -> J = {func(*states(Q - alpha_text * g)):.6e}")
print(f"  corrected, -<r,φ+ψ>/‖φ+ψ‖²:        α = {alpha_true:.6f}  -> J = {func(*states(Q - alpha_true * g)):.6e}")
print(f"  ‖g‖²/‖Mg‖² (same thing):           α = {alpha_simple:.6f}")
print(f"  J at start = {func(X1, X2):.6e}")

# --- 4. Точный шаг и CR на той же задаче ------------------------------------
def sd_exact(tol=1e-20):
    q = np.full(NM, 25.0); r = np.subtract(*states(q))
    for k in range(10000):
        if 0.5 * r @ r < tol: return k
        g = grad(r); mg = m_apply(g); a = (g @ g) / (mg @ mg)
        q = q - a * g; r = r + a * mg
    return None

def cr(tol=1e-20):
    q = np.full(NM, 25.0); r = np.subtract(*states(q))
    # r = c - M q, M = A1⁻¹ + A2⁻¹ (симметрична, > 0)
    mr = m_apply(r); p, mp = r.copy(), mr.copy(); rmr = r @ mr
    for k in range(10000):
        if 0.5 * r @ r < tol: return k
        a = rmr / (mp @ mp); q = q + a * p; r = r - a * mp
        mr = m_apply(r); rmr_new = r @ mr
        p = r + (rmr_new / rmr) * p; mp = mr + (rmr_new / rmr) * mp; rmr = rmr_new
    return None

print(f"\niterations to FUNC < 1e-20: thesis step {it}, exact step {sd_exact()}, CR {cr()}")
