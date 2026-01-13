import pandas as pd, numpy as np

src = "C:\\Users\\2003h\\OneDrive\\Máy tính\\doan_2025.1\\Quadrotor-Control-System\\src\\archive\\controls_to_5_5_zero_angles.csv"
df = pd.read_csv(src)

dt = 0.02
t_grid = np.round(np.arange(0.0, 2.0, dt), 12)  # 2 seconds: 0.00 ... 1.98 => 100 samples

t_src = df["time"].to_numpy(dtype=float)

def interp_col(col):
    if col not in df.columns:
        return np.zeros_like(t_grid, dtype=float)
    return np.interp(t_grid, t_src, df[col].to_numpy(dtype=float))

u1 = interp_col("u1")
u2 = interp_col("u2")
u3 = interp_col("u3")

out = pd.DataFrame({
    "time": t_grid,
    "u1": np.round(u1, 0).astype(int),
    "u2": np.round(u2, 0).astype(int),
    "u3": np.round(u3, 0).astype(int),
})

path = "C:\\Users\\2003h\\OneDrive\\Máy tính\\doan_2025.1\\Quadrotor-Control-System\\src\\archive\\thuan1.csv"
out.to_csv(path, index=False)

path, out.head(12), out.tail(5), out.shape