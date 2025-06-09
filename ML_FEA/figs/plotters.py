import os
import math
import pathlib
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

FIG_DIR   = pathlib.Path('figs')
FIG_DIR.mkdir(parents=True, exist_ok=True)
os.makedirs("figs", exist_ok=True)

def savefig(name, dpi=300):
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{name}.png", dpi=dpi)
    plt.close()


def animate_shot_loop(shot, df, models_dict, save_path="figs/shot_animation.mp4"):
    proj, ang, lam, v0 = shot
    sub = df[(df['Projectile']==proj)&(df['Angle']==ang)&
             (df['Laminate']==lam)&(df['Velocity']==v0)].sort_values('Time')
    if sub.empty:
        print(f"no data for {shot}"); return

    time   = sub['Time'].values
    actual = sub['Residual Velcity'].values
    preds  = {name: sub[col].values for name, col in models_dict.items()}

    all_vals     = [actual] + list(preds.values())
    y_min        = min(map(np.min, all_vals))
    y_max        = max(map(np.max, all_vals))
    x_min, x_max = np.min(time), np.max(time)
    fig, ax      = plt.subplots(figsize=(6, 4))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min - 5, y_max + 5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Residual Velocity (m/s)')
    ax.set_title(f'{proj}, {lam}, {ang}°, V₀={v0} m/s')

    actual_line, = ax.plot([], [], 'o-', label='FEM (Actual)')
    pred_lines = {
        name: ax.plot([], [], '--', label=name)[0]
        for name in preds
    }
    ax.legend()

    def init():
        actual_line.set_data([], [])
        for line in pred_lines.values():
            line.set_data([], [])
        return [actual_line] + list(pred_lines.values())

    def update(frame):
        total_frames = len(time)
        step = total_frames

        if frame < step:
            actual_line.set_data(time[:frame], actual[:frame])
            for line in pred_lines.values():
                line.set_data([], [])
        else:
            actual_line.set_data(time, actual)
            for i, (name, line) in enumerate(pred_lines.items()):
                model_start = step * (i + 1)
                model_end = step * (i + 2)
                if frame < model_start:
                    line.set_data([], [])
                else:
                    line.set_data(time[:frame - model_start], preds[name][:frame - model_start])
        return [actual_line] + list(pred_lines.values())

    total_frames = len(time) * (len(preds) + 1)
    ani = FuncAnimation(fig, update, frames=total_frames, init_func=init, blit=True, interval=100, repeat=True)
    
    save_path = os.path.abspath(save_path)
    ani.save(save_path, writer='ffmpeg')
    plt.close()


def animate_residuals_over_time(df, shot, models_dict, save_path="figs/residuals_animation.mp4"):
    proj, ang, lam, v0 = shot
    sub = df[
        (df['Projectile'] == proj) &
        (df['Angle'] == ang) &
        (df['Laminate'] == lam) &
        (df['Velocity'] == v0)
    ].sort_values('Time')

    if sub.empty:
        print(f"no data for shot: {shot}")
        return

    time = sub['Time'].values
    actual = sub['Residual Velcity'].values

    residuals = {name: np.abs(sub[pred_col].values - actual) for name, pred_col in models_dict.items()}

    fig, ax = plt.subplots(figsize=(8, 5))
    lines = {}
    for i, (name, res) in enumerate(residuals.items()):
        lines[name], = ax.plot([], [], label=f'{name} Residual', lw=2)

    ax.set_xlim(time.min(), time.max())
    ax.set_ylim(0, max(max(res) for res in residuals.values()) * 1.1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Residual Error (|Prediction - Actual|)")
    ax.set_title(f"Residual Error Over Time\n{proj}, {lam}, {ang}°, V₀={v0} m/s")
    ax.legend()

    def init():
        for line in lines.values():
            line.set_data([], [])
        return lines.values()

    def update(frame):
        for name, res in residuals.items():
            lines[name].set_data(time[:frame], res[:frame])
        return lines.values()

    ani = animation.FuncAnimation(
        fig, update, frames=len(time), init_func=init,
        blit=True, repeat=True, interval=300
    )

    ani.save(save_path, writer='ffmpeg')
    plt.close()


def plot_actual_vs_pred(models, df):
    n = len(models)
    save_name = "plot_actual_vs_pred"
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (model_name, pred_col) in zip(axes, models.items()):
        y_true = df['Residual Velcity']
        y_pred = df[pred_col]
        ax.scatter(y_true, y_pred, alpha=0.6)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        ax.set_xlabel("Actual Residual Velocity")
        ax.set_ylabel("Predicted Residual Velocity")
        ax.set_title(f"{model_name}: Actual vs Predicted")
        ax.grid(True)
    
    plt.tight_layout()
    savefig(save_name)


def plot_residual_distribution(df, pred_cols, save_name=None):
    plt.figure(figsize=(7, 4))
    for label, col in pred_cols.items():
        res = df["Residual"] - (df["Residual"] - df[col])  
        res = df["Residual"] - df[col]
        sns.kdeplot(res, label=label, fill=False, bw_adjust=1.2)
    plt.axvline(0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Residual (y_true − y_pred)")
    plt.ylabel("Density")
    plt.title("Residual Distribution per Model")
    plt.legend()
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=300)
    # plt.show()
    

def plot_metrics_by_laminate(metrics_df, metric="R²", save_name=None):
    """
    metrics_df: DataFrame with columns ["Model","Laminate","MRE (%)","R²"]
    metric: either "R²" or "MRE (%)"
    """
   
    pivot = metrics_df.pivot(index="Laminate", columns="Model", values=metric)
    pivot = pivot.sort_index()
    
    plt.figure(figsize=(6, 4))
    for model in pivot.columns:
        plt.plot(pivot.index, pivot[model], marker="o", label=model)
    
    plt.xlabel("Laminate ID")
    plt.ylabel(metric)
    plt.title(f"{metric} by Laminate for Each Model")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=300)
    # plt.show()


def plot_residual_vs_pred(models, df) -> None:
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    axes = axes.ravel()      

    for ax, (name, pred_col) in zip(axes, models.items()):
        y_true = df["Residual Velcity"]
        y_pred = df[pred_col]
        resid  = y_true - y_pred          

        ax.scatter(y_pred, resid, alpha=0.6, s=20)
        ax.axhline(0, ls="--", c="red", lw=1)   
        ax.set_xlabel("Predicted residual velocity")
        ax.set_ylabel("Residual = actual − predicted")
        ax.set_title(f"{name}: residual vs predicted")
        ax.grid(True, ls="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("figs/residual_vs_pred.png", dpi=150)
    plt.close()


def save_all_shot_time_series(df,models_dict,target_col="Residual Velcity",time_col="Time",group_cols=("Projectile", "Laminate", "Angle", "Velocity"),out_dir="figs/shots",figsize=(6,4),dpi=150):
    os.makedirs(out_dir, exist_ok=True)
    
    for shot_keys, shot_df in df.groupby(list(group_cols)):
        proj, lam, ang, vel = shot_keys
        shot_df = shot_df.sort_values(time_col)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(
            shot_df[time_col], shot_df[target_col],
            label="Actual", marker="o", linewidth=1
        )
        for name, pred_col in models_dict.items():
            ax.plot(
                shot_df[time_col], shot_df[pred_col],
                label=name, marker="x", linewidth=1
            )
        
        ax.set(
            xlabel="Time",
            ylabel="Residual Velocity",
            title=f"{proj} | {lam} | {ang}° | {vel} m/s"
        )
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        
        fname = f"{proj}_{lam}_{ang}_{vel}.png".replace(" ", "_")
        fig.savefig(os.path.join(out_dir, fname), dpi=dpi)
        plt.close(fig)
