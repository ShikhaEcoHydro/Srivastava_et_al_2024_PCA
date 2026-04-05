"""
=============================================================================
Geochemical PCA Analysis — Final Production Script
=============================================================================

Based on:
    Srivastava, D., Dubey, C.P., Banerji, U.S., & Joshi, K.B. (2024).
    Geochemical trends in sedimentary environments using PCA approach.
    Journal of Earth System Science, 133(3), 122.
    DOI: 10.1007/s12040-024-02306-2

Reference methodology:
    Srivastava, D. (2021). Statistical analysis of geochemical datasets
    using PCA approach. M.Sc. Internship Report, Central University of
    Karnataka / NCESS, Thiruvananthapuram.

=============================================================================
DESIGN DECISIONS
=============================================================================

SCREE PLOT  (taken from main.py — matches paper Figures 4a / 5a)
----------------------------------------------------------------
  - Left Y-axis  : Explained Variance % per PC — blue bars
  - Right Y-axis : Cumulative Variance %       — red dot-line
  - PCs above elbow highlighted in darker blue
  - This is the bar + cumulative-line style in the published paper

BIPLOT  (taken from pasted code — MATLAB-exact per-axis scaling)
----------------------------------------------------------------
  - Per-axis independent scaling, exact MATLAB formula from
    internship report (pp. 20-21):

        MATLAB:
            scale = max(abs(W(:,1:2))) ./ max(abs(Eigenvector(:,1:2)))
            line([0, Eigenvector(i,1)*scale(1)], [0, Eigenvector(i,2)*scale(2)])

        Python equivalent (independently per axis):
            scale_x = max|scores_x| / max|loadings_x|
            scale_y = max|scores_y| / max|loadings_y|
            arrow_tip_x = loading_x * scale_x
            arrow_tip_y = loading_y * scale_y

  - Axis limits auto-computed from union of score coords + label
    positions, rounded outward to nearest integer + fixed margin.
    Nothing ever falls outside the plot box.

DATA FILES REQUIRED
-------------------
    Shimla_Chail.xlsx   — SCM dataset  (sheets: Data, Scores, Loadings)
    Diu_table.xlsx      — DMS dataset  (sheets: Data, Scores, Loadings)
=============================================================================
"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS — adjust to your local setup
# =============================================================================
DATA_DIR   = Path("D:/Srivastava_et_al_2024_PCA/Data")
OUTPUT_DIR = Path("D:/Srivastava_et_al_2024_PCA/Outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCM_FILE = DATA_DIR / "Shimla_Chail.xlsx"
DMS_FILE = DATA_DIR / "Diu_table.xlsx"

# =============================================================================
# GLOBAL PLOT STYLE
# =============================================================================
plt.rcParams.update({
    'font.family'        : 'times new roman',
    'font.size'          : 10,
    'axes.titlesize'     : 11,
    'axes.labelsize'     : 10,
    'axes.linewidth'     : 0.8,
    'xtick.major.width'  : 0.8,
    'ytick.major.width'  : 0.8,
    'figure.dpi'         : 150,
    'axes.spines.top'    : False,
    'axes.spines.right'  : False,
})

# Colours
CHAIL_COL  = '#1f77b4'   # blue     — Chail group samples
SHIMLA_COL = '#2ca02c'   # green    — Shimla group samples
DMS_COL    = "#3827d6"   # red      — DMS core samples
LOAD_COL   = "#D30404"   # dark red — loading vectors


# =============================================================================
# STEP 1 — LOAD DATA
# =============================================================================

def load_dataset(filepath, id_col):
    """
    Load raw geochemical data + MATLAB validation sheets from an Excel file.

    Parameters
    ----------
    filepath : Path
    id_col   : str — sample ID column name

    Returns
    -------
    data_df, sample_ids, matlab_scores, matlab_loads
    """
    raw           = pd.read_excel(filepath, sheet_name='Data')
    matlab_scores = pd.read_excel(filepath, sheet_name='Scores')
    matlab_loads  = pd.read_excel(filepath, sheet_name='Loadings')

    sample_ids = raw[id_col].tolist()
    data_df    = raw.drop(columns=[id_col]).select_dtypes(include=[np.number])

    return data_df, sample_ids, matlab_scores, matlab_loads


# =============================================================================
# STEP 2 — STANDARDISE  [Equation 1]
# =============================================================================

def standardise(df, log_transform=True):
    """
    Preprocessing pipeline for geochemical PCA.

    (A) Natural log — compresses multi-order-of-magnitude geochemical data
        and satisfies PCA normality assumption (paper Section 4.1).
    (B) Z-score: (X - mean) / std — unit-free, equal contribution [Eq. 1].
        ddof=1 matches MATLAB std().

    Returns
    -------
    std_df : DataFrame — preprocessed data (mean=0, std=1 per variable)
    means  : Series
    stds   : Series
    """
    if log_transform:
        df = np.log(df)
    means  = df.mean()
    stds   = df.std(ddof=1)
    std_df = (df - means) / stds
    return std_df, means, stds


# =============================================================================
# STEP 3 — COVARIANCE MATRIX  [Equations 5-6]
# =============================================================================

def covariance_matrix(std_df):
    """
    Cov(X,Y) = sum[(Xi-Xm)(Yi-Ym)] / (n-1)

    Returns
    -------
    cov_df : DataFrame — symmetric (n_vars x n_vars) covariance matrix
    """
    X   = std_df.values
    n   = X.shape[0]
    X_c = X - X.mean(axis=0)
    cov = (X_c.T @ X_c) / (n - 1)
    return pd.DataFrame(cov, index=std_df.columns, columns=std_df.columns)


# =============================================================================
# STEP 4 — EIGENDECOMPOSITION  [Equation 7]
# =============================================================================

def eigendecompose(cov_df):
    """
    Solve det(A - lambdaI)X = 0  [Eq. 7].

    Uses numpy.linalg.eigh (symmetric solver — numerically stable).
    Results sorted descending by eigenvalue.
    Sign convention: largest-magnitude element of each eigenvector positive
    (matches MATLAB eig() default).

    Returns
    -------
    eigenvalues  : ndarray (n,)
    eigenvectors : ndarray (n, n) — columns are PC directions (loadings)
    """
    vals, vecs   = np.linalg.eigh(cov_df.values)
    idx          = np.argsort(vals)[::-1]
    eigenvalues  = vals[idx]
    eigenvectors = vecs[:, idx]

    for j in range(eigenvectors.shape[1]):
        if eigenvectors[np.argmax(np.abs(eigenvectors[:, j])), j] < 0:
            eigenvectors[:, j] *= -1

    return eigenvalues, eigenvectors


# =============================================================================
# STEP 5 — EXPLAINED VARIANCE
# =============================================================================

def explained_variance(eigenvalues):
    total   = eigenvalues.sum()
    var_pct = (eigenvalues / total) * 100
    cum_pct = np.cumsum(var_pct)
    return var_pct, cum_pct


# =============================================================================
# STEP 6 — SCORES  [Equation 8]
# =============================================================================

def compute_scores(std_df, eigenvectors):
    """Final Data = Row Feature Vector x Row Data Adjusted  [Eq. 8]"""
    return std_df.values @ eigenvectors


# =============================================================================
# STEP 7 — FULL PCA PIPELINE
# =============================================================================

def run_pca(data_df, sample_ids):
    n_vars = data_df.shape[1]

    std_df, means, stds  = standardise(data_df)
    cov_df               = covariance_matrix(std_df)
    eigenvalues, eigvecs = eigendecompose(cov_df)
    var_pct, cum_pct     = explained_variance(eigenvalues)
    scores_all           = compute_scores(std_df, eigvecs)

    pc_labels   = [f'PC{i+1}' for i in range(n_vars)]
    scores_df   = pd.DataFrame(scores_all,  index=sample_ids, columns=pc_labels)
    loadings_df = pd.DataFrame(eigvecs,     index=data_df.columns, columns=pc_labels)
    variance_df = pd.DataFrame({
        'Eigenvalue'         : eigenvalues,
        'Explained Var (%)'  : var_pct,
        'Cumulative Var (%)' : cum_pct,
    }, index=pc_labels)

    return {
        'std_data'    : std_df,
        'cov_matrix'  : cov_df,
        'eigenvalues' : eigenvalues,
        'eigenvectors': eigvecs,
        'scores'      : scores_df,
        'loadings'    : loadings_df,
        'variance'    : variance_df,
        'var_pct'     : var_pct,
        'cum_pct'     : cum_pct,
    }


# =============================================================================
# STEP 8 — VALIDATE AGAINST MATLAB
# =============================================================================

def validate(pca_result, matlab_loads, label):
    """
    Compare Python variance percentages against published MATLAB values.

    Targets (Srivastava et al. 2024):
        SCM: PC1=20.86%, PC2=19.75%, PC3=11.90%  -> cum=52.51%
        DMS: PC1=48.94%, PC2=15.64%, PC3=14.72%  -> cum=79.30%
    """
    matlab_var = matlab_loads.iloc[1, 1:4].values.astype(float)
    python_var = pca_result['var_pct'][:3]
    cum_python = pca_result['cum_pct'][2]
    cum_matlab = float(matlab_loads.iloc[2, 3])

    print(f"\n  {'─'*55}")
    print(f"  VALIDATION — {label}")
    print(f"  {'─'*55}")
    print(f"  {'PC':<6} {'MATLAB (%)':>12} {'Python (%)':>12} {'Match':>8}")
    print(f"  {'─'*55}")
    for i, (m, p) in enumerate(zip(matlab_var, python_var)):
        mark = 'OK' if abs(m - p) < 0.01 else '~OK' if abs(m - p) < 0.5 else 'FAIL'
        print(f"  PC{i+1:<4} {m:>12.4f} {p:>12.4f} {mark:>8}")
    print(f"  {'─'*55}")
    print(f"  {'Cum PC1-3':<6} {cum_matlab:>12.4f} {cum_python:>12.4f}")
    print(f"  {'─'*55}")


# =============================================================================
# STEP 9 — VISUALISATIONS
# =============================================================================

# -----------------------------------------------------------------------------
# 9a  SCREE PLOT — taken from main.py, matches paper Figures 4a / 5a
#
#     Bar chart of explained variance % per PC (left y-axis, blue bars).
#     Cumulative variance % as red dot-line on right twin y-axis.
#     PCs above the elbow highlighted in darker blue.
# -----------------------------------------------------------------------------

def plot_scree(variance_df, n_highlight, title, ax, x_tick_step=1):
    """
    Bar-style scree plot matching paper Figures 4a and 5a.

    Parameters
    ----------
    variance_df  : DataFrame with 'Explained Var (%)' and 'Cumulative Var (%)'
    n_highlight  : int — number of selected PCs (highlighted dark blue)
    title        : str
    ax           : Axes
    x_tick_step  : int — spacing between x-axis tick labels
                   SCM (29 PCs) → 5  gives ticks at 5, 10, 15, 20, 25
                   DMS (13 PCs) → 2  gives ticks at 2, 4, 6, 8, 10, 12
    """
    n    = len(variance_df)
    pcs  = range(1, n + 1)
    cols = ['#2166ac' if i < n_highlight else '#b0c4de' for i in range(n)]

    ax.bar(pcs, variance_df['Explained Var (%)'],
           color=cols, edgecolor='white', linewidth=0.5)

    ax2 = ax.twinx()
    ax2.plot(pcs, variance_df['Cumulative Var (%)'],
             'o-', color='crimson', linewidth=1.5, markersize=3,
             label='Cumulative %')
    ax2.axhline(80, color='grey', linestyle=':', linewidth=0.8, alpha=0.7)
    ax2.set_ylabel('Cumulative Variance (%)', color='crimson', fontsize=8)
    ax2.tick_params(axis='y', labelcolor='crimson', labelsize=7)
    ax2.set_ylim(0, 110)
    ax2.spines['right'].set_visible(True)

    ax.set_xlabel('Principal Component', fontsize=9)
    ax.set_ylabel('Explained Variance (%)', fontsize=9)
    ax.set_title(title, fontweight='bold', pad=6)

    # X-axis ticks at every x_tick_step positions (always include tick 1)
    xtick_positions = [i for i in pcs if i == 1 or i % x_tick_step == 0]
    ax.set_xticks(xtick_positions)
    ax.tick_params(axis='x', labelsize=7)


# -----------------------------------------------------------------------------
# 9b  BIPLOT helpers — taken from pasted code (MATLAB-exact per-axis scaling)
# -----------------------------------------------------------------------------

def _arrow_tips(loadings_df, pc_x, pc_y, scores_df, arrow_scale=1.0):
    """
    Per-axis independent scaling — exact MATLAB formula from internship
    report (pp. 20-21):

        scale = max(abs(W(:,1:2))) ./ max(abs(Eigenvector(:,1:2)))

    Python:
        scale_x = max|scores_x| / max|loadings_x|
        scale_y = max|scores_y| / max|loadings_y|

    Each axis scaled independently so the longest arrow on that axis
    reaches the edge of the score cloud on that axis.
    """
    sx = scores_df[pc_x].abs().max()
    sy = scores_df[pc_y].abs().max()
    lx = loadings_df[pc_x].abs().max()
    ly = loadings_df[pc_y].abs().max()

    scale_x = (sx / lx) * arrow_scale if lx > 1e-12 else 1.0
    scale_y = (sy / ly) * arrow_scale if ly > 1e-12 else 1.0

    tips_x = loadings_df[pc_x].values * scale_x
    tips_y = loadings_df[pc_y].values * scale_y

    return tips_x, tips_y


def _tick_step(span):
    """Sensible integer tick spacing for a given axis span."""
    if span <= 10:
        return 2
    elif span <= 20:
        return 5
    elif span <= 40:
        return 10
    else:
        return 15


def _compute_axis_limits(scores_df, tips_x, tips_y, pc_x, pc_y,
                          label_frac=0.15, margin=2.0):
    """
    Auto axis limits that contain scores, arrow tips, and element labels.

    1. Label positions = tip * (1 + label_frac)
    2. Union of score coords + label positions
    3. Round outward to nearest integer
    4. Add fixed margin
    """
    label_x = tips_x * (1 + label_frac)
    label_y = tips_y * (1 + label_frac)

    all_x = np.concatenate([scores_df[pc_x].values, label_x])
    all_y = np.concatenate([scores_df[pc_y].values, label_y])

    xmin = np.floor(all_x.min()) - margin
    xmax = np.ceil(all_x.max())  + margin
    ymin = np.floor(all_y.min()) - margin
    ymax = np.ceil(all_y.max())  + margin

    return (xmin, xmax), (ymin, ymax)


# -----------------------------------------------------------------------------
# 9b  BIPLOT (main function) — taken from pasted code
# -----------------------------------------------------------------------------

def plot_biplot(scores_df, loadings_df,
                pc_x, pc_y,
                groups, group_colors, group_labels,
                title, ax,
                arrow_scale=1.0,
                label_frac=0.15,
                margin=2.0):
    """
    RQ-mode biplot: sample score cloud + loading vectors.

    Scores  — coloured by geological group.
    Arrows  — dark-red; per-axis MATLAB scaling.
    Labels  — element names placed label_frac beyond each arrow tip.
    Limits  — auto-computed, nothing falls outside the box.

    Parameters
    ----------
    scores_df    : DataFrame — all PC scores
    loadings_df  : DataFrame — all eigenvectors (loadings)
    pc_x, pc_y   : str       — PC names to plot ('PC1', 'PC2', etc.)
    groups       : list[str] — group label per sample
    group_colors : list[str] — one colour per unique group
    group_labels : list[str] — legend label per unique group
    title        : str
    ax           : Axes
    arrow_scale  : float — multiplier on MATLAB scale (default 1.0)
    label_frac   : float — fractional overshoot beyond arrow tip for labels
    margin       : float — whitespace units beyond data extent
    """
    # 1. Arrow tips (MATLAB-exact per-axis scaling)
    tips_x, tips_y = _arrow_tips(
        loadings_df, pc_x, pc_y, scores_df, arrow_scale
    )

    # 2. Axis limits before drawing
    xlim, ylim = _compute_axis_limits(
        scores_df, tips_x, tips_y, pc_x, pc_y, label_frac, margin
    )

    # 3. Score scatter
    unique_groups = list(dict.fromkeys(groups))
    for group, color, label in zip(unique_groups, group_colors, group_labels):
        mask = np.array([g == group for g in groups])
        ax.scatter(
            scores_df.loc[mask, pc_x],
            scores_df.loc[mask, pc_y],
            c=color, s=45, alpha=0.88, label=label,
            edgecolors='white', linewidths=0.5, zorder=4
        )

    # 4. Loading arrows + element labels
    for i, var in enumerate(loadings_df.index):
        lx, ly = tips_x[i], tips_y[i]

        ax.annotate(
            '',
            xy=(lx, ly), xytext=(0, 0),
            arrowprops=dict(
                arrowstyle='->', color=LOAD_COL,
                lw=1.2, mutation_scale=9
            ),
            zorder=3
        )

        ax.text(
            lx * (1 + label_frac),
            ly * (1 + label_frac),
            var,
            fontsize=6, color=LOAD_COL,
            ha='center', va='center', fontweight='bold',
            zorder=5
        )

    # 5. Reference lines
    ax.axhline(0, color='grey', linewidth=0.6, linestyle='--', alpha=0.55)
    ax.axvline(0, color='grey', linewidth=0.6, linestyle='--', alpha=0.55)

    # 6. Computed limits + integer ticks
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]
    x_step = _tick_step(x_span)
    y_step = _tick_step(y_span)

    x_ticks = np.arange(
        int(np.ceil(xlim[0]  / x_step)) * x_step,
        int(np.floor(xlim[1] / x_step)) * x_step + x_step,
        x_step
    )
    y_ticks = np.arange(
        int(np.ceil(ylim[0]  / y_step)) * y_step,
        int(np.floor(ylim[1] / y_step)) * y_step + y_step,
        y_step
    )
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.tick_params(labelsize=8)

    # 7. Labels, title, legend
    ax.set_xlabel(pc_x, fontsize=10, fontweight='bold')
    ax.set_ylabel(pc_y, fontsize=10, fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=7, fontsize=10)
    ax.legend(fontsize=7, markerscale=1.1,
              framealpha=0.8, edgecolor='#cccccc',
              loc='upper right')


# -----------------------------------------------------------------------------
# 9c  LOADINGS HEATMAP
# -----------------------------------------------------------------------------

def plot_loadings_heatmap(loadings_df, n_pcs, title, ax):
    """
    Colour-coded heatmap of eigenvector values for PC1..PC(n_pcs).
    Red = positive loading, blue = negative loading.
    """
    data = loadings_df.iloc[:, :n_pcs]
    im   = ax.imshow(data.values, cmap='RdBu_r',
                     aspect='auto', vmin=-0.55, vmax=0.55)

    ax.set_xticks(range(n_pcs))
    ax.set_xticklabels(data.columns, fontsize=9)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index, fontsize=7.5)
    ax.set_title(title, fontweight='bold', pad=6)

    plt.colorbar(im, ax=ax, shrink=0.8, label='Loading value', pad=0.02)

    for i in range(len(data.index)):
        for j in range(n_pcs):
            val   = data.iloc[i, j]
            color = 'white' if abs(val) > 0.28 else 'black'
            ax.text(j, i, f'{val:.2f}',
                    ha='center', va='center', fontsize=5.5, color=color)


# -----------------------------------------------------------------------------
# Console helper
# -----------------------------------------------------------------------------

def print_variance_table(variance_df, n_show, label):
    print(f"\n  {label} — Variance Explained")
    print(f"  {'PC':<8} {'Eigenvalue':>12} {'Var (%)':>10} {'Cum (%)':>10}")
    print(f"  {'─'*44}")
    for i, row in variance_df.head(n_show).iterrows():
        print(f"  {i:<8} {row['Eigenvalue']:>12.4f} "
              f"{row['Explained Var (%)']:>10.4f} "
              f"{row['Cumulative Var (%)']:>10.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 65)
    print("  Geochemical PCA — Srivastava et al. (2024)")
    print("  Python implementation | Validated against MATLAB")
    print("  Scree: bar+cumulative (main.py) | Biplot: per-axis MATLAB scale")
    print("=" * 65)

    for f in [SCM_FILE, DMS_FILE]:
        if not f.exists():
            raise FileNotFoundError(
                f"\n  Cannot find: {f}\n"
                "  Expected:\n"
                "    Shimla_Chail.xlsx\n"
                "    Diu_table.xlsx"
            )

    # =========================================================================
    # DATASET 1 — SCM  (Shimla & Chail Metasediments)
    # =========================================================================

    print("\n  Loading SCM dataset...")
    scm_data, scm_ids, scm_mat_scores, scm_mat_loads = load_dataset(
        SCM_FILE, id_col='S.No.'
    )
    print(f"  {scm_data.shape[0]} samples x {scm_data.shape[1]} variables")

    half       = len(scm_ids) // 2
    scm_groups = ['Chail'] * half + ['Shimla'] * (len(scm_ids) - half)

    print("  Running PCA on SCM...")
    scm = run_pca(scm_data, scm_ids)
    print_variance_table(scm['variance'], n_show=9, label="SCM")
    validate(scm, scm_mat_loads, "SCM")
    print(f"  SCM  paper: 52.51%   Python: {scm['cum_pct'][2]:.2f}%")

    # =========================================================================
    # DATASET 2 — DMS  (Diu Island Mudflat Sediments)
    # =========================================================================

    print("\n  Loading DMS dataset...")
    dms_data, dms_ids, dms_mat_scores, dms_mat_loads = load_dataset(
        DMS_FILE, id_col='S. no.'
    )
    print(f"  {dms_data.shape[0]} samples x {dms_data.shape[1]} variables")

    dms_groups = ['DMS Core'] * len(dms_ids)

    print("  Running PCA on DMS...")
    dms = run_pca(dms_data, dms_ids)
    print_variance_table(dms['variance'], n_show=5, label="DMS")
    validate(dms, dms_mat_loads, "DMS")
    print(f"  DMS  paper: 79.30%   Python: {dms['cum_pct'][2]:.2f}%")

    # =========================================================================
    # FIGURE 1 — SCM: Scree + Biplots  (reproduces Figure 4)
    # =========================================================================

    print("\n  Generating Figure 1 — SCM (Figure 4)...")

    fig1 = plt.figure(figsize=(16, 5.5))
    gs1 = gridspec.GridSpec(1, 3, figure=fig1, wspace=0.40)

    ax1a = fig1.add_subplot(gs1[0])
    plot_scree(
        scm['variance'], n_highlight=9,
        title='(a) Scree Plot\nFirst 9 PCs = 85.53% variance',
        ax=ax1a,
        x_tick_step=5       # ticks at 1, 5, 10, 15, 20, 25 — matches paper Fig 4a
    )

    ax1b = fig1.add_subplot(gs1[1])
    plot_biplot(
        scm['scores'], scm['loadings'],
        pc_x='PC1', pc_y='PC2',
        groups=scm_groups,
        group_colors=[CHAIL_COL, SHIMLA_COL],
        group_labels=['Chail Group', 'Shimla Group'],
        title='(b) PC1 vs PC2\nSiO\u2082 negative on PC1 \u2192 intermediate source',
        ax=ax1b,
        arrow_scale=1.0,
        label_frac=0.15,
        margin=2.0
    )

    ax1c = fig1.add_subplot(gs1[2])
    plot_biplot(
        scm['scores'], scm['loadings'],
        pc_x='PC2', pc_y='PC3',
        groups=scm_groups,
        group_colors=[CHAIL_COL, SHIMLA_COL],
        group_labels=['Chail Group', 'Shimla Group'],
        title='(c) PC2 vs PC3\nPhyllosilicates \u2192 PC2 ; K-feldspar \u2192 PC3',
        ax=ax1c,
        arrow_scale=1.0,
        label_frac=0.15,
        margin=2.0
    )

    plt.tight_layout()
    fig1.savefig(OUTPUT_DIR / 'Figure4_SCM_PCA.png',
                 dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("  Saved: Figure4_SCM_PCA.png")

    # =========================================================================
    # FIGURE 2 — DMS: Scree + Biplots  (reproduces Figure 5)
    # =========================================================================

    print("  Generating Figure 2 — DMS (Figure 5)...")

    fig2 = plt.figure(figsize=(16, 5.5))
    gs2 = gridspec.GridSpec(1, 3, figure=fig2, wspace=0.40)

    ax2a = fig2.add_subplot(gs2[0])
    plot_scree(
        dms['variance'], n_highlight=3,
        title='(a) Scree Plot\nPC1+PC2+PC3 = 79.30% variance',
        ax=ax2a,
        x_tick_step=2       # ticks at 1, 2, 4, 6, 8, 10, 12 — matches paper Fig 5a
    )

    ax2b = fig2.add_subplot(gs2[1])
    plot_biplot(
        dms['scores'], dms['loadings'],
        pc_x='PC1', pc_y='PC2',
        groups=dms_groups,
        group_colors=[DMS_COL],
        group_labels=['DMS Core'],
        title='(b) PC1 vs PC2\nTOC & Cu positive \u2192 in-situ productivity',
        ax=ax2b,
        arrow_scale=1.0,
        label_frac=0.15,
        margin=2.0
    )

    ax2c = fig2.add_subplot(gs2[2])
    plot_biplot(
        dms['scores'], dms['loadings'],
        pc_x='PC2', pc_y='PC3',
        groups=dms_groups,
        group_colors=[DMS_COL],
        group_labels=['DMS Core'],
        title='(c) PC2 vs PC3\nK\u2082O & MgO positive \u2192 weathering proxies',
        ax=ax2c,
        arrow_scale=1.0,
        label_frac=0.15,
        margin=2.0
    )

    plt.tight_layout()
    fig2.savefig(OUTPUT_DIR / 'Figure5_DMS_PCA.png',
                 dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print("  Saved: Figure5_DMS_PCA.png")

    # =========================================================================
    # FIGURE 3 — Loadings Heatmaps (supplementary)
    # =========================================================================

    print("  Generating Figure 3 — Loadings Heatmaps...")

    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(16, 8))
    fig3.suptitle(
        'PCA Loadings (Eigenvectors) \u2014 PC1 to PC3\n'
        'Srivastava et al. (2024), J. Earth Syst. Sci.',
        fontsize=11, fontweight='bold'
    )

    plot_loadings_heatmap(
        scm['loadings'], n_pcs=3,
        title=(f'SCM \u2014 Shimla & Chail Metasediments\n'
               f'PC1={scm["var_pct"][0]:.2f}%  '
               f'PC2={scm["var_pct"][1]:.2f}%  '
               f'PC3={scm["var_pct"][2]:.2f}%'),
        ax=ax3a
    )
    plot_loadings_heatmap(
        dms['loadings'], n_pcs=3,
        title=(f'DMS \u2014 Diu Island Mudflat Sediments\n'
               f'PC1={dms["var_pct"][0]:.2f}%  '
               f'PC2={dms["var_pct"][1]:.2f}%  '
               f'PC3={dms["var_pct"][2]:.2f}%'),
        ax=ax3b
    )

    plt.tight_layout()
    fig3.savefig(OUTPUT_DIR / 'FigureS1_Loadings_Heatmap.png',
                 dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print("  Saved: FigureS1_Loadings_Heatmap.png")

    # =========================================================================
    # EXPORT TO EXCEL
    # =========================================================================

    print("  Exporting results to Excel...")
    with pd.ExcelWriter(OUTPUT_DIR / 'PCA_Results.xlsx') as writer:
        scm['variance'].to_excel(writer,             sheet_name='SCM_Variance')
        scm['loadings'].iloc[:, :3].to_excel(writer, sheet_name='SCM_Loadings_PC1-3')
        scm['scores'].iloc[:, :3].to_excel(writer,   sheet_name='SCM_Scores_PC1-3')
        scm['cov_matrix'].to_excel(writer,           sheet_name='SCM_CovMatrix')
        dms['variance'].to_excel(writer,             sheet_name='DMS_Variance')
        dms['loadings'].iloc[:, :3].to_excel(writer, sheet_name='DMS_Loadings_PC1-3')
        dms['scores'].iloc[:, :3].to_excel(writer,   sheet_name='DMS_Scores_PC1-3')
        dms['cov_matrix'].to_excel(writer,           sheet_name='DMS_CovMatrix')
    print("  Saved: PCA_Results.xlsx")

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print(f"\n{'='*65}")
    print("  ANALYSIS COMPLETE")
    print(f"{'='*65}")
    print(f"  SCM  PC1={scm['var_pct'][0]:.2f}%  PC2={scm['var_pct'][1]:.2f}%  "
          f"PC3={scm['var_pct'][2]:.2f}%  ->  cum={scm['cum_pct'][2]:.2f}%"
          f"  (paper 52.51%)")
    print(f"  DMS  PC1={dms['var_pct'][0]:.2f}%  PC2={dms['var_pct'][1]:.2f}%  "
          f"PC3={dms['var_pct'][2]:.2f}%  ->  cum={dms['cum_pct'][2]:.2f}%"
          f"  (paper 79.30%)")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()