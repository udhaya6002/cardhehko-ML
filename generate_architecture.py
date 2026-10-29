import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(14, 8), dpi=200)
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis("off")

# Colors
bg_color = "#f8f9fa"
box_raw = "#e3f2fd"
box_feat = "#e8f5e9"
box_model = "#fff3e0"
box_eval = "#f3e5f5"
stroke_color = "#37474f"
accent_blue = "#1976d2"
accent_green = "#388e3c"
accent_orange = "#f57c00"
accent_purple = "#7b1fa2"

# Title
ax.text(7, 7.6, "System & Methodology Architecture: End-to-End ML Pipeline", 
        ha="center", va="center", fontsize=16, fontweight="bold", color="#263238")
ax.text(7, 7.25, "Used Car Price Prediction (CarDekho Dataset) - Anti-Leakage Compliant", 
        ha="center", va="center", fontsize=11, style="italic", color="#546e7a")

def draw_box(x, y, w, h, bg, border, title, items):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15", 
                                  edgecolor=border, facecolor=bg, linewidth=1.8)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 0.35, title, ha="center", va="center", 
            fontsize=11, fontweight="bold", color=border)
    line_y = y + h - 0.7
    for item in items:
        ax.text(x + 0.2, line_y, item, ha="left", va="center", fontsize=8.5, color="#212121")
        line_y -= 0.32

# 1. Raw Data & Audit
draw_box(0.5, 3.8, 2.7, 3.0, box_raw, accent_blue, "1. Data Audit & Prep", [
    "- Car details v3.csv",
    "- 8,128 rows, 13 features",
    "- Audit missing & nulls",
    "- Drop 1,679 duplicates",
    "- Check distributions",
    "- Target: selling_price",
    "- Safe logging & checks"
])

# 2. Feature Engineering & Split
draw_box(3.8, 3.8, 3.0, 3.0, box_feat, accent_green, "2. Features & Split", [
    "- Strip units: kmpl, CC, bhp",
    "- Torque parser (Nm, RPM)",
    "- Brand extraction (from name)",
    "- car_age = 2024 - year",
    "- km_per_year = km / age",
    "- Target transform: log1p(y)",
    "- 80/20 train/test split"
])

# 3. Anti-leakage Preprocessing
draw_box(7.4, 3.8, 3.0, 3.0, box_feat, accent_green, "3. Preprocessing (Train-only)", [
    "- 1st-99th percentile cap",
    "- Median impute (numeric)",
    "- Mode impute (categorical)",
    "- One-hot encode (drop 1st)",
    "- Align test columns to train",
    "- StandardScaler (linear only)",
    "- 0 data leakage guaranteed"
])

# 4. Model Training & Tuning
draw_box(10.8, 3.8, 2.7, 3.0, box_model, accent_orange, "4. Models & CV", [
    "- 5-Fold Stratified KFold",
    "- Linear Regression (Base)",
    "- Ridge (alpha tuning)",
    "- Lasso (L1 sparsity)",
    "- Decision Tree (depth/leaf)",
    "- Random Forest (RS tuning)",
    "- Gradient Boosting (best)"
])

# 5. Bottom Evaluation / Artifacts layer
draw_box(0.5, 0.4, 3.8, 2.8, box_eval, accent_purple, "5. Evaluation & Metrics", [
    "- Dual-scale metrics (log & rupee)",
    "- R2, RMSE, MAE comparison",
    "- Residual & scatter plots",
    "- Best Model Test R2: 0.9132",
    "- Best Model Rupee R2: 0.8987",
    "- Rupee MAE: Rs. 73,299"
])

draw_box(4.8, 0.4, 4.2, 2.8, box_eval, accent_purple, "6. Interpretability & Ablation", [
    "- Ridge vs Lasso coefficient weights",
    "- Lasso zeroed redundant features",
    "- Random Forest feature importance",
    "- Gradient Boosting feature importance",
    "- Permutation importance on test set",
    "- Ablation: delta R2 = +0.0428 for",
    "  engineered car_age & km_per_year"
])

draw_box(9.5, 0.4, 4.0, 2.8, box_eval, accent_purple, "7. Artifacts & Deliverables", [
    "- model_comparison.csv",
    "- error_analysis.csv (segments)",
    "- top_features.csv & ablation.csv",
    "- Colab notebook (main.ipynb)",
    "- 15-section final report (PDF/MD)",
    "- PPT Presentation (14 slides)",
    "- Deliverables archive (zip)"
])

# Arrows
arrow_kw = dict(arrowstyle="->", lw=2, color="#455a64", mutation_scale=15)
ax.annotate("", xy=(3.8, 5.3), xytext=(3.2, 5.3), arrowprops=arrow_kw)
ax.annotate("", xy=(7.4, 5.3), xytext=(6.8, 5.3), arrowprops=arrow_kw)
ax.annotate("", xy=(10.8, 5.3), xytext=(10.4, 5.3), arrowprops=arrow_kw)

# Downward flow arrows
ax.annotate("", xy=(12.1, 3.2), xytext=(12.1, 3.8), arrowprops=arrow_kw)
ax.annotate("", xy=(6.9, 3.2), xytext=(6.9, 3.8), arrowprops=arrow_kw)
ax.annotate("", xy=(2.4, 3.2), xytext=(2.4, 3.8), arrowprops=arrow_kw)

plt.tight_layout()
plt.savefig("figures/architecture.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved figures/architecture.png")
