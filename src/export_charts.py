"""
export_charts.py
----------------
Generates high-resolution, presentation-ready visualizations of the key EDA
and Machine Learning results and exports them to assets/charts/ for embedding
in README.md and documentation.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'WA_Fn-UseC_-HR-Employee-Attrition.csv')
CHARTS_DIR = os.path.join(BASE_DIR, 'assets', 'charts')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(CHARTS_DIR, exist_ok=True)

# Styling Palette (Linear / Stripe / Tableau Inspired)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#E2E8F0'
plt.rcParams['axes.linewidth'] = 1.0

PALETTE = {
    'primary_blue': '#2563EB',
    'deep_indigo': '#4F46E5',
    'coral_red': '#EF4444',
    'emerald_green': '#059669',
    'slate_dark': '#0F172A',
    'slate_muted': '#64748B',
    'bg_white': '#FFFFFF',
    'grid_color': '#F1F5F9'
}


def load_dataset():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return pd.read_csv(os.path.join(BASE_DIR, 'WA_Fn-UseC_-HR-Employee-Attrition.csv'))


def export_attrition_overview(df):
    """1. Attrition Overview Donut Chart"""
    counts = df['Attrition'].value_counts()
    labels = [f"Retained (No)\n{counts['No']:,} ({counts['No']/len(df):.1%})",
              f"Departed (Yes)\n{counts['Yes']:,} ({counts['Yes']/len(df):.1%})"]
    colors = [PALETTE['primary_blue'], PALETTE['coral_red']]

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=PALETTE['bg_white'])
    wedges, texts = ax.pie(
        [counts['No'], counts['Yes']],
        labels=labels,
        colors=colors,
        startangle=140,
        wedgeprops=dict(width=0.4, edgecolor='white', linewidth=3),
        textprops=dict(color=PALETTE['slate_dark'], fontsize=11, fontweight='medium')
    )
    ax.text(0, 0, f"{counts['Yes']/len(df):.1%}\nAttrition",
            ha='center', va='center', fontsize=16, fontweight='bold', color=PALETTE['slate_dark'])
    ax.set_title("Workforce Retention & Attrition Distribution", fontsize=14, fontweight='bold', pad=20, color=PALETTE['slate_dark'])
    
    out_path = os.path.join(CHARTS_DIR, 'attrition_overview.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Exported: {out_path}")


def export_department_attrition(df):
    """2. Department Attrition Rate"""
    dept_agg = df.groupby(['Department', 'Attrition']).size().unstack(fill_value=0)
    dept_agg['Total'] = dept_agg['No'] + dept_agg['Yes']
    dept_agg['Rate'] = (dept_agg['Yes'] / dept_agg['Total'] * 100).round(1)
    dept_agg = dept_agg.sort_values(by='Rate', ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=PALETTE['bg_white'])
    ax.set_facecolor(PALETTE['bg_white'])
    bars = ax.bar(dept_agg.index, dept_agg['Rate'], color=[PALETTE['coral_red'], '#3B82F6', '#60A5FA'], width=0.55, edgecolor='none')
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.6, f"{yval:.1f}%", ha='center', va='bottom', fontsize=11, fontweight='bold', color=PALETTE['slate_dark'])

    ax.set_ylabel("Attrition Rate (%)", fontsize=11, color=PALETTE['slate_muted'], fontweight='medium')
    ax.set_title("Attrition Rate (%) by Department", fontsize=14, fontweight='bold', pad=15, color=PALETTE['slate_dark'])
    ax.set_ylim(0, max(dept_agg['Rate']) + 4)
    ax.grid(axis='y', linestyle='--', color=PALETTE['grid_color'], alpha=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    out_path = os.path.join(CHARTS_DIR, 'department_attrition.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Exported: {out_path}")


def export_job_role_attrition(df):
    """3. Job Role Attrition Ranking"""
    role_agg = df.groupby(['JobRole', 'Attrition']).size().unstack(fill_value=0)
    role_agg['Total'] = role_agg['No'] + role_agg['Yes']
    role_agg['Rate'] = (role_agg['Yes'] / role_agg['Total'] * 100).round(1)
    role_agg = role_agg.sort_values(by='Rate', ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=PALETTE['bg_white'])
    ax.set_facecolor(PALETTE['bg_white'])
    
    colors = [PALETTE['coral_red'] if r > 20 else '#3B82F6' for r in role_agg['Rate']]
    bars = ax.barh(role_agg.index, role_agg['Rate'], color=colors, height=0.6)
    
    for bar in bars:
        xval = bar.get_width()
        ax.text(xval + 0.8, bar.get_y() + bar.get_height()/2.0, f"{xval:.1f}%", ha='left', va='center', fontsize=10, fontweight='bold', color=PALETTE['slate_dark'])

    ax.set_xlabel("Attrition Rate (%)", fontsize=11, color=PALETTE['slate_muted'], fontweight='medium')
    ax.set_title("Attrition Risk Ranking by Job Role", fontsize=14, fontweight='bold', pad=15, color=PALETTE['slate_dark'])
    ax.set_xlim(0, max(role_agg['Rate']) + 6)
    ax.grid(axis='x', linestyle='--', color=PALETTE['grid_color'], alpha=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    out_path = os.path.join(CHARTS_DIR, 'job_role_attrition.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Exported: {out_path}")


def export_overtime_impact(df):
    """4. Overtime Impact Comparison"""
    ot_agg = df.groupby(['OverTime', 'Attrition']).size().unstack(fill_value=0)
    ot_agg['Total'] = ot_agg['No'] + ot_agg['Yes']
    ot_agg['Rate'] = (ot_agg['Yes'] / ot_agg['Total'] * 100).round(1)

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=PALETTE['bg_white'])
    ax.set_facecolor(PALETTE['bg_white'])
    bars = ax.bar(["Standard Hours (No Overtime)", "Mandatory Overtime"],
                  [ot_agg.loc['No', 'Rate'], ot_agg.loc['Yes', 'Rate']],
                  color=['#3B82F6', PALETTE['coral_red']], width=0.45)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.1f}%", ha='center', va='bottom', fontsize=12, fontweight='bold', color=PALETTE['slate_dark'])

    ax.set_ylabel("Attrition Rate (%)", fontsize=11, color=PALETTE['slate_muted'])
    ax.set_title("The Overtime Burnout Multiplier (~3x Turnover Surge)", fontsize=13, fontweight='bold', pad=15, color=PALETTE['slate_dark'])
    ax.set_ylim(0, 36)
    ax.grid(axis='y', linestyle='--', color=PALETTE['grid_color'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    out_path = os.path.join(CHARTS_DIR, 'overtime_impact.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Exported: {out_path}")


def export_model_benchmark_and_evaluation():
    """5, 6, 7, 8: Model comparison, ROC curves, confusion matrix, feature importance"""
    comp_file = os.path.join(MODELS_DIR, 'model_comparison.pkl')
    best_file = os.path.join(MODELS_DIR, 'best_model.pkl')
    if not os.path.exists(comp_file) or not os.path.exists(best_file):
        print("Model artifacts missing. Run train_model.py first.")
        return

    comp_data = joblib.load(comp_file)
    best_data = joblib.load(best_file)
    df_comp = pd.DataFrame(comp_data['comparison_table'])

    # 5. Multi-Metric Benchmark Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=PALETTE['bg_white'])
    ax.set_facecolor(PALETTE['bg_white'])
    
    models = df_comp['Model'].tolist()
    x = np.arange(len(models))
    width = 0.16
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    colors = ['#2563EB', '#4F46E5', PALETTE['coral_red'], '#059669', '#D97706']
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax.bar(x + i*width, df_comp[metric], width, label=metric, color=color)
        
    ax.set_ylabel('Score (0.0 to 1.0)', fontsize=11, color=PALETTE['slate_muted'])
    ax.set_title('Supervised Classification Algorithm Benchmark', fontsize=14, fontweight='bold', pad=15, color=PALETTE['slate_dark'])
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(models, fontsize=10, fontweight='medium', color=PALETTE['slate_dark'])
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True, facecolor=PALETTE['bg_white'], edgecolor=PALETTE['grid_color'], loc='upper right')
    ax.grid(axis='y', linestyle='--', color=PALETTE['grid_color'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    out_path = os.path.join(CHARTS_DIR, 'model_benchmark.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Exported: {out_path}")

    # 6. ROC Curves Comparison
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=PALETTE['bg_white'])
    ax.set_facecolor(PALETTE['bg_white'])
    
    roc_colors = ['#2563EB', PALETTE['coral_red'], '#059669', '#4F46E5']
    for idx, (m_name, m_info) in enumerate(comp_data['models_data'].items()):
        curve = m_info.get('roc_curve', {})
        fpr = curve.get('fpr', [])
        tpr = curve.get('tpr', [])
        auc = curve.get('roc_auc', 0.5)
        ax.plot(fpr, tpr, label=f"{m_name} (AUC = {auc:.3f})", color=roc_colors[idx % len(roc_colors)], linewidth=2.2)

    ax.plot([0, 1], [0, 1], 'k--', color='#94A3B8', linewidth=1.5, label='Random Baseline (AUC = 0.500)')
    ax.set_xlabel('False Positive Rate', fontsize=11, color=PALETTE['slate_muted'])
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=11, color=PALETTE['slate_muted'])
    ax.set_title('ROC Curves: Model Discrimination Capability', fontsize=14, fontweight='bold', pad=15, color=PALETTE['slate_dark'])
    ax.legend(frameon=True, facecolor=PALETTE['bg_white'], edgecolor=PALETTE['grid_color'], loc='lower right')
    ax.grid(True, linestyle='--', color=PALETTE['grid_color'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    out_path = os.path.join(CHARTS_DIR, 'roc_curves.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Exported: {out_path}")

    # 7. Champion Confusion Matrix
    cm = np.array(best_data['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=PALETTE['bg_white'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted: Stay (No)', 'Predicted: Leave (Yes)'],
                yticklabels=['Actual: Stay (No)', 'Actual: Leave (Yes)'],
                annot_kws={'fontsize': 14, 'fontweight': 'bold'}, ax=ax)
    ax.set_title(f"Champion Model Confusion Matrix ({best_data['model_name']})", fontsize=12, fontweight='bold', pad=15, color=PALETTE['slate_dark'])
    
    out_path = os.path.join(CHARTS_DIR, 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Exported: {out_path}")

    # 8. Feature Importance Ranking
    feat_dict = best_data.get('feature_importances', {})
    if feat_dict:
        top_feats = list(feat_dict.items())[:12]
        df_feats = pd.DataFrame(top_feats, columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=True)
        df_feats['Clean'] = df_feats['Feature'].apply(lambda s: s.replace('num__', '').replace('cat__', '').replace('_', ' '))

        fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=PALETTE['bg_white'])
        ax.set_facecolor(PALETTE['bg_white'])
        bars = ax.barh(df_feats['Clean'], df_feats['Importance'], color='#2563EB', height=0.6)
        
        for bar in bars:
            xval = bar.get_width()
            ax.text(xval + 0.005, bar.get_y() + bar.get_height()/2.0, f"{xval:.3f}", ha='left', va='center', fontsize=9.5, fontweight='bold', color=PALETTE['slate_dark'])

        ax.set_xlabel('Absolute Relative Predictive Weight', fontsize=11, color=PALETTE['slate_muted'])
        ax.set_title('Top 12 Most Influential Features in Champion Model', fontsize=14, fontweight='bold', pad=15, color=PALETTE['slate_dark'])
        ax.grid(axis='x', linestyle='--', color=PALETTE['grid_color'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        out_path = os.path.join(CHARTS_DIR, 'feature_importance.png')
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Exported: {out_path}")


def main():
    print("=" * 60)
    print("Exporting High-Resolution Visualizations for README...")
    print("=" * 60)
    df = load_dataset()
    export_attrition_overview(df)
    export_department_attrition(df)
    export_job_role_attrition(df)
    export_overtime_impact(df)
    export_model_benchmark_and_evaluation()
    print("=" * 60)
    print(f"All visualizations exported cleanly to: {CHARTS_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
