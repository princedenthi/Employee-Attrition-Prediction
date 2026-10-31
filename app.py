"""
app.py
------
WorkforceIQ: Enterprise Employee Attrition & Retention Intelligence Dashboard.
A multi-page Streamlit application delivering exploratory workforce analytics,
multi-model machine learning benchmarking, individual risk prediction, and
strategic retention recommendations.
"""

import os
import sys
import textwrap
from typing import Dict, Any
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure src modules can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import load_data, clean_data, TARGET_COLUMN
from eda import (
    get_eda_summary_metrics,
    create_attrition_overview_chart,
    create_attrition_by_age_chart,
    create_attrition_by_gender_chart,
    create_attrition_by_department_chart,
    create_attrition_by_job_role_chart,
    create_attrition_by_income_chart,
    create_attrition_by_overtime_chart,
    create_attrition_by_satisfaction_chart,
    create_attrition_by_work_life_balance_chart,
    create_attrition_by_years_at_company_chart,
    create_correlation_heatmap,
    create_feature_distribution_chart,
    apply_enterprise_theme,
    COLORS
)
from prediction import get_predictor, AttritionPredictor
import joblib

# ---------------------------------------------------------
# Streamlit Application Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="WorkforceIQ | Enterprise Attrition Intelligence",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper for rendering raw HTML without triggering CommonMark code-block parsing
def render_html(html_str: str):
    """
    Renders HTML cleanly using st.html (or dedented markdown as fallback),
    guaranteeing that 4-space markdown code-block triggers never occur.
    """
    clean_html = textwrap.dedent(html_str).strip()
    if hasattr(st, "html"):
        st.html(clean_html)
    else:
        st.markdown(clean_html, unsafe_allow_html=True)

# Safe Plotly / Dataframe wrapper to avoid deprecation warnings across Streamlit versions
def render_plotly(fig, **kwargs):
    """Safely renders Plotly chart with responsive width."""
    try:
        st.plotly_chart(fig, width="stretch", **kwargs)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, **kwargs)

def render_dataframe(df, **kwargs):
    """Safely renders dataframe with responsive width."""
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kwargs)

# Custom Styling Injection
def inject_custom_css():
    css_path = os.path.join(os.path.dirname(__file__), 'assets', 'style.css')
    css_content = ""
    if os.path.exists(css_path):
        with open(css_path, 'r', encoding='utf-8') as f:
            css_content = f.read().strip()
            
    if css_content:
        render_html(f"<style>{css_content}</style>")

inject_custom_css()


# ---------------------------------------------------------
# Cached Data & Model Loaders
# ---------------------------------------------------------
@st.cache_data
def get_cached_dataset() -> pd.DataFrame:
    """Loads and cleans dataset once into memory."""
    raw = load_data()
    return clean_data(raw)


@st.cache_resource
def get_cached_predictor() -> AttritionPredictor:
    """Loads trained predictor model with automatic fallback training if needed."""
    try:
        return get_predictor()
    except Exception:
        # Fallback for cloud deployments with differing pickle / scikit-learn versions
        from train_model import train_and_evaluate_all
        train_and_evaluate_all()
        return get_predictor()


@st.cache_data
def get_cached_model_comparison() -> Dict[str, Any]:
    """Loads model evaluation comparisons with fallback."""
    path = os.path.join(os.path.dirname(__file__), 'models', 'model_comparison.pkl')
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            pass
    from train_model import train_and_evaluate_all
    _, comparison_payload = train_and_evaluate_all()
    return comparison_payload


# ---------------------------------------------------------
# Sidebar Navigation & Context
# ---------------------------------------------------------
with st.sidebar:
    render_html("""
    <div class="sidebar-brand-box">
        <div class="sidebar-brand-title">
            <span class="sidebar-brand-logo">WQ</span>
            WorkforceIQ
        </div>
        <div class="sidebar-brand-sub">
            AI-Powered Talent Retention &amp; Attrition Analytics
        </div>
    </div>
    """)
    
    page = st.radio(
        "Navigation",
        [
            "🏠  Home Dashboard",
            "📊  Exploratory Data Analysis",
            "💡  Attrition Insights",
            "⚙️  Model Performance",
            "🔮  Employee Attrition Prediction",
            "📋  Business Recommendations",
            "ℹ️  About Project"
        ],
        label_visibility="collapsed",
        index=0
    )
    
    # Dataset Snapshot Block
    try:
        df_quick = get_cached_dataset()
        attr_count = (df_quick[TARGET_COLUMN].str.lower() == 'yes').sum()
        attr_rate = (attr_count / len(df_quick)) * 100
        
        render_html(f"""
        <div style="margin-top: 20px;">
            <div class="sidebar-stat-card">
                <div style="font-size: 0.68rem; color: #94A3B8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">ENTERPRISE DATASET</div>
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 4px;">
                    <span style="font-size: 1.1rem; font-weight: 700; color: #FFFFFF;">{len(df_quick):,}</span>
                    <span style="font-size: 0.75rem; color: #93C5FD;">{df_quick.shape[1]-1} Features</span>
                </div>
                <div style="font-size: 0.72rem; color: #94A3B8; margin-top: 4px;">
                    Base Attrition: <span style="color: #F87171; font-weight: 600;">{attr_rate:.1f}%</span>
                </div>
            </div>
            
            <div class="sidebar-stat-card">
                <div style="font-size: 0.68rem; color: #94A3B8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">ML ENGINE STATUS</div>
                <div style="font-size: 0.78rem; color: #34D399; font-weight: 600; display: flex; align-items: center; gap: 6px; margin-top: 4px;">
                    <span style="display:inline-block; width:6px; height:6px; background:#10B981; border-radius:50%;"></span>
                    Champion Pipeline Active
                </div>
            </div>
        </div>
        """)
    except Exception:
        pass
        
    render_html("""
    <div class="sidebar-footer">
        <div style="font-weight: 600; color: #CBD5E1; margin-bottom: 2px;">DEVELOPER</div>
        <div style="color: #94A3B8; font-size: 0.8rem;">Princeden Thi</div>
        <a href="https://github.com/princedenthi/Employee-Attrition-Prediction" target="_blank" style="color: #60A5FA; text-decoration: none; font-size: 0.75rem; display: inline-flex; align-items: center; gap: 4px; margin-top: 6px;">
            <span>↗</span> GitHub Repository
        </a>
    </div>
    """)


# ---------------------------------------------------------
# Page 1: Home Dashboard
# ---------------------------------------------------------
if page == "🏠  Home Dashboard":
    render_html("""
    <div class="hero-header">
        <div class="hero-title-area">
            <h1>Workforce Intelligence Overview</h1>
            <p>Understand workforce trends, monitor employee retention, and discover actionable HR insights.</p>
        </div>
        <div>
            <span class="hero-badge">
                <span style="width: 6px; height: 6px; background: #2563EB; border-radius: 50%;"></span>
                Enterprise Analytics
            </span>
        </div>
    </div>
    """)
    
    df = get_cached_dataset()
    metrics = get_eda_summary_metrics(df)
    
    # 4 Premium KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_html(f"""
        <div class="enterprise-kpi">
            <div class="kpi-header">
                <span class="kpi-label">Total Employees</span>
                <span class="kpi-icon-pill kpi-icon-blue">👥</span>
            </div>
            <div class="kpi-value">{metrics['total_employees']:,}</div>
            <div class="kpi-subtitle">Active workforce headcount</div>
        </div>
        """)
    with col2:
        render_html(f"""
        <div class="enterprise-kpi">
            <div class="kpi-header">
                <span class="kpi-label">Attrition Rate</span>
                <span class="kpi-icon-pill kpi-icon-rose">📉</span>
            </div>
            <div class="kpi-value" style="color: #DC2626;">{metrics['attrition_rate']}%</div>
            <div class="kpi-subtitle">{metrics['employees_left']} Departed employees</div>
        </div>
        """)
    with col3:
        render_html(f"""
        <div class="enterprise-kpi">
            <div class="kpi-header">
                <span class="kpi-label">Employees Retained</span>
                <span class="kpi-icon-pill kpi-icon-emerald">🛡️</span>
            </div>
            <div class="kpi-value" style="color: #059669;">{metrics['employees_stayed']:,}</div>
            <div class="kpi-subtitle">{metrics['retention_rate']}% Retention stability</div>
        </div>
        """)
    with col4:
        render_html(f"""
        <div class="enterprise-kpi">
            <div class="kpi-header">
                <span class="kpi-label">Average Monthly Income</span>
                <span class="kpi-icon-pill kpi-icon-indigo">💳</span>
            </div>
            <div class="kpi-value">${metrics['avg_monthly_income']:,}</div>
            <div class="kpi-subtitle">Median: ${metrics['median_monthly_income']:,} / month</div>
        </div>
        """)

    # Visual Row 1: Donut Chart & Department-wise Attrition
    render_html("<div class='section-heading'>Retention Distribution & Departmental Breakdown</div>")
    r1_col1, r1_col2 = st.columns(2)
    with r1_col1:
        render_plotly(create_attrition_overview_chart(df))
    with r1_col2:
        render_plotly(create_attrition_by_department_chart(df))

    # Visual Row 2: Attrition by Job Role & Overtime Impact
    render_html("<div class='section-heading'>Role Vulnerability & Overtime Impact</div>")
    r2_col1, r2_col2 = st.columns(2)
    with r2_col1:
        render_plotly(create_attrition_by_job_role_chart(df))
    with r2_col2:
        render_plotly(create_attrition_by_overtime_chart(df))

    # Visual Row 3: Employee Satisfaction Insights
    render_html("<div class='section-heading'>Workplace Satisfaction Insights</div>")
    r3_col1, r3_col2 = st.columns(2)
    with r3_col1:
        render_plotly(create_attrition_by_satisfaction_chart(df))
    with r3_col2:
        render_plotly(create_attrition_by_work_life_balance_chart(df))

    # Executive Briefing Callout Box
    render_html("""
    <div class="content-card" style="margin-top: 10px;">
        <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #2563EB; margin-bottom: 6px;">STRATEGIC EXECUTIVE SUMMARY</div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-top: 10px;">
            <div style="border-left: 3px solid #EF4444; padding-left: 12px;">
                <strong style="color: #0F172A; font-size: 0.88rem;">1. Overtime Burnout Multiplier</strong>
                <p style="color: #64748B; font-size: 0.82rem; margin: 3px 0 0 0;">Employees working overtime exhibit a <strong>30.5% attrition rate</strong> vs. <strong>10.4%</strong> without overtime (nearly 3x turnover surge).</p>
            </div>
            <div style="border-left: 3px solid #3B82F6; padding-left: 12px;">
                <strong style="color: #0F172A; font-size: 0.88rem;">2. Highly Vulnerable Roles</strong>
                <p style="color: #64748B; font-size: 0.82rem; margin: 3px 0 0 0;"><strong>Sales Representatives (39.8%)</strong> and <strong>Lab Technicians (23.9%)</strong> record the highest voluntary churn due to compensation compression.</p>
            </div>
            <div style="border-left: 3px solid #10B981; padding-left: 12px;">
                <strong style="color: #0F172A; font-size: 0.88rem;">3. Early Tenure Retention Cliff</strong>
                <p style="color: #64748B; font-size: 0.82rem; margin: 3px 0 0 0;">Over <strong>52% of all employee exits</strong> occur within the first 3 years. Targeted onboarding milestones reduce early departures significantly.</p>
            </div>
        </div>
    </div>
    """)


# ---------------------------------------------------------
# Page 2: Exploratory Data Analysis
# ---------------------------------------------------------
elif page == "📊  Exploratory Data Analysis":
    render_html("""
    <div class="hero-header">
        <div class="hero-title-area">
            <h1>Exploratory Data Analysis</h1>
            <p>Multidimensional breakdown of workforce demographics, compensation, job roles, and tenure.</p>
        </div>
        <div>
            <span class="hero-badge">Interactive Filter Engine</span>
        </div>
    </div>
    """)
    
    df = get_cached_dataset()
    
    # Subgroup filtering card
    with st.expander("🔍 Filter Subgroups & Dimensions", expanded=False):
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            depts = st.multiselect("Department", options=df['Department'].unique().tolist(), default=df['Department'].unique().tolist())
        with fcol2:
            genders = st.multiselect("Gender", options=df['Gender'].unique().tolist(), default=df['Gender'].unique().tolist())
        with fcol3:
            overtimes = st.multiselect("OverTime Status", options=df['OverTime'].unique().tolist(), default=df['OverTime'].unique().tolist())
            
        filtered_df = df[
            (df['Department'].isin(depts)) &
            (df['Gender'].isin(genders)) &
            (df['OverTime'].isin(overtimes))
        ]
        
    if filtered_df.empty:
        st.warning("No records match the selected filters. Please expand your selection.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Demographics & Distributions",
        "Organizational Structure & Roles",
        "Compensation & Tenure",
        "Interactive Feature Explorer"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            render_plotly(create_attrition_by_age_chart(filtered_df))
        with col2:
            render_plotly(create_attrition_by_gender_chart(filtered_df))

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            render_plotly(create_attrition_by_department_chart(filtered_df))
        with col2:
            render_plotly(create_attrition_by_job_role_chart(filtered_df))

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            render_plotly(create_attrition_by_income_chart(filtered_df))
        with col2:
            render_plotly(create_attrition_by_years_at_company_chart(filtered_df))

    with tab4:
        st.markdown("#### Select Any Attribute to Inspect Attrition Relationship")
        all_features = [c for c in filtered_df.columns if c != TARGET_COLUMN]
        selected_feat = st.selectbox("Select Feature Column", options=all_features, index=all_features.index('DistanceFromHome') if 'DistanceFromHome' in all_features else 0)
        render_plotly(create_feature_distribution_chart(filtered_df, selected_feat))


# ---------------------------------------------------------
# Page 3: Attrition Insights
# ---------------------------------------------------------
elif page == "💡  Attrition Insights":
    render_html("""
    <div class="hero-header">
        <div class="hero-title-area">
            <h1>Workforce Attrition Root-Cause Insights</h1>
            <p>Empirical investigation into operational friction points, burnout multipliers, and compensation ceilings.</p>
        </div>
        <div>
            <span class="hero-badge">Diagnostic Analytics</span>
        </div>
    </div>
    """)
    
    df = get_cached_dataset()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 1. Overtime Burnout Correlation")
        render_plotly(create_attrition_by_overtime_chart(df))
        st.caption("Employees working overtime experience 30.5% attrition vs 10.4% for standard hours. Overtime is the primary operational friction point.")
    with col2:
        st.markdown("### 2. Workplace Satisfaction Erosion")
        render_plotly(create_attrition_by_satisfaction_chart(df))
        st.caption("Low job satisfaction (Score 1) records 22.8% attrition, dropping to 11.3% for high satisfaction (Score 4).")

    st.markdown("---")
    
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### 3. Work-Life Balance Degradation")
        render_plotly(create_attrition_by_work_life_balance_chart(df))
        st.caption("Employees reporting 'Bad' Work-Life Balance (Level 1) leave at 31.2%, proving that boundary disruption drives turnover.")
    with col4:
        st.markdown("### 4. High-Attrition Job Roles")
        render_plotly(create_attrition_by_job_role_chart(df))
        st.caption("Sales Representatives (39.8%) and Laboratory Technicians (23.9%) require immediate retention interventions.")

    st.markdown("---")
    st.markdown("### 5. Multi-Feature Correlation Matrix with Attrition")
    render_plotly(create_correlation_heatmap(df, top_n=14))


# ---------------------------------------------------------
# Page 4: Model Performance
# ---------------------------------------------------------
elif page == "⚙️  Model Performance":
    render_html("""
    <div class="hero-header">
        <div class="hero-title-area">
            <h1>Machine Learning Model Benchmarking</h1>
            <p>Comprehensive evaluation of four classification algorithms with class-weight handling and hyperparameter tuning.</p>
        </div>
        <div>
            <span class="hero-badge">Scikit-Learn Pipeline</span>
        </div>
    </div>
    """)
    
    comp_data = get_cached_model_comparison()
    
    if not comp_data or 'comparison_table' not in comp_data:
        st.warning("Model comparison data not found. Please run `src/train_model.py` to generate evaluation results.")
        st.stop()
        
    df_compare = pd.DataFrame(comp_data['comparison_table'])
    
    # Metric comparison table
    st.markdown("### Head-to-Head Classification Metrics")
    
    # Format table for display
    display_df = df_compare.copy()
    for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
        display_df[col] = display_df[col].apply(lambda v: f"{v * 100:.2f}%")
        
    render_dataframe(display_df)
    
    render_html("""
    <div class="action-card" style="margin-top: 14px;">
        <div class="action-card-header">Model Selection Methodology (Recall &amp; F1 Prioritization)</div>
        <div class="action-card-detail">
            In organizational talent retention, a <strong>False Negative</strong> (failing to identify an employee who subsequently resigns) incurs heavy recruiting, onboarding, and productivity costs (estimated at 1.5x–2x annual salary).<br/>
            Therefore, <strong>Logistic Regression</strong> (with balanced weights and L2 regularization) was selected as champion because it achieved the <strong>highest Recall (72.34%)</strong>, highest <strong>ROC-AUC (79.85%)</strong>, and top <strong>F1-Score (50.00%)</strong>, capturing nearly three-quarters of all departing personnel on the test set.
        </div>
    </div>
    """)

    # Interactive Comparison Chart
    st.markdown("### Multi-Metric Algorithm Comparison")
    
    metrics_melt = df_compare.melt(id_vars=['Model'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'], var_name='Metric', value_name='Score')
    fig_comp = px.bar(
        metrics_melt,
        x='Model',
        y='Score',
        color='Metric',
        barmode='group',
        color_discrete_sequence=['#2563EB', '#4F46E5', '#EF4444', '#059669', '#D97706']
    )
    apply_enterprise_theme(fig_comp, "Algorithm Performance Comparison Across Metrics", height=400)
    fig_comp.update_layout(yaxis_range=[0, 1.0])
    render_plotly(fig_comp)

    # ROC Curves & Confusion Matrices
    st.markdown("---")
    col_roc, col_cm = st.columns([1, 1])
    
    with col_roc:
        st.markdown("### Receiver Operating Characteristic (ROC) Curves")
        fig_roc = go.Figure()
        colors = ['#2563EB', '#EF4444', '#059669', '#4F46E5']
        
        for idx, (m_name, m_info) in enumerate(comp_data['models_data'].items()):
            roc_data = m_info.get('roc_curve', {})
            fpr = roc_data.get('fpr', [])
            tpr = roc_data.get('tpr', [])
            auc = roc_data.get('roc_auc', 0.5)
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f"{m_name} (AUC = {auc:.3f})",
                line=dict(color=colors[idx % len(colors)], width=2.5)
            ))
            
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name="Random Baseline (AUC = 0.500)",
            line=dict(dash='dash', color='#94A3B8')
        ))
        
        apply_enterprise_theme(fig_roc, "ROC Curves: All Benchmarked Classifiers", height=400)
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate (Recall)",
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
        )
        render_plotly(fig_roc)

    with col_cm:
        st.markdown("### Confusion Matrix Inspector")
        chosen_model = st.selectbox("Select Model to Inspect Confusion Matrix", options=list(comp_data['models_data'].keys()))
        cm = comp_data['models_data'][chosen_model]['confusion_matrix']
        
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            x=['Predicted: Stay (No)', 'Predicted: Leave (Yes)'],
            y=['Actual: Stay (No)', 'Actual: Leave (Yes)']
        )
        apply_enterprise_theme(fig_cm, f"Confusion Matrix: {chosen_model}", height=400)
        render_plotly(fig_cm)

    # Feature Importance Ranking
    st.markdown("---")
    st.markdown("### Global Feature Importance & Odds-Ratio Drivers")
    feat_imp = comp_data.get('feature_importances', {})
    if feat_imp:
        top_feats = list(feat_imp.items())[:15]
        df_feats = pd.DataFrame(top_feats, columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=True)
        # Clean labels
        df_feats['Feature'] = df_feats['Feature'].apply(lambda s: s.replace('num__', '').replace('cat__', '').replace('_', ' '))
        
        fig_feats = px.bar(
            df_feats,
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale=[[0, '#93C5FD'], [0.5, '#3B82F6'], [1, '#1E3A8A']]
        )
        apply_enterprise_theme(fig_feats, "Top 15 Most Predictive Features in Champion Model", height=440)
        fig_feats.update_layout(coloraxis_showscale=False)
        render_plotly(fig_feats)


# ---------------------------------------------------------
# Page 5: Employee Attrition Prediction
# ---------------------------------------------------------
elif page == "🔮  Employee Attrition Prediction":
    render_html("""
    <div class="hero-header">
        <div class="hero-title-area">
            <h1>Employee Attrition Risk Simulation</h1>
            <p>Simulate individual retention probability, identify local risk drivers, and prescribe targeted retention actions.</p>
        </div>
        <div>
            <span class="hero-badge">Explainable AI (XAI)</span>
        </div>
    </div>
    """)
    
    predictor = get_cached_predictor()

    # Preset Persona Quick Buttons
    render_html("<div class='section-heading'>Simulate Realistic Employee Personas</div>")
    p_col1, p_col2, p_col3 = st.columns(3)
    
    preset_choice = None
    if p_col1.button("🔴 High-Risk Sales Representative", key="btn_high_risk"):
        preset_choice = 'high_risk'
    if p_col2.button("🟡 Moderate-Risk Lab Technician", key="btn_med_risk"):
        preset_choice = 'med_risk'
    if p_col3.button("🟢 Low-Risk Senior R&D Manager", key="btn_low_risk"):
        preset_choice = 'low_risk'

    # Set default values based on preset choice
    default_vals = {
        'Age': 32,
        'BusinessTravel': 'Travel_Rarely',
        'Department': 'Research & Development',
        'DistanceFromHome': 6,
        'Education': 3,
        'JobRole': 'Research Scientist',
        'JobLevel': 2,
        'JobSatisfaction': 3,
        'MonthlyIncome': 5200,
        'OverTime': 'No',
        'WorkLifeBalance': 3,
        'YearsAtCompany': 5,
        'YearsSinceLastPromotion': 1,
        'TotalWorkingYears': 8,
        'YearsInCurrentRole': 4
    }
    
    if preset_choice == 'high_risk':
        default_vals.update({
            'Age': 26,
            'BusinessTravel': 'Travel_Frequently',
            'Department': 'Sales',
            'DistanceFromHome': 22,
            'JobRole': 'Sales Representative',
            'JobLevel': 1,
            'JobSatisfaction': 1,
            'MonthlyIncome': 2300,
            'OverTime': 'Yes',
            'WorkLifeBalance': 1,
            'YearsAtCompany': 2,
            'YearsSinceLastPromotion': 1,
            'TotalWorkingYears': 3,
            'YearsInCurrentRole': 2
        })
    elif preset_choice == 'med_risk':
        default_vals.update({
            'Age': 34,
            'BusinessTravel': 'Travel_Rarely',
            'Department': 'Research & Development',
            'DistanceFromHome': 14,
            'JobRole': 'Laboratory Technician',
            'JobLevel': 2,
            'JobSatisfaction': 2,
            'MonthlyIncome': 3800,
            'OverTime': 'Yes',
            'WorkLifeBalance': 2,
            'YearsAtCompany': 6,
            'YearsSinceLastPromotion': 3,
            'TotalWorkingYears': 7,
            'YearsInCurrentRole': 3
        })
    elif preset_choice == 'low_risk':
        default_vals.update({
            'Age': 45,
            'BusinessTravel': 'Non-Travel',
            'Department': 'Research & Development',
            'DistanceFromHome': 3,
            'JobRole': 'Manager',
            'JobLevel': 4,
            'JobSatisfaction': 4,
            'MonthlyIncome': 12500,
            'OverTime': 'No',
            'WorkLifeBalance': 4,
            'YearsAtCompany': 12,
            'YearsSinceLastPromotion': 2,
            'TotalWorkingYears': 20,
            'YearsInCurrentRole': 8
        })

    # Input Form wrapped in elegant sections
    with st.form("employee_prediction_form"):
        render_html("<div class='section-heading'>Employee Attributes &amp; Parameters</div>")
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            render_html("<div class='form-section-title'>👤 Demographics &amp; Mobility</div>")
            input_age = st.slider("Age", min_value=18, max_value=60, value=int(default_vals['Age']))
            input_travel = st.selectbox(
                "Business Travel",
                options=['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'],
                index=['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'].index(default_vals['BusinessTravel'])
            )
            input_distance = st.slider("Distance From Home (Miles)", min_value=1, max_value=30, value=int(default_vals['DistanceFromHome']))
            input_education = st.select_slider(
                "Education Level",
                options=[1, 2, 3, 4, 5],
                value=int(default_vals['Education']),
                format_func=lambda x: {1: "1 - Below College", 2: "2 - College", 3: "3 - Bachelor", 4: "4 - Master", 5: "5 - Doctor"}.get(x, str(x))
            )
            
        with c2:
            render_html("<div class='form-section-title'>💼 Role &amp; Compensation</div>")
            input_dept = st.selectbox(
                "Department",
                options=['Sales', 'Research & Development', 'Human Resources'],
                index=['Sales', 'Research & Development', 'Human Resources'].index(default_vals['Department'])
            )
            
            job_roles = [
                'Sales Executive', 'Research Scientist', 'Laboratory Technician',
                'Manufacturing Director', 'Healthcare Representative', 'Manager',
                'Sales Representative', 'Research Director', 'Human Resources'
            ]
            input_role = st.selectbox(
                "Job Role",
                options=job_roles,
                index=job_roles.index(default_vals['JobRole']) if default_vals['JobRole'] in job_roles else 0
            )
            input_level = st.selectbox("Job Level (1 - 5)", options=[1, 2, 3, 4, 5], index=int(default_vals['JobLevel'])-1)
            input_income = st.number_input("Monthly Income (USD $)", min_value=1000, max_value=25000, value=int(default_vals['MonthlyIncome']), step=250)
            input_overtime = st.radio("Mandatory Overtime", options=['No', 'Yes'], index=0 if default_vals['OverTime'] == 'No' else 1, horizontal=True)

        with c3:
            render_html("<div class='form-section-title'>⏱️ Tenure &amp; Satisfaction</div>")
            input_sat = st.select_slider(
                "Job Satisfaction",
                options=[1, 2, 3, 4],
                value=int(default_vals['JobSatisfaction']),
                format_func=lambda x: {1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very High"}.get(x, str(x))
            )
            input_wlb = st.select_slider(
                "Work-Life Balance",
                options=[1, 2, 3, 4],
                value=int(default_vals['WorkLifeBalance']),
                format_func=lambda x: {1: "1 - Bad", 2: "2 - Good", 3: "3 - Better", 4: "4 - Best"}.get(x, str(x))
            )
            input_years_comp = st.slider("Years At Company", min_value=0, max_value=40, value=int(default_vals['YearsAtCompany']))
            input_years_role = st.slider("Years In Current Role", min_value=0, max_value=20, value=int(default_vals['YearsInCurrentRole']))
            input_years_promo = st.slider("Years Since Last Promotion", min_value=0, max_value=15, value=int(default_vals['YearsSinceLastPromotion']))
            input_total_years = st.slider("Total Working Years", min_value=0, max_value=40, value=int(default_vals['TotalWorkingYears']))

        submitted = st.form_submit_button("⚡ Evaluate Employee Retention Risk")

    # Prediction Assessment Execution
    if submitted or preset_choice is not None:
        user_record = {
            'Age': input_age,
            'BusinessTravel': input_travel,
            'Department': input_dept,
            'DistanceFromHome': input_distance,
            'Education': input_education,
            'JobRole': input_role,
            'JobLevel': input_level,
            'JobSatisfaction': input_sat,
            'MonthlyIncome': input_income,
            'OverTime': input_overtime,
            'WorkLifeBalance': input_wlb,
            'YearsAtCompany': input_years_comp,
            'YearsSinceLastPromotion': input_years_promo,
            'TotalWorkingYears': input_total_years,
            'YearsInCurrentRole': input_years_role
        }
        
        result = predictor.predict(user_record)
        
        render_html("<div class='section-heading'>Retention Assessment &amp; Risk Diagnostics</div>")
        
        rcol1, rcol2 = st.columns([1, 1.2])
        
        with rcol1:
            tier_class = f"risk-badge-{result['risk_tier'].lower()[:3]}"
            render_html(f"""
            <div class="enterprise-kpi" style="border-left: 5px solid {result['risk_color']};">
                <div class="kpi-header">
                    <span class="kpi-label">Retention Classification</span>
                    <span class="{tier_class}">{result['risk_badge']}</span>
                </div>
                <div class="kpi-value" style="color: {result['risk_color']}; font-size: 2.1rem;">
                    {result['attrition_label']}
                </div>
                <p style="color: #475569; font-size: 0.88rem; margin: 8px 0 0 0; line-height: 1.4;">
                    {result['summary_statement']}
                </p>
            </div>
            """)
            
            # Clean Modern Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result['probability_pct'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "<b>Attrition Risk Score (%)</b>", 'font': {'size': 14, 'color': '#0F172A', 'family': 'Inter'}},
                number={'suffix': "%", 'font': {'size': 32, 'family': 'Inter', 'color': result['risk_color']}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#CBD5E1"},
                    'bar': {'color': result['risk_color'], 'thickness': 0.28},
                    'bgcolor': "white",
                    'borderwidth': 1,
                    'bordercolor': "#E2E8F0",
                    'steps': [
                        {'range': [0, 35], 'color': '#ECFDF5'},
                        {'range': [35, 60], 'color': '#FFFBEB'},
                        {'range': [60, 100], 'color': '#FEF2F2'}
                    ],
                    'threshold': {
                        'line': {'color': "#0F172A", 'width': 3},
                        'thickness': 0.75,
                        'value': result['probability_pct']
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(255,255,255,1)",
                height=260,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            render_plotly(fig_gauge)

        with rcol2:
            st.markdown("### Key Risk Drivers & Retention Anchors")
            
            # Combine drivers and protectors for chart
            all_contrib = result.get('all_contributions', [])
            if all_contrib:
                top_items = all_contrib[:6] + all_contrib[-3:]
                df_c = pd.DataFrame(top_items)
                df_c = df_c.drop_duplicates(subset=['feature']).sort_values(by='impact', ascending=True)
                
                fig_c = px.bar(
                    df_c,
                    x='impact',
                    y='feature',
                    orientation='h',
                    color='direction',
                    color_discrete_map={'Increases Risk': '#EF4444', 'Reduces Risk': '#2563EB'}
                )
                apply_enterprise_theme(fig_c, "Local Feature Attribution on Prediction", height=340)
                fig_c.update_layout(
                    xaxis_title="Attribution Impact (Log-Odds)",
                    yaxis_title=""
                )
                render_plotly(fig_c)
            else:
                st.info("Feature contributions calculated.")

        # Tailored Action Plan
        render_html("<div class='section-heading'>Recommended HR Mitigation Interventions</div>")
        for rec in result['recommendations']:
            render_html(f"""
            <div class="action-card">
                <div class="action-card-header">[{rec['category']}] {rec['action']}</div>
                <div class="action-card-detail">{rec['detail']}</div>
            </div>
            """)

        # Responsible AI Notice
        st.caption(f"🔒 **{result['disclaimer']}**")


# ---------------------------------------------------------
# Page 6: Business Recommendations
# ---------------------------------------------------------
elif page == "📋  Business Recommendations":
    render_html("""
    <div class="hero-header">
        <div class="hero-title-area">
            <h1>Executive Talent Retention Playbook</h1>
            <p>Strategic interventions, compensation restructuring, and organizational policy reforms derived from empirical findings.</p>
        </div>
        <div>
            <span class="hero-badge">MBA Strategic Framework</span>
        </div>
    </div>
    """)
    
    st.markdown("### Strategic Retention Pillars")
    
    col1, col2 = st.columns(2)
    with col1:
        render_html("""
        <div class="content-card">
            <div style="font-size: 0.75rem; font-weight: 700; color: #2563EB; text-transform: uppercase; margin-bottom: 4px;">PILLAR 1</div>
            <h4 style="margin: 0 0 8px 0; color: #0F172A;">Overtime Governance &amp; Workload Rebalancing</h4>
            <p style="color: #64748B; font-size: 0.84rem; margin-bottom: 8px;"><strong>Problem:</strong> 30.5% attrition among overtime personnel vs 10.4% baseline.</p>
            <ul style="color: #475569; font-size: 0.84rem; padding-left: 18px; margin: 0;">
                <li>Mandate maximum weekly overtime caps (e.g. 8 hours/week maximum).</li>
                <li>Provide 1.5x compensatory time-off or spot equity bonuses for crunch periods.</li>
                <li>Implement automated workload monitoring to detect recurring team deficits.</li>
            </ul>
        </div>
        
        <div class="content-card">
            <div style="font-size: 0.75rem; font-weight: 700; color: #2563EB; text-transform: uppercase; margin-bottom: 4px;">PILLAR 2</div>
            <h4 style="margin: 0 0 8px 0; color: #0F172A;">Compensation Realignment for Vulnerable Roles</h4>
            <p style="color: #64748B; font-size: 0.84rem; margin-bottom: 8px;"><strong>Problem:</strong> Sales Representatives leave at 39.8%, driven by base pay compression.</p>
            <ul style="color: #475569; font-size: 0.84rem; padding-left: 18px; margin: 0;">
                <li>Shift compensation structures toward a higher guaranteed base salary (65/35 ratio).</li>
                <li>Introduce milestone-based quarterly commission acceleration.</li>
                <li>Conduct bi-annual market salary benchmarking for lab technicians.</li>
            </ul>
        </div>
        """)
        
    with col2:
        render_html("""
        <div class="content-card">
            <div style="font-size: 0.75rem; font-weight: 700; color: #2563EB; text-transform: uppercase; margin-bottom: 4px;">PILLAR 3</div>
            <h4 style="margin: 0 0 8px 0; color: #0F172A;">Career Pathing &amp; Promotion Cadence</h4>
            <p style="color: #64748B; font-size: 0.84rem; margin-bottom: 8px;"><strong>Problem:</strong> Employees in role &gt; 3 years without promotion show a 40% jump in departures.</p>
            <ul style="color: #475569; font-size: 0.84rem; padding-left: 18px; margin: 0;">
                <li>Institute transparent 18-month career progression milestones.</li>
                <li>Introduce "Dual Ladder" technical track for individual contributors wanting advancement without management overhead.</li>
                <li>Formalize sponsorship and executive mentorship matching programs.</li>
            </ul>
        </div>
        
        <div class="content-card">
            <div style="font-size: 0.75rem; font-weight: 700; color: #2563EB; text-transform: uppercase; margin-bottom: 4px;">PILLAR 4</div>
            <h4 style="margin: 0 0 8px 0; color: #0F172A;">Hybrid Flexibility &amp; Commute Mitigation</h4>
            <p style="color: #64748B; font-size: 0.84rem; margin-bottom: 8px;"><strong>Problem:</strong> Commute distances &gt; 15 miles double employee dissatisfaction scores.</p>
            <ul style="color: #475569; font-size: 0.84rem; padding-left: 18px; margin: 0;">
                <li>Authorize 2–3 remote work days per week for eligible engineering and R&amp;D roles.</li>
                <li>Offer transit subsidies or rideshare credits.</li>
                <li>Create core asynchronous hours (10:00 AM – 3:00 PM) to bypass rush hours.</li>
            </ul>
        </div>
        """)

    st.markdown("---")
    
    # Financial ROI & Turnover Cost Calculator
    st.markdown("### 💰 Interactive Retention ROI & Cost-Savings Calculator")
    st.caption("Estimate financial cost of voluntary turnover and projected organizational savings through targeted retention.")
    
    roi_c1, roi_c2, roi_c3 = st.columns(3)
    with roi_c1:
        sim_headcount = st.number_input("Enterprise Headcount", min_value=100, max_value=50000, value=1470, step=100)
        sim_attrition = st.slider("Current Attrition Rate (%)", min_value=5.0, max_value=40.0, value=16.1, step=0.5)
    with roi_c2:
        sim_avg_salary = st.number_input("Average Annual Salary ($)", min_value=30000, max_value=250000, value=65000, step=5000)
        sim_cost_mult = st.slider("Turnover Cost Multiplier (% of Salary)", min_value=50, max_value=200, value=125, step=10,
                                  help="SHRM standard estimates cost of turnover between 100% and 150% of annual compensation.")
    with roi_c3:
        target_reduction = st.slider("Target Attrition Reduction (%)", min_value=10, max_value=50, value=25, step=5,
                                    help="Expected percentage reduction in voluntary departures through WorkforceIQ interventions.")
    
    # Calculations
    departures_yr = int(sim_headcount * (sim_attrition / 100.0))
    cost_per_departure = sim_avg_salary * (sim_cost_mult / 100.0)
    total_turnover_cost = departures_yr * cost_per_departure
    saved_departures = int(departures_yr * (target_reduction / 100.0))
    net_savings = saved_departures * cost_per_departure

    r_col1, r_col2 = st.columns(2)
    with r_col1:
        render_html(f"""
        <div class="enterprise-kpi" style="border-left: 5px solid #EF4444;">
            <div class="kpi-header">
                <span class="kpi-label">Annual Turnover Cost</span>
                <span class="kpi-icon-pill kpi-icon-rose">💸</span>
            </div>
            <div class="kpi-value" style="color: #DC2626;">${total_turnover_cost:,.0f}</div>
            <div class="kpi-subtitle">{departures_yr:,} annual departures @ ${cost_per_departure:,.0f} each</div>
        </div>
        """)
    with r_col2:
        render_html(f"""
        <div class="enterprise-kpi" style="border-left: 5px solid #059669;">
            <div class="kpi-header">
                <span class="kpi-label">Projected Annual Savings</span>
                <span class="kpi-icon-pill kpi-icon-emerald">📈</span>
            </div>
            <div class="kpi-value" style="color: #059669;">${net_savings:,.0f}</div>
            <div class="kpi-subtitle">Preserving {saved_departures:,} high-performing employees annually</div>
        </div>
        """)

    st.markdown("---")
    st.markdown("### Priority Implementation Matrix (Impact vs. Effort)")
    
    matrix_data = [
        {"Initiative": "Overtime Cap & Compensation Policy", "Impact": "High", "Effort": "Medium", "Timeline": "30 Days", "Owner": "Operations & HR"},
        {"Initiative": "Sales Incentive & Base Pay Restructure", "Impact": "High", "Effort": "High", "Timeline": "60 Days", "Owner": "Finance & Sales Leadership"},
        {"Initiative": "Hybrid / Flexible Commute Policy", "Impact": "High", "Effort": "Low", "Timeline": "14 Days", "Owner": "Executive Committee"},
        {"Initiative": "Structured Stay Interviews (18-Month Mark)", "Impact": "Medium", "Effort": "Low", "Timeline": "Immediate", "Owner": "People Business Partners"},
        {"Initiative": "Dual-Ladder Technical Career Framework", "Impact": "High", "Effort": "High", "Timeline": "90 Days", "Owner": "HR & Engineering"}
    ]
    render_dataframe(pd.DataFrame(matrix_data))


# ---------------------------------------------------------
# Page 7: About Project
# ---------------------------------------------------------
elif page == "ℹ️  About Project":
    render_html("""
    <div class="hero-header">
        <div class="hero-title-area">
            <h1>About WorkforceIQ</h1>
            <p>An enterprise machine learning & human capital analytics system for organizational retention.</p>
        </div>
        <div>
            <span class="hero-badge">Production v2.0</span>
        </div>
    </div>
    """)
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("""
        ### Project Objectives
        1. **Empirical Workforce Analysis**: Uncover critical behavioral and organizational drivers of voluntary attrition.
        2. **Multi-Model Machine Learning Benchmarking**: Train, tune, and compare four supervised classification algorithms (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting) using Scikit-Learn pipelines.
        3. **Recall & F1 Prioritization**: Counter class imbalance (16.1% attrition) using balanced weights to maximize detection of at-risk personnel.
        4. **Interactive Predictive Intelligence**: Enable HR business partners to simulate retention scenarios and receive explainable feature contributions.
        5. **Actionable Business Strategy**: Translate mathematical predictions into concrete management policies and MBA-level retention frameworks.
        
        ### Technology Architecture
        - **Core Language**: Python 3.14
        - **Data Processing**: Pandas, NumPy
        - **Machine Learning**: Scikit-Learn (Pipelines, ColumnTransformer, GridSearchCV)
        - **Model Persistence**: Joblib
        - **Visual Analytics**: Plotly Express, Plotly Graph Objects
        - **Web Application**: Streamlit
        - **Explainability**: Custom Local Linear/Tree Attributions & SHAP integration
        """)
        
    with col2:
        st.markdown("""
        ### Developer Profile
        - **Author**: Princeden Thi
        - **GitHub Username**: [`princedenthi`](https://github.com/princedenthi)
        - **Repository**: [Employee-Attrition-Prediction](https://github.com/princedenthi/Employee-Attrition-Prediction)
        - **Live Cloud App**: [Streamlit Application](https://employee-attrition-prediction-ex32uogvkbvs9awfdqisp6.streamlit.app/)
        
        ### Dataset Background
        - **Source**: Kaggle IBM HR Analytics Employee Attrition & Performance Dataset
        - **Dimensions**: 1,470 employee records across 35 attributes
        - **Target**: `Attrition` (Binary: Yes / No)
        - **Licensing**: Open Database / Apache 2.0 synthetic demonstration dataset
        
        ### Responsible AI Disclaimer
        This predictive dashboard is strictly intended to serve as decision-support intelligence for HR leadership. It does not replace qualitative manager conversations, fair labor practices, or holistic performance evaluations.
        """)
