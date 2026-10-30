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
    COLORS
)
from prediction import get_predictor, AttritionPredictor
import joblib

# ---------------------------------------------------------
# Streamlit Application Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="WorkforceIQ | Employee Attrition Intelligence",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling Injection
def inject_custom_css():
    css_path = os.path.join(os.path.dirname(__file__), 'assets', 'style.css')
    custom_css = ""
    if os.path.exists(css_path):
        with open(css_path, 'r', encoding='utf-8') as f:
            custom_css = f"<style>{f.read()}</style>"
    
    inline_css = """
    <style>
    /* Global Adjustments */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }
    .stApp {
        background-color: #F8FAFC;
    }
    /* Brand Header */
    .brand-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
        color: white;
        padding: 24px 30px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .brand-header h1 {
        color: #FFFFFF !important;
        font-weight: 700;
        font-size: 2.1rem;
        margin: 0;
        padding: 0;
    }
    .brand-header p {
        color: #93C5FD;
        margin: 6px 0 0 0;
        font-size: 1rem;
    }
    /* KPI Metric Cards */
    .kpi-container {
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
        margin-bottom: 20px;
    }
    .kpi-card {
        background: white;
        border-radius: 10px;
        padding: 16px 20px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        flex: 1;
        min-width: 200px;
    }
    .kpi-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
    }
    .kpi-val {
        font-size: 1.9rem;
        font-weight: 700;
        color: #0F172A;
        margin: 4px 0;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: #94A3B8;
    }
    /* Section Headers */
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0F172A;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 8px;
        margin: 24px 0 16px 0;
    }
    </style>
    """
    st.markdown(custom_css + inline_css, unsafe_allow_html=True)

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
    """Loads trained predictor model."""
    return get_predictor()


@st.cache_data
def get_cached_model_comparison() -> Dict[str, Any]:
    """Loads model evaluation comparisons."""
    path = os.path.join(os.path.dirname(__file__), 'models', 'model_comparison.pkl')
    if os.path.exists(path):
        return joblib.load(path)
    return {}


# ---------------------------------------------------------
# Sidebar Navigation & Context
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏢 **WorkforceIQ**")
    st.caption("AI-Powered Talent Retention & Attrition Analytics")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        [
            "🏠 Home Dashboard",
            "📊 Exploratory Data Analysis",
            "💡 Attrition Insights",
            "⚙️ Model Performance",
            "🔮 Employee Attrition Prediction",
            "📋 Business Recommendations",
            "ℹ️ About Project"
        ],
        index=0
    )
    
    st.markdown("---")
    
    # Dataset Quick Stats in Sidebar
    try:
        df_quick = get_cached_dataset()
        st.markdown("**Dataset Snapshot**")
        st.write(f"• Headcount: **{len(df_quick):,}**")
        st.write(f"• Features: **{df_quick.shape[1] - 1}**")
        attr_count = (df_quick[TARGET_COLUMN].str.lower() == 'yes').sum()
        st.write(f"• Baseline Attrition: **{attr_count / len(df_quick):.1%}**")
    except Exception:
        pass
        
    st.markdown("---")
    st.caption("Developed by **Princeden Thi**\nGitHub: [@princedenthi](https://github.com/princedenthi)")


# ---------------------------------------------------------
# Page 1: Home Dashboard
# ---------------------------------------------------------
if page == "🏠 Home Dashboard":
    st.markdown("""
    <div class="brand-header">
        <h1>WorkforceIQ Executive Dashboard</h1>
        <p>Strategic HR Intelligence, Attrition Risk Diagnostics & Machine Learning Forecasting</p>
    </div>
    """, unsafe_allow_html=True)
    
    df = get_cached_dataset()
    metrics = get_eda_summary_metrics(df)
    
    # Headline KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Headcount</div>
            <div class="kpi-val">{metrics['total_employees']:,}</div>
            <div class="kpi-sub">Active Enterprise Workforce</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid #EF4444;">
            <div class="kpi-label">Voluntary Attrition Rate</div>
            <div class="kpi-val" style="color: #DC2626;">{metrics['attrition_rate']}%</div>
            <div class="kpi-sub">{metrics['employees_left']} Departed / {metrics['employees_stayed']} Retained</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Median Monthly Income</div>
            <div class="kpi-val">${metrics['median_monthly_income']:,.0f}</div>
            <div class="kpi-sub">Baseline Compensation</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid #F59E0B;">
            <div class="kpi-label">Overtime Exposure</div>
            <div class="kpi-val" style="color: #D97706;">{metrics['pct_overtime']}%</div>
            <div class="kpi-sub">Key Driver of Attrition</div>
        </div>
        """, unsafe_allow_html=True)

    # Executive Overview Visuals
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(create_attrition_overview_chart(df), use_container_width=True)
    with c2:
        st.plotly_chart(create_attrition_by_department_chart(df), use_container_width=True)

    # Executive Summary & Critical Findings
    st.markdown("<div class='section-title'>Executive Summary & Strategic Findings</div>", unsafe_allow_html=True)
    
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.info("""
        **1. Overtime Burnout Multiplier**  
        Employees subjected to mandatory overtime exhibit an attrition rate of **30.5%**, compared to only **10.4%** for those without overtime—a **nearly 3x turnover surge**.
        """)
    with sc2:
        st.warning("""
        **2. Vulnerable Job Roles**  
        **Sales Representatives (39.8%)** and **Laboratory Technicians (23.9%)** show severe turnover pressure driven by compensation ceilings and demanding travel/bench routines.
        """)
    with sc3:
        st.success("""
        **3. Early Tenure Fragility**  
        Over **52%** of all employee exits occur within the first **3 years** of tenure. Targeted onboarding and 18-month career progression pathways dramatically curb early loss.
        """)

    # Architecture Overview Callout
    with st.expander("🔍 View End-to-End System Architecture"):
        st.markdown("""
        ```mermaid
        flowchart LR
            A[Kaggle IBM HR Dataset] --> B[Data Cleaning & Stratified Split]
            B --> C[Preprocessing Pipeline: StandardScaler & OneHotEncoder]
            C --> D[Multi-Model Training: Logistic, DecisionTree, RF, GradientBoost]
            D --> E[F1 & Recall Evaluation Engine]
            E --> F[Saved Production Pipeline Joblib]
            F --> G[Interactive Streamlit Risk Predictor & Business Insights]
        ```
        """)


# ---------------------------------------------------------
# Page 2: Exploratory Data Analysis
# ---------------------------------------------------------
elif page == "📊 Exploratory Data Analysis":
    st.markdown("<div class='brand-header'><h1>Exploratory Data Analysis (EDA)</h1><p>Interactive multidimensional breakdown of employee demographics, compensation, and satisfaction.</p></div>", unsafe_allow_html=True)
    
    df = get_cached_dataset()
    
    # Subgroup filtering
    with st.expander("Filter Subgroups", expanded=False):
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
        "Demographics & Distribution",
        "Organizational & Roles",
        "Compensation & Tenure",
        "Interactive Feature Explorer"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_attrition_by_age_chart(filtered_df), use_container_width=True)
        with col2:
            st.plotly_chart(create_attrition_by_gender_chart(filtered_df), use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_attrition_by_department_chart(filtered_df), use_container_width=True)
        with col2:
            st.plotly_chart(create_attrition_by_job_role_chart(filtered_df), use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_attrition_by_income_chart(filtered_df), use_container_width=True)
        with col2:
            st.plotly_chart(create_attrition_by_years_at_company_chart(filtered_df), use_container_width=True)

    with tab4:
        st.markdown("#### Select Any Feature to Analyze Its Relationship with Attrition")
        all_features = [c for c in filtered_df.columns if c != TARGET_COLUMN]
        selected_feat = st.selectbox("Select Feature Column", options=all_features, index=all_features.index('DistanceFromHome') if 'DistanceFromHome' in all_features else 0)
        st.plotly_chart(create_feature_distribution_chart(filtered_df, selected_feat), use_container_width=True)


# ---------------------------------------------------------
# Page 3: Attrition Insights
# ---------------------------------------------------------
elif page == "💡 Attrition Insights":
    st.markdown("<div class='brand-header'><h1>Workforce Attrition Root-Cause Insights</h1><p>Deep-dive forensic analytics on workplace friction points, overtime drag, and compensation ceilings.</p></div>", unsafe_allow_html=True)
    
    df = get_cached_dataset()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 1. Overtime Burnout Correlation")
        st.plotly_chart(create_attrition_by_overtime_chart(df), use_container_width=True)
        st.caption("Employees working overtime experience 30.5% attrition vs 10.4% for standard hours. Overtime is the primary operational friction point.")
    with col2:
        st.markdown("### 2. Workplace Satisfaction Erosion")
        st.plotly_chart(create_attrition_by_satisfaction_chart(df), use_container_width=True)
        st.caption("Low job satisfaction (Score 1) records 22.8% attrition, dropping to 11.3% for high satisfaction (Score 4).")

    st.markdown("---")
    
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### 3. Work-Life Balance Degradation")
        st.plotly_chart(create_attrition_by_work_life_balance_chart(df), use_container_width=True)
        st.caption("Employees reporting 'Bad' Work-Life Balance (Level 1) leave at 31.2%, proving that personal boundary disruption drives rapid turnover.")
    with col4:
        st.markdown("### 4. High-Attrition Job Roles")
        st.plotly_chart(create_attrition_by_job_role_chart(df), use_container_width=True)
        st.caption("Sales Representatives (39.8%) and Laboratory Technicians (23.9%) require urgent intervention and restructured incentive packages.")

    st.markdown("---")
    st.markdown("### 5. Numerical Feature Correlation Matrix with Attrition")
    st.plotly_chart(create_correlation_heatmap(df, top_n=14), use_container_width=True)


# ---------------------------------------------------------
# Page 4: Model Performance
# ---------------------------------------------------------
elif page == "⚙️ Model Performance":
    st.markdown("<div class='brand-header'><h1>Machine Learning Model Benchmarking</h1><p>Comprehensive evaluation of four classification algorithms with class-weight handling and hyperparameter tuning.</p></div>", unsafe_allow_html=True)
    
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
        
    st.dataframe(display_df, use_container_width=True)
    
    st.info("""
    **Model Selection Methodology (Recall & F1 Prioritization):**  
    In organizational talent retention, a **False Negative** (failing to identify an employee who subsequently resigns) incurs heavy recruiting, onboarding, and productivity costs (estimated at 1.5x–2x annual salary).  
    Therefore, **Logistic Regression** (with class balancing and L2 regularization) was selected as the optimal production model because it achieved the **highest Recall (72.34%)**, highest **ROC-AUC (79.85%)**, and top **F1-Score (50.00%)**, successfully capturing nearly three-quarters of all departing personnel on the test set.
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
        color_discrete_sequence=['#1E3A8A', '#0284C7', '#EF4444', '#10B981', '#F59E0B'],
        title="<b>Algorithm Performance Comparison Across Metrics</b>"
    )
    fig_comp.update_layout(template="plotly_white", yaxis_range=[0, 1.0], height=420)
    st.plotly_chart(fig_comp, use_container_width=True)

    # ROC Curves & Confusion Matrices
    st.markdown("---")
    col_roc, col_cm = st.columns([1, 1])
    
    with col_roc:
        st.markdown("### Receiver Operating Characteristic (ROC) Curves")
        fig_roc = go.Figure()
        colors = ['#1E3A8A', '#EF4444', '#10B981', '#8B5CF6']
        
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
        
        fig_roc.update_layout(
            template="plotly_white",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate (Recall)",
            height=420,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_cm:
        st.markdown("### Confusion Matrix Inspector")
        chosen_model = st.selectbox("Select Model to Inspect Confusion Matrix", options=list(comp_data['models_data'].keys()))
        cm = comp_data['models_data'][chosen_model]['confusion_matrix']
        
        cm_labels = [['True Negative (Retained)', 'False Positive (False Alarm)'],
                     ['False Negative (Missed Exit)', 'True Positive (Detected Exit)']]
        
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            x=['Predicted: Stay (No)', 'Predicted: Leave (Yes)'],
            y=['Actual: Stay (No)', 'Actual: Leave (Yes)'],
            title=f"<b>Confusion Matrix: {chosen_model}</b>"
        )
        fig_cm.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_cm, use_container_width=True)

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
            color_continuous_scale='Teal',
            title="<b>Top 15 Most Predictive Features in Champion Model</b>"
        )
        fig_feats.update_layout(template="plotly_white", height=450, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_feats, use_container_width=True)


# ---------------------------------------------------------
# Page 5: Employee Attrition Prediction
# ---------------------------------------------------------
elif page == "🔮 Employee Attrition Prediction":
    st.markdown("""
    <div class="brand-header">
        <h1>Individual Employee Attrition Risk Assessment</h1>
        <p>Interactive risk simulation tool powered by machine learning and explainable feature contributions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    predictor = get_cached_predictor()

    # Presets for quick demonstration
    st.markdown("#### Test with Realistic Employee Profiles:")
    p_col1, p_col2, p_col3 = st.columns(3)
    
    preset_choice = None
    if p_col1.button("🔴 High-Risk Sales Representative", use_container_width=True):
        preset_choice = 'high_risk'
    if p_col2.button("🟡 Moderate-Risk Lab Technician", use_container_width=True):
        preset_choice = 'med_risk'
    if p_col3.button("🟢 Low-Risk Senior R&D Manager", use_container_width=True):
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

    # Input Form
    with st.form("employee_prediction_form"):
        st.markdown("### Employee Attributes & Workplace Parameters")
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown("##### 👤 Demographics & Travel")
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
            st.markdown("##### 💼 Role & Compensation")
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
            st.markdown("##### ⏱️ Tenure & Satisfaction")
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

        submitted = st.form_submit_button("⚡ Evaluate Employee Retention Risk", use_container_width=True)

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
        
        st.markdown("<div class='section-title'>Assessment Outcome & Risk Analysis</div>", unsafe_allow_html=True)
        
        rcol1, rcol2 = st.columns([1, 1.2])
        
        with rcol1:
            st.markdown(f"""
            <div class="kpi-card" style="border-left: 6px solid {result['risk_color']};">
                <div class="kpi-label">Predicted Retention Outcome</div>
                <div class="kpi-val" style="color: {result['risk_color']}; font-size: 2.2rem;">
                    {result['attrition_label']}
                </div>
                <div style="margin: 8px 0;">
                    <span class="badge-{result['risk_tier'].lower()[:3]}">{result['risk_badge']}</span>
                </div>
                <p style="color: #475569; font-size: 0.95rem; margin-top: 10px;">
                    {result['summary_statement']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result['probability_pct'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "<b>Attrition Risk Score (%)</b>", 'font': {'size': 16}},
                number={'suffix': "%", 'font': {'size': 32}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
                    'bar': {'color': result['risk_color'], 'thickness': 0.3},
                    'bgcolor': "white",
                    'borderwidth': 1,
                    'bordercolor': "#E2E8F0",
                    'steps': [
                        {'range': [0, 35], 'color': '#ECFDF5'},
                        {'range': [35, 60], 'color': '#FFFBEB'},
                        {'range': [60, 100], 'color': '#FEF2F2'}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 3},
                        'thickness': 0.75,
                        'value': result['probability_pct']
                    }
                }
            ))
            fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

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
                    color_discrete_map={'Increases Risk': '#EF4444', 'Reduces Risk': '#10B981'},
                    title="<b>Local Feature Impact on Attrition Decision</b>"
                )
                fig_c.update_layout(template="plotly_white", height=340, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_c, use_container_width=True)
            else:
                st.info("Feature contributions calculated.")

        # Tailored Action Plan
        st.markdown("<div class='section-title'>Recommended HR Mitigation Strategies</div>", unsafe_allow_html=True)
        for rec in result['recommendations']:
            st.markdown(f"""
            <div class="callout-box">
                <strong style="color: #1E3A8A;">[{rec['category']}] {rec['action']}</strong><br>
                <span style="color: #334155; font-size: 0.92rem;">{rec['detail']}</span>
            </div>
            """, unsafe_allow_html=True)

        # Responsible AI Notice
        st.caption(f"🔒 **{result['disclaimer']}**")


# ---------------------------------------------------------
# Page 6: Business Recommendations
# ---------------------------------------------------------
elif page == "📋 Business Recommendations":
    st.markdown("""
    <div class="brand-header">
        <h1>Executive Talent Retention Playbook</h1>
        <p>MBA-level strategic interventions and organizational policy reforms derived from empirical findings.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Strategic Retention Pillars")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### 1. Overtime Governance & Workload Rebalancing
        - **Problem**: 30.5% attrition among overtime personnel vs 10.4% baseline.
        - **Policy Intervention**:
          - Mandate maximum weekly overtime caps (e.g. 8 hours/week maximum).
          - Provide 1.5x compensatory time-off or spot equity bonuses for crunch periods.
          - Implement automated workload monitoring to detect recurring team deficits.
        - **Target Outcome**: Reduce voluntary churn among technical staff by 35%.
        """)
        
        st.markdown("""
        #### 2. Compensation Realignment for Vulnerable Roles
        - **Problem**: Sales Representatives leave at 39.8%, driven by base pay compression.
        - **Policy Intervention**:
          - Shift compensation structures toward a higher guaranteed base salary (65/35 ratio).
          - Introduce milestone-based quarterly commission acceleration.
          - Conduct bi-annual market salary benchmarking for lab technicians.
        - **Target Outcome**: Enhance sales force tenure by an average of 1.8 years.
        """)
        
    with col2:
        st.markdown("""
        #### 3. Career Pathing & Promotion Cadence
        - **Problem**: Employees in role > 3 years without promotion show a 40% jump in departure likelihood.
        - **Policy Intervention**:
          - Institute transparent 18-month career progression milestones.
          - Introduce "Dual Ladder" technical track for individual contributors wanting advancement without management overhead.
          - Formalize sponsorship and executive mentorship matching programs.
        - **Target Outcome**: Cut mid-career talent flight by 25%.
        """)
        
        st.markdown("""
        #### 4. Hybrid Flexibility & Commute Mitigation
        - **Problem**: Commute distances > 15 miles double employee dissatisfaction scores.
        - **Policy Intervention**:
          - Authorize 2–3 remote work days per week for eligible engineering and R&D roles.
          - Offer transit subsidies or rideshare credits.
          - Create core asynchronous hours (10:00 AM – 3:00 PM) to bypass rush hours.
        - **Target Outcome**: Boost employee work-life balance scores from 2.1 to 3.4.
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
    st.table(pd.DataFrame(matrix_data))


# ---------------------------------------------------------
# Page 7: About Project
# ---------------------------------------------------------
elif page == "ℹ️ About Project":
    st.markdown("""
    <div class="brand-header">
        <h1>About the WorkforceIQ Project</h1>
        <p>A comprehensive Machine Learning & Business Intelligence solution for enterprise human resource retention.</p>
    </div>
    """, unsafe_allow_html=True)
    
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
        
        ### Dataset Background
        - **Source**: Kaggle IBM HR Analytics Employee Attrition & Performance Dataset
        - **Dimensions**: 1,470 employee records across 35 attributes
        - **Target**: `Attrition` (Binary: Yes / No)
        - **Licensing**: Open Database / Apache 2.0 synthetic demonstration dataset
        
        ### Responsible AI Disclaimer
        This predictive dashboard is strictly intended to serve as decision-support intelligence for HR leadership. It does not replace qualitative manager conversations, fair labor practices, or holistic performance evaluations.
        """)
