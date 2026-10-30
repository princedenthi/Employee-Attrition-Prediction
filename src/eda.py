"""
eda.py
------
Exploratory Data Analysis (EDA) module using Plotly for enterprise-grade visualizations.
Generates refined, Tableau/Power BI inspired visual analytics on demographics,
compensation, job roles, workplace satisfaction, and attrition drivers.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data_preprocessing import load_data, clean_data, TARGET_COLUMN

# Enterprise SaaS Color Palette (Linear / Stripe / Tableau Inspired)
COLORS = {
    'primary_blue': '#2563EB',    # Royal Blue
    'indigo': '#4F46E5',          # Deep Indigo
    'sky': '#0284C7',             # Clean Sky Blue
    'emerald': '#059669',         # Enterprise Emerald (Stayed)
    'rose': '#EF4444',            # Clean Coral Red (Attrition)
    'slate_dark': '#0F172A',      # Typography Primary
    'slate_muted': '#64748B',     # Typography Secondary
    'border': '#E2E8F0',          # Subtle Card Borders
    'grid': '#F1F5F9',            # Minimal Plotly Grid
    'card_bg': '#FFFFFF'          # Clean Card Canvas
}

ATTRITION_COLORS = {'Yes': COLORS['rose'], 'No': COLORS['primary_blue']}
ATTRITION_STATUS_COLORS = {'Stayed (No)': COLORS['primary_blue'], 'Departed (Yes)': COLORS['rose']}


def apply_enterprise_theme(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    """Applies a clean, modern Tableau/SaaS aesthetic to any Plotly figure."""
    fig.update_layout(
        title={
            'text': f"<b>{title}</b>" if title else "",
            'font': {'family': 'Inter, -apple-system, sans-serif', 'size': 14, 'color': COLORS['slate_dark']},
            'x': 0.02,
            'xanchor': 'left'
        },
        font=dict(family="Inter, -apple-system, sans-serif", size=11, color=COLORS['slate_muted']),
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(255,255,255,1)",
        height=height,
        hoverlabel=dict(
            bgcolor=COLORS['slate_dark'],
            font_size=12,
            font_family="Inter, -apple-system, sans-serif",
            font_color="#FFFFFF",
            bordercolor=COLORS['slate_dark']
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            linecolor=COLORS['border'],
            tickfont=dict(size=11, color=COLORS['slate_muted'])
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=COLORS['grid'],
            zeroline=False,
            linecolor=COLORS['border'],
            tickfont=dict(size=11, color=COLORS['slate_muted'])
        ),
        margin=dict(l=24, r=24, t=48, b=24),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=COLORS['slate_muted'])
        )
    )
    return fig


def get_eda_summary_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes headline workforce metrics."""
    total = len(df)
    left = int((df[TARGET_COLUMN].str.strip().str.lower() == 'yes').sum())
    stayed = total - left
    rate = (left / total) * 100 if total > 0 else 0
    
    return {
        'total_employees': total,
        'employees_left': left,
        'employees_stayed': stayed,
        'attrition_rate': round(rate, 1),
        'retention_rate': round(100.0 - rate, 1),
        'avg_age': round(df['Age'].mean(), 1) if 'Age' in df else 0,
        'avg_monthly_income': int(round(df['MonthlyIncome'].mean())) if 'MonthlyIncome' in df else 0,
        'median_monthly_income': int(round(df['MonthlyIncome'].median())) if 'MonthlyIncome' in df else 0,
        'avg_years_at_company': round(df['YearsAtCompany'].mean(), 1) if 'YearsAtCompany' in df else 0,
        'pct_overtime': round((df['OverTime'] == 'Yes').mean() * 100, 1) if 'OverTime' in df else 0
    }


def create_attrition_overview_chart(df: pd.DataFrame) -> go.Figure:
    """Enterprise donut chart showing overall headcount split with central KPI."""
    counts = df[TARGET_COLUMN].value_counts()
    labels = counts.index.tolist()
    values = counts.values.tolist()
    
    colors = [ATTRITION_COLORS.get(label, '#94A3B8') for label in labels]
    attr_rate = (counts.get('Yes', 0) / len(df) * 100) if len(df) > 0 else 0
    
    fig = go.Figure(data=[go.Pie(
        labels=['Departed (Yes)' if l == 'Yes' else 'Retained (No)' for l in labels],
        values=values,
        hole=0.68,
        marker=dict(colors=colors, line=dict(color='#FFFFFF', width=3)),
        textinfo='percent',
        textposition='outside',
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>"
    )])
    
    fig.add_annotation(
        text=f"<span style='font-size:24px;font-weight:700;color:{COLORS['slate_dark']}'>{attr_rate:.1f}%</span><br><span style='font-size:11px;color:{COLORS['slate_muted']};font-weight:600;'>ATTRITION RATE</span>",
        x=0.5, y=0.5, showarrow=False, align="center"
    )
    
    apply_enterprise_theme(fig, "Workforce Headcount & Retention Overview", height=360)
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5)
    )
    return fig


def create_attrition_by_age_chart(df: pd.DataFrame) -> go.Figure:
    """Age distribution split by Attrition with smooth overlay."""
    fig = px.histogram(
        df,
        x="Age",
        color=TARGET_COLUMN,
        barmode="overlay",
        nbins=24,
        color_discrete_map=ATTRITION_COLORS,
        opacity=0.72,
        labels={TARGET_COLUMN: "Status", "Age": "Employee Age (Years)"}
    )
    apply_enterprise_theme(fig, "Age Distribution by Attrition Status", height=360)
    fig.update_layout(xaxis_title="Age (Years)", yaxis_title="Employee Count")
    return fig


def create_attrition_by_gender_chart(df: pd.DataFrame) -> go.Figure:
    """Attrition rate and counts by Gender."""
    agg = df.groupby(['Gender', TARGET_COLUMN]).size().reset_index(name='Count')
    total_by_gender = df.groupby('Gender').size().reset_index(name='Total')
    agg = agg.merge(total_by_gender, on='Gender')
    agg['Rate'] = (agg['Count'] / agg['Total'] * 100).round(1)
    
    fig = px.bar(
        agg,
        x="Gender",
        y="Count",
        color=TARGET_COLUMN,
        barmode="group",
        color_discrete_map=ATTRITION_COLORS,
        text=agg.apply(lambda r: f"{r['Count']} ({r['Rate']}%)", axis=1)
    )
    fig.update_traces(textposition='outside', marker_line_width=0)
    apply_enterprise_theme(fig, "Attrition Headcount by Gender", height=360)
    fig.update_layout(yaxis_title="Employee Count", xaxis_title="")
    return fig


def create_attrition_by_department_chart(df: pd.DataFrame) -> go.Figure:
    """Department-wise attrition rate comparison."""
    dept_agg = df.groupby(['Department', TARGET_COLUMN]).size().unstack(fill_value=0)
    dept_agg['Total'] = dept_agg.get('No', 0) + dept_agg.get('Yes', 0)
    dept_agg['Attrition_Rate_%'] = (dept_agg.get('Yes', 0) / dept_agg['Total'] * 100).round(1)
    dept_agg = dept_agg.reset_index().sort_values(by='Attrition_Rate_%', ascending=False)

    fig = px.bar(
        dept_agg,
        x="Department",
        y="Attrition_Rate_%",
        color="Attrition_Rate_%",
        color_continuous_scale=[[0, '#93C5FD'], [0.5, '#3B82F6'], [1, '#EF4444']],
        text=dept_agg["Attrition_Rate_%"].apply(lambda v: f"{v}%")
    )
    fig.update_traces(textposition='outside', marker_line_width=0)
    apply_enterprise_theme(fig, "Attrition Rate (%) by Department", height=360)
    fig.update_layout(
        yaxis_title="Attrition Rate (%)",
        xaxis_title="",
        coloraxis_showscale=False
    )
    return fig


def create_attrition_by_job_role_chart(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart ranking Job Roles by Attrition Rate."""
    role_agg = df.groupby(['JobRole', TARGET_COLUMN]).size().unstack(fill_value=0)
    role_agg['Total'] = role_agg.get('No', 0) + role_agg.get('Yes', 0)
    role_agg['Rate'] = (role_agg.get('Yes', 0) / role_agg['Total'] * 100).round(1)
    role_agg = role_agg.sort_values(by='Rate', ascending=True).reset_index()

    fig = px.bar(
        role_agg,
        x="Rate",
        y="JobRole",
        orientation="h",
        color="Rate",
        color_continuous_scale=[[0, '#93C5FD'], [0.4, '#3B82F6'], [1, '#EF4444']],
        text=role_agg.apply(lambda r: f"{r['Rate']}%  ({r.get('Yes', 0)}/{r['Total']})", axis=1)
    )
    fig.update_traces(textposition='outside', marker_line_width=0)
    apply_enterprise_theme(fig, "Attrition Risk Ranking by Job Role", height=420)
    fig.update_layout(
        xaxis_title="Attrition Rate (%)",
        yaxis_title="",
        coloraxis_showscale=False,
        margin=dict(l=20, r=45, t=48, b=24)
    )
    return fig


def create_attrition_by_income_chart(df: pd.DataFrame) -> go.Figure:
    """Clean box plot of Monthly Income comparison."""
    fig = px.box(
        df,
        x=TARGET_COLUMN,
        y="MonthlyIncome",
        color=TARGET_COLUMN,
        color_discrete_map=ATTRITION_COLORS,
        points="outliers"
    )
    apply_enterprise_theme(fig, "Monthly Income Distribution by Attrition", height=360)
    fig.update_layout(
        yaxis_title="Monthly Income ($)",
        xaxis_title="Attrition Status",
        showlegend=False
    )
    return fig


def create_attrition_by_overtime_chart(df: pd.DataFrame) -> go.Figure:
    """Overtime vs Attrition comparison."""
    ot_agg = df.groupby(['OverTime', TARGET_COLUMN]).size().unstack(fill_value=0)
    ot_agg['Total'] = ot_agg['No'] + ot_agg['Yes']
    ot_agg['Rate'] = (ot_agg['Yes'] / ot_agg['Total'] * 100).round(1)
    ot_agg = ot_agg.reset_index()

    fig = px.bar(
        ot_agg,
        x="OverTime",
        y="Rate",
        color="OverTime",
        color_discrete_map={'Yes': COLORS['rose'], 'No': COLORS['primary_blue']},
        text=ot_agg['Rate'].apply(lambda v: f"{v}% Attrition")
    )
    fig.update_traces(textposition='outside', marker_line_width=0)
    apply_enterprise_theme(fig, "Impact of Overtime on Turnover Rate", height=360)
    fig.update_layout(
        yaxis_title="Attrition Rate (%)",
        xaxis_title="Mandatory / Regular Overtime",
        showlegend=False
    )
    return fig


def create_attrition_by_satisfaction_chart(df: pd.DataFrame) -> go.Figure:
    """Attrition rate across Job Satisfaction levels."""
    sat_labels = {1: '1 - Low', 2: '2 - Medium', 3: '3 - High', 4: '4 - Very High'}
    df_copy = df.copy()
    df_copy['Satisfaction_Label'] = df_copy['JobSatisfaction'].map(sat_labels)
    
    sat_agg = df_copy.groupby(['Satisfaction_Label', TARGET_COLUMN]).size().unstack(fill_value=0)
    sat_agg['Total'] = sat_agg.get('No', 0) + sat_agg.get('Yes', 0)
    sat_agg['Rate'] = (sat_agg.get('Yes', 0) / sat_agg['Total'] * 100).round(1)
    sat_agg = sat_agg.reindex([sat_labels[i] for i in range(1, 5)]).reset_index()

    fig = px.bar(
        sat_agg,
        x="Satisfaction_Label",
        y="Rate",
        color="Rate",
        color_continuous_scale=[[0, '#93C5FD'], [0.5, '#3B82F6'], [1, '#EF4444']],
        text=sat_agg['Rate'].apply(lambda v: f"{v}%")
    )
    fig.update_traces(textposition='outside', marker_line_width=0)
    apply_enterprise_theme(fig, "Attrition Rate by Job Satisfaction Score", height=360)
    fig.update_layout(
        yaxis_title="Attrition Rate (%)",
        xaxis_title="Job Satisfaction Rating",
        coloraxis_showscale=False
    )
    return fig


def create_attrition_by_work_life_balance_chart(df: pd.DataFrame) -> go.Figure:
    """Attrition rate across Work-Life Balance levels."""
    wlb_labels = {1: '1 - Bad', 2: '2 - Good', 3: '3 - Better', 4: '4 - Best'}
    df_copy = df.copy()
    df_copy['WLB_Label'] = df_copy['WorkLifeBalance'].map(wlb_labels)
    
    wlb_agg = df_copy.groupby(['WLB_Label', TARGET_COLUMN]).size().unstack(fill_value=0)
    wlb_agg['Total'] = wlb_agg.get('No', 0) + wlb_agg.get('Yes', 0)
    wlb_agg['Rate'] = (wlb_agg.get('Yes', 0) / wlb_agg['Total'] * 100).round(1)
    wlb_agg = wlb_agg.reindex([wlb_labels[i] for i in range(1, 5)]).reset_index()

    fig = px.bar(
        wlb_agg,
        x="WLB_Label",
        y="Rate",
        color="Rate",
        color_continuous_scale=[[0, '#A78BFA'], [0.5, '#6366F1'], [1, '#EF4444']],
        text=wlb_agg['Rate'].apply(lambda v: f"{v}%")
    )
    fig.update_traces(textposition='outside', marker_line_width=0)
    apply_enterprise_theme(fig, "Attrition Rate by Work-Life Balance Rating", height=360)
    fig.update_layout(
        yaxis_title="Attrition Rate (%)",
        xaxis_title="Work-Life Balance Rating",
        coloraxis_showscale=False
    )
    return fig


def create_attrition_by_years_at_company_chart(df: pd.DataFrame) -> go.Figure:
    """Attrition count by Years At Company (Tenure)."""
    fig = px.histogram(
        df,
        x="YearsAtCompany",
        color=TARGET_COLUMN,
        barmode="group",
        nbins=18,
        color_discrete_map=ATTRITION_COLORS
    )
    apply_enterprise_theme(fig, "Attrition Headcount across Tenure (Years at Company)", height=360)
    fig.update_layout(
        xaxis_title="Years at Company",
        yaxis_title="Employee Count"
    )
    return fig


def create_correlation_heatmap(df: pd.DataFrame, top_n: int = 14) -> go.Figure:
    """Clean correlation heatmap with minimal visual noise."""
    df_clean = clean_data(df)
    df_corr = df_clean.copy()
    df_corr['Attrition_Numeric'] = df_corr[TARGET_COLUMN].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
    
    num_cols = df_corr.select_dtypes(include=['int64', 'float64']).columns.tolist()
    corrs_with_target = df_corr[num_cols].corr()['Attrition_Numeric'].abs().sort_values(ascending=False)
    selected_cols = corrs_with_target.head(top_n).index.tolist()
    
    corr_matrix = df_corr[selected_cols].corr().round(2)
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        zmin=-1,
        zmax=1
    )
    apply_enterprise_theme(fig, f"Correlation Matrix (Top {top_n} Factors Correlated with Attrition)", height=480)
    return fig


def create_feature_distribution_chart(df: pd.DataFrame, feature_col: str) -> go.Figure:
    """Distribution chart for any selected feature."""
    if feature_col not in df.columns:
        raise ValueError(f"Feature '{feature_col}' not in dataset.")
        
    if pd.api.types.is_numeric_dtype(df[feature_col]):
        fig = px.histogram(
            df,
            x=feature_col,
            color=TARGET_COLUMN,
            marginal="box",
            barmode="overlay",
            color_discrete_map=ATTRITION_COLORS,
            opacity=0.75
        )
    else:
        fig = px.histogram(
            df,
            x=feature_col,
            color=TARGET_COLUMN,
            barmode="group",
            color_discrete_map=ATTRITION_COLORS
        )
    apply_enterprise_theme(fig, f"Feature Distribution: {feature_col} by Attrition", height=380)
    return fig
