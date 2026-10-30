"""
eda.py
------
Exploratory Data Analysis (EDA) module using Plotly for interactive visualizations.
Generates comprehensive visual analytics on demographics, compensation, job roles,
workplace satisfaction, and attrition drivers.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_preprocessing import load_data, clean_data, TARGET_COLUMN


# Corporate Color Palette
COLORS = {
    'primary': '#1E3A8A',       # Deep Navy
    'secondary': '#0284C7',     # Bright Cyan/Sky
    'accent_red': '#EF4444',    # Crimson Red (Attrition = Yes)
    'accent_green': '#10B981',  # Emerald Green (Attrition = No)
    'neutral_light': '#F8FAFC', # Slate Light
    'neutral_dark': '#1E293B',  # Slate Dark
    'amber': '#F59E0B',         # Warning Amber
    'purple': '#8B5CF6'         # Royal Purple
}

ATTRITION_COLORS = {'Yes': COLORS['accent_red'], 'No': COLORS['accent_green']}


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
        'attrition_rate': round(rate, 2),
        'avg_age': round(df['Age'].mean(), 1) if 'Age' in df else 0,
        'median_monthly_income': round(df['MonthlyIncome'].median(), 0) if 'MonthlyIncome' in df else 0,
        'avg_years_at_company': round(df['YearsAtCompany'].mean(), 1) if 'YearsAtCompany' in df else 0,
        'pct_overtime': round((df['OverTime'] == 'Yes').mean() * 100, 1) if 'OverTime' in df else 0
    }


def create_attrition_overview_chart(df: pd.DataFrame) -> go.Figure:
    """Donut chart showing overall headcount split by Attrition status."""
    counts = df[TARGET_COLUMN].value_counts()
    labels = counts.index.tolist()
    values = counts.values.tolist()
    
    colors = [ATTRITION_COLORS.get(label, '#94A3B8') for label in labels]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)),
        textinfo='label+percent+value',
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>"
    )])
    
    fig.update_layout(
        title="<b>Workforce Attrition Overview</b>",
        font=dict(family="Inter, sans-serif"),
        template="plotly_white",
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    return fig


def create_attrition_by_age_chart(df: pd.DataFrame) -> go.Figure:
    """Age distribution split by Attrition."""
    fig = px.histogram(
        df,
        x="Age",
        color=TARGET_COLUMN,
        barmode="overlay",
        nbins=25,
        color_discrete_map=ATTRITION_COLORS,
        opacity=0.75,
        labels={TARGET_COLUMN: "Attrition", "Age": "Employee Age (Years)"},
        title="<b>Age Distribution by Attrition Status</b>"
    )
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, sans-serif"),
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
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
        text=agg.apply(lambda r: f"{r['Count']} ({r['Rate']}%)", axis=1),
        title="<b>Attrition Headcount & Proportions by Gender</b>"
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, sans-serif"),
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_attrition_by_department_chart(df: pd.DataFrame) -> go.Figure:
    """Department-wise attrition count and percentage."""
    dept_agg = df.groupby(['Department', TARGET_COLUMN]).size().unstack(fill_value=0)
    dept_agg['Total'] = dept_agg['No'] + dept_agg['Yes']
    dept_agg['Attrition_Rate_%'] = (dept_agg['Yes'] / dept_agg['Total'] * 100).round(1)
    dept_agg = dept_agg.reset_index()

    fig = px.bar(
        dept_agg,
        x="Department",
        y="Attrition_Rate_%",
        color="Attrition_Rate_%",
        color_continuous_scale="Reds",
        text=dept_agg["Attrition_Rate_%"].apply(lambda v: f"{v}%"),
        title="<b>Attrition Rate (%) by Department</b>"
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, sans-serif"),
        yaxis_title="Attrition Rate (%)",
        height=380,
        margin=dict(l=20, r=20, t=50, b=20)
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
        color_continuous_scale="OrRd",
        text=role_agg.apply(lambda r: f"{r['Rate']}% ({r.get('Yes', 0)}/{r['Total']})", axis=1),
        title="<b>Attrition Rate (%) Ranked by Job Role</b>"
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, sans-serif"),
        xaxis_title="Attrition Rate (%)",
        yaxis_title="",
        height=450,
        margin=dict(l=20, r=40, t=50, b=20)
    )
    return fig


def create_attrition_by_income_chart(df: pd.DataFrame) -> go.Figure:
    """Box plot of Monthly Income comparison."""
    fig = px.box(
        df,
        x=TARGET_COLUMN,
        y="MonthlyIncome",
        color=TARGET_COLUMN,
        color_discrete_map=ATTRITION_COLORS,
        points="outliers",
        title="<b>Monthly Income Distribution by Attrition Status</b>"
    )
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, sans-serif"),
        yaxis_title="Monthly Income ($)",
        xaxis_title="Attrition",
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
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
        color_discrete_map={'Yes': COLORS['accent_red'], 'No': COLORS['secondary']},
        text=ot_agg['Rate'].apply(lambda v: f"{v}% Attrition"),
        title="<b>Impact of Overtime on Employee Turnover</b>"
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, sans-serif"),
        yaxis_title="Attrition Rate (%)",
        xaxis_title="Overtime Mandatory / Active",
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False
    )
    return fig


def create_attrition_by_satisfaction_chart(df: pd.DataFrame) -> go.Figure:
    """Attrition rate across Job Satisfaction levels (1: Low to 4: Very High)."""
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
        color_continuous_scale="Blues_r",
        text=sat_agg['Rate'].apply(lambda v: f"{v}%"),
        title="<b>Attrition Rate by Job Satisfaction Score</b>"
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, sans-serif"),
        yaxis_title="Attrition Rate (%)",
        xaxis_title="Job Satisfaction Score",
        height=380,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def create_attrition_by_work_life_balance_chart(df: pd.DataFrame) -> go.Figure:
    """Attrition rate across Work-Life Balance levels (1: Bad to 4: Best)."""
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
        color_continuous_scale="Purples_r",
        text=wlb_agg['Rate'].apply(lambda v: f"{v}%"),
        title="<b>Attrition Rate by Work-Life Balance Rating</b>"
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, sans-serif"),
        yaxis_title="Attrition Rate (%)",
        xaxis_title="Work-Life Balance Score",
        height=380,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def create_attrition_by_years_at_company_chart(df: pd.DataFrame) -> go.Figure:
    """Attrition count by Years At Company (Tenure)."""
    fig = px.histogram(
        df,
        x="YearsAtCompany",
        color=TARGET_COLUMN,
        barmode="group",
        nbins=20,
        color_discrete_map=ATTRITION_COLORS,
        title="<b>Attrition Distribution across Tenure (Years at Company)</b>"
    )
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, sans-serif"),
        xaxis_title="Years at Company",
        yaxis_title="Employee Count",
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_correlation_heatmap(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Interactive correlation heatmap for key numerical variables with Attrition."""
    df_clean = clean_data(df)
    df_corr = df_clean.copy()
    df_corr['Attrition_Numeric'] = df_corr[TARGET_COLUMN].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
    
    num_cols = df_corr.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Sort features by absolute correlation with Attrition
    corrs_with_target = df_corr[num_cols].corr()['Attrition_Numeric'].abs().sort_values(ascending=False)
    selected_cols = corrs_with_target.head(top_n).index.tolist()
    
    corr_matrix = df_corr[selected_cols].corr().round(2)
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title=f"<b>Correlation Heatmap (Top {top_n} Factors Correlated with Attrition)</b>"
    )
    fig.update_layout(
        font=dict(family="Inter, sans-serif"),
        height=520,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def create_feature_distribution_chart(df: pd.DataFrame, feature_col: str) -> go.Figure:
    """Distribution chart for any chosen numeric or categorical feature."""
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
            opacity=0.75,
            title=f"<b>Distribution of {feature_col} by Attrition</b>"
        )
    else:
        fig = px.histogram(
            df,
            x=feature_col,
            color=TARGET_COLUMN,
            barmode="group",
            color_discrete_map=ATTRITION_COLORS,
            title=f"<b>Distribution of {feature_col} by Attrition</b>"
        )
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, sans-serif"),
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig
