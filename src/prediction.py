"""
prediction.py
-------------
Inference and Explainable AI (XAI) engine for Employee Attrition.
Loads the trained pipeline, predicts attrition probability, classifies risk tier,
computes local feature contributions, and generates actionable HR recommendations.
"""

import os
from typing import Dict, Any, Tuple, List
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'best_model.pkl')


class AttritionPredictor:
    """Predictor class that encapsulates the model, preprocessing, and explainability."""

    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained model not found at '{model_path}'. "
                "Please run 'src/train_model.py' to generate the model artifacts."
            )
        self.artifacts = joblib.load(model_path)
        self.model_name = self.artifacts['model_name']
        self.pipeline = self.artifacts['pipeline']
        self.preprocessor = self.pipeline.named_steps['preprocessor']
        self.classifier = self.pipeline.named_steps['classifier']
        self.feature_names = self.artifacts['feature_names']
        self.numerical_cols = self.artifacts['numerical_cols']
        self.categorical_cols = self.artifacts['categorical_cols']
        self.categorical_options = self.artifacts.get('categorical_options', {})
        self.defaults = self.artifacts['defaults']
        self.metrics = self.artifacts.get('metrics', {})

    def format_input(self, user_inputs: Dict[str, Any]) -> pd.DataFrame:
        """
        Merges provided user inputs with dataset defaults for any unspecified columns,
        ensuring schema conformity with the trained pipeline.
        """
        full_record = self.defaults.copy()
        full_record.update(user_inputs)
        df_input = pd.DataFrame([full_record])
        return df_input

    def predict(self, user_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates attrition prediction, probability, risk tier, explanations, and advice.
        """
        df_input = self.format_input(user_inputs)
        
        # Binary prediction (0 = Stay/No, 1 = Leave/Yes)
        pred_class = int(self.pipeline.predict(df_input)[0])
        
        # Risk probability
        if hasattr(self.pipeline, "predict_proba"):
            proba = float(self.pipeline.predict_proba(df_input)[0, 1])
        elif hasattr(self.pipeline, "decision_function"):
            decision = float(self.pipeline.decision_function(df_input)[0])
            proba = 1.0 / (1.0 + np.exp(-decision))
        else:
            proba = float(pred_class)
            
        # Determine Risk Category
        if proba < 0.35:
            risk_tier = "Low"
            risk_color = "#10B981"  # Emerald green
            risk_badge = "Low Risk Profile"
            summary_statement = "The employee exhibits strong retention indicators and organizational stability."
        elif proba < 0.60:
            risk_tier = "Medium"
            risk_color = "#F59E0B"  # Amber
            risk_badge = "Moderate Risk Profile"
            summary_statement = "Notable friction points detected. Proactive HR touchpoints are recommended."
        else:
            risk_tier = "High"
            risk_color = "#EF4444"  # Red
            risk_badge = "High Attrition Risk"
            summary_statement = "Critical risk indicators flagged. Prompt strategic retention intervention is strongly advised."

        # Compute local feature contributions
        contributions = self._compute_feature_contributions(df_input)
        
        # Generate targeted HR recommendations based on specific user inputs
        recommendations = self._generate_recommendations(user_inputs, proba)

        return {
            'prediction': 'Yes' if pred_class == 1 or proba >= 0.50 else 'No',
            'attrition_label': 'At Risk of Leaving' if proba >= 0.50 else 'Likely to Stay',
            'probability': proba,
            'probability_pct': round(proba * 100, 1),
            'risk_tier': risk_tier,
            'risk_color': risk_color,
            'risk_badge': risk_badge,
            'summary_statement': summary_statement,
            'top_drivers': contributions['top_positive'],  # Features increasing attrition
            'top_protectors': contributions['top_negative'],  # Features reducing attrition
            'all_contributions': contributions['all'],
            'recommendations': recommendations,
            'disclaimer': (
                "Responsible AI Notice: Predictions represent statistical probabilities based on historical patterns "
                "and should never be treated as guaranteed outcomes or used as automated employment decisions. "
                "They are designed to guide supportive HR conversations and proactive talent retention."
            )
        }

    def _compute_feature_contributions(self, df_input: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes sample-level attribution.
        For linear models, calculates (x - mean) * weight.
        For tree models, calculates feature presence * global importance.
        """
        try:
            X_trans = self.preprocessor.transform(df_input)
            
            if hasattr(self.classifier, 'coef_'):
                weights = self.classifier.coef_[0]
                # Log-odds contribution
                raw_contributions = X_trans[0] * weights
            elif hasattr(self.classifier, 'feature_importances_'):
                importances = self.classifier.feature_importances_
                # Scaled directional approximation
                raw_contributions = X_trans[0] * importances
            else:
                raw_contributions = np.zeros(len(self.feature_names))

            contrib_items = []
            for name, score in zip(self.feature_names, raw_contributions):
                if abs(score) > 1e-4:
                    clean_name = name.replace('num__', '').replace('cat__', '').replace('_', ' ')
                    contrib_items.append({
                        'feature': clean_name,
                        'impact': float(score),
                        'direction': 'Increases Risk' if score > 0 else 'Reduces Risk'
                    })

            # Sort: highest positive impact first (increases risk)
            sorted_by_impact = sorted(contrib_items, key=lambda x: x['impact'], reverse=True)
            top_positive = [item for item in sorted_by_impact if item['impact'] > 0][:5]
            top_negative = [item for item in sorted(contrib_items, key=lambda x: x['impact']) if item['impact'] < 0][:5]

            return {
                'top_positive': top_positive,
                'top_negative': top_negative,
                'all': sorted_by_impact
            }
        except Exception as e:
            return {'top_positive': [], 'top_negative': [], 'all': []}

    def _generate_recommendations(self, inputs: Dict[str, Any], proba: float) -> List[Dict[str, str]]:
        """Generates dynamic tailored HR action items based on the employee's specific profile."""
        recs = []
        
        # Overtime trigger
        if str(inputs.get('OverTime', '')).strip().lower() == 'yes':
            recs.append({
                'category': 'Workload & Well-being',
                'action': 'Conduct Overtime Audit & Workload Rebalancing',
                'detail': 'Overtime is the #1 correlated driver of employee attrition. Evaluate resource allocation, eliminate non-essential administrative burdens, and establish compensatory time-off policies.'
            })
            
        # Work-Life Balance trigger
        wlb = inputs.get('WorkLifeBalance', 3)
        if isinstance(wlb, (int, float)) and wlb <= 2:
            recs.append({
                'category': 'Flexibility & Culture',
                'action': 'Introduce Flexible / Hybrid Work Arrangements',
                'detail': 'Low work-life balance rating indicates burnout risk. Offer core working hours, remote options, or flexible scheduling to support restorative balance.'
            })

        # Job Satisfaction & Environment trigger
        js = inputs.get('JobSatisfaction', 3)
        if isinstance(js, (int, float)) and js <= 2:
            recs.append({
                'category': 'Engagement & Role Design',
                'action': 'Conduct 1-on-1 Stay Interview & Role Alignment',
                'detail': 'Engage the employee in a confidential career check-in to identify daily frustrations, autonomy blockers, and opportunities for meaningful job redesign.'
            })

        # Promotion Stagnation
        years_promo = inputs.get('YearsSinceLastPromotion', 0)
        years_comp = inputs.get('YearsAtCompany', 0)
        if isinstance(years_promo, (int, float)) and years_promo >= 4:
            recs.append({
                'category': 'Career Advancement',
                'action': 'Review Promotion Timeline & Lateral Growth Pathways',
                'detail': f'The employee has not received a promotion in {years_promo} years. Map clear promotion criteria, professional development milestones, or high-visibility project assignments.'
            })

        # Distance from Home
        dist = inputs.get('DistanceFromHome', 0)
        if isinstance(dist, (int, float)) and dist >= 15:
            recs.append({
                'category': 'Commute & Ergonomics',
                'action': 'Commute Subsidy or Remote Work Allowance',
                'detail': f'Commute distance ({dist} miles) is creating ongoing physical fatigue. Consider 2-3 work-from-home days per week or transit support.'
            })

        # Compensation alignment
        income = inputs.get('MonthlyIncome', 5000)
        level = inputs.get('JobLevel', 2)
        if isinstance(income, (int, float)) and income < 3500 and level >= 2:
            recs.append({
                'category': 'Compensation & Rewards',
                'action': 'Market Compensation Review',
                'detail': 'Monthly compensation appears lower than industry benchmark for this job level. Consider a merit salary adjustment or equity/performance incentive.'
            })

        # Frequent travel
        travel = str(inputs.get('BusinessTravel', '')).strip().lower()
        if 'frequently' in travel:
            recs.append({
                'category': 'Travel Policy',
                'action': 'Mitigate Travel Fatigue',
                'detail': 'Frequent travel significantly elevates turnover risk. Introduce travel rest days and explore video-conferencing alternatives where possible.'
            })

        # Fallback if profile is already very stable
        if not recs:
            recs.append({
                'category': 'Talent Retention',
                'action': 'Maintain Continuous Recognition & Leadership Mentorship',
                'detail': 'This employee exhibits high organizational stability. Continue nurturing their trajectory through executive mentorship and leadership succession planning.'
            })

        return recs


# Module-level singleton
_predictor_instance = None

def get_predictor() -> AttritionPredictor:
    """Returns a singleton instance of AttritionPredictor."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = AttritionPredictor()
    return _predictor_instance


if __name__ == '__main__':
    predictor = get_predictor()
    print(f"Loaded predictor for model: {predictor.model_name}")
    test_employee = {
        'Age': 29,
        'BusinessTravel': 'Travel_Frequently',
        'Department': 'Sales',
        'DistanceFromHome': 18,
        'Education': 2,
        'JobRole': 'Sales Representative',
        'JobLevel': 1,
        'JobSatisfaction': 1,
        'MonthlyIncome': 2400,
        'OverTime': 'Yes',
        'WorkLifeBalance': 1,
        'YearsAtCompany': 2,
        'YearsSinceLastPromotion': 1,
        'TotalWorkingYears': 3,
        'YearsInCurrentRole': 2
    }
    result = predictor.predict(test_employee)
    print("\n--- Test Prediction Result ---")
    print(f"Prediction: {result['prediction']} ({result['attrition_label']})")
    print(f"Probability: {result['probability_pct']}% | Risk Tier: {result['risk_tier']}")
    print(f"Summary: {result['summary_statement']}")
    print("\nTop Risk Drivers:")
    for d in result['top_drivers']:
        print(f"  + {d['feature']}: {d['impact']:.4f}")
    print("\nRecommendations:")
    for r in result['recommendations']:
        print(f"  * [{r['category']}] {r['action']}: {r['detail']}")
