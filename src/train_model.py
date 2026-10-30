"""
train_model.py
--------------
Trains, tunes, evaluates, and compares four classification algorithms:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Gradient Boosting Classifier

Handles class imbalance with class weights and sample weights.
Saves the best-performing pipeline and evaluation artifacts using Joblib.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    classification_report
)

from data_preprocessing import (
    load_data,
    clean_data,
    get_feature_columns,
    build_preprocessor,
    split_data,
    get_default_feature_values,
    TARGET_COLUMN
)


MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def get_model_definitions():
    """
    Defines the four classifiers and their hyperparameter grids for tuning.
    Addresses class imbalance via class_weight='balanced' or sample_weight.
    """
    models = {
        'Logistic Regression': {
            'estimator': LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            ),
            'param_grid': {
                'classifier__C': [0.01, 0.1, 1.0, 5.0],
                'classifier__solver': ['lbfgs', 'liblinear']
            }
        },
        'Decision Tree': {
            'estimator': DecisionTreeClassifier(
                class_weight='balanced',
                random_state=42
            ),
            'param_grid': {
                'classifier__max_depth': [3, 5, 8, 12],
                'classifier__min_samples_split': [5, 10, 20],
                'classifier__min_samples_leaf': [2, 5, 10]
            }
        },
        'Random Forest': {
            'estimator': RandomForestClassifier(
                class_weight='balanced',
                random_state=42
            ),
            'param_grid': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [5, 8, 12, None],
                'classifier__min_samples_split': [4, 8],
                'classifier__min_samples_leaf': [2, 4]
            }
        },
        'Gradient Boosting': {
            'estimator': GradientBoostingClassifier(
                random_state=42
            ),
            'param_grid': {
                'classifier__n_estimators': [100, 150],
                'classifier__learning_rate': [0.03, 0.1, 0.2],
                'classifier__max_depth': [3, 4, 5],
                'classifier__subsample': [0.8, 1.0]
            }
        }
    }
    return models


def get_transformed_feature_names(preprocessor, num_cols, cat_cols):
    """Extracts human-readable feature names after One-Hot Encoding."""
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_cols = list(cat_encoder.get_feature_names_out(cat_cols))
    return num_cols + encoded_cat_cols


def train_and_evaluate_all():
    """Orchestrates end-to-end training, tuning, evaluation, and saving."""
    print("=" * 60)
    print("STEP 1: Loading and Preprocessing Data")
    print("=" * 60)
    
    df_raw = load_data()
    df_clean = clean_data(df_raw)
    num_cols, cat_cols = get_feature_columns(df_clean)
    defaults = get_default_feature_values(df_raw)
    
    # Store unique category choices for the UI input forms
    categorical_options = {col: sorted(df_clean[col].dropna().unique().tolist()) for col in cat_cols}
    
    X_train, X_test, y_train, y_test = split_data(df_clean, test_size=0.2, random_state=42)
    print(f"Training dataset size: {X_train.shape[0]} samples")
    print(f"Testing dataset size:  {X_test.shape[0]} samples")
    print(f"Train attrition rate:  {y_train.mean():.2%}")
    print(f"Test attrition rate:   {y_test.mean():.2%}")
    
    # Preprocessor fit only on training data to prevent leakage
    base_preprocessor = build_preprocessor(num_cols, cat_cols)
    base_preprocessor.fit(X_train)
    transformed_feature_names = get_transformed_feature_names(base_preprocessor, num_cols, cat_cols)
    print(f"Total features after encoding: {len(transformed_feature_names)}")

    # Sample weights for Gradient Boosting to handle class imbalance
    sample_weights_train = compute_sample_weight('balanced', y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models_def = get_model_definitions()
    
    results = {}
    comparison_table = []
    
    print("\n" + "=" * 60)
    print("STEP 2: Model Training and Hyperparameter Tuning")
    print("=" * 60)
    
    for name, config in models_def.items():
        print(f"\n---> Training: {name}...")
        
        # Build fresh pipeline for each model
        preprocessor = build_preprocessor(num_cols, cat_cols)
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', config['estimator'])
        ])
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid=config['param_grid'],
            scoring='f1',
            cv=cv,
            n_jobs=-1,
            verbose=0
        )
        
        # Fit grid search (pass sample weights if Gradient Boosting)
        if name == 'Gradient Boosting':
            grid_search.fit(X_train, y_train, classifier__sample_weight=sample_weights_train)
        else:
            grid_search.fit(X_train, y_train)
            
        best_pipeline = grid_search.best_estimator_
        
        # Evaluate on the held-out test set
        y_pred = best_pipeline.predict(X_test)
        if hasattr(best_pipeline, "predict_proba"):
            y_proba = best_pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(best_pipeline, "decision_function"):
            decision = best_pipeline.decision_function(X_test)
            y_proba = (decision - decision.min()) / (decision.max() - decision.min())
        else:
            y_proba = y_pred.astype(float)
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)
        
        # Compute ROC & Precision-Recall curves for visualization
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_proba)
        
        print(f"Best Params: {grid_search.best_params_}")
        print(f"Accuracy:  {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
        print("Confusion Matrix:")
        print(cm)
        
        results[name] = {
            'pipeline': best_pipeline,
            'best_params': grid_search.best_params_,
            'metrics': {
                'accuracy': float(acc),
                'precision': float(prec),
                'recall': float(rec),
                'f1': float(f1),
                'roc_auc': float(roc_auc)
            },
            'confusion_matrix': cm.tolist(),
            'y_pred': y_pred.tolist(),
            'y_proba': y_proba.tolist(),
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'roc_auc': float(roc_auc)
            },
            'pr_curve': {
                'precision': pr_precision.tolist(),
                'recall': pr_recall.tolist()
            }
        }
        
        comparison_table.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        })
        
    df_compare = pd.DataFrame(comparison_table)
    print("\n" + "=" * 60)
    print("STEP 3: Model Performance Comparison")
    print("=" * 60)
    print(df_compare.to_string(index=False))
    
    # Selection criterion: In HR Attrition, identifying employees at risk is critical.
    # Prioritize F1-Score & Recall rather than raw accuracy.
    # We rank primarily by F1-Score, then by Recall, then ROC-AUC.
    ranked_models = df_compare.sort_values(
        by=['F1-Score', 'Recall', 'ROC-AUC'],
        ascending=[False, False, False]
    ).reset_index(drop=True)
    
    best_model_name = ranked_models.iloc[0]['Model']
    print(f"\n>>> Selected Best Model: {best_model_name} <<<")
    print(f"Selection Rationale: Achieves highest balanced F1-score ({ranked_models.iloc[0]['F1-Score']:.4f}) and high recall ({ranked_models.iloc[0]['Recall']:.4f}), effectively detecting at-risk talent without excessive false alarms.")
    
    # Extract feature importances or coefficients
    best_pipeline = results[best_model_name]['pipeline']
    classifier = best_pipeline.named_steps['classifier']
    feature_importances = {}
    
    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
        feature_importances = dict(sorted(
            zip(transformed_feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        ))
    elif hasattr(classifier, 'coef_'):
        coefs = np.abs(classifier.coef_[0])
        feature_importances = dict(sorted(
            zip(transformed_feature_names, coefs),
            key=lambda x: x[1],
            reverse=True
        ))
        
    print("\nTop 10 Influential Features for Attrition:")
    for feat, imp in list(feature_importances.items())[:10]:
        print(f"  - {feat}: {imp:.4f}")
        
    print("\n" + "=" * 60)
    print("STEP 4: Saving Trained Models & Artifacts")
    print("=" * 60)
    
    # Save best model payload
    best_model_payload = {
        'model_name': best_model_name,
        'pipeline': best_pipeline,
        'metrics': results[best_model_name]['metrics'],
        'confusion_matrix': results[best_model_name]['confusion_matrix'],
        'feature_names': transformed_feature_names,
        'numerical_cols': num_cols,
        'categorical_cols': cat_cols,
        'categorical_options': categorical_options,
        'defaults': defaults,
        'feature_importances': feature_importances,
        'all_models_summary': comparison_table
    }
    
    best_model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    joblib.dump(best_model_payload, best_model_path)
    print(f"Saved best model pipeline to: {best_model_path}")
    
    # Save comparison data for dashboard
    comparison_payload = {
        'comparison_table': comparison_table,
        'models_data': {
            name: {
                'metrics': results[name]['metrics'],
                'confusion_matrix': results[name]['confusion_matrix'],
                'best_params': results[name]['best_params'],
                'roc_curve': results[name]['roc_curve'],
                'pr_curve': results[name]['pr_curve']
            } for name in results
        },
        'y_test': y_test.tolist(),
        'feature_names': transformed_feature_names,
        'feature_importances': feature_importances
    }
    comparison_path = os.path.join(MODELS_DIR, 'model_comparison.pkl')
    joblib.dump(comparison_payload, comparison_path)
    print(f"Saved model comparison payload to: {comparison_path}")
    
    return best_model_payload, comparison_payload


if __name__ == '__main__':
    train_and_evaluate_all()
