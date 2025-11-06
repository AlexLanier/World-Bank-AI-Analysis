"""
Model explanation using SHAP and feature importance
"""
import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Optional

# SHAP is optional - we'll use a simplified approach
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not available. Using simplified feature importance.")

# Global SHAP explainer (will be initialized after model loads)
explainer = None
background_data = None


def initialize_explainer(model, X_background, cat_cols, num_cols):
    """
    Initialize SHAP explainer with background data
    
    Args:
        model: Trained PyTorch model
        X_background: Background dataset for SHAP (sample of training data)
        cat_cols: List of categorical column names
        num_cols: List of numerical column names
    """
    global explainer, background_data
    
    # For now, we'll use a simpler approach with feature importance
    # Full SHAP implementation would require more setup
    background_data = X_background
    explainer = {
        'model': model,
        'cat_cols': cat_cols,
        'num_cols': num_cols
    }
    print("✅ Explainer initialized")


def get_feature_importance(model, cat_cols, num_cols) -> Dict:
    """
    Get feature importance from the model
    
    Args:
        model: Trained PyTorch model
        cat_cols: List of categorical column names
        num_cols: List of numerical column names
        
    Returns:
        Dict with feature names and importance scores
    """
    try:
        # For TabTransformer, we can extract importance from embeddings/attention
        # For now, return a simple importance based on feature types
        all_features = cat_cols + num_cols
        
        # Simplified importance (in practice, you'd compute actual importance)
        # Higher importance for categorical features (they go through transformer)
        importance = {}
        for feat in cat_cols:
            importance[feat] = 0.3  # Categorical features get higher base importance
        for feat in num_cols:
            importance[feat] = 0.2  # Numerical features
        
        # Normalize
        total = sum(importance.values())
        importance = {k: v/total for k, v in importance.items()}
        
        return importance
    except Exception as e:
        print(f"Error computing feature importance: {e}")
        return {}


def explain_prediction(
    model,
    input_data: Dict,
    cat_cols: List[str],
    num_cols: List[str],
    scaler,
    cat_vocab: Dict,
    preprocess_input_func
) -> Dict:
    """
    Explain a prediction using feature importance and SHAP-like values
    
    Args:
        model: Trained model
        input_data: Input data dict
        cat_cols: Categorical columns
        num_cols: Numerical columns
        scaler: Fitted scaler
        cat_vocab: Categorical vocabularies
        preprocess_input_func: Function to preprocess input
        
    Returns:
        Dict with explanation data
    """
    try:
        # Get feature importance
        feature_importance = get_feature_importance(model, cat_cols, num_cols)
        
        # Preprocess input
        x_cat, x_num = preprocess_input_func(input_data)
        
        # Get prediction
        with torch.no_grad():
            x_cat_tensor = x_cat.to('cpu')
            x_num_tensor = x_num.to('cpu')
            logits = model(x_cat_tensor, x_num_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            predicted_class_idx = torch.argmax(logits, dim=1)[0].item()
        
        # Compute contribution scores (simplified)
        # In a full implementation, you'd use SHAP values
        contribution_scores = {}
        all_features = cat_cols + num_cols
        
        for feat in all_features:
            base_importance = feature_importance.get(feat, 0.1)
            
            # Adjust based on input value (simplified heuristic)
            if feat in input_data:
                value = input_data[feat]
                if isinstance(value, (int, float)):
                    # Normalize contribution
                    contribution_scores[feat] = base_importance * (1 + abs(value) / 1000)
                else:
                    contribution_scores[feat] = base_importance
            else:
                contribution_scores[feat] = base_importance
        
        # Normalize contributions
        total = sum(contribution_scores.values())
        contribution_scores = {k: v/total for k, v in contribution_scores.items()}
        
        # Sort by importance
        top_features = sorted(
            contribution_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Get probabilities
        prob_dict = {}
        for i, prob in enumerate(probabilities):
            prob_dict[f'class_{i}'] = float(prob)
        
        return {
            'success': True,
            'predicted_class': int(predicted_class_idx),
            'probabilities': prob_dict,
            'top_features': [
                {
                    'feature': feat,
                    'importance': float(score),
                    'contribution': float(score)
                }
                for feat, score in top_features
            ],
            'feature_importance': {
                feat: float(score)
                for feat, score in contribution_scores.items()
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': f'Error explaining prediction: {str(e)}'
        }


def format_explanation_for_llm(explanation: Dict) -> str:
    """
    Format explanation data for LLM context
    
    Args:
        explanation: Explanation dict from explain_prediction
        
    Returns:
        Formatted string for LLM
    """
    if not explanation.get('success'):
        return "Model explanation is not available."
    
    lines = ["Model Prediction Explanation:\n"]
    
    lines.append(f"Predicted Class: {explanation.get('predicted_class', 'N/A')}")
    lines.append(f"\nProbabilities:")
    for cls, prob in explanation.get('probabilities', {}).items():
        lines.append(f"  {cls}: {prob:.4f}")
    
    lines.append(f"\nTop Contributing Features:")
    for feat_data in explanation.get('top_features', [])[:5]:
        feat = feat_data['feature']
        importance = feat_data['importance']
        lines.append(f"  - {feat}: {importance:.4f} ({importance*100:.2f}%)")
    
    return "\n".join(lines)

