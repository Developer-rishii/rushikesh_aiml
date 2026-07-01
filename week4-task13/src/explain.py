import os
import pickle
import pandas as pd
import numpy as np
import shap
import sys

# Suppress some shap/sklearn warnings for clean demo output
import warnings
warnings.filterwarnings('ignore')

class SessionExplainer:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            self.pipeline = pickle.load(f)
            
        self.preprocessor = self.pipeline.named_steps['preprocessor']
        self.model = self.pipeline.named_steps['classifier']
        
        # We need to use TreeExplainer or LinearExplainer depending on the model
        if hasattr(self.model, 'get_booster'):
            # XGBoost
            self.explainer = shap.TreeExplainer(self.model)
            self.is_tree = True
        else:
            # Logistic Regression
            # Load background data from test_set.csv
            test_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'test_set.csv')
            df_bg = pd.read_csv(test_path).drop(columns=['true_violation']).head(100)
            self.explainer = shap.LinearExplainer(self.model, self.preprocessor.transform(df_bg))
            self.is_tree = False
            
        # Get feature names from preprocessor
        num_features = self.preprocessor.transformers_[0][2]
        cat_encoder = self.preprocessor.transformers_[1][1].named_steps['onehot']
        cat_features = self.preprocessor.transformers_[1][2]
        
        if hasattr(cat_encoder, 'get_feature_names_out'):
            cat_feature_names = cat_encoder.get_feature_names_out(cat_features)
        else:
            cat_feature_names = cat_encoder.get_feature_names(cat_features)
            
        self.feature_names = list(num_features) + list(cat_feature_names)

    def explain(self, raw_input_df):
        """
        Takes a raw pandas DataFrame (1 row) and returns the plain English reason.
        """
        # Get prediction
        prob = self.pipeline.predict_proba(raw_input_df)[0, 1]
        pred_label = "True Violation" if prob > 0.5 else "False Positive"
        
        # Transform input for SHAP
        transformed_input = self.preprocessor.transform(raw_input_df)
        
        # Get SHAP values
        if self.is_tree:
            shap_values = self.explainer.shap_values(transformed_input)
            if isinstance(shap_values, list): # For some versions of SHAP/XGB
                shap_values = shap_values[1]
        else:
            shap_values = self.explainer.shap_values(transformed_input)
            
        # shap_values shape should be (1, n_features)
        if len(shap_values.shape) == 2:
            shap_vals = shap_values[0]
        else:
            shap_vals = shap_values
            
        # Map back to original features for readability
        feature_impacts = []
        for i, val in enumerate(shap_vals):
            if abs(val) > 0.05: # Only consider somewhat impactful features
                feat_name = self.feature_names[i]
                
                # Reverse mapping for categorical if needed, but for English it's fine
                orig_val = None
                
                # Try to find original value in raw df
                for raw_col in raw_input_df.columns:
                    if raw_col in feat_name:
                        orig_val = raw_input_df.iloc[0][raw_col]
                        break
                        
                feature_impacts.append({
                    'feature': feat_name,
                    'impact': val,
                    'value': orig_val
                })
                
        # Sort by absolute impact
        feature_impacts.sort(key=lambda x: abs(x['impact']), reverse=True)
        
        # Build plain English string
        reason_parts = []
        for f in feature_impacts[:3]: # Top 3 reasons
            direction = "increases" if f['impact'] > 0 else "decreases"
            val_str = f"({f['value']})" if f['value'] is not None else ""
            
            # Map nice names
            nice_name = f['feature'].replace('_', ' ').title()
            
            if "decreases" in direction:
                reason_parts.append(f"{nice_name} {val_str} lowers the risk score")
            else:
                reason_parts.append(f"{nice_name} {val_str} raises the risk score")
                
        reason_str = f"Flagged as {pred_label} (confidence: {prob:.2f}). Reason: "
        if not reason_parts:
            reason_str += "No single feature had a major impact, model relied on complex combinations."
        else:
            reason_str += ", and ".join(reason_parts) + "."
            
        return pred_label, prob, reason_str

if __name__ == "__main__":
    # Test the explainer with a single example
    model_path = os.path.join(os.path.dirname(__file__), '..', 'best_model.pkl')
    test_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'test_set.csv')
    
    if os.path.exists(model_path) and os.path.exists(test_path):
        df_test = pd.read_csv(test_path)
        sample = df_test.iloc[[0]].drop(columns=['true_violation'])
        
        explainer = SessionExplainer(model_path)
        label, prob, reason = explainer.explain(sample)
        print("--- One Example Walkthrough ---")
        print(f"Input features:\n{sample.iloc[0].to_dict()}")
        print(f"\nModel Output: {label}")
        print(f"Explanation:\n{reason}")
    else:
        print("Model or test data not found.")
