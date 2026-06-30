import pandas as pd
import re
import os

class OntologyError(Exception):
    pass

class SkillsOntology:
    def __init__(self, filepath):
        self.filepath = filepath
        self.skills_df = self.load_and_validate()
        
        # Precompute mappings
        self.canonical_names = set(self.skills_df['canonical_name'])
        
        # Build mapping: text (lower) -> canonical_name
        self.alias_to_canonical = {}
        for _, row in self.skills_df.iterrows():
            canonical = row['canonical_name']
            aliases_str = row['aliases']
            
            # Add canonical itself
            self.alias_to_canonical[canonical.lower()] = canonical
            
            # Add aliases
            if pd.notna(aliases_str) and aliases_str.strip():
                for alias in aliases_str.split('|'):
                    alias = alias.strip().lower()
                    if alias:
                        self.alias_to_canonical[alias] = canonical
                        
        # Store for rapidfuzz
        self.all_known_variants = list(self.alias_to_canonical.keys())

    def load_and_validate(self):
        if not os.path.exists(self.filepath):
            raise OntologyError(f"Ontology file not found at {self.filepath}")
            
        df = pd.read_csv(self.filepath)
        
        # Required columns
        required_cols = {'canonical_name', 'category', 'aliases'}
        if not required_cols.issubset(set(df.columns)):
            missing = required_cols - set(df.columns)
            raise OntologyError(f"Ontology missing required columns: {missing}")
            
        # Duplicate canonical names
        if df['canonical_name'].duplicated().any():
            dupes = df[df['canonical_name'].duplicated()]['canonical_name'].tolist()
            raise OntologyError(f"Duplicate canonical names found: {dupes}")
            
        # Empty alias lists
        if df['aliases'].isna().any() or (df['aliases'].str.strip() == '').any():
            raise OntologyError("Empty alias lists are not allowed.")
            
        return df

# Singleton instance initialized lazily
_instance = None

def get_ontology(filepath='data/skills_ontology.csv'):
    global _instance
    if _instance is None:
        _instance = SkillsOntology(filepath)
    return _instance
