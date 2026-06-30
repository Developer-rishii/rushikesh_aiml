import re

def extract_baseline(text, ontology):
    """
    Dumb baseline: raw case-insensitive exact-string keyword match 
    against canonical names only.
    """
    if not text or not str(text).strip():
        return []
        
    extracted = set()
    text_lower = text.lower()
    
    for canonical in ontology.canonical_names:
        canonical_lower = canonical.lower()
        # Use regex to ensure word boundaries to avoid matching "C" in "Mac" if "C" was a skill.
        # But some skills like "C++" have special chars.
        # A simpler way: exact substring match, but that has substring traps. 
        # The prompt implies baseline is dumb. We will use simple word boundary match 
        # or exact substring if word boundaries fail for punctuation.
        
        # Simple substring match with word boundary check
        escaped = re.escape(canonical_lower)
        # using \b might fail for C++ or C# because +/# are not word characters
        # So we create a custom boundary pattern
        pattern = r'(?<!\w)' + escaped + r'(?!\w)'
        if re.search(pattern, text_lower):
            extracted.add(canonical)
            
    return list(extracted)
