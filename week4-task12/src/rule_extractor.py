import re
from rapidfuzz import fuzz, process

# Negation cues: patterns that indicate a skill is NOT possessed
NEGATION_CUES = [
    r'\bno\s+experience\s+(?:with|in)\b',
    r'\bnot\s+familiar\s+with\b',
    r'\bnot\s+(?:proficient|skilled|experienced)\s+(?:in|with)\b',
    r'\black(?:s|ing)?\s+(?:experience|knowledge|skills?)\s+(?:in|with)\b',
    r'\bwithout\s+(?:experience|knowledge)\s+(?:in|with)\b',
    r'\bno\s+knowledge\s+of\b',
    r'\bdo\s+not\s+(?:know|have|use)\b',
    r"\bdon'?t\s+(?:know|have|use)\b",
    r'\bnever\s+(?:used|worked\s+with)\b',
    r'\bnot\s+(?:a\s+)?(?:fan\s+of)\b',
]

# Short / ambiguous tokens that are prone to substring false positives
SHORT_AMBIGUOUS_TOKENS = {'r', 'c', 'go', 'ai', 'dl', 'ml', 'ds', 'np', 'pd', 'js', 'ts', 'rb', 'kt', 'tf', 'cv'}


def _is_negated(text, match_start, match_end):
    """Check if the match is preceded by a negation cue within the SAME clause/sentence."""
    # Find the sentence boundary before the match — split on . ; ! ? and newlines
    sentence_start = match_start
    for i in range(match_start - 1, max(0, match_start - 120) - 1, -1):
        if text[i] in '.;!?\n':
            sentence_start = i + 1
            break
    else:
        sentence_start = max(0, match_start - 120)
    
    # Also find clause boundaries: "but", "however", "although", commas
    # This prevents "not familiar with AWS but I use Python" from negating Python
    clause_start = sentence_start
    clause_text = text[sentence_start:match_start].lower()
    for sep in [' but ', ' however ', ' although ', ', ']:
        last_idx = clause_text.rfind(sep)
        if last_idx >= 0:
            abs_pos = sentence_start + last_idx + len(sep)
            if abs_pos > clause_start:
                clause_start = abs_pos
    
    # Only look within the same clause, up to 50 chars before the match
    window_start = max(clause_start, match_start - 50)
    context_before = text[window_start:match_start].lower()
    
    for pattern in NEGATION_CUES:
        if re.search(pattern, context_before):
            return True
    
    # Also check a narrow wrap-around for cues that span the skill mention
    # e.g. "I have no experience with Docker" — here "no experience with" is before "Docker"
    window = text[window_start:match_end + 5].lower()
    for pattern in NEGATION_CUES:
        if re.search(pattern, window):
            return True
    return False


def _is_safe_word_boundary(text, start, end):
    """
    Check that a match at [start:end] is bounded by non-alphanumeric chars
    (or string start/end). This prevents matching 'R' inside 'Director'.
    """
    if start > 0 and text[start - 1].isalnum():
        return False
    if end < len(text) and text[end].isalnum():
        return False
    return True


def _is_short_token_false_positive(text, match_text, start, end):
    """
    Extra guard for very short tokens (1-2 chars). 
    For single-letter tokens like 'R', require it to appear as a standalone
    token, not part of abbreviations like 'R&D' or in a word like 'Director'.
    """
    match_lower = match_text.lower()
    if match_lower not in SHORT_AMBIGUOUS_TOKENS:
        return False
    
    # Already checked word boundaries in _is_safe_word_boundary,
    # but add extra checks for short tokens:
    
    # Check for R&D pattern
    after = text[end:end+3] if end + 3 <= len(text) else text[end:]
    if after.startswith('&'):
        return True
    
    # Check if it's inside a larger token-like pattern 
    # e.g. "r-script" should be matched via alias, not as standalone "R"
    # For single-char matches, require surrounding whitespace or punctuation
    before_char = text[start - 1] if start > 0 else ' '
    after_char = text[end] if end < len(text) else ' '
    
    # If surrounded by dashes/dots, it might be part of a compound word
    if before_char in '-.' or after_char in '-.':
        return True
    
    return False


def extract_candidates(text, ontology):
    """
    Stage 3a: Rule-based candidate generator.
    Returns list of candidate dicts with features for the ML model.
    """
    if not text or not str(text).strip():
        return []
    
    candidates = []
    seen_canonicals = {}  # canonical_name -> best candidate
    text_lower = text.lower()
    
    # --- Pass 1: Exact canonical name match ---
    for canonical in ontology.canonical_names:
        canonical_lower = canonical.lower()
        escaped = re.escape(canonical_lower)
        pattern = r'(?<!\w)' + escaped + r'(?!\w)'
        
        for m in re.finditer(pattern, text_lower):
            start, end = m.start(), m.end()
            
            if not _is_safe_word_boundary(text_lower, start, end):
                continue
            
            is_short = canonical_lower in SHORT_AMBIGUOUS_TOKENS
            if is_short and _is_short_token_false_positive(text, text_lower, start, end):
                continue
            
            negated = _is_negated(text, start, end)
            context = text[max(0, start - 40):min(len(text), end + 40)]
            
            cand = {
                'canonical_name': canonical,
                'match_type': 'exact',
                'matched_text': text[start:end],
                'offset_start': start,
                'offset_end': end,
                'fuzzy_score': 100.0,
                'is_negated': negated,
                'is_short_token': is_short,
                'token_length': len(canonical),
                'context': context,
            }
            
            key = canonical
            if key not in seen_canonicals or (not negated and seen_canonicals[key].get('is_negated', False)):
                seen_canonicals[key] = cand
    
    # --- Pass 2: Alias match ---
    for alias, canonical in ontology.alias_to_canonical.items():
        if canonical in seen_canonicals and seen_canonicals[canonical]['match_type'] == 'exact' and not seen_canonicals[canonical]['is_negated']:
            continue  # Already matched exactly and not negated
        
        escaped = re.escape(alias)
        pattern = r'(?<!\w)' + escaped + r'(?!\w)'
        
        for m in re.finditer(pattern, text_lower):
            start, end = m.start(), m.end()
            
            if not _is_safe_word_boundary(text_lower, start, end):
                continue
            
            is_short = alias in SHORT_AMBIGUOUS_TOKENS
            if is_short and _is_short_token_false_positive(text, text_lower, start, end):
                continue
            
            negated = _is_negated(text, start, end)
            context = text[max(0, start - 40):min(len(text), end + 40)]
            
            cand = {
                'canonical_name': canonical,
                'match_type': 'alias',
                'matched_text': text[start:end],
                'offset_start': start,
                'offset_end': end,
                'fuzzy_score': 100.0,
                'is_negated': negated,
                'is_short_token': is_short,
                'token_length': len(alias),
                'context': context,
            }
            
            key = canonical
            if key not in seen_canonicals or (not negated and seen_canonicals[key].get('is_negated', False)):
                seen_canonicals[key] = cand
    
    # --- Pass 3: Fuzzy matching for near-misses ---
    # Extract tokens/phrases from text to check against ontology
    words = re.findall(r'[a-zA-Z0-9#+\-.]+', text)
    # Also try bigrams and trigrams
    phrases = words[:]
    for i in range(len(words) - 1):
        phrases.append(words[i] + ' ' + words[i+1])
    for i in range(len(words) - 2):
        phrases.append(words[i] + ' ' + words[i+1] + ' ' + words[i+2])
    
    for phrase in phrases:
        phrase_lower = phrase.lower()
        if len(phrase_lower) < 3:
            continue  # Skip very short tokens for fuzzy matching
        
        results = process.extract(phrase_lower, ontology.all_known_variants, 
                                   scorer=fuzz.ratio, limit=3, score_cutoff=75)
        
        for matched_variant, score, _ in results:
            canonical = ontology.alias_to_canonical[matched_variant]
            
            if canonical in seen_canonicals:
                existing = seen_canonicals[canonical]
                if existing['match_type'] in ('exact', 'alias') and not existing['is_negated']:
                    continue  # Already have a better match
            
            if score >= 100:
                continue  # This would have been caught by exact/alias passes
            
            # Find the phrase in the original text
            idx = text_lower.find(phrase_lower)
            if idx < 0:
                continue
            start, end = idx, idx + len(phrase_lower)
            
            negated = _is_negated(text, start, end)
            context = text[max(0, start - 40):min(len(text), end + 40)]
            
            cand = {
                'canonical_name': canonical,
                'match_type': 'fuzzy',
                'matched_text': text[start:end],
                'offset_start': start,
                'offset_end': end,
                'fuzzy_score': score,
                'is_negated': negated,
                'is_short_token': len(phrase_lower) <= 2,
                'token_length': len(phrase_lower),
                'context': context,
            }
            
            key = canonical
            if key not in seen_canonicals or (not negated and score > seen_canonicals[key].get('fuzzy_score', 0)):
                seen_canonicals[key] = cand
    
    return list(seen_canonicals.values())
