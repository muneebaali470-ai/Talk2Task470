"""
task_extractor.py
-----------------
Upgraded to handle formatted text, "Task:" labels, and dates on new lines.
"""

import re
from datetime import datetime, timedelta
from nlp_processor import (
    analyze_text, tokenize_words, pos_tag_tokens, ACTION_VERBS, TASK_CUES
)

WEEKDAYS = r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
MONTHS = (r"(january|february|march|april|may|june|july|august|"
          r"september|october|november|december|jan|feb|mar|apr|jun|jul|"
          r"aug|sep|sept|oct|nov|dec)")

DATE_PATTERNS = [
    re.compile(rf"\b(?:by|before|on|until|till|due)\s+{WEEKDAYS}\b", re.I),
    re.compile(r"\b(day\s+after\s+tomorrow|tomorrow(?:\s+(?:morning|afternoon|evening|night))?|tonight|today(?:\s+(?:morning|afternoon|evening))?)\b", re.I),
    re.compile(r"\b(next\s+(?:week|month|year)|this\s+(?:week|month|weekend)|end\s+of\s+(?:day|week|month)|eod|eow|eom)\b", re.I),
    re.compile(r"\b(as\s+soon\s+as\s+possible|asap)\b", re.I),
    re.compile(r"\bin\s+\d+\s+(?:day|days|hour|hours|week|weeks)\b", re.I),
    re.compile(rf"\b{WEEKDAYS}\s+(?:morning|afternoon|evening|night)\b", re.I),
    re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTHS}(?:\s+\d{{2,4}})?\b", re.I),
    re.compile(rf"\b{MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:\s*,?\s*\d{{2,4}})?\b", re.I),
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b", re.I),
    re.compile(r"\b\d{1,2}:\d{2}\b"),
    # Added: to catch explicit "Deadline: 08 November 2026"
    re.compile(rf"\b(?:deadline|date|due)[\s:]*\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTHS}(?:\s+\d{{2,4}})?\b", re.I),
]

HIGH_WORDS = {"urgent", "urgently", "immediately", "asap", "today", "critical", "important", "priority", "right away", "right now", "as soon as possible", "high priority", "top priority", "must", "essential", "emergency", "deadline", "overdue"}
MEDIUM_WORDS = {"soon", "this week", "upcoming", "moderate", "medium", "normal", "next week", "planned", "scheduled", "before"}
LOW_WORDS = {"whenever", "optional", "no rush", "low priority", "someday", "eventually", "if possible", "nice to have", "later"}

def detect_priority(sentence: str) -> str:
    low = sentence.lower()
    if any(w in low for w in HIGH_WORDS): return "High"
    if any(w in low for w in MEDIUM_WORDS): return "Medium"
    if any(w in low for w in LOW_WORDS): return "Low"
    return "Medium"

def extract_deadline(sentence: str):
    matches = []
    for pattern in DATE_PATTERNS:
        for m in pattern.finditer(sentence):
            # Clean up the matched string (remove "Deadline:", "Date:", etc.)
            val = re.sub(r"^(deadline|date|due)[\s:]*", "", m.group(0), flags=re.I).strip()
            matches.append((m.start(), val))
    if not matches: return None
    matches.sort(key=lambda x: x[0])
    return matches[0][1]

def resolve_deadline(expression: str):
    if not expression: return "No deadline detected"
    expr = expression.strip()
    low = expr.lower()
    today = datetime.today()
    try:
        if low == "today": return f"Today ({today.strftime('%d %b %Y')})"
        if low == "tomorrow": return f"Tomorrow ({(today + timedelta(days=1)).strftime('%d %b %Y')})"
        if low.startswith("tomorrow"):
            d = (today + timedelta(days=1)).strftime('%d %b %Y')
            return f"{expr} ({d})"
        if low == "tonight": return f"Tonight ({today.strftime('%d %b %Y')})"
        return expr
    except Exception:
        return expr

def build_description(sentence: str, action_verbs) -> str:
    text = sentence.strip().rstrip(".!?")
    
    # Clean up task labels
    text = re.sub(r"^(task|tasks|action item|to-do):\s*", "", text, flags=re.I)
    text = re.sub(r"^(important|note):\s*", "", text, flags=re.I)

    clauses = re.split(r"[,;]|(?:\s+and\s+(?=\w))", text)
    clauses = [c.strip() for c in clauses if c.strip()]

    chosen = text
    if action_verbs:
        verb_forms = {v[0].lower() for v in action_verbs}
        for clause in clauses:
            if any(vt in clause.lower() for vt in verb_forms):
                chosen = clause
                break

    for cue in sorted(TASK_CUES, key=len, reverse=True):
        chosen = re.sub(rf"^\s*{re.escape(cue)}\s+", "", chosen, flags=re.I)
        chosen = re.sub(rf"\b{re.escape(cue)}\b", "", chosen, flags=re.I)

    for pattern in DATE_PATTERNS:
        chosen = pattern.sub("", chosen)

    chosen = re.sub(r"\b(by|before|on|until|till|due)\s*$", "", chosen, flags=re.I)
    chosen = re.sub(r"\s{2,}", " ", chosen).strip(" ,.-")

    if chosen: chosen = chosen[0].upper() + chosen[1:]
    else: chosen = sentence.strip().rstrip(".!?")

    return chosen

def pick_action_keyword(action_verbs, sentence: str) -> str:
    if action_verbs: return action_verbs[0][0].lower()
    for w in tokenize_words(sentence):
        if w.lower() in ACTION_VERBS: return w.lower()
    for word, tag in pos_tag_tokens(tokenize_words(sentence)):
        if tag.startswith("VB"): return word.lower()
    return "action"

def extract_tasks(text: str):
    if not text or not text.strip(): return [], []
    
    # Upgraded: If the text contains "Task:", we treat each block separated by double newlines as a task.
    if "task:" in text.lower():
        blocks = re.split(r"\n\s*\n", text)
        analysed = []
        tasks = []
        for block in blocks:
            if not block.strip(): continue
            # Analyze the block
            analysed.extend(analyze_text(block))
            
            lines = [l.strip() for l in block.split("\n") if l.strip()]
            desc = ""
            deadline_str = "No deadline detected"
            priority = "Medium"
            
            for line in lines:
                if line.lower().startswith("task:"):
                    desc = line[5:].strip()
                elif line.lower().startswith(("deadline:", "date:", "time:")):
                    # Combine date and time if present
                    extracted = extract_deadline(line)
                    if extracted:
                        deadline_str = resolve_deadline(extracted)
                elif line.lower().startswith("important:"):
                    priority = "High"
            
            if desc:
                tasks.append({
                    "description": desc,
                    "deadline": deadline_str,
                    "priority": priority,
                    "action_keyword": pick_action_keyword([], desc),
                    "original_sentence": block,
                    "completed": False,
                })
        return tasks, analysed

    # Fallback to sentence-by-sentence NLP extraction
    analysed = analyze_text(text)
    tasks = []
    for item in analysed:
        if not item["is_task"]: continue
        sentence = item["sentence"]
        verbs = item["action_verbs"]
        description = build_description(sentence, verbs)
        if not description or len(description.split()) < 2: continue
        
        deadline_raw = extract_deadline(sentence)
        deadline = resolve_deadline(deadline_raw)
        priority = detect_priority(sentence)
        keyword = pick_action_keyword(verbs, sentence)
        
        tasks.append({
            "description": description,
            "deadline": deadline,
            "priority": priority,
            "action_keyword": keyword,
            "original_sentence": sentence,
            "completed": False,
        })
        
    seen = set()
    unique = []
    for t in tasks:
        key = t["description"].lower()
        if key in seen: continue
        seen.add(key)
        unique.append(t)
    return unique, analysed