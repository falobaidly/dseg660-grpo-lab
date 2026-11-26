"""Verifier tests. Run: python test_verifier.py | tee results/verifier_tests.txt"""
from verifier import extract_choice, reward

# (category, completion, reference, expected_choice, expected_reward)
CASES = [
    ("correct", "A", "A", "A", 1.0),
    ("correct", "B.", "B", "B", 1.0),
    ("correct", "(C)", "C", "C", 1.0),
    ("correct", "**A**", "A", "A", 1.0),
    ("correct", "Answer: B", "B", "B", 1.0),
    ("correct", " C\n", "C", "C", 1.0),
    ("correct", "A. Pneum", "A", "A", 1.0),
    ("correct", "<think>\n\n</think>\n\nB", "B", "B", 1.0),
    ("incorrect", "B", "A", "B", 0.0),
    ("incorrect", "C.", "A", "C", 0.0),
    ("incorrect", "Answer: A", "C", "A", 0.0),
    ("malformed", "", "A", None, 0.0),
    ("malformed", "The patient", "A", None, 0.0),
    ("malformed", "D", "A", None, 0.0),
    ("malformed", "a", "A", None, 0.0),
    ("malformed", "Answer", "A", None, 0.0),
    ("malformed", "Pneumonia", "A", None, 0.0),
    ("malformed", "AB", "A", None, 0.0),
    ("malformed", None, "A", None, 0.0),
    ("ambiguous", "A or B", "A", None, 0.0),
    ("ambiguous", "A, C", "A", None, 0.0),
    ("ambiguous", "B/C", "B", None, 0.0),
    ("ambiguous", "A (not B", "A", None, 0.0),
]

if __name__ == "__main__":
    fails = 0
    print(f"{'category':<10} {'completion':<28} {'ref':<4} {'parsed':<7} {'reward':<6} result")
    for cat, text, ref, exp_c, exp_r in CASES:
        c, r = extract_choice(text), reward(text, ref)
        ok = c == exp_c and r == exp_r
        fails += not ok
        print(f"{cat:<10} {repr(text):<28} {ref:<4} {str(c):<7} {r:<6} {'PASS' if ok else 'FAIL'}")
    print(f"\n{len(CASES) - fails}/{len(CASES)} passed")
    assert fails == 0
