
from utils.helpers import format_score, generate_feedback


# Test score formatting
print("========== SCORE FORMATTING TEST ==========")

print(format_score(82.5))
print(format_score(95))
print(format_score(None))


# Sample result from the confidence scoring module
sample_results = {
    "final_score": 82.5,
    "components": {
        "Estimated gaze": 90.0,
        "Head stability": 80.0,
        "Expression indicator": 75.0
    },
    "message": (
        "Experimental behavioral indicator; "
        "not a validated measure of actual confidence."
    )
}


# Test feedback generation
print("\n========== FEEDBACK TEST ==========")

feedback = generate_feedback(sample_results)

for number, message in enumerate(feedback, start=1):
    print(f"{number}. {message}")


print("\n========== TEST COMPLETED ==========")
