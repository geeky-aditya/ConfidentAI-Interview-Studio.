
from webcam.confidence_score import calculate_confidence_score

# Sample inputs for testing the scoring formula
gaze_scores = [90, 95, 85]
head_stability_scores = [80, 75, 85]
expression_scores = [70, 75, 80]

result = calculate_confidence_score(
    gaze_scores,
    head_stability_scores,
    expression_scores
)

print("\n========== CONFIDENCE SCORE TEST ==========")

print("Final Score:", result["final_score"])

print("\nComponent Scores:")

for name, score in result["components"].items():
    print(f"{name}: {score}/100")

print("\nMessage:", result["message"])

print("===========================================")
