
from webcam.session_metrics import SessionMetrics

metrics = SessionMetrics()

print("SESSION METRICS TEST")

# Test data only; these are not real interview results.
metrics.record_sample(
    gaze_score=90,
    head_stability_score=80,
    expression_score=75,
    emotion="Neutral"
)

metrics.record_sample(
    gaze_score=70,
    head_stability_score=60,
    expression_score=85,
    emotion="Happy"
)

summary = metrics.get_summary()

print("Frames analyzed:", summary["frames_analyzed"])
print("Gaze average:", summary["gaze_score"])
print("Head stability average:", summary["head_stability_score"])
print("Expression average:", summary["expression_score"])
print("Emotion distribution:", summary["emotion_distribution"])
print("Dominant emotion:", summary["dominant_emotion"])

assert summary["frames_analyzed"] == 2
assert summary["gaze_score"] == 80.0
assert summary["head_stability_score"] == 70.0
assert summary["expression_score"] == 80.0

print("All tests passed!")
