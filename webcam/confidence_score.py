
def calculate_confidence_score(
    gaze_scores,
    head_stability_scores,
    expression_scores
):
    """
    Combine behavioral indicators into an experimental
    score from 0 to 100.

    These weights are initial design choices, not validated
    measures of interview confidence.
    """

    components = {
        "Estimated gaze": (gaze_scores, 0.40),
        "Head stability": (head_stability_scores, 0.30),
        "Expression indicator": (expression_scores, 0.30)
    }

    available = []

    for name, (values, weight) in components.items():

        if values:
            average_score = sum(values) / len(values)

            average_score = max(
                0.0,
                min(100.0, average_score)
            )

            available.append(
                (name, average_score, weight)
            )

    if not available:
        return {
            "final_score": None,
            "components": {},
            "message": "Not enough valid measurements."
        }

    total_weight = sum(
        weight for _, _, weight in available
    )

    final_score = sum(
        score * weight
        for _, score, weight in available
    ) / total_weight

    return {
        "final_score": round(final_score, 2),
        "components": {
            name: round(score, 2)
            for name, score, _ in available
        },
        "message": (
            "Experimental behavioral indicator; "
            "not a validated measure of actual confidence."
        )
    }
