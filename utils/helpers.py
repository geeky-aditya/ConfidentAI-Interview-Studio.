
def format_score(value):
    """
    Format a score for display.
    Example: 82.5 becomes '82.50/100'.
    """
    if value is None:
        return "Not available"

    return f"{value:.2f}/100"


def generate_feedback(results):
    """
    Generate feedback from the component scores
    returned by calculate_confidence_score().
    """

    feedback = []

    components = results.get("components", {})

    gaze = components.get("Estimated gaze")
    stability = components.get("Head stability")
    expression = components.get("Expression indicator")

    # Feedback about estimated gaze
    if gaze is not None:
        if gaze >= 70:
            feedback.append(
                "Your estimated gaze was frequently near "
                "the camera direction."
            )
        else:
            feedback.append(
                "Try positioning the interview window near "
                "the camera. Lighting, glasses, and head "
                "position can affect this estimate."
            )

    # Feedback about head stability
    if stability is not None:
        if stability >= 70:
            feedback.append(
                "Your estimated head orientation was "
                "relatively consistent during the session."
            )
        else:
            feedback.append(
                "Your head orientation varied during the "
                "session. Natural movement is normal; "
                "review the recording for context."
            )

    # Feedback about expression
    if expression is not None:
        feedback.append(
            "Your expression indicator is based on the "
            "CNN's Happy and Neutral probabilities. It "
            "does not directly measure confidence."
        )

    if not feedback:
        feedback.append(
            "There is not enough valid data to generate "
            "feedback. Try another session with your face "
            "clearly visible."
        )

    return feedback
