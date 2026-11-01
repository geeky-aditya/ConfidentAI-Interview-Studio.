
import threading
from collections import Counter


class SessionMetrics:
    """
    Stores measurements collected during one interview session.

    Scores are collected from actual frame analysis.
    This class does not generate or assume scores itself.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        """Clear measurements before a new interview."""

        with getattr(self, "_lock", threading.Lock()):
            self._gaze_scores = []
            self._head_stability_scores = []
            self._expression_scores = []
            self._emotion_counts = Counter()
            self._frames_analyzed = 0

    @staticmethod
    def _valid_score(score):
        """Accept only numeric scores between 0 and 100."""

        if score is None:
            return None

        try:
            score = float(score)
        except (TypeError, ValueError):
            return None

        if not 0 <= score <= 100:
            return None

        return score

    def record_sample(
        self,
        gaze_score=None,
        head_stability_score=None,
        expression_score=None,
        emotion=None
    ):
        """
        Record measurements from one processed frame.

        Missing measurements are ignored rather than treated as zero.
        """

        gaze_score = self._valid_score(gaze_score)
        head_stability_score = self._valid_score(
            head_stability_score
        )
        expression_score = self._valid_score(
            expression_score
        )

        with self._lock:
            self._frames_analyzed += 1

            if gaze_score is not None:
                self._gaze_scores.append(gaze_score)

            if head_stability_score is not None:
                self._head_stability_scores.append(
                    head_stability_score
                )

            if expression_score is not None:
                self._expression_scores.append(
                    expression_score
                )

            if emotion:
                self._emotion_counts[str(emotion)] += 1

    @staticmethod
    def _average(values):
        if not values:
            return None

        return round(sum(values) / len(values), 2)

    def get_summary(self):
        """
        Return a snapshot of the measurements collected so far.
        """

        with self._lock:
            total_emotions = sum(
                self._emotion_counts.values()
            )

            if total_emotions:
                emotion_distribution = {
                    emotion: round(
                        count / total_emotions * 100, 2
                    )
                    for emotion, count
                    in self._emotion_counts.items()
                }
            else:
                emotion_distribution = {}

            return {
                "frames_analyzed": self._frames_analyzed,

                "gaze_score": self._average(
                    self._gaze_scores
                ),

                "head_stability_score": self._average(
                    self._head_stability_scores
                ),

                "expression_score": self._average(
                    self._expression_scores
                ),

                "emotion_distribution": emotion_distribution,

                "dominant_emotion": (
                    self._emotion_counts.most_common(1)[0][0]
                    if self._emotion_counts
                    else None
                )
            }
