
import numpy as np


def prediction_entropy(probs, eps=1e-9):
    """
    Computes average entropy of prediction probabilities.
    Higher entropy = more uncertainty.
    """
    probs = np.clip(probs, eps, 1 - eps)

    entropy = -(
        probs * np.log(probs) +
        (1 - probs) * np.log(1 - probs)
    )

    return entropy.mean()


def entropy_status(entropy, reference_entropy):
    """
    Compare current entropy with reference entropy
    """
    delta = entropy - reference_entropy

    if delta > 0.15:
        return "🚨 CONFIDENCE COLLAPSE"
    elif delta > 0.05:
        return "⚠️ CONFIDENCE DROPPING"
    else:
        return "OK"
