

from scipy.stats import ks_2samp


def calculate_ks(reference, production):
    """
    Returns KS statistic and p-value.
    KS statistic closer to 1 => larger difference.
    p-value < 0.05 => statistically significant difference.
    """
    ks_stat, p_value = ks_2samp(reference, production)
    return ks_stat, p_value


def ks_status(ks_stat, p_value):
    """
    Convert KS results into human-readable status.
    """
    if ks_stat > 0.2 and p_value < 0.05:
        return "🚨 DRIFT"
    elif ks_stat > 0.1:
        return "⚠️ WARNING"
    else:
        return "OK"
