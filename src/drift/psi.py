import numpy as np

def calculate_psi(expected,actual,bins=10):
    """Calculate population stability index (PSI)"""

    expected=np.array(expected)
    actual=np.array(actual)

    breakpoints=np.percentile(expected, np.linspace(0,100,bins+1)) #it is used to create bins based on expected distribution
    expected_counts = np.histogram(expected, breakpoints)[0] #it is used to count the number of observations in each bin
    actual_counts= np.histogram(actual, breakpoints)[0] #it is used to count the number of observations in each bin

    expected_perc=expected_counts/len(expected)
    actual_perc=actual_counts/len(actual)

    psi=np.sum(
        (actual_perc-expected_perc)*np.log((actual_perc+1e-6)/(expected_perc+1e-6))
    )
    return psi



    