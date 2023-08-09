import math
import numpy as np
 
def get_digits(n):
    """
    Returns the number of digits in a given integer.

    Parameters:
    n (int): The integer to count the digits of.

    Returns:
    int: The number of digits in the given integer.

    Example:
    get_digits(12345)
    Output: 5
    """
    if n > 0:
        return int(math.log10(n))+1
    elif n == 0:
        return 1
    else:
        return int(math.log10(-n))+2 # +1 if you don't count the '-' 


def signif(x, digits=6):
    """
    Rounds a given number to a specified number of significant digits.

    Parameters:
    x (float): The number to round.
    digits (int): The number of significant digits. Defaults to 6.

    Returns:
    float: The rounded number with the specified number of significant digits.

    Example:
    signif(3.14159, 3)
    Output: 3.14
    """
    if x == 0 or not math.isfinite(x):
        return x
    digits -= math.ceil(math.log10(abs(x)))
    r_val = round(x, digits)

    if get_digits(r_val) > digits:
        r_val = int(r_val)
        
    return r_val

def get_barlabel_from_histbar(bar, ndigits=2):
    """
    Formats the label for a histogram bar based on the bar height.

    Parameters:
    bar (matplotlib.container.BarContainer): The histogram bar to format the label for.
    ndigits (int): The number of significant digits. Defaults to 2.

    Returns:
    tuple: A tuple containing the formatted bar label and the unit.

    Example:
    get_barlabel_from_histbar(bar, 3)
    Output: (0.123, 'ms')
    """
    barlabel = bar.get_height()
    print(barlabel)
    if barlabel >= 1:
        barlabel = np.round(barlabel, 1)
        unit = 's'
    elif barlabel >= 1e-3:
        barlabel = np.round(barlabel*1e3, 1)
        unit = 'ms'
    elif barlabel >= 1e-6:
        barlabel = np.round(barlabel*1e6, 1)
        unit = 'us'
    else: 
        barlabel = np.round(barlabel*1e9, 1)
        unit = 'ns'
    barlabel = signif(barlabel, digits=ndigits)
    return barlabel, unit