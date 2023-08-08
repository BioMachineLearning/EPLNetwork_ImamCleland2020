import math
import numpy as np

def get_digits(n):
    if n > 0:
        return int(math.log10(n))+1
    elif n == 0:
        return 1
    else:
        return int(math.log10(-n))+2 # +1 if you don't count the '-' 


def signif(x, digits=6):
    if x == 0 or not math.isfinite(x):
        return x
    digits -= math.ceil(math.log10(abs(x)))
    r_val = round(x, digits)

    if get_digits(r_val) > digits:
        r_val = int(r_val)
        
    return r_val

def get_barlabel_from_histbar(bar, ndigits=2):
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