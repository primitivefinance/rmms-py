from math import inf

def illinois(f, a, b, y, margin=1e-2):

    ''' 
    Implementation from: https://towardsdatascience.com/mastering-root-searching-algorithms-in-python-7120c335a2a8
    
    Bracketed approach of Root-finding with illinois method
    Parameters
    ----------
    f: callable, continuous function
    a: float, lower bound to be searched
    b: float, upper bound to be searched
    y: float, target value
    margin: float, margin of error in absolute term
    Returns
    -------
    A float c, where f(c) is within the margin of y
    '''
    assert y >= (lower := f(a)), f"y is smaller than the lower bound. {y} < {lower}"
    assert y <= (upper := f(b)), f"y is larger than the upper bound. {y} > {upper}"

    stagnant = 0

    while 1:
        c = ((a * (upper - y)) - (b * (lower - y))) / (upper - lower)
        if abs((y_c := f(c)) - y) < margin:
            # found!
            return c
        elif y < y_c:
            b, upper = c, y_c
            if stagnant == -1:
                # Lower bound is stagnant!
                lower += (y - lower) / 2
            stagnant = -1
        else:
            a, lower = c, y_c
            if stagnant == 1:
                # Upper bound is stagnant!
                upper -= (upper - y) / 2
            stagnant = 1