import pandas as pd
import numpy as np


def find_center_mass(x,y,m):
    """
    function, that recievs some points and find point - center mass
    x - np.array of x-coordinates
    y - np.array of y-coordinates
    m - np.array of mass (number of people)
    """
    cgx = np.sum(x*m)/np.sum(m)
    cgy = np.sum(y*m)/np.sum(m)
    sum_m = sum(m)
    return cgx, cgy, sum_m
