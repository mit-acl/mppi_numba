#!/usr/bin/env python3
import numpy as np

# Normal unicycle model
def normalize_angle(th):
    pi_2 = 2*np.pi
    # reduce the angle  
    th =  th % pi_2; 

    # force it to be the positive remainder, so that 0 <= angle < 360  
    th = (th + pi_2) % (pi_2);  

    # force into the minimum absolute value residue class, so that -180 < angle <= 180  
    if th > np.pi:
        th -= pi_2

    return th

def normalize_angle_np(th):
    pi_2 = 2*np.pi
    # reduce the angle  
    th =  th % pi_2; 

    # force it to be the positive remainder, so that 0 <= angle < 360  
    th = (th + pi_2) % (pi_2);  

    # force into the minimum absolute value residue class, so that -180 < angle <= 180  
    mask = (th > np.pi)
    th[mask] = th[mask] - pi_2

    return th
