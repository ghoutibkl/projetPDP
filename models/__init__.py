import numpy as np



def model_1(Z,Ac,Wc,Lc):
    """
    h(t)=Z+Σ Ac.cos(Wc.t-Lc)
    Z:hauteur de moyenne.
    Ac:amplitude.
    Wc:fréquence (speed).
    Lc:phase
    [R(t)= hauteur d’eau au temps t ]= Model
    remarque:
    Wc:ne dépend pas du port.
    Lc:dépend du port.
    Ac:dépend du port. 

    """
    t=np.arange(0,365,1)
    return Z+np.sum(Ac*np.cos(Wc*t-Lc))


