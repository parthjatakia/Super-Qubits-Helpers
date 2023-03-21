"""
Author : Parth Jatakia
Date : 21 March 2023
Description : This script is used to brute force the parameter space of the fluxoniuum to calculate different quantifiers
              to help choose a high coherence fluxonium
"""

import numpy as np
import scqubits as scq
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

def fluxoniumSampling(Ejrange, Ecrange, Elrange, fname, **kwargs):
    """
    This function is used to brute force the parameter space of the fluxonium to calculate different quantifiers
    to help choose a high coherence fluxonium. The current quantifiers are
    1. Qubit Frequency
    2. Qubit Anharmonicity
    3. Charge Matrix Element

    Parameters
    ----------
    Ejrange : np.array
        List of Ej values to be sampled
    Ecrange : np.array
        List of Ec values to be sampled
    Elrange : np.array
        List of El values to be sampled
    fname : str
        Name of the file + location to be saved
    **kwargs : dict
        Dictionary of other parameters to be passed to the fluxonium class.
        1. flux : float (default = 0.5)

    Returns
    -------
    A pickle file containing the data. The data is a dictiorary with the following keys
    1. Ej : "Josephson Energy"
    2. Ec : "Capacitive Energy"
    3. El : "Inductive Energy"
    4. w01 : "Qubit Frequency"
    5. w12 : "Qubit Anharmonicity"
    6. n01 : "Charge Matrix Element"
    """
    # Initialize the data dictionary
    data = {}
    data['Ej'] = []
    data['Ec'] = []
    data['El'] = []
    data['w01'] = []
    data['w12'] = []
    data['n01'] = []

    # Handle the kwargs
    if 'flux' in kwargs:
        flux = kwargs['flux']
    else:
        flux = 0.5

    # Loop over the parameters
    for Ej in tqdm(Ejrange):
        for Ec in Ecrange:
            for El in Elrange:
                # Append the parameters
                data['Ej'].append(Ej)
                data['Ec'].append(Ec)
                data['El'].append(El)

                # Create the fluxonium
                fluxonium = scq.Fluxonium(Ej, Ec, El, flux = flux, cutoff= 110)

                # Calculate the quantifiers
                energy = fluxonium.eigenvals(evals_count = 3)
                data['w01'].append(energy[1] - energy[0])
                data['w12'].append(energy[2] - energy[1])
                data['n01'].append(fluxonium.matrixelement_table("n_operator")[0,1])

    # Save the data
    with open(fname, 'wb') as f:
        pickle.dump(data, f)

    return data

def bestfluxoniumSearch(conditions):
    """
    This function is used to search the best fluxonium from the brute force parameter space search.

    Parameters
    ----------
    conditions : dict
        Dictionary of conditions to be satisfied by the fluxonium. The keys are
        1. w01_min : float
            Qubit Frequency
        2. w01_max : float
            Qubit Frequency
        3. w12_min : float
            Qubit Anharmonicity
        4. w12_max : float
            Qubit Anharmonicity
        5. n01_min : float
            Charge Matrix Element
        6. n01_max : float
            Charge Matrix Element

    Returns
    -------
    A list of the best fluxoniums satisfying the conditions.
    """
    # Load the data
    with open('test.pkl', 'rb') as f:
        data = pickle.load(f)

    # Initialize the list of best fluxoniums
    bestfluxoniums = []

    # Loop over the data
    for i in range(len(data['Ej'])):
        if data['w01'][i] >= conditions['w01_min'] and data['w01'][i] <= conditions['w01_max'] and data['w12'][i] >= conditions['w12_min'] and data['w12'][i] <= conditions['w12_max'] and data['n01'][i] >= conditions['n01_min'] and data['n01'][i] <= conditions['n01_max']:
            bestfluxoniums.append([data['Ej'][i], data['Ec'][i], data['El'][i]])

    return bestfluxoniums



