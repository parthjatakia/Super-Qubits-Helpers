"""
Author : Parth Jatakia
Date : 21 March 2023
Description : This script is used to brute force the parameter space of the fluxonium to calculate different quantifiers
              to help choose a high coherence fluxonium
"""

import numpy as np
import scqubits as scq
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import scipy.constants as ct

def calcT1_capacitive(Ej, Ec, El, Qcal, Tr):
    """
    Function to Calculate out the T1 decay due to capacitive loss
    :param Ej:
    :param Ec:
    :param El:
    :param Qcal:
    :param Tr:
    :return:
    """

    # Define Constants
    hbar = ct.hbar
    kb = ct.k

    # Defining Fluxonium
    fluxonium = scq.Fluxonium(EJ = Ej, EC = Ec, EL = El, flux = 0.5, cutoff = 110)

    # Calculating Qubit Frequency in radians
    energies =  fluxonium.eigenvals(evals_count = 4)
    w01 = (energies[1] - energies[0])*1e9*2*np.pi

    # Calculating Capacitive Energy in J
    Ec_q = hbar*fluxonium.EC*1e9*2*np.pi

    # Calculating phi matrix element between 0 and 1
    phimat01 = np.abs(fluxonium.matrixelement_table("phi_operator")[0,1])

    # Calculating the rate of decay
    G1_q = (( hbar * w01**2 / (4*Ec_q* Qcal) ) * (1 / np.tanh( hbar*w01 / (2*kb*Tr) )) * phimat01**2)
    T1_q = 1/G1_q

    return T1_q*1e6

def thermalPopulation(Ej, Ec, El, flux, T, levels = 4):
    """
    Function to calculate the thermal population of the fluxonium up till a level. The default is 4
    :param Ej: float (GHz)
    :param Ec: float (GHz)
    :param El: float (GHz)
    :param flux: float
    :param T: float (K)
    :return: np.array (length = levels)
    """
    # Define Constants
    hbar = ct.hbar
    kb = ct.k

    # Defining Fluxonium
    fluxonium = scq.Fluxonium(EJ = Ej, EC = Ec, EL = El, flux = flux, cutoff = 110)

    # Calculating Population
    cutoff = 3*levels + 10
    energies =  fluxonium.eigenvals(evals_count = cutoff)
    population = np.zeros(cutoff)
    for i in tqdm(range(cutoff)):
        population[i] = np.exp(-(hbar*energies[i]*1e9*2*np.pi)/(kb*T))
    # Normalizing the population
    population = population/np.sum(population)

    return population[:levels]


def fluxoniumSampling(Ejrange, Ecrange, Elrange, fname, **kwargs):
    """
    This function is used to brute force the parameter space of the fluxonium to calculate different quantifiers
    to help choose a high coherence fluxonium. The current quantifiers are
    1. Qubit Frequency
    2. Qubit Anharmonicity
    3. Charge Matrix Element
    4. T1 decay due to capacitive loss

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
                data['n01'].append(np.abs(fluxonium.matrixelement_table("n_operator")[0,1]))

    # Save the data
    with open(fname, 'wb') as f:
        pickle.dump(data, f)

    return data

def bestfluxoniumSearch(conditions, tempRange, fname, **kwargs):
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
    tempRange : np.array
        List of temperatures to be sampled
    **kwargs : dict
        Dictionary of other parameters to be passed to the fluxonium class.
        1. Qcal : float (default = 1e6)
            The capacitive Qubit Quality Factor
    Returns
    -------
    A list of the best fluxoniums satisfying the conditions.
    """

    # Handle the kwargs
    if 'Qcal' in kwargs:
        Qcal = kwargs['Qcal']
    else:
        Qcal = 5e6

    # Load the data
    with open(fname, 'rb') as f:
        data = pickle.load(f)

    # Initialize the list of best fluxoniums
    validfluxoniums = []

    print("Searching for the fluxoniums that satisfy the conditions...")
    # Loop over the data to find the valid fluxoniums
    for i in tqdm(range(len(data['Ej']))):
        if data['w01'][i] >= conditions['w01_min'] and data['w01'][i] <= conditions['w01_max'] and data['w12'][i] >= conditions['w12_min'] and data['w12'][i] <= conditions['w12_max'] and data['n01'][i] >= conditions['n01_min'] and data['n01'][i] <= conditions['n01_max']:
            validfluxoniums.append([data['Ej'][i], data['Ec'][i], data['El'][i], data['w01'][i], data['w12'][i], data['n01'][i]])

    if len(validfluxoniums) == 0:
        print("No valid fluxoniums found. Please try again with different conditions.")
        return

    print("Calculating the best fluxoniums with highest T1...")
    # Find the fluxonium with the  higest T1 among the valid fluxoniums for each temperature
    bestfluxoniums = []
    for temp in tqdm(tempRange):
        T1 = []
        for fluxonium in validfluxoniums:
            T1.append(calcT1_capacitive(fluxonium[0], fluxonium[1], fluxonium[2], Qcal, temp))
        bestfluxoniums.append((validfluxoniums[np.argmax(T1)], np.max(T1)))

    return bestfluxoniums



