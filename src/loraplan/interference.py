"""
The ``interference`` module contains tools for modeling wireless interference.

A central object is a (LoRa) ``Traffic`` object. This represents a collection
of LoRa transmissions observed within a specified ``TimeWindow``. To study interference
two additional concepts are required. First is a ``TrafficGenerator`` that can
randomly sample ``Traffic`` objects from some specified distribution.
Non-stochastic generators are also supported, such as e.g. models in which
devices tramsit at regular intervals or with fixed parameters. Second, ``Traffic``
should be classified into successfull or failed, depending on the degree of
interference. We use a ``CollisionModel`` class that takes ``Traffic``-objects
and assigns labels to each transmission.
"""

import numpy as np
from abc import ABC, abstractmethod
from scipy.spatial import distance_matrix




def airtime(payloadSize, sf, bw=125, codingRate='4/5',
            lowDrOptimize=False, explicitHeader=True,
            preambleLength=8):
    """
    Airtime of LoRa transmission in milliseconds.


    Parameters
    ----------

    payloadSize : int or array_like
        Payload size (bytes). For LoRaWAN this includes the MAC header
        (about 9 bytes when no MAC commands are included),
        the application payload, and the MIC (4 bytes).

    sf : int or array_like
        Spreading factor 6 through 12 (no 6 in LoRaWAN).

    bw : int, optional
       Bandwidth (KHz), typically 125, 250, or 500 KHz.

    codingRate : str, optional
        '4/5', '4/6', '4/7' or '4/8'. For LoRaWAN '4/5'.
        Assumes all packets have the same codingRate.

    lowDrOptimize : bool, optional
        Indicates whether low data rate optiomization is enabled.
        This is usually enabled for low data rates, to avoid
        issues with drift of the crystal reference oscillator
        due to either temperature change or motion. When
        enabled, specifically for 125 kHz bandwidth and SF11
        and SF12, this adds a small overhead to increase
        robustness to reference frequency variations over
        the timescale of the packet.
        Assumes all packets have the same value.

    explicitHeader: bool, optional
        Indicates if the LoRa header is present. This is the
        low-level header that defines, e.g., coding rate,
        payload length and the presence of a CRC checksum.
        In plain LoRa it can be left out if each transmission
        uses the very same parameters and the receiver is
        aware of those. For LoRaWAN, where at least the
        payload length is not fixed, the low-level LoRa
        header is always enabled. Assumes all packets have
        the same value.

    preambleLength : int or array_like, optional
        Number of preamble symbols. Default is 8 (LoRaWAN standard).
        Assumes all packets have the same codingRate.|


    Returns
    -------

    airtime : float or array_like
       The time-on-air in milliseconds.
       
       
    Raises
    ------
    
    ValueError
        If codingRate is not one of '4/5', '4/6', '4/7', or '4/8'.
        
        
    Notes
    -----
    
    This function is translated from on the (javascript)
    airtime-calculator by avbentem [1]_.
    
    Any errors and or ommisions are due to me.

    For use in LoRaWAN, it seems a +13 to payloadSize is needed.
    This mathces our values with the airtime-calculator.
    
    References
    ----------
    .. [1] A. V. Bentem. "Airtime-calculator". Available at https://github.com/avbentem/airtime-calculator
    
    """

    codingRates = {'4/5': 1, '4/6': 2, '4/7': 3, '4/8': 4}

    if not codingRate in codingRates:
        raise ValueError(
            f"Bad CodingRate. Try one of {list(codingRates.keys())}.")

    tSymbol = (2**sf) / bw

    tPreamble = (preambleLength + 4.25) * tSymbol

    h = not explicitHeader  # 0 when explicitHeader, else 1

    de = lowDrOptimize

    if not isinstance(de, bool):

        # allow 1 if 'auto' and additional conditions
        if de == 'auto' and bw == 125 and sf >= 11:
            de = 1

        else:
            de = 0

    cr = codingRates[codingRate]

    symPayload = 8 + np.maximum(np.ceil((8 * payloadSize - 4 *
                                sf + 44 - 20 * h) / (4 * (sf - 2 * de)))
                                * (cr + 4), 0)

    tPayload = symPayload * tSymbol

    return tPreamble + tPayload




#=========================================================
#        Wireless Traffic
#=========================================================

class Traffic():
    """
    A collcetion of LoRa wireless traffic.
    
    Attributes
    ----------
    nPackets : int
        Number of packets in traffic instance.
    start : array-like of float
        Start times of packets.
    airtime : array-like of float
        Time-on-air of packets.
    channel : array-like of int
        Channels of packets.
    sf : array-like of int
        Spreading factors of packets.
    power : array-like of float
        Power of packets.
        This can be taken to be either transmitted or received
        power (usually dBm), depending on context. 

    Notes
    -----
    A Traffic object represents a collection of LoRa
    transmissions, each with associated parameters.

    We take these to represent the packets as observed by
    a receiver. Parameters related to transmission rather
    that reception e.g. transmitted power, are not relevant.
    Instead, factors related to reception and possible failure
    thereof are mandatory.
    
    More variables may be added. The main idea is that any
    ``CollisionModel`` can process ``Traffic``-objects
    to form predictions over successful receptions.
    """

    def __init__(self, nPackets, start, airtime,
                 channel, sf, power):
        """
        Initialize LoRa traffic instance.

        Parameters
        ----------

        nPackets : int
            Number of transmissions (packets).

        start : float or array_like
            Start times of packets; shape = (nPakets, ).

        airtime : float or array_like
            Airtimes of packets; shape = (nPackets, ).

        channel : int or array_like
            Channels of packets, shape = (nPackets, ).

        sf : int or array_like
            Spreading factors; of packets shape (nPackets, ).

        power : float or  array_like
             Powers (dBm) of packets; shape (nPackets, ).

        """
        
        self.nObs = nPackets
        self.start = start
        self.airtime = airtime
        self.channel = channel
        self.sf = sf
        self.power = power
    
    def __repr__(self):
        return f"Traffic({self.__dict__})"
    
    def __str__(self):
        nObs = self.nObs
        return f"Traffic({nObs=})"
    
    def to_numpy(self, copy=False):
        """
        A NumPy ndarray representing the stored data.
        
        Parameters
        ----------
        copy : bool, optional
            Whether data is explicitly copied or possibly
            a view to arrays stored in instance. Default False.
        
        Returns
        -------
        np.ndarray
        """

        result = np.zeros(shape=(self.nObs, 5))
        
        
        for i, vals in enumerate(self.__dict__.values()):
            if i > 0:
                result[:, i-1] = vals
        
        if copy:
            return result.copy()
        
        return result
        
        
        
    
#=========================================================
#        LoRaParameters-object
#=========================================================
    
class LoRaParameters():
    """
    An object class for LoRa Network Parameters.

    Attributes
    ---------
    nChannels : int
        Number of channels permitted.

    freq : float
        (Central) frequency used by network.

    bandwidth : float
        Bandwidth of uplink channel.

    spreadingFactors : list of int
        List of spreading factors allowed on network.

    overhead : int
        Number of bytes of overhead in payload.
        
    maxPow : float
        Maximum transmitted power (dBm). 

    dwellTime : float
        Maximum permitted dwell time in ms. Use `None` if not restricted.

    dutyCycle : float
        Maximum duty cycle permitted. Use `None` if not resttrictred.
        

    Notes
    -----
    
    FUTURE - would be nice to read data from e.g.
        https://github.com/TheThingsNetwork/lorawan-frequency-plans

    For now parameters are specified manually.

    Defaults are for North America ISM band.
    """

    def __init__(self, nChannels=1, freq=915, bandwidth=125,
                 spreading=None, overhead=13, maxPow=30,
                 dwellTime=400, dutyCycle=None, **kwargs):
        """
        Instantiate LoRaParemeters instance.

        Parameters
        ----------
        nChannels : int
            Number of channels permitted, Default 1.

        freq : float
            (Central) frequency used by network, default 915 MHz.

        bandwidth : float
            Bandwidth of uplink channel, default 125 kHz.

        spreadingFactors : list of int
            List of spreading factors allowed on network, default [7, 8, 9, 10].

        overhead : int
            Number of bytes of overhead in payload.
        
        maxPow : float
            Maximum transmitted power (dBm).

        dwellTime : float
            Maximum permitted dwell time in ms. Use `None` if not restricted.

        dutyCycle : float
            Maximum duty cycle permitted. Use `None` if not resttrictred.
        
        """
        
        if spreading is None:
            spreading = [7, 8, 9, 10]

        self.nChannels = nChannels
        self.freq = freq
        self.bw = bandwidth
        self.sf = spreading
        self.overhead = overhead
        self.maxPow = maxPow
        self.dwellTime = dwellTime
        self.dutyCycle = dutyCycle
        
        # save keyword arguments
        self.__dict__.update(kwargs)

    def __repr__(self):
        return "LoRaParameters(%r)" % self.__dict__

    def __str__(self):
        out = "A LoRa Parameters object with params:"
        for k, v in self.__dict__.items():
            out += f"\n\t {k}: {v}"
        return out




#=========================================================
#        Traffic Generators
#=========================================================

class TrafficGenerator(ABC):
    """
    ABC for traffic-generating objects.
    
    Typical ``TrafficGenerators`` are  compositions. Often, a generator
    takes an ``ArrivalProcess`` and a ``ParameterDistribution``.
    ``Traffic``-objects are generated by sampling arrivals and equipping
    these with sampled parameters. Dependence between arrivals and
    parameters should also be supported.
    """
    
    @abstractmethod
    def __call__(self, seed=None):
        """
        Get a sample.
        """
        
class IndependentTraffic(TrafficGenerator):
    """A class for LoRa traffic generation.
    independent sampling of wireless parameters.
    
    Attributes
    ----------
    
    arrivals : ArrivalProcess
        An arrival process from ``loraplan.distributions``.
    parameters : ParameterDistribution
        A distribution over wireless parameters.
    """
    def __init__(self, arrivals=None, paramDist=None, loraParams=None):
        """
        Instantiate an independent LoRa traffic generator object.

        Parameters
        ----------
        arrivals : ArrivalProcess
            An arrival process describing for sampling nPackets and arrival times.

        paramDist : parameterDistribution
            A distribution from which to sample LoRa
            
        loraParams: LoRaWANParameters, optional
            A parameter object against which to verify validity of distributions.
        """
        
        self.arrivals = arrivalProcess
        self.parameters = paramDist
        
    def __call__(self, seed=None):
        pass # TBD

    