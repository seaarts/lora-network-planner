import numpy as np
from abc import ABC, abstractmethod
from scipy.spatial import distance_matrix



#=========================================================
#        LoRa / LoRaWAN utilities
#=========================================================


def airtime(payloadSize, sf, bw=125, codingRate='4/5',
            lowDrOptimize=False, explicitHeader=True,
            preambleLength=8):
    """
    Calculate the LoRa airtime in milliseconds.


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
#        Modeling Wireless Traffic
#=========================================================


class Traffic():
    """
    A collcetions of LoRa wireless traffic.
    
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
    CollisionModel can process LoRaTraffic instances and form
    predictions over successful receptions.
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
        # WANT: Reference to the distribution that generated it.
        
        return f"Traffic(nPackets={self.nObs})"


#=========================================================
#        Simulation tools and utilities
#=========================================================


class TimeWindow():
    """A time interval.
    
    Attributes
    ----------
    tMin : float
        The left-hand end point of the time interval.
    tMax : float
        The right-hand end point of the time interval.
    buff : float
        Number of time units by which to buffer the interval
        in both left and right directions.
        
    """
    
    def __init__(self, tMin, tMax, buffer=0):
        """Specify a time-window by start and end points.
        
        Parameters
        ----------
        tMin : float
            The left-hand end point of the time interval.
        tMax : float
            The right-hand end point of the time interval.
        buff : float, optional
            Number of time units by which to buffer the interval
            in both left and right directions. Default 0.
            
        Raises
        ------
        ValueError
            If right hand end-point smaller than left-hand one.
        """
        
        if tMin >= tMax:
            
            raise ValueError("Init failed. tMin >= tMax.")
            
        
        self.tMin = tMin
        self.tMax = tMax
        self.buff = buffer
        
    def __repr__(self):
        return f"TimeWindow({self.tMin}, {self.tMax}, buffer={self.buff})"
    
    @property
    def length(self):
        """
        Length of TimeWindow including buffers.
        """
        
        return self.tMax - self.tMin + 2*self.buff
    
    
    def contains(self, x):
        """
        Check whether time window contains given point.
        
        Parameters
        ----------
        x : float or array_like
            Position of a point on real line.
            
        Returns
        -------
        contains : bool
            Indicates whether x in contained in TimeWindow
        """
        
        return (self.tMin <= x) * (x <= self.tMax)
   
    
    def uniform(self, size=None, seed=None):
        """
        Sample points uniformly over buffered time window.
        
        Parameters
        ----------
        size : int or tuple of ints, optional
            Shape of uniform random vector.
            Passed to `np.random.unfiorm()`
        seed : int, optional
            Seed passed to `np.random.default_rng()`.
            
        Returns
        -------
        u : ndarray or scalar
            
        """
        
        rng = np.random.default_rng(seed=seed)
        
        low = self.tMin - self.buff
        high = self.tMax + self.buff
        
        return rng.uniform(low=low, high=high, size=size)
    
    
class ArrivalProcess(ABC):
    """Stochastic arrival process on TimeWindow"""
    
    def __init__(self):
        """
        Instantiate arrival process.
        """
    
    @abstractmethod
    def sample(self):
        pass
    

class PoissonProcess(ArrivalProcess):
    """Homogeneous Poisson arrival processes
    
    
    Attributes
    ----------
    rate : float
        Poisson arrival rate.
    """
    
    def __init__(self, TimeWindow, rate=None):
        """Instantiate homogeneous Poisson process.
        
        Parameters
        ----------
        timeWindow : timeWindow
             Time window over which to simulate process.
        
        rate : float
             Poisson arrival rate.
             
             
        """
        super().__init__()
        self.timeWindow = TimeWindow
        self.rate = rate
        
    def __repr__(self):
        return f"HomogeneousPoissonProcess({self.timeWindow}, rate={self.rate})"
        
    def sample(self, size=None, rate=None, seed=None, stream=None):
        """
        Sample Poisson process over TimeWindow.
        
        Parameters
        ----------
        
        rate : float, optional
            Poisson arrival rate. May be `None` if stored in `ArrivalProcess.rate`.
        
        size : int, optional
            The number of samples to be taken. Limited to integer
            rather than tuples, because each sample may vary in
            size. Default 1.
            
        stream : list of np.random.generators, optional
            List of generators for the individual Process samples.
            Passed as seeds to to TimeWindow.sample(). This enables
            the use of Common Random Numbers in placing the points,
            when e.g. the arrival rates are varied over multiple
            experiments. Default `None`.
            
            
        Returns
        -------
        samples : list of ndarrays
            The length of `samples` is `size`.
            
            
        Raises
        ------
        ValueError
            If `self.rate` and `rate` are `None`.
        """
        
        rng = np.random.default_rng(seed=seed)
           
        
        if not rate and self.rate:
            rate = self.rate
        else:
            raise ValueError("""No rate given. Try passing`rate=1.0`,\
 or set a rate for the ArrivalProcess."""
                            )
       
        if size is None:
            size = 1
    
        if stream is None:
            stream = [rng for i in range(size)]  # pointers to same rng
        
        
        window = self.timeWindow
        
        mu = window.length * rate
        
        nPoints = np.random.poisson(mu, size=size)
        
        
        return [window.uniform(nPoints[i], seed=stream[i]) for i in range(size)]