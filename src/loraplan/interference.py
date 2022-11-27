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

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections as mc

from loraplan.probability import distributions as lpd
from loraplan.probability import point_processes as lpp
from loraplan.probability import thinning_determinantal, thinning_matern


def airtime(
    payloadSize,
    sf,
    bw=125,
    codingRate="4/5",
    lowDrOptimize=False,
    explicitHeader=True,
    preambleLength=8,
):
    """
    Airtime of LoRa transmission in seconds.


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

    This function is translated from the (javascript)
    airtime-calculator written by avbentem [1]_.

    Any errors and or ommisions are due to me.

    For use in LoRaWAN, it seems a +13 to payloadSize is needed
    (see refereces on avbentem's repo).
    This mathces our values with the airtime-calculator.

    References
    ----------
    .. [1] A. V. Bentem. "Airtime-calculator". Available at https://github.com/avbentem/airtime-calculator

    """

    codingRates = {"4/5": 1, "4/6": 2, "4/7": 3, "4/8": 4}

    if codingRate not in codingRates:
        raise ValueError(f"Bad CodingRate, try {list(codingRates.keys())}.")

    tSymbol = (2**sf) / bw

    tPreamble = (preambleLength + 4.25) * tSymbol

    h = not explicitHeader  # 0 when explicitHeader, else 1

    de = lowDrOptimize

    if not isinstance(de, bool):

        # allow 1 if 'auto' and additional conditions
        if de == "auto" and bw == 125 and sf >= 11:
            de = 1

        else:
            de = 0

    cr = codingRates[codingRate]

    symPayload = 8 + np.maximum(
        np.ceil((8 * payloadSize - 4 * sf + 44 - 20 * h) / (4 * (sf - 2 * de)))
        * (cr + 4),
        0,
    )

    tPayload = symPayload * tSymbol

    return (tPreamble + tPayload) / 1000


# =========================================================
#        Wireless Traffic
# =========================================================


class Traffic:
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

    def __init__(self, nPackets, start, airtime, channel, sf, power):
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
        self.start = np.array(start)
        self.airtime = np.array(airtime)
        self.channel = np.array(channel)
        self.sf = np.array(sf)
        self.power = np.array(power)

    @property
    def midpoint(self):
        """
        Get midpoint of transmissions.
        """
        return self.start + (self.airtime / 2)

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

        result = np.zeros((len(self.__dict__), self.nObs))

        for i, vals in enumerate(self.__dict__.values()):
            if i > 0:
                result[:, i - 1] = vals

        if copy:
            return result.copy()

        return result

    def to_dict(self, addConstant=True):
        """
        Return Traffic data as a dictionary.

        Parameters
        ----------
        addConstant : bool, optional
            Whether to add a data column ``constant`` of ``1``s to dictionary.
            Defaults to ``True``.
        """

        data = self.__dict__.copy()

        if addConstant:
            data["constant"] = np.ones(self.nObs)

        return data

    def thinALOHA(self):
        """
        Get retention labels under ALOHA-style thinning.

        See Also
        --------
        Our module on
        :mod:`Matérn thinning <loraplan.probability.thinning_matern>`.
        """
        if self.nObs == 0:
            return []

        points = np.vstack((self.midpoint, self.channel, self.sf)).T

        radii = self.airtime / 2

        eps = 1.0e-30  # essentially 0 for which 0/0 is 0.

        return thinning_matern.maternThinningI(
            points, radii, metric="seuclidean", V=[1, eps, eps]
        )

    def thinDeterminantal(self, ensemble, params):
        """
        Sample retention labels under given determinantal thinning model.

        Determinantal thinning takes a determinantal model, in the for mof an
        L-ensemble ``enesemble``, and is parameterers ``params`` and samples
        the retained packets based on the model. It is assumed that the model
        exclusively uses variables that are available in the Traffic-object's
        ``self.__dict__``. If not, an error is raised.

        Parameters
        ----------
        ensemble : loraplan.probability.determinantal_thinning.EllEnsemble
            An L-ensemble objcet defined over variables in Traffic.
        params : dict
            Parameter dictionary passed to L-ensemble.

        See Also
        --------
        Our module on
        :mod:`determinantal thinning <loraplan.probability.thinning_determinantal>`
        for a more detailed description.
        """
        if self.nObs == 0:
            return []
        return ensemble.sample(self.to_dict(), params)

    def plot(
        self, y_variable="index", labels=None, text=False, linewidths=25, **kwargs
    ):
        """
        Plot wireless Traffic.

        Parameters
        ----------
        y_variable : str, optional
            The variable in ``Traffic.to_dict()`` to be used as y-axis location.
            The default ``'index'`` gives each packet a unique y-position.

        labels : array_like of bools
            Labels for whether the transmission is successful or not.

        text : bool, optional
            Whether text containing additional packet information is plotted.
            For wide time-windows, or Traffic including many (> 20) packets
            this is best set to ``False`` or the text will likely overlap.

        linewidth : int, optional
            Sets the vertical width (height) of the packet lines.

        kwargs
            Passed to ``plt.subplots()``

        """

        # collect data
        data = self.to_dict()

        data["index"] = list(range(data["nObs"]))

        starts = list(zip(data["start"], data[y_variable]))
        ends = list(zip(data["start"] + data["airtime"], data[y_variable]))

        packets = [[s, t] for (s, t) in zip(starts, ends)]

        # color by label if present
        if hasattr(labels, "__len__"):
            cmap = {True: "cornflowerblue", False: "darkorange"}
            color = [cmap[label] for label in labels]
        else:
            color = "cornflowerblue"

        # build plot
        lc = mc.LineCollection(packets, linewidths=linewidths, color=color, alpha=0.4)

        fig, ax = plt.subplots(**kwargs)
        ax.add_collection(lc)

        # scale plot and add labels
        ax.autoscale()
        ax.set_ylabel(f"{y_variable}")
        ax.set_xlabel("time (s)")

        # get range of x-values
        x_min = data["start"].min()
        x_max = (data["start"] + data["airtime"]).max()

        # plot additional information
        if text:
            for p in range(data["nObs"]):
                x, y = data["start"][p], data[y_variable][p]
                # construct text
                text = f"ch={data['channel'][p]}, sf={data['sf'][p]}"
                offset = +0.02 * (x_max - x_min)
                plt.text(x + offset, y, text, color="k", fontsize=12)

        ax.margins(0.25)

        return fig, ax


# =========================================================
#        LoRaParameters-object
# =========================================================


class LoRaParameters:
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

    def __init__(
        self,
        nChannels=1,
        freq=915,
        bandwidth=125,
        spreading=None,
        overhead=13,
        maxPow=30,
        dwellTime=0.4,
        dutyCycle=None,
        codingRate="4/5",
        **kwargs,
    ):
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

        codingRate : str, optional
            '4/5', '4/6', '4/7' or '4/8'. LoRaWAN uses '4/5'.

        dwellTime : float
            Maximum permitted dwell time in s. Use `None` if not restricted.

        dutyCycle : float
            Maximum duty cycle permitted. Use `None` if not resttrictred.

        Raises
        ------
        ValueError
            If Coding rate not one of '4/5', '4/6', '4/7', '4/8'.

        """

        codingRates = {"4/5": 1, "4/6": 2, "4/7": 3, "4/8": 4}

        if not codingRate in codingRates:
            raise ValueError(f"Bad CodingRate. Try one of {list(codingRates.keys())}.")

        if spreading is None:
            spreading = [7, 8, 9, 10]

        self.nChannels = nChannels
        self.freq = freq
        self.bw = bandwidth
        self.sf = spreading
        self.overhead = overhead
        self.maxPow = maxPow
        self.codingRate = codingRate
        self.dwellTime = dwellTime
        self.dutyCycle = dutyCycle

        # save keyword arguments
        self.__dict__.update(kwargs)

    # def __repr__(self):
    #    return "LoRaParameters(%r)" % self.__dict__

    def __repr__(self):
        out = "LoraParameters("
        for key, val in self.__dict__.items():
            out += f"\n\t {key}: {val}"
        out += ")"
        return out


# =========================================================
#        Traffic Generators
# =========================================================


class TrafficGenerator(ABC):
    """
    ABC for traffic-generating objects.

    Typical ``TrafficGenerators`` are  compositions. Often, a generator
    takes an ``ArrivalProcess`` and a ``ParameterDistribution``.
    ``Traffic``-objects are generated by sampling arrivals and equipping
    these with sampled parameters. Dependence between arrivals and
    parameters should also be supported.

    Notes
    -----
    ``Traffic``-objects are market point processes. As such, each sample
    has a variable number of observations. This makes it difficult to
    store samples as arrays. Instead, a sample of ``size = k`` is a list
    of length k, in which each entry is an array of shape ``(nObs[k], d)``
    where ``d`` is the number of parameters. Due to this, much of the code
    includes list comprehensions.
    """

    @abstractmethod
    def sample(self):
        pass

    def __call__(self, seed=None, **kwargs):
        """
        Get a sample.
        """
        sample = self.sample(seed=None, **kwargs)


class IndependentLoRaGenerator(TrafficGenerator):
    """
    Composition of a TimeWindow, LoRaParameters, and (parameter) Distributions.

    The ``IndependentLoRaGenerator`` is a ``TrafficGenerator``-object for when
    parameters are distributed independently of each other and of the arrivals.
    The main purpose of this object is to collect various model components in one
    place. In particular, an ``ArrivalProcess``, ``LoRaParameters`` and
    ``Distribution``-objects over the various wireless parameters.


    Attributes
    ----------
    arrivals : ArrivalProcess
        An arrival process for generating nr. of packets and arrival times.

    params : LoRaParameters
        A LoRaParameters-object from which essential parameters are pulled.

    channelDist : Distribution
        A distribution over LoRa channels

    spreadingDist : Distribution
        A distirbution over spreading factors

    payloadDist : Distribution
        A distribution over payload length in bytes.

    powerDist : Distribution
        A distribution over received transmission power.
        If transmissions are equipped with link parameters
        one can change this for `transmittedPower - pathLoss`.

    """

    def __init__(
        self, arrivals, params, channelDist, spreadingDist, payloadDist, powerDist
    ):
        """
        Initialize LoRa traffic generator with independent parameter distributions.
        """

        self.arrivals = arrivals
        self.params = params
        self.channelDist = channelDist
        self.spreadingDist = spreadingDist
        self.payloadDist = payloadDist
        self.powerDist = powerDist

    def __repr__(self):
        out = f"IndependentLoRaGenerator("
        for key, val in self.__dict__.items():
            out += f"\n\t{key}: {val}"
        out += ")"
        return out

    @classmethod
    def from_parameters(
        cls,
        params,
        tMin=0,
        tMax=1,
        buffer=1,
        rate=3,
        p_channel=None,
        p_spread=None,
        a_payload=11,
        p_payload=None,
        loc_pow=-80,
        scale_pow=10,
    ):
        """
        Make and IndependentLoRaGenerator from LoRaParameters.

        This is a convenience function that makes the simplest distribution
        over a LoRaParameters object, with some additions.

        Parameters
        ----------
        params : LoRaParameters
        tMin : float
        tMax : float
        buffer : float
        rate : int
        p_channel : array_like of floats
        p_spread : array_like of floats
        a_payload : int, or array_like of ints
        p_payload : array_like of floats
        loc_pow : float
        scale_pow : float

        """

        if not isinstance(params, LoRaParameters):
            raise ValueError("params must be a LoRaParameters-object.")

        # default arrival process and param distributions
        window = lpp.TimeWindow(tMin, tMax, buffer=buffer)
        arrivals = lpp.PoissonArrivals(timeWindow=window, rate=rate)

        # channel distribution
        channelDist = lpd.Choice(params.nChannels, p=p_channel)

        # sperading factor distirbution
        spreadingDist = lpd.Choice(params.sf, p=p_spread)

        # payload distribution
        payloadDist = lpd.Choice(a=a_payload, p=p_payload)

        # a power distribution
        powerDist = lpd.Normal(loc=loc_pow, scale=scale_pow)

        return cls(arrivals, params, channelDist, spreadingDist, payloadDist, powerDist)

    def sample(self, size=None, seed=None, **kwargs):
        """
        Sample LoRa wireless traffic.

        The sample first queries the ArrivalProces for the number of packets and
        their individual arrival times. Given the number of packets nObs, the
        remaining distributions are sampled to compile a matrix of parameter data.
        The airtime is computed last using ``interference.airtime()`` and stored
        network-level parameters ``self.parameters`` as well as packet-level
        parameters such as the spreading factor ``SF`` and ``payload``.

        See Also
        --------
        ``loraplan.interference.airtime``

        """

        arrivals = self.arrivals.sample(size=size, **kwargs)

        N = len(arrivals)  # N can be `None` up to here

        nObs = [arr.shape[0] for arr in arrivals]

        # sample prameters from respective distributions
        channels = [self.channelDist(size=n) for n in nObs]
        spreadings = [self.spreadingDist(size=n) for n in nObs]
        powers = [self.powerDist(size=n) for n in nObs]
        payloads = [self.payloadDist(size=n) for n in nObs]

        # compute airtimes
        cr = self.params.codingRate
        bw = self.params.bw
        oh = self.params.overhead

        airtimes = [
            airtime(payloads[i] + oh, spreadings[i], bw=bw, codingRate=cr)
            for i in range(N)
        ]

        # form Traffic-objects
        sample = [
            Traffic(
                nObs[i], arrivals[i], airtimes[i], channels[i], spreadings[i], powers[i]
            )
            for i in range(N)
        ]

        if len(sample) == 1:
            return sample[0]
        else:
            return sample
