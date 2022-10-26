import numpy
from abc import ABC, abstractmethod

class LogLinearPathLoss(ABC):
    """Generic log-linear path loss models.
    """ 
    
    def __init__(self, name, const, coeff):
        """
        Initialize instance of log-linear path loss model.
        
        Parameters
        ----------
        name : str
        const : foat
            Constant used in linear model
        coeff : float
            Coefficient used in linear model
        
        Notes
        -----
        Log Linear Path Loss models are of the form
            PL(dist) = const + coef*log(dist)
        """
        
        self.name = name
        self.const = const
        self.coeff = coeff
    
    def __call__(self, dist):
        """Evaluate path loss given distance.
        
        Parameters
        ----------
        dist : float
            Distance of link in meters.
        """
        return self.const + self.coeff * np.log10(dist) 


class FreeSpacePathLoss(LogLinearPathLoss):
    """ A free-space path loss model.
    
    Notes
    -----
    Add reference!
    """
    def __init__(self, freq):
        """Initialize free-space path loss for given frequency.
        
        Parameters
        ----------
        frequency : float
            Frequency of transmission in MHz.
        """
        
        # compute constant and coefficient
        const = 20 * np.log10(freq) - 27.55
        coeff = 20
        
        # initialize parent class
        super().__init__("Free-space path loss", const, coeff)
        
        self.freq = freq
        
    @property
    def frequency(self):
        """Return frequency used in model
        """
        return self.freq    


class HataPathLoss(LogLinearPathLoss):
    """
    A Hata Path loss model.
    
    Notes
    -----
    The Hata model is an empirical model designed for urban settings.
    
    Read more about the Hata path loss model at `https://en.wikipedia.org/wiki/Hata_model`.
    """
    def __init__(self, freq=915, h_rx=30, h_tx=1.5, kind="urban", city_size="small"):
        """
        Initialize Hata model.
        
        Parameters
        ----------
        frequency : float,
            Transmission frequency (MHz)
        h_rx : float
            Receiver height (m)
        h_tx : float
            Transmitter height (m)
        kind : str, optional
            Categorical either "rural", "suburban", or "urban".
            Specifies the type of terrain the model covers.
        city_size : str, optional
            Categorical, either "small", "medium", or "large".
            Specifies the size of the city modeled.
        """
        
        if kind not in ["open", "suburban", "urban"]:
            raise ValueError('kind must be "rural", "suburban", or "urban".')
        
        if city_size not in ["small", "medium", "large"]:
            raise ValueError('city_size must be "small", "medium" or "large".')
        
        
        ### compute constant
        const = 69.55 + 26.16 * np.log10(freq) - 13.82 * np.log10(h_rx)
        
        # height correction (open, suburban, small cities, medium cities)
        corr = 0.8 + h_tx * (1.1 * np.log10(freq) - 0.7) - 1.56 * np.log10(freq)
        
        # adjust antenna height correction factor
        if kind == "urban" and city_size == "large":
            if freq <= 200:
                corr = 8.5*np.log10(1.54*h_tx)**2 - 1.1 
            elif freq > 200:
                corr = 3.2*np.log10(11.75*h_tx)**2 - 4.97
            
        const += corr # add correction factor to constant
        
        # environment type correction
        if kind == "suburban":
            # compute path loss reduction for suburban
            reduction = 2*np.log10(freq/28)**2 + 5.4
            const -= reduction
            
        elif kind == "open":
            # compute path loss recution for open space
            reduction = 4.78*np.log10(freq)**2 - 18.33*np.log10(freq) + 40.94
            const -= reduction
        

        coeff = (44.9 - 6.55*np.log10(h_rx))
        
        const -= 3*coeff # as we use m and not km
        
        
        # initialize parent class
        super().__init__("Hata path loss (%s)" % kind, const, coeff)
        
        self.freq = freq
        self.h_tx = h_tx
        self.h_rx = h_rx
        self.kind = kind
        self.city_size = city_size
    
    
    @property
    def frequency(self):
        """Return frequency used in model
        """
        return self.freq
    
    
class IndoorPathLoss(LogLinearPathLoss):
    """
    ITU indoor path loss.
    
    Notes
    -----
    An indoor path loss models. Read more on wikipedia at
    `https://en.wikipedia.org/wiki/ITU_model_for_indoor_attenuation`.
    """
    
    def __init__(self, freq, n_floors, kind='office'):
        """
        Initialize model for given frequency and number of floors.
        
        Parameters
        ----------
        freq :  float
            Transmission frequency (MHz)
        nFloors : int
            (Typical) number of floors to penertrate. Should not exceed 3.
        kind : str, optional
            Kind of building modeled. Either "office" or "commercial".
        """
        
        if not isinstance(nFloors, int):
            raise ValueError("nFloors should be integer")
        if nFloors < 0:
            raise ValueError("nFloors must be non-negative")
        if n_floors > 3:
            raise ValueError("nFloors must not exceed 3")
            
        if kind not in ["office", "commercial"]:
            raise ValueError("The `kind` must be either 'office' or 'commercial'")
            
        ###  only implemented for ca 900 MHz
        
        # floor penetration factor and coefficient
        if kind == 'commercial':
            # no penetration, smaller coefficient
            penetration = 0
            coeff = 20
            
        elif kind == 'office':
            # floor-pependent penetration, large coefficients
            penetration_map = {0:0, 1:9, 2:19, 3:24}
            penetration = penetration_map[n_floors]
            coeff = 33
        
        const = 20*np.log10(freq) + penetration - 28
        
        # initialize parent class
        super().__init__("ITU indoor path loss (%s)" % kind, const, coeff)
        
        self.freq = freq
        self.n_floors = n_floors
        self.kind = kind