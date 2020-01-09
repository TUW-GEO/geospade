import re
import warnings
import osr
from pyproj.crs import CRS
from cartopy import crs as ccrs
from shapely.geometry import LineString


class SpatialRef():
    """
    This class represents any OGC compliant spatial reference system. Internally, the
    GDAL OSR Spatial Reference class is used, which offers access to and control over
    different representations, such as EPSG, PROJ4 or WKT. Additionally, it can also
    create an instance of a Cartopy projection subclass (PROJ4Projection), which
    can be used to plot geometries or data.
    """

    def __init__(self, arg, sref_type=None):
        """
        Constructs a ˋSpatialRefˋ instance based on ˋargˋ.
        If the type is not provided by the type argument ˋsref_typeˋ, the constructor tries
        to determine the type itself.:


        Parameters
        ----------
        arg: int or dict or str
            Represents a given spatial reference system. Regarding spatial reference type determination,
            different cases are distinguished:
            - If ˋargˋ is an integer, it tries to interpret it as an EPSG Code (e.g. 4326).
            - If ˋargˋ is a dict, it assumes that PROJ4 parameters are given as input
              (e.g. {'proj': 'longlat', 'ellps': 'WGS84', 'datum': 'WGS84', 'no_defs' : True}).
            - If ˋargˋ is a string, the constructor tries to check whether it contains an
              'EPSG' prefix (=> EPSG, e.g., 'EPSG: 4326'), a plus '+' (=> PROJ4, e.g.,
              '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs') or 'GEODCS' (=> WKT, e.g.,
              'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],
              AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",
              0.01745329251994328,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]').
        sref_type: str, optional
            String defining the type of ˋargˋ. It can be: 'proj4', 'wkt' or 'epsg'.
            If it is None, the spatial reference type of ˋargˋ is guessed.
        """

        self.osr_sref = osr.SpatialReference()

        if sref_type is None:
            if isinstance(arg, int):  # integer is interpreted as EPSG
                sref_type = 'epsg'
            elif isinstance(arg, dict):  # dictionary representing PROJ4 parameters
                sref_type = 'proj4'
            elif isinstance(arg, str):  # string can be EPSG, PROJ4 or WKT
                if arg.lower().startswith('epsg'):
                    sref_type = 'epsg'
                elif '+' == arg[0]:  # first character is '+' => PROJ4
                    sref_type = 'proj4'
                elif 'GEOGCS[' in arg:  # there is a GEOGCS tag in the given string => WKT
                    sref_type = 'wkt'
            else:
                err_msg = "Spatial reference type for '{}' is unknown.".format(arg)
                raise ValueError(err_msg)

        # internal variables to store the different spatial reference representations
        self._proj4 = None
        self._epsg = None
        self._wkt = None
        # set the spatial reference types
        if sref_type == 'proj4':
            self.proj4 = arg
        elif sref_type == 'epsg':
            self.epsg = arg
        elif sref_type == 'wkt':
            self.wkt = arg
        else:
            err_msg = "Spatial reference type '{}' is unknown. Use 'epsg', 'wkt' or 'proj4'."
            raise Exception(err_msg.format(sref_type))

        # defines a reference spatial reference representation, which is necessary for checking the consistency of
        # transformations between different representations
        self._sref_type = sref_type

    @classmethod
    def from_osr(cls, osr_sref):
        """
        Creates a `SpatialRef` object from an OSR spatial reference object. To allow this transformation,
        the PROJ4 is used.

        Parameters
        ----------
        osr_sref: osr.SpatialReference
            OSR spatial reference.

        Returns
        -------
        SpatialRef
            Spatial reference defined by the exported PROJ4 string from an OSR spatial reference object.
        """

        proj4_string = osr_sref.ExportToProj4()
        return cls(proj4_string, sref_type='proj4')

    @property
    def proj4(self):
        """
        str: PROJ4 string representation
        """

        if self._proj4 is None:
            success = self._check_conversion("proj4")
            if success:
                self._proj4 = self.osr_to_proj4(self.osr_sref)
        return self._proj4

    @proj4.setter
    def proj4(self, proj4_params):
        """
        Sets internal spatial reference from PROJ4 parameters.

        Parameters
        ----------
        proj4_params: str or dict
            PROJ4 parameters as a string (e.g., '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs') or a dictionary
            (e.g., {'proj': 'longlat', 'ellps': 'WGS84', 'datum': 'WGS84', 'no_defs' : True})
        """

        self.osr_sref = self.proj4_to_osr(proj4_params)
        self._sref_type = 'proj4'
        if self._proj4 != self.proj4:
            self._proj4 = self.proj4
            self._epsg = self.epsg
            self._wkt = self.wkt

    @property
    def epsg(self):
        """
        int: EPSG code representation as an integer.
        """

        if self._epsg is None:
            success = self._check_conversion("epsg")
            if success:
                self._epsg = self.osr_to_epsg(self.osr_sref)

        return self._epsg

    @epsg.setter
    def epsg(self, epsg_code):
        """
        Sets internal spatial reference from an EPSG code.

        Parameters
        ----------
        epsg_code: int or str
            EPSG Code as a string (e.g., 'EPSG: 4326') or integer (e.g., 4326).
        """

        self.osr_sref = self.epsg_to_osr(epsg_code)
        self._sref_type = 'epsg'
        if self._epsg != self.epsg:
            self._epsg = self.epsg
            self._wkt = self.wkt
            self._proj4 = self.proj4

    @property
    def wkt(self):
        """
        str: Well Known Text (WKT) representation of the spatial reference without tabs or line breaks.
        """

        if self._wkt is None:
            success = self._check_conversion("wkt")
            if success:
                self._wkt = self.osr_to_wkt(self.osr_sref)

        return self._wkt

    @wkt.setter
    def wkt(self, wkt_string):
        """
        Sets internal spatial reference from a Well Known Text (WKT) string.

        Parameters
        ----------
        wkt_string: str
            WKT string, e.g., 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,
            AUTHORITY["EPSG","7030"]], AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],
            UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'.
        """

        self.osr_sref = self.wkt_to_osr(wkt_string)
        self._sref_type = 'wkt'
        if self._wkt != self.wkt:
            self._wkt = self.wkt
            self._epsg = self.epsg
            self._proj4 = self.proj4

    def to_pretty_wkt(self):
        """
        str: Well Known Text (WKT) representation of the spatial reference formatted with tabs and line breaks.
        """

        return self.osr_sref.ExportToPrettyWkt()

    def to_proj4_dict(self):
        """
        dict: Converts internal PROJ4 parameter string to a dictionary, where the keys do not contain a plus and
        the values are converted to non-string values if possible.
        """

        return self._proj4_str_to_dict(self.proj4)

    def to_cartopy_crs(self, bounds=None):
        """
        Creates a PROJ4Projection object that can be used as an argument of the
        cartopy `projection` and `transfrom` kwargs. (`PROJ4Projection` is a
        subclass of a `cartopy.crs.Projection` class)

        Parameters
        ----------
        bounds: 4-tuple, optional
            Boundary of the projection (lower left x, upper right x,
                                        lower left y, upper right y).

        Returns
        -------
        PROJ4Projection
            `PROJ4Projection` instance representing the spatial reference.
        """

        return PROJ4Projection(self.proj4, bounds=bounds)

    @staticmethod
    def osr_to_proj4(osr_sref):
        """
        Converts an `osr.SpatialReference` instance to a PROJ4 string.

        Parameters
        ----------
        osr_sref: osr.SpatialReference
            OSR spatial reference.

        Returns
        -------
        str
            PROJ4 string.
        """

        return osr_sref.ExportToProj4()[:-1]

    @staticmethod
    def proj4_to_osr(proj4_params):
        """
        Converts PROJ4 parameters to an OSR spatial reference object.

        Parameters
        ----------
        proj4_params: str or dict
            PROJ4 parameters as a string (e.g., '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs') or a dictionary
            (e.g., {'proj': 'longlat', 'ellps': 'WGS84', 'datum': 'WGS84', 'no_defs' : True})

        Returns
        -------
        osr.SpatialReference
            OSR spatial reference.
        """

        # convert to string because GDAL takes proj4  as string only
        if isinstance(proj4_params, dict):
            arg = ''
            for k, v in proj4_params.items():
                arg += '+{}={} '.format(k, v)
        elif isinstance(proj4_params, str):
            if '+' == proj4_params[0]:
                arg = proj4_params
            else:
                err_msg = "'{}' does not start with/contain a plus, which is mandatory for a Proj4 string."
                raise Exception(err_msg)
        else:
            err_msg = "Proj4 parameters have to be either given as a string or a dict (see docs)."
            raise Exception(err_msg)

        osr_sref = osr.SpatialReference()
        osr_sref.ImportFromProj4(arg)

        return osr_sref

    @staticmethod
    def osr_to_epsg(osr_sref):
        """
        Converts an `osr.SpatialReference` instance to an EPSG code.

        Parameters
        ----------
        osr_sref: osr.SpatialReference
            OSR spatial reference.

        Returns
        -------
        int
            EPSG code.
        """

        osr_sref.AutoIdentifyEPSG()
        epsg_code_str = osr_sref.GetAuthorityCode(None)
        epsg_code = None if epsg_code_str is None else int(epsg_code_str)
        return epsg_code

    @staticmethod
    def epsg_to_osr(epsg_code):
        """
        Converts an EPSG code to an OSR spatial reference object.

        Parameters
        ----------
        epsg_code: int or str
            EPSG Code as a string (e.g., 'EPSG:4326') or integer (e.g., 4326).

        Returns
        -------
        osr.SpatialReference
            OSR spatial reference.
        """

        if isinstance(epsg_code, int):
            arg = epsg_code
        elif isinstance(epsg_code, str) and epsg_code.lower().startswith('epsg'):
            epsg_code_fndngs = re.findall(r'\d+')  # extract the numerical value
            if 4 <= len(epsg_code_fndngs[0]) <= 5:  # correct length of EPSG code
                arg = epsg_code_fndngs[0]
            else:
                err_msg = "'{}' is not an EPSG conform string. Either use the EPSG code as an integer or as a string, e.g. 'EPSG:4326'"
                raise Exception(err_msg.format(epsg_code))
        else:
            err_msg = "The EPSG code has to be either given as a string or an integer (see docs)."
            raise Exception(err_msg)

        osr_sref = osr.SpatialReference()
        osr_sref.ImportFromEPSG(arg)

        return osr_sref

    @staticmethod
    def osr_to_wkt(osr_sref):
        """
        Converts an `osr.SpatialReference` instance to a  Well Known Text (WKT) string.

        Parameters
        ----------
        osr_sref: osr.SpatialReference
            OSR spatial reference.

        Returns
        -------
        str
            WKT string.
        """

        return osr_sref.ExportToWkt()

    @staticmethod
    def wkt_to_osr(wkt_string):
        """
        Converts a Well Known Text (WKT) string to an OSR spatial reference object.

        Parameters
        ----------
        wkt_string: str
            WKT string, e.g., 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,
            AUTHORITY["EPSG","7030"]], AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],
            UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'.

        Returns
        -------
        osr.SpatialReference
            OSR spatial reference.
        """

        if isinstance(wkt_string, str):
            if 'GEOGCS[' in wkt_string:
                arg = wkt_string
            else:
                err_msg = "'{}' is not a valid WKT string."
                raise Exception(err_msg)
        else:
            err_msg = "The argument has to be provided as a string."
            raise ValueError(err_msg)

        osr_sref = osr.SpatialReference()
        osr_sref.ImportFromWkt(arg)

        return osr_sref

    def __convert_proj4_pairs_to_dict(self, proj4_pairs):
        """
        Converts PROJ4 parameters to floats if possible.

        Parameters
        ----------
        proj4_pairs: list of tuples
            List of (key, value) pairs coming from a PROJ4 object, e.g., [('proj', 'longlat'), ('ellps', 'WGS84')].

        Returns
        -------
        dict:
            Dictionary containing PROJ4 parameters.
        """

        proj4_dict = dict()
        for x in proj4_pairs:
            if len(x) == 1 or x[1] is True:
                proj4_dict[x[0]] = True
                continue
            try:
                proj4_dict[x[0]] = float(x[1])
            except ValueError:
                proj4_dict[x[0]] = x[1]

        return proj4_dict

    def _proj4_str_to_dict(self, proj4_string):
        """
        Converts PROJ4 compatible string to dictionary.

        Parameters
        ----------
        proj4_string: str
            PROJ4 parameters as a string (e.g., '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs').

        Returns
        -------
        dict:
            Dictionary containing PROJ4 parameters.

        Notes
        -----
        Key only parameters will be assigned a value of `True`.
        EPSG codes should be provided as "EPSG:XXXX" where "XXXX"
        is the EPSG number code. It can also be provided as
        "+init=EPSG:XXXX" as long as the underlying PROJ library
        supports it (deprecated in PROJ 6.0+).
        """

        # convert EPSG codes to equivalent PROJ4 string definition
        if proj4_string.lower().startswith('epsg:'):
            crs = CRS(proj4_string)
            return crs.to_dict()
        else:
            proj4_pairs = (x.split('=', 1) for x in proj4_string.replace('+', '').split(" "))
            return self.__convert_proj4_pairs_to_dict(proj4_pairs)

    def _check_conversion(self, sref_type):
        """
        Checks whether two spatial reference types can be neatly transformed in both directions
        (e.g., WKT <-> EPSG).

        Parameters
        ----------
        sref_type: str
            String defining the spatial reference type to check. It can be: 'proj4', 'wkt' or 'epsg'.

        Returns
        -------
        bool
            If False, the bijective conversion between two spatial reference types is not possible.
        """

        sref_types = ["proj4", "wkt", "epsg"]
        srefs_to = {'proj4': lambda x: SpatialRef.osr_to_proj4(x),
                    'wkt': lambda x: SpatialRef.osr_to_wkt(x),
                    'epsg': lambda x: SpatialRef.osr_to_epsg(x)}
        srefs_from = {'proj4': lambda x: SpatialRef.proj4_to_osr(x),
                      'wkt': lambda x: SpatialRef.wkt_to_osr(x),
                      'epsg': lambda x: SpatialRef.epsg_to_osr(x)}
        warn_msg = "Transformation between '{}' and '{}' is not bijective."

        if sref_type not in sref_types:
            err_msg = "Spatial reference type '{}' is unknown. Use 'epsg', 'wkt' or 'proj4'."
            raise ValueError(err_msg.format(sref_type))

        if self._sref_type != sref_type:
            this_sref_val = srefs_to[self._sref_type](self.osr_sref)
            other_sref_val = srefs_to[sref_type](self.osr_sref)
            if other_sref_val is None:
                warnings.warn(warn_msg.format(self._sref_type.upper(), sref_type.upper()))
                return False
            # do forth and back-conversion
            other_sref = srefs_from[sref_type](other_sref_val)
            this_sref_val_check = srefs_to[self._sref_type](other_sref)
            if this_sref_val != this_sref_val_check:
                warnings.warn(warn_msg.format(self._sref_type.upper(), sref_type.upper()))
                return False
            else:
                return True
        else:
            return True

    def __eq__(self, other):
        """ Checks if this and another `SpatialRef` object are equal according to their PROJ4 strings. """
        return self.proj4 == other.proj4

    def __ne__(self, other):
        """ Checks if this and another `SpatialRef` object are unequal according to their PROJ4 strings. """
        return not self == other


class PROJ4Projection(ccrs.Projection):
    """
    This class represents any cartopy projection based on its PROJ4 parameters.
    Instances of this class can be parsed as 'projection' and 'transform' kwargs
    as regular Cartopy projections, because it is a subclass of a
    `cartopy.crs.Projection` class.
    """

    # proj4 to cartopy.crs.Globe parameter dictionary
    _GLOBE_PARAMS = {'datum': 'datum',
                     'ellps': 'ellipse',
                     'a': 'semimajor_axis',
                     'b': 'semiminor_axis',
                     'f': 'flattening',
                     'rf': 'inverse_flattening',
                     'towgs84': 'towgs84',
                     'nadgrids': 'nadgrids'}

    def __init__(self, terms, globe=None, bounds=None):
        """
        Creates an instance from PROJ4 parameters, globe and projection boundaries.

        Parameters
        ----------
        terms : dict
            Dictionary containing PROj4 parameters of a projection.
        globe : cartopy.crs.Globe, optional
            A Cartopy `Globe` instance. If omitted, constructor creates it from the terms itself.
        bounds : 4-tuple
            Boundary of the projection (lower left x, upper right x, lower left y, upper right y)
        """

        globe = self._globe_from_proj4(terms) if globe is None else globe

        other_terms = []  # terms that are not defining the datum / globe
        for term in terms.items():
            if term[0] not in self._GLOBE_PARAMS:
                other_terms.append(term)
        super(PROJ4Projection, self).__init__(other_terms, globe)

        self.bounds = bounds

    def __repr__(self):
        """ String representation of `PROJ4Projection`. """
        return 'PROJ4Projection({})'.format(self.proj4_init)

    @property
    def boundary(self):
        """ shapely.geometry.LineString : Linestring describing the boundaries of the projection. """
        x0, x1, y0, y1 = self.bounds
        return LineString([(x0, y0), (x0, y1), (x1, y1), (x1, y0),
                           (x0, y0)])

    @property
    def x_limits(self):
        """ tuple: x coordinate limits. """
        x0, x1, y0, y1 = self.bounds
        return (x0, x1)

    @property
    def y_limits(self):
        """ tuple: y coordinate limits. """
        x0, x1, y0, y1 = self.bounds
        return (y0, y1)

    @property
    def threshold(self):
        """ float: Threshold defined by the minimum coordinate extent. """
        x0, x1, y0, y1 = self.bounds
        return min(abs(x1 - x0), abs(y1 - y0)) / 100.

    def _globe_from_proj4(self, proj4_terms):
        """
        Create a `Globe` object from PROJ4 parameters.

        Parameters
        ----------
        proj4_terms: dict
            PROJ4 parameters including terms that are irrelevant for the globe.
        """

        globe_terms = filter(lambda term: term[0] in self._GLOBE_PARAMS,
                             proj4_terms.items())
        globe = ccrs.Globe(**{self._GLOBE_PARAMS[name]: value for name, value in
                              globe_terms})
        return globe
