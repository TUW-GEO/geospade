""" Coordinate Reference System (CRS) module. """

import re

import cartopy.crs
import pyproj
import warnings
from osgeo import osr
from pyproj.crs import CRS
from cartopy import crs as ccrs


class SpatialRef:
    """
    This class represents any OGC compliant spatial reference system. Internally, the
    GDAL OSR SpatialReference class is used, which offers access to and control over
    different representations, such as EPSG, PROJ4 or WKT. Additionally, it can also
    create an instance of a Cartopy Projection, which can be used to plot geometries
    or data.

    """

    def __init__(self, arg, sref_type=None):
        """
        Constructs a ˋSpatialRefˋ instance based on ˋargˋ.
        If the type is not provided by the type argument ˋsref_typeˋ, the constructor tries
        to determine the type itself.

        Parameters
        ----------
        arg : int or dict or str
            Represents a given spatial reference system. Regarding spatial reference type determination,
            different cases are distinguished:
            - If ˋargˋ is an integer, it tries to interpret it as an EPSG Code (e.g. 4326).
            - If ˋargˋ is a dict, it assumes that PROJ4 parameters are given as input
              (e.g. {'proj': 'longlat', 'ellps': 'WGS84', 'datum': 'WGS84', 'no_defs' : True}).
            - If ˋargˋ is a string, the constructor tries to check whether it contains
                - an 'EPSG' prefix (=> EPSG, e.g., 'EPSG: 4326'),
                - a plus '+' (=> PROJ4, e.g., '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'), or
                - 'GEODCS' (=> WKT, e.g., 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",
                6378137,298.257223563,AUTHORITY["EPSG","7030"]], AUTHORITY["EPSG","6326"]],
                PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,
                AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]').
        sref_type : str, optional
            String defining the type of ˋargˋ. It can be: 'proj4', 'wkt' or 'epsg'.
            If it is None, the spatial reference type of ˋargˋ is guessed.

        """

        self.osr_sref = None

        if sref_type is None:  # argument type guessing
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
            self.osr_sref = self.proj4_to_osr(arg)
            self._proj4 = self.osr_to_proj4(self.osr_sref)
        elif sref_type == 'epsg':
            self.osr_sref = self.epsg_to_osr(arg)
            self._epsg = self.osr_to_epsg(self.osr_sref)
        elif sref_type == 'wkt':
            self.osr_sref = self.wkt_to_osr(arg)
            self._wkt = self.osr_to_wkt(self.osr_sref)
        else:
            err_msg = "Spatial reference type '{}' is unknown. Use 'epsg', 'wkt' or 'proj4'."
            raise Exception(err_msg.format(sref_type))

        # defines a reference spatial reference representation, which is necessary for checking the consistency of
        # transformations between different representations
        self._sref_type = sref_type

    @classmethod
    def from_osr(cls, osr_sref) -> "SpatialRef":
        """
        Creates a `SpatialRef` object from an OSR spatial reference object. To allow this transformation,
        PROJ4 is used.

        Parameters
        ----------
        osr_sref : osr.SpatialReference
            OSR spatial reference.

        Returns
        -------
        SpatialRef
            Spatial reference defined by the exported PROJ4 string from an OSR spatial reference object.

        """

        proj4_string = osr_sref.ExportToProj4()
        return cls(proj4_string, sref_type='proj4')

    @property
    def proj4(self) -> str:
        """ PROJ4 string representation. """

        if self._proj4 is None:
            _ = self._check_conversion("proj4")
            self._proj4 = self.osr_to_proj4(self.osr_sref)

        return self._proj4

    @property
    def epsg(self) -> int:
        """ EPSG code representation as an integer. """

        if self._epsg is None:
            _ = self._check_conversion("epsg")
            self._epsg = self.osr_to_epsg(self.osr_sref)

        return self._epsg

    @property
    def wkt(self) -> str:
        """ Well Known Text (WKT) representation of the spatial reference without tabs or line breaks. """

        if self._wkt is None:
            _ = self._check_conversion("wkt")
            self._wkt = self.osr_to_wkt(self.osr_sref)

        return self._wkt

    def to_pretty_wkt(self) -> str:
        """ Well Known Text (WKT) representation of the spatial reference formatted with tabs and line breaks. """

        return self.osr_sref.ExportToPrettyWkt()

    def to_proj4_dict(self) -> dict:
        """
        Converts internal PROJ4 parameter string to a dictionary, where the keys do not contain a plus and
        the values are converted to non-string values if possible.

        """

        return self._proj4_str_to_dict(self.proj4)

    def to_cartopy_proj(self) -> cartopy.crs.Projection:
        """
        Creates a `cartopy.crs.Projection` instance from PROJ4 parameters.

        Returns
        -------
        cartopy.crs.projection
            Cartopy projection representing the projection of the spatial reference system.

        """
        proj4_params = self.to_proj4_dict()
        proj4_name = proj4_params.get('proj')
        central_longitude = proj4_params.get('lon_0', 0.)
        central_latitude = proj4_params.get('lat_0', 0.)
        false_easting = proj4_params.get('x_0', 0.)
        false_northing = proj4_params.get('y_0', 0.)
        scale_factor = proj4_params.get('k', 1.)
        standard_parallels = (proj4_params.get('lat_1', 20.),
                              proj4_params.get('lat_2', 50.))

        if proj4_name == 'longlat':
            ccrs_proj = ccrs.PlateCarree(central_longitude)
        elif proj4_name == 'aeqd':
            ccrs_proj = ccrs.AzimuthalEquidistant(central_longitude,
                                                  central_latitude,
                                                  false_easting,
                                                  false_northing)
        elif proj4_name == 'merc':
            ccrs_proj = ccrs.Mercator(central_longitude,
                                      false_easting=false_easting,
                                      false_northing=false_northing,
                                      scale_factor=scale_factor)
        elif proj4_name == 'eck1':
            ccrs_proj = ccrs.EckertI(central_longitude,
                                     false_easting,
                                     false_northing)
        elif proj4_name == 'eck2':
            ccrs_proj = ccrs.EckertII(central_longitude,
                                      false_easting,
                                      false_northing)
        elif proj4_name == 'eck3':
            ccrs_proj = ccrs.EckertIII(central_longitude,
                                       false_easting,
                                       false_northing)
        elif proj4_name == 'eck4':
            ccrs_proj = ccrs.EckertIV(central_longitude,
                                      false_easting,
                                      false_northing)
        elif proj4_name == 'eck5':
            ccrs_proj = ccrs.EckertV(central_longitude,
                                     false_easting,
                                     false_northing)
        elif proj4_name == 'eck6':
            ccrs_proj = ccrs.EckertVI(central_longitude,
                                      false_easting,
                                      false_northing)
        elif proj4_name == 'aea':
            ccrs_proj = ccrs.AlbersEqualArea(central_longitude,
                                             central_latitude,
                                             false_easting,
                                             false_northing,
                                             standard_parallels)
        elif proj4_name == 'eqdc':
            ccrs_proj = ccrs.EquidistantConic(central_longitude,
                                              central_latitude,
                                              false_easting,
                                              false_northing,
                                              standard_parallels)
        elif proj4_name == 'gnom':
            ccrs_proj = ccrs.Gnomonic(central_longitude,
                                      central_latitude)
        elif proj4_name == 'laea':
            ccrs_proj = ccrs.LambertAzimuthalEqualArea(central_longitude,
                                                       central_latitude,
                                                       false_easting,
                                                       false_northing)
        elif proj4_name == 'lcc':
            ccrs_proj = ccrs.LambertConformal(central_longitude,
                                              central_latitude,
                                              false_easting,
                                              false_northing,
                                              standard_parallels=standard_parallels)
        elif proj4_name == 'mill':
            ccrs_proj = ccrs.Miller(central_longitude)
        elif proj4_name == 'moll':
            ccrs_proj = ccrs.Mollweide(central_longitude,
                                       false_easting=false_easting,
                                       false_northing=false_northing)
        elif proj4_name == 'stere':
            ccrs_proj = ccrs.Stereographic(central_latitude,
                                           central_longitude,
                                           false_easting,
                                           false_northing,
                                           scale_factor=scale_factor)
        elif proj4_name == 'ortho':
            ccrs_proj = ccrs.Orthographic(central_longitude,
                                          central_latitude)
        elif proj4_name == 'robin':
            ccrs_proj = ccrs.Robinson(central_longitude,
                                      false_easting=false_easting,
                                      false_northing=false_northing)
        elif proj4_name == 'sinus':
            ccrs_proj = ccrs.Sinusoidal(central_longitude,
                                        false_easting,
                                        false_northing)
        elif proj4_name == 'tmerc':
            ccrs_proj = ccrs.TransverseMercator(central_longitude,
                                                central_latitude,
                                                false_easting,
                                                false_northing,
                                                scale_factor)
        else:
            err_msg = "Projection '{}' is not supported.".format(proj4_name)
            raise ValueError(err_msg)

        return ccrs_proj

    def to_pyproj_crs(self) -> pyproj.CRS:
        """ PYPROJ coordinate reference system instance. """
        return pyproj.CRS.from_user_input(self.wkt)

    @staticmethod
    def osr_to_proj4(osr_sref) -> str:
        """
        Converts an `osr.SpatialReference` instance to a PROJ4 string.

        Parameters
        ----------
        osr_sref : osr.SpatialReference
            OSR spatial reference.

        Returns
        -------
        str
            PROJ4 string.

        """

        return osr_sref.ExportToProj4()[:-1]

    @staticmethod
    def proj4_to_osr(proj4_params) -> osr.SpatialReference:
        """
        Converts PROJ4 parameters to an OSR spatial reference object.

        Parameters
        ----------
        proj4_params : str or dict
            PROJ4 parameters as a string (e.g., '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs') or a dictionary
            (e.g., {'proj': 'longlat', 'ellps': 'WGS84', 'datum': 'WGS84', 'no_defs' : True})

        Returns
        -------
        osr.SpatialReference
            OSR spatial reference.

        """

        # convert to string because GDAL takes PROJ4 as string only
        if isinstance(proj4_params, dict):
            arg = ''
            for k, v in proj4_params.items():
                arg += '+{}={} '.format(k, v)
        elif isinstance(proj4_params, str):
            if '+' == proj4_params[0]:
                arg = proj4_params
            else:
                err_msg = "'{}' does not start with a plus, which is mandatory for a PROJ4 string."
                raise ValueError(err_msg)
        else:
            err_msg = "PROJ4 parameters have to be either given as a string or a dict."
            raise ValueError(err_msg)

        osr_sref = osr.SpatialReference()
        osr_sref.ImportFromProj4(arg)

        return osr_sref

    @staticmethod
    def osr_to_epsg(osr_sref) -> int:
        """
        Converts an `osr.SpatialReference` instance to an EPSG code.

        Parameters
        ----------
        osr_sref : osr.SpatialReference
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
    def epsg_to_osr(epsg_code) -> osr.SpatialReference:
        """
        Converts an EPSG code to an OSR spatial reference object.

        Parameters
        ----------
        epsg_code : int or str
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
                raise ValueError(err_msg.format(epsg_code))
        else:
            err_msg = "The EPSG code has to be either given as a string or an integer (see docs)."
            raise ValueError(err_msg)

        osr_sref = osr.SpatialReference()
        osr_sref.ImportFromEPSG(arg)

        return osr_sref

    @staticmethod
    def osr_to_wkt(osr_sref) -> str:
        """
        Converts an `osr.SpatialReference` instance to a  Well Known Text (WKT) string.

        Parameters
        ----------
        osr_sref : osr.SpatialReference
            OSR spatial reference.

        Returns
        -------
        str
            WKT string.

        """

        return osr_sref.ExportToWkt()

    @staticmethod
    def wkt_to_osr(wkt_string) -> osr.SpatialReference:
        """
        Converts a Well Known Text (WKT) string to an OSR spatial reference object.

        Parameters
        ----------
        wkt_string : str
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
                raise ValueError(err_msg)
        else:
            err_msg = "The argument has to be provided as a string."
            raise ValueError(err_msg)

        osr_sref = osr.SpatialReference()
        osr_sref.ImportFromWkt(arg)

        return osr_sref

    def __convert_proj4_pairs_to_dict(self, proj4_pairs) -> dict:
        """
        Converts PROJ4 parameters to floats if possible.

        Parameters
        ----------
        proj4_pairs : list of tuples
            List of (key, value) pairs coming from a PROJ4 object, e.g., [('proj', 'longlat'), ('ellps', 'WGS84')].

        Returns
        -------
        dict
            Dictionary containing parsed PROJ4 parameters.

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

    def _proj4_str_to_dict(self, proj4_string) -> dict:
        """
        Converts PROJ4 compatible string to dictionary.

        Parameters
        ----------
        proj4_string : str
            PROJ4 parameters as a string (e.g., '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs').

        Returns
        -------
        dict
            Dictionary containing PROJ4 parameters.

        Notes
        -----
        Key only parameters will be assigned a value of `True`.
        EPSG codes should be provided as "EPSG:XXXX" where "XXXX"
        is the EPSG code number. It can also be provided as
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

    def _check_conversion(self, tar_sref_type) -> bool:
        """
        Checks whether this spatial reference type can be neatly transformed to another spatial reference type
        (e.g., WKT -> EPSG).

        Parameters
        ----------
        tar_sref_type : str
            String defining the target spatial reference type to convert to. It can be: 'proj4', 'wkt' or 'epsg'.

        Returns
        -------
        bool
            If False, the conversion between this and the target spatial reference is not possible.

        """

        sref_types = ["proj4", "wkt", "epsg"]
        srefs_to = {'proj4': lambda x: SpatialRef.osr_to_proj4(x),
                    'wkt': lambda x: SpatialRef.osr_to_wkt(x),
                    'epsg': lambda x: SpatialRef.osr_to_epsg(x)}
        warn_msg = "Conversion from '{}' to '{}' is not possible."

        if tar_sref_type not in sref_types:
            err_msg = "Spatial reference type '{}' is unknown. Use 'epsg', 'wkt' or 'proj4'."
            raise ValueError(err_msg.format(tar_sref_type))

        is_valid = False
        if self._sref_type != tar_sref_type:
            tar_sref_val = srefs_to[tar_sref_type](self.osr_sref)
            if tar_sref_val is None:
                warnings.warn(warn_msg.format(self._sref_type.upper(), tar_sref_type.upper()))
            else:
                is_valid = True
        else:
            is_valid = True

        return is_valid

    def __eq__(self, other) -> bool:
        """ bool : Checks if this and another `SpatialRef` object are equal according to their PROJ4 strings. """

        return self.proj4 == other.proj4

    def __ne__(self, other) -> bool:
        """ bool : Checks if this and another `SpatialRef` object are unequal according to their PROJ4 strings. """

        return not self == other

    def __deepcopy__(self, memo) -> "SpatialRef":
        """
        Deepcopy method of the `SpatialRef` class.

        Parameters
        ----------
        memo : dict

        Returns
        -------
        SpatialRef
            Deepcopy of a spatial reference object.

        """

        return SpatialRef.from_osr(self.osr_sref)
