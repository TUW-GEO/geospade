import re

from gdal import osr
from pyproj.crs import CRS
from cartopy import crs as ccrs
from collections import OrderedDict
from shapely.geometry import LineString

# TODO: osr.SpatialReference conversion
class SpatialRef(object):
    """
    This class represents any OGC compliant spatial reference system. Internally, the
    GDAL OSR Spatial Reference class is used, which offers access to different representations, such as EPSG,
    Proj4 or WKT. Additionally, it can also
    create an instance of the Cartopy projection subclass (PROJ4Projection), which
    can be used to plot geometries or data in other classes.
    """

    def __init__(self, arg, spref_type=None):
        """
        Constructs a SpatialRef instance based on 'arg' which can be
        representing given spatial reference either as:
            EPSG Code
            Proj4 string or Proj4 dict
            WKT
        If the type is not provided by the type argument, the constructor tries
        to determine the type itself.
        If arg is an integer, it tries to create
        an osr SpatialReference obejct from an EPSG Code.
        If arg is a dict, it presumes, that the user tries to create
        spatial reference from proj4 parameters. In this case the leading
        '+' sign are to be omitted in the parsed dictionary keys.
        If arg is a string, the constructor tries to check whether the EPSG code
        hasn't been parsed as string or with the 'EPSG:' prefix, or if the
        string isn't in fact a proj4 string, or if the string conrains 'GEODCS'
        tag, which indicates a WKT String

        Parameters
        ----------

        arg: int or dict or str
            argument containing the Spatial Reference definition
        :param type: str (optional)
            either: 'proj4' or 'wkt' or 'epsg'
        """
        self.__arg = arg  # internal class variable used for cross-checking the different representations
        self.spref = osr.SpatialReference()

        if spref_type is None:
            # cases : integer -> EPSG, dict -> Proj4, string -> WKT
            if isinstance(arg, int):
                spref_type = 'epsg'
            elif isinstance(arg, dict):
                spref_type = 'proj4'
                # convert to string because GDAL takes proj4  as string only
                string = ''
                for k, v in arg.items():
                    string += '+{}={} '.format(k, v)
                arg = string
            elif isinstance(arg, str):
                if 'epsg' in arg[0:4].lower():  # EPSG prefix has been parsed too
                    spref_type = 'epsg'
                    epsg_code = re.findall(r'\d+')  # extract the numerical value
                    if 4 <= len(epsg_code[0]) <= 5:  # correct length of epsg code
                        arg = epsg_code[0]
                elif '+' == arg[0]:  # first character is '+' => proj4 as string
                    spref_type = 'proj4'
                elif 'GEOGCS[' in arg:  # there is a GEOGCS tag in string => WKT
                    spref_type = 'wkt'
            else:
                raise ValueError('Spatial reference type is unknown')

        if spref_type == 'proj4':
            self.spref.ImportFromProj4(arg)
        elif spref_type == 'epsg':
            self.spref.ImportFromEPSG(arg)
        elif spref_type == 'wkt':
            self.spref.ImportFromWkt(arg)

        self.spref_type = spref_type

    @property
    def proj4(self):
        """
        :return: dict
            proj4 representation as dict without leading '+' sign in dict keys
        """
        proj4_string = self.proj4_string
        if proj4_string is None:
            return None
        else:
            success = self.__check_conversion(proj4_string, "proj4")
            if success:
                return self._proj4_str_to_dict(proj4_string)
            else:
                raise Warning("Conversion to Proj4 string is not bijective.")
                return None

    @property
    def proj4_string(self):
        """
        :return: str
            proj4 representation as string
        """
        return self.spref.ExportToProj4()[:-1]

    @property
    def epsg(self):
        """
        :return: int or None
            EPSG code of the Spatial reference, if known. Else None
        """
        try:
            code = int(self.spref.GetAttrValue('AUTHORITY', 1))
        except TypeError:
            # EPSG Code unknown
            code = None

        success = self.__check_conversion(code, "epsg")
        if success:
            return code
        else:
            raise Warning("Conversion to EPSG code is not bijective.")
            return None

            # proj4 string may provide unsufficient information about Spatial
            # Reference. Consider following code:
            # >>> sref = SpatialRef(31259) #EPSG for MGI 34
            # >>> sref = SpatialRef(sref.proj4_string)
            # >>> sref.epsg is None
            # True

    @property
    def wkt(self):
        """
        :return: str
            Well Known Text representation of the Spatial reference without
            tabs or line breaks
        """
        wkt_string = self.spref.ExportToWkt()
        success = self.__check_conversion(wkt_string, "wkt")
        if success:
            return wkt_string
        else:
            raise Warning("Conversion to WKT string is not bijective.")
            return None

    @property
    def pretty_wkt(self):
        # TODO: check if it is necessary
        """
        :return: str
            Well Known Text representation of the Spatial reference formatted
            with tabs or line breaks
        """
        pretty_wkt_string = self.spref.ExportToPrettyWkt()
        success = self.__check_conversion(pretty_wkt_string, "pretty_wkt")
        if success:
            return pretty_wkt_string
        else:
            raise Warning("Conversion to pretty WKT string is not bijective.")
            return None

    def to_osr(self):
        """
        Converts this class into an OSGEO/GDAL spatial reference representation.

        Returns
        -------
        osr.SpatialReference
            spatial reference object from GDAL/OSGEO
        """

        sref = osr.SpatialReference()
        sref.ImportFromWkt(self.wkt)
        return sref

    def get_cartopy_crs(self, bounds=None):
        """
        Get a PROJ4Projection object that can be used as an argument of the
        cartopy 'projection' and 'transfrom' kwargs. (PROJ4Projection is a
        subclass of a cartopy.crs.Projection class)
        :param bounds: 4-tuple
            Boundary of the projection (lower-left-x, upper-right-x,
                                        lower-left-y, upper-right-y)
        :return: PROJ4Projection
            PROJ4Projection instance representing the sptatial ref.
        """
        return PROJ4Projection(self.proj4, bounds=bounds)

    def __convert_proj4_pairs_to_dict(self, proj_pairs):
        """Convert PROJ.4 parameters to floats if possible."""
        proj_dict = OrderedDict()
        for x in proj_pairs:
            if len(x) == 1 or x[1] is True:
                proj_dict[x[0]] = True
                continue

            try:
                proj_dict[x[0]] = float(x[1])
            except ValueError:
                proj_dict[x[0]] = x[1]

        return dict(proj_dict)

    def _proj4_str_to_dict(self, proj4_str):
        """Convert PROJ.4 compatible string definition to dict
        EPSG codes should be provided as "EPSG:XXXX" where "XXXX"
        is the EPSG number code. It can also be provided as
        ``"+init=EPSG:XXXX"`` as long as the underlying PROJ library
        supports it (deprecated in PROJ 6.0+).
        Note: Key only parameters will be assigned a value of `True`.
        """
        # TODO: @shahn I would got for one solution, i.e. PyProj > 2.2
        # TODO: test if this can be simplified
        # # convert EPSG codes to equivalent PROJ4 string definition
        if proj4_str.startswith('EPSG:'):
            crs = CRS(proj4_str)
            return crs.to_dict()
        else:
            proj4_pairs = (x.split('=', 1) for x in proj4_str.replace('+', '').split(" "))
            return self.__convert_proj4_pairs_to_dict(proj4_pairs)

    def __check_conversion(self, check_spref_value, check_spref_type)
        if self.spref_type != check_spref_type:
            spref_other = SpatialRef(check_spref_value, type=check_spref_type)
            spref_value_this = get_attribute(spref_other, self.spref_type)
            if (spref_value_this == self.arg):
                return True
            else:
                return False
        else:
            return True

    def __eq__(self, other):
        this_proj4 = self.proj4
        other_proj4 = other.proj4
        if this_proj4 == other_proj4:
            return True
        else:
            return False

    def __ne__(self, other):
        return not self == other


class PROJ4Projection(ccrs.Projection):
    """
    This class represents any cartopy projection based on its proj4 parameters.
    Instances of this class can be parsed as 'projection' and 'transfrom' kwargs
    as regular cartopy projections, because it is a subclass of a
    cartopy.crs.Projection class. (Sort of a workaround - Big thanks going to
    guys from pytroll/pyresample)
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
        Creates an instance from proj4 params, globe and projection boundaries.
        :param terms: dict
            dictionary containing proj4 params of a projection
        :param globe: carotpy.crs.Globe (optional)
            A cartopy Globe instance. If omitted, constructor creates it from
            the terms itself.
        :param bounds: 4-tuple
            Boundary of the projection (lower-left-x, upper-right-x,
                                        lower-left-y, upper-right-y)
        """
        globe = self._globe_from_proj4(terms) if globe is None else globe

        other_terms = []  # terms that are not defining the datum / globe
        for term in terms.items():
            if term[0] not in self._GLOBE_PARAMS:
                other_terms.append(term)
        super(PROJ4Projection, self).__init__(other_terms, globe)

        self.bounds = bounds

    def __repr__(self):
        return 'PROJ4Projection({})'.format(self.proj4_init)

    @property
    def boundary(self):
        x0, x1, y0, y1 = self.bounds
        return LineString([(x0, y0), (x0, y1), (x1, y1), (x1, y0),
                           (x0, y0)])

    @property
    def x_limits(self):
        x0, x1, y0, y1 = self.bounds
        return (x0, x1)

    @property
    def y_limits(self):
        x0, x1, y0, y1 = self.bounds
        return (y0, y1)

    @property
    def threshold(self):
        x0, x1, y0, y1 = self.bounds
        return min(abs(x1 - x0), abs(y1 - y0)) / 100.

    def _globe_from_proj4(self, proj4_terms):
        """
        Create a `Globe` object from PROJ.4 parameters.
        :param proj4_terms: dict
            proj4 parmas including terms that are irrelevant for the globe.
        """

        globe_terms = filter(lambda term: term[0] in self._GLOBE_PARAMS,
                             proj4_terms.items())
        globe = ccrs.Globe(**{self._GLOBE_PARAMS[name]: value for name, value in
                              globe_terms})
        return globe
