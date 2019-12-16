class GeometryUnkown(Exception):
    """ Class to handle exceptions thrown by unknown geometry types."""

    def __init__(self, geometry):
        """
        Constructor of `GeometryUnknown`.

        Parameters
        ----------
        geometry : ogr.Geometry or shapely.geometry or list or tuple, optional
            A vector geometry.
        """

        self.message = "The given geometry type '{}' cannot be used.".format(type(geometry))

    def __str__(self):
        """ String representation of this class. """

        return self.message