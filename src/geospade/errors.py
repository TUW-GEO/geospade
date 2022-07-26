""" Module collecting some often used error messages. """


class GeometryUnknown(Exception):
    """ Class to handle exceptions thrown by unknown geometry types."""

    def __init__(self, geometry):
        """
        Constructor of `GeometryUnknown`.

        Parameters
        ----------
        geometry : ogr.Geometry
            An OGR geometry.

        """

        self.message = "The given geometry type '{}' cannot be used.".format(type(geometry))

    def __str__(self) -> str:
        """ String representation of this class. """

        return self.message


class SrefUnknown(Exception):
    """ Class to handle exceptions thrown by an unknown spatial reference system. """

    def __init__(self):
        """ Constructor of `SrefUnknown`. """

        self.message = "No spatial reference system information is supplied."

    def __str__(self) -> str:
        """ String representation of this class. """

        return self.message