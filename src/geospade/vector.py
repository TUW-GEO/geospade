""" Module containing class definitions for vectors. """


class VectorGeometry:
    """
    Represents a vector geometry.

    """

    def __init__(self):
        """
        Constructor of the `VectorGeometry` class.

        """
        err_msg = "'VectorGeometry' is not implemented yet."
        raise NotImplementedError(err_msg)


class SwathGeometry:
    """
    Represents the geometry of a satellite swath grid.

    """

    def __init__(self, xs, ys, sref, name=None, description=None):
        """
        Constructor of the `SwathGeometry` class.

        Parameters
        ----------
        xs : list of numbers
            East-west coordinates.
        ys : list of numbers
            South-north coordinates.
        sref : geospade.crs.SpatialRef
            Spatial reference of the geometry.
        name : int or str, optional
            Name of the geometry.
        description : string, optional
            Verbal description of the geometry.

        """
        err_msg = "'SwathGeometry' is not implemented yet."
        raise NotImplementedError(err_msg)