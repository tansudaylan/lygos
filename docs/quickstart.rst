Quickstart
==========

The same five steps apply to TESS and Roman images. Retrieve an
:class:`~lygos.imagestack.ImageStack`, choose an aperture, measure a light curve, plot,
and animate.

.. code-block:: python

   import lygos

   # TESS: a 15 by 15 pixel TESSCut time series of RR Lyr in Sector 14
   stack = lygos.get_tess_cutout("RR Lyr", sector=14, size=15)[0].good()
   aperture = stack.threshold_aperture(threshold=5.0)
   light_curve = stack.aperture_photometry(aperture)
   lygos.plot_image(stack, "rrlyr_median", aperture=aperture)
   lygos.plot_light_curve(light_curve, "rrlyr_light_curve")
   lygos.animate_stack(stack.select(slice(0, 120)), "rrlyr_animation", aperture=aperture,
                       light_curve=stack.select(slice(0, 120)).aperture_photometry(aperture))

   # Roman: aligned 41 by 41 pixel cutouts of a simulated supernova in the time-domain survey
   roman = lygos.get_roman_cutout((9.6632, -43.88128), band="J129", size=41,
                                  mjd_range=(62150, 62550)).subtract_background()
   lygos.animate_stack(roman, "supernova_difference", difference=True,
                       light_curve=roman.aperture_photometry(roman.circular_aperture(3.0)))

Targets
-------

Every retrieval function accepts a target as an :class:`astropy.coordinates.SkyCoord`, a
pair of right ascension and declination in degrees, or a name that the Sesame service
resolves, such as ``"RR Lyr"`` or ``"TIC 22529346"``. See
:func:`~lygos.tess.resolve_coordinate`.

Units and times
---------------

TESS pixels are calibrated count rates [e-/s] and times are TESS barycentric Julian
dates, BTJD = BJD - 2457000 [day]. Roman pixels are count rates [counts/s] and times are
modified Julian dates (MJD) [day]. Each stack records these in ``unit`` and
``time_format``, and all plots label them.
