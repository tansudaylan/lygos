Processing
==========

All processing methods belong to :class:`~lygos.imagestack.ImageStack` and return new
stacks or plain arrays, so they chain.

.. code-block:: python

   stack = lygos.get_tess_cutout("RR Lyr", sector=14, size=15)[0]
   light_curve = stack.good().subtract_background().aperture_photometry(stack.circular_aperture(3.0))

Selecting frames
----------------

:meth:`~lygos.imagestack.ImageStack.good`
   keeps frames with zero quality flags and finite pixels.
:meth:`~lygos.imagestack.ImageStack.select`
   keeps frames by index, slice, or Boolean mask, for example ``stack.select(stack.time < 1690.0)``.
:meth:`~lygos.imagestack.ImageStack.concatenate`
   joins two stacks on one pixel grid, such as consecutive sectors, in time order.
:meth:`~lygos.imagestack.ImageStack.bin_time`
   averages frames in time bins of a given width [day].

Background and differences
--------------------------

:meth:`~lygos.imagestack.ImageStack.subtract_background`
   removes a sigma-clipped median sky level from each frame (``method="median"``), or a
   fitted tilted plane (``method="plane"``). Pass a ``mask`` of source pixels to exclude
   them from the fit.
:meth:`~lygos.imagestack.ImageStack.median_image`
   returns the per-pixel median of good frames, a low-noise reference image.
:meth:`~lygos.imagestack.ImageStack.difference`
   subtracts a reference image, the median image by default, from every frame. Constant
   sources cancel, so variable stars, transients, and moving objects stand out.
:meth:`~lygos.imagestack.ImageStack.variability_map`
   returns the robust per-pixel scatter over time divided by the median pixel
   uncertainty. Pixels of constant sources sit near one. Subtract a varying sky level
   first.

Apertures and photometry
------------------------

:meth:`~lygos.imagestack.ImageStack.circular_aperture`
   selects pixels within a radius [pixel] of a (column, row) position.
:meth:`~lygos.imagestack.ImageStack.threshold_aperture`
   selects the connected pixels around a position that are brighter than a number of sky
   standard deviations in the median image.
:meth:`~lygos.imagestack.ImageStack.aperture_photometry`
   sums each frame within an aperture, optionally after subtracting the median of a mask
   of sky pixels, and propagates the pixel uncertainties. It returns a dictionary with
   ``time``, ``flux``, ``flux_error``, and ``quality``.
:meth:`~lygos.imagestack.ImageStack.centroid`
   returns the flux-weighted (column, row) position of every frame, which reveals pointing
   jitter and blending.
:meth:`~lygos.imagestack.ImageStack.pixel_of`
   converts a sky coordinate into a (column, row) position.

Resampling
----------

:func:`~lygos.imagestack.resample_image` interpolates an image from one WCS onto the
pixel grid of another, and :func:`~lygos.imagestack.tangent_plane_wcs` builds a north-up
grid of any size and pixel scale. lygos uses them to align Roman exposures, and they apply
equally to TESS images, for example to place two sectors on one grid.

Photometric pipeline
--------------------

:func:`lygos.init` remains the image-based TESS photometry pipeline. It fits stars with
the TESS point-spread function and produces light curves and diagnostic figures for
known targets.
