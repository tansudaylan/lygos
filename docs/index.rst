lygos
=====

lygos retrieves, processes, visualizes, and animates images from the Transiting
Exoplanet Survey Satellite (TESS) and the Nancy Grace Roman Space Telescope. Every
retrieval returns one :class:`~lygos.imagestack.ImageStack`, a time-ordered cube of
images with times, uncertainties, quality flags, and sky coordinates. The same
background subtraction, difference imaging, aperture photometry, plotting, and
animation tools therefore apply to every mission and product.

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Product
     - Source
     - lygos function
   * - TESS cutouts
     - MAST TESSCut service
     - :func:`~lygos.tess.get_tess_cutout`
   * - TESS full-frame images
     - public MAST bucket on Amazon Web Services
     - :func:`~lygos.tess.get_tess_ffi`, :func:`~lygos.tess.get_tess_ffi_cutout`
   * - Roman detector images
     - OpenUniverse 2024 simulation at IRSA
     - :func:`~lygos.roman.read_roman_image`
   * - Roman cutouts
     - OpenUniverse 2024 simulation at IRSA
     - :func:`~lygos.roman.get_roman_cutout`

No archive account is needed. Only the pixels that are needed are transferred, so a
cutout time series from hundreds of TESS full-frame images or Roman exposures takes
seconds to minutes.

.. image:: ../examples/roman_time_domain_supernova/visuals/roman_supernova_difference_animation.gif
   :alt: Roman J129 difference images of a simulated Type Ia supernova with its light curve
   :width: 420px
   :align: center

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   tess
   roman
   processing
   visualization
   examples
   api
   data_sources
