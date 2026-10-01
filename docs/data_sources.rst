Data sources and citations
==========================

Cite the data and services behind lygos products when you publish them.

TESS
   Ricker, G. R., et al. 2015, Journal of Astronomical Telescopes, Instruments, and
   Systems, 1, 014003. https://doi.org/10.1117/1.JATIS.1.1.014003
TESSCut
   Brasseur, C. E., Phillip, C., Fleming, S. W., Mullally, S. E., and White, R. L. 2019,
   Astrocut, Astrophysics Source Code Library, ascl:1905.007.
TESS full-frame images
   Calibrated by the Science Processing Operations Center (SPOC) at NASA Ames Research
   Center (Jenkins et al. 2016, Proceedings of the SPIE, 9913, 99133E) and distributed by
   MAST, including through the public Amazon Web Services bucket ``stpubdata``.
OpenUniverse 2024 Roman simulation
   OpenUniverse et al. 2025, "OpenUniverse2024: A shared, simulated view of the sky for
   the next generation of cosmological surveys", distributed by the NASA/IPAC Infrared
   Science Archive (IRSA). https://irsa.ipac.caltech.edu/data/theory/openuniverse2024/
Astropy
   Astropy Collaboration, 2022, The Astrophysical Journal, 935, 167.

Data access
-----------

All archives are read anonymously. TESSCut is a web service with request limits, so
lygos caches its files. Amazon Web Services transfers from the two public buckets are
free. Section reads of TESS full-frame images and streamed reads of Roman images transfer
only the bytes that a cutout needs.
