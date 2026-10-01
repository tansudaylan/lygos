Installation
============

Install lygos from its repository in editable mode.

.. code-block:: bash

   git clone https://github.com/tansudaylan/lygos.git
   cd lygos
   pip install -e .

lygos needs Python 3.10 or later. The imaging layer uses ``numpy``, ``scipy``,
``astropy``, ``matplotlib``, ``Pillow``, ``astroquery`` for TESSCut and name
resolution, ``s3fs`` for anonymous Amazon Web Services access, and ``tdpy`` for shared
plotting, animation, and TESS pointing helpers. The image-based TESS photometry
pipeline :func:`lygos.init` also needs ``miletos``, ``pcat``, ``nicomedia``,
``h5py``, and ``pandas``.

Download cache
--------------

Downloaded files are kept so that repeated calls do not contact the archives. When the
``LYGOS_PATH`` environment variable names the repository root, the cache is
``$LYGOS_PATH/data/cache/<mission>``, which Git ignores. Otherwise it is
``~/.lygos/cache/<mission>``. Every retrieval function also takes a ``cache_dir``
argument. :func:`lygos.paths.get_cache_path` returns the default location.

.. code-block:: bash

   export LYGOS_PATH=/path/to/lygos

Testing
-------

The offline tests use synthetic images and run in a few seconds. One additional test
queries TESSCut and the Roman simulation archive.

.. code-block:: bash

   python -m pytest tests
   LYGOS_TEST_NETWORK=1 python -m pytest tests -k public
