from setuptools import setup, find_packages

setup(
    name = 'lygos',
    packages = find_packages(),
    version = '1.14',
    description = 'PSF modeling of images in time domain astronomy', 
    author = 'Tansu Daylan',
    author_email = 'tansu.daylan@gmail.com',
    url = 'https://github.com/tansudaylan/lygos',
    download_url = 'https://github.com/tansudaylan/lygos',
    license='MIT',
    classifiers=['Development Status :: 4 - Beta',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python'],
    #install_requires=['numpy', 'scipy', 'tqdm', 'h5py', 'tdpy'],
    python_requires='>=3'
    #include_package_data = True
    )

