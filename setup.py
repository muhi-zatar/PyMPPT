from setuptools import find_packages, setup


setup(
    name='PyMPPT',
    version='v0.0.1',
    author='Muhy Zater',
    author_email='muhizatar95@gmail.com',
    description='Easy-to-use library to implement ANN-based Maximum Power Point Tracker (MPPT)',
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    keywords=['MPPT', 'ANN-MPPT', 'tensorflow', 'PyMPPT'],
    license='Apache',
    url='https://github.com/muhi-zatar',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'tensorflow',
        'keras',
        'pandas'
    ],
    classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ]
)

