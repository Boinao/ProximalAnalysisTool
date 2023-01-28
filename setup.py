from setuptools import setup, Command, find_packages

PACKAGE_NAME = "SDAT"

setup(
    name=PACKAGE_NAME,
    version=__version__,
    description="TDP Project for spectral data analysis",
    url="https://github.com/",
    author="Anand SS, Ross Lyngdoh, Nidhin, Pradyuman",
    author_email='anandss@sac.isro.gov.com',
    license="Public Domain",
    keywords='Spectral Library SDAT Field Spectra',
    package_dir={'ProximalAnalysisTool': 'ProximalAnalysisTool'},
    packages=find_packages(),
    #install_requires =['sklearn', 'Py6S', 'pandas'],
    entry_points={
        'gui_scripts': [
            'SDAT=ProximalAnalysisTool.ProximalAnalysisTool.__main__:run'
        ],
    },
    classifiers=[
        #'License :: Unlicense',
        'Programming Language :: Python :: 3.7',
        'Operating System :: Microsoft :: Windows',
        'Topic :: Scientific/Engineering',
    ]
)
