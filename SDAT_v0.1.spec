# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = ['pandas', 'seaborn', 'sklearn', 'sklearn.feature_selection', 'sklearn.pipeline', 'sklearn.metrics', 'sklearn.model_selection', 'sklearn.cross_decomposition', 'sklearn.linear_model', 'sklearn.svm', 'sklearn.ensemble', 'sklearn.mixture', 'sklearn.ensemble.RandomForestClassifier', 'cv2', 'pysptools.spectro', 'scipy.io']
hiddenimports += collect_submodules('Ui')
hiddenimports += collect_submodules('modules')
hiddenimports += collect_submodules('Py6S')
hiddenimports += collect_submodules('spectral')
hiddenimports += collect_submodules('specdal')
hiddenimports += collect_submodules('utils')
hiddenimports += collect_submodules('external')
hiddenimports += collect_submodules('prosail')
hiddenimports += collect_submodules('Ui.SpectralPre_TransformUi')
hiddenimports += collect_submodules('specdal.containers.spectrum')
hiddenimports += collect_submodules('specdal.containers.collection')
hiddenimports += collect_submodules('Ui.About_SDATUi')
hiddenimports += collect_submodules('Ui.ClassificationUi')
hiddenimports += collect_submodules('Ui.DerivativeUi')
hiddenimports += collect_submodules('Ui.DimensionalityReductionUi')
hiddenimports += collect_submodules('Ui.FeatureSelectionUI')
hiddenimports += collect_submodules('Ui.featureSelectionMultipleClassesUi')
hiddenimports += collect_submodules('Ui.FgccUi')
hiddenimports += collect_submodules('Ui.GridSearchCVUi')
hiddenimports += collect_submodules('Ui.Multivariate_RegressionUi')
hiddenimports += collect_submodules('Ui.Resampling_SACUi')
hiddenimports += collect_submodules('Ui.ProsailUi')
hiddenimports += collect_submodules('Ui.Py6sUi')
hiddenimports += collect_submodules('Ui.indices')
hiddenimports += collect_submodules('Ui.ResamplingUi')
hiddenimports += collect_submodules('Ui.SigreaderUi')
hiddenimports += collect_submodules('Ui.Spectra_Library_matchUI')
hiddenimports += collect_submodules('Ui.Spectra_Library_searchUi')
hiddenimports += collect_submodules('Ui.Spectra_simulation_SACUi')
hiddenimports += collect_submodules('Ui.SpectraEvolution_viewdataUi')
hiddenimports += collect_submodules('Ui.SpectralDistancesUi')
hiddenimports += collect_submodules('Ui.TimeseriesUi')
hiddenimports += collect_submodules('Ui.Univariate_RegressionUi')
hiddenimports += collect_submodules('Ui.ViewDataUi')
hiddenimports += collect_submodules('Ui.VisualizerUi')


block_cipher = None


a = Analysis(['D:/Proximal/ProximalAnalysisTool/MainViewer.py'],
             pathex=[],
             binaries=[],
             datas=[('D:/Proximal/ProximalAnalysisTool/Vegetation_index_sig_asd.csv', '.'), ('D:/Proximal/ProximalAnalysisTool/external', 'external/'), ('D:/Proximal/ProximalAnalysisTool/external/SpectralLibrary', 'external/SpectralLibrary/'), ('D:/Proximal/ProximalAnalysisTool/external/about', 'external/about/'), ('D:/Proximal/ProximalAnalysisTool/images', 'images/'), ('D:/Proximal/ProximalAnalysisTool/prosail', 'prosail/'), ('D:/Proximal/ProximalAnalysisTool/prosail/*.txt', 'prosail/'), ('D:/Proximal/ProximalAnalysisTool/Py6S', 'Py6S/'), ('D:/Proximal/ProximalAnalysisTool/Py6S/SixSHelpers', 'Py6S/SixSHelpers/'), ('D:/Proximal/ProximalAnalysisTool/Py6S/Params', 'Py6S/Params/'), ('D:/Proximal/ProximalAnalysisTool/Required/numpy', 'numpy/'), ('D:/Proximal/ProximalAnalysisTool/Required/pandas', 'pandas/'), ('D:/Proximal/ProximalAnalysisTool/Required/scipy', 'scipy/'), ('D:/Proximal/ProximalAnalysisTool/Required/seaborn', 'seaborn/'), ('D:/Proximal/ProximalAnalysisTool/Required/sklearn', 'sklearn/'), ('D:/Proximal/ProximalAnalysisTool/Ui', 'Ui/'), ('D:/Proximal/ProximalAnalysisTool/Ui/*.ui', 'Ui/'), ('D:/Proximal/ProximalAnalysisTool/spectral', 'spectral/'), ('D:/Proximal/ProximalAnalysisTool/specdal', 'specdal/'), ('D:/Proximal/ProximalAnalysisTool/extradll', 'extradll/')],
             hiddenimports=hiddenimports,
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=['jedi', 'scipy.io', 'tornado', 'cryptography', 'h5py', 'notebook', 'torch', 'parso', 'dask', 'certifi', 'cvxopt', 'tk', 'tcl', 'tcl8', 'IPython', 'babel'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts, 
          [],
          exclude_binaries=True,
          name='SDAT_v0.1',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas, 
               strip=False,
               upx=True,
               upx_exclude=[],
               name='SDAT_v0.1')
