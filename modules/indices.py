# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 13:31:35 2019

@author: Trainee
"""
import numpy as np
import math
import pandas as pd


def calculateValue(df, str1, a1, a2, a3, a4, a5, c1, c2, c3, c4):
    if (a1 != 0):
        v1 = df.loc[a1]
    if (a2 != 0):
        v2 = df.loc[a2]
    if (a3 != 0):
        v3 = df.loc[a3]
    if (a4 != 0):
        v4 = df.loc[a4]
    # if(a5!=0):
    # v5=df.loc[a5]
    if str1 == "ARI":
        return (1 / v1) - (1 / v2)

    if str1 == "ARI2":
        return v1 * ((1 / v2) - (1 / v3))
    if str1 == "BG1":
        return v1 / v2
    if str1 == "BG2":
        return v1 / v2
    if str1 == "BRI1":
        return v1 / v2
    if str1 == "BRI2":
        return v1 / v2
    if str1 == "CAI":
        return c1 * (v1 + v2) - v3
    if str1 == "CRI550":
        return (1 / v1) / (1 / v2)
    if str1 == "CRI700":
        return (1 / v1) / (1 / v2)
    if str1 == "DI1":
        return v1 - v2
    if str1 == "GM1":
        return v1 / v2
    if str1 == "GM2":
        return v1 / v2
    if str1 == "Greenness index(G)":
        return v1 / v2
    if str1 == "LIC3":
        return v1 / v2
    if str1 == "HI":
        return ((v1 - v2) / (v1 + v2)) - c1 * v3
    if str1 == "MCARI":
        return ((v1 - v2) - c1 * (v1 - v3)) * (v1 / v2)
    if str1 == "MRENDVI":
        return (v1 - v2) / (v1 + v2 - c1 * v3)
    if str1 == "MRESR":
        return (v1 - v2) / (v1 + v2)
    if str1 == "MSAVI":
        return c1 * (c2 * v1 + 1 - np.sqrt(((c2 * v1 + 1) ** 2) - c3 * (v1 - v2)))

    if str1 == "MSR":
        return ((v1 / v2) - 1) / (np.sqrt(v1 / v2 + 1))
    if str1 == "MSI":
        return v1 / v2
    if str1 == "MTVI":
        return c1 * (c1 * (v1 - v2) - c2 * (v3 - v2))
    if str1 == "NDI1":
        return (v1 - v2) / (v1 - v3)
    if str1 == "NDI2":
        return (v1 - v2) / (v1 - v3)
    if str1 == "NDI3":
        return (v1 - v2) / (v3 - v4)

    if str1 == "NDNI":
        return (np.log(1 / v1) - np.log(1 / v2)) / (np.log(1 / v1) + np.log(1 / v2))
    if str1 == "NDLI":
        return (np.log(1 / v1) - np.log(1 / v2)) / (np.log(1 / v1) + np.log(1 / v2))

    if str1 == "NDVI":
        return (v1 - v2) / (v1 + v2)
    if str1 == "NMDI":
        return (v1 - (v2 - v3)) / (v1 + (v2 - v3))
    if str1 == "OSAVI":
        return ((1 + c1) * (v1 - v2)) / (v1 + v2 + c2)
    if str1 == "PRI":
        return (v1 - v2) / (v1 + v2)
    if str1 == "PSRI":
        return (v1 - v2) / v3
    if str1 == "PSNDc":
        return (v1 - v2) / (v1 + v2)
    if str1 == "PSSRa":
        return v1 / v2
    if str1 == "PSSRb":
        return v1 / v2
    if str1 == "PSSRc":
        return v1 / v2
    if str1 == "RARS":
        return v1 / v2
    if str1 == "RDVI":
        return (v1 - v2) / (np.sqrt(v1 + v2))
    if str1 == "RENDVI":
        return (v1 - v2) / (v1 + v2)
    if str1 == "SIPI":
        return (v1 - v2) / (v1 - v3)
    if str1 == "SR":
        return v1 / v2
    if str1 == "SR2":
        return v1 / v2
    if str1 == "SR3":
        return v1 / v2
    if str1 == "SR4":
        return v1 / v2
    if str1 == "SR5":
        return v1 / (v2 * v3)
    if str1 == "SR6":
        return v1 / (v2 * v3)
    if str1 == "SR7":
        return v1 / (v2 * v3)
    if str1 == "TCARI":
        return c1 * ((v1 - v2) - c2 * (v1 - v3) * (v1 / v2))
    if str1 == "VREI1":
        return v1 / v2
    if str1 == "VREI2":
        return (v1 - v2) / (v3 - v4)
    if str1 == "WBI":
        return v1 / v2
    if str1 == "TGI":
        return c1 * ((v1 - v2) * (v3 - v4) - (v1 - v4) * (v1 - v2))
    if str1 == "ZM":
        return v1 / v2
    if str1 == "LMVI1":
        return (v1 - v2) / (v1 + v2)
    if str1 == "RVI":
        return v1 / v2
    if str1 == 'ALL':
        # ari=v1 / v2
        # rvi=v1 / v2
        # return {"ARI":ari,"RVI":rvi}
        ARI = (1 / v1) - (1 / v2)
        ARI2 = v1 * ((1 / v2) - (1 / v3))
        BG1 = v1 / v2
        BG2 = v1 / v2
        BRI1 = v1 / v2
        BRI2 = v1 / v2
        CAI = c1 * (v1 + v2) - v3
        CRI550 = (1 / v1) / (1 / v2)
        CRI700 = (1 / v1) / (1 / v2)
        DI1 = v1 - v2
        GM1 = v1 / v2
        GM2 = v1 / v2
        Greenness_index = v1 / v2
        LIC3 = v1 / v2
        HI = ((v1 - v2) / (v1 + v2)) - c1 * v3
        MCARI = ((v1 - v2) - c1 * (v1 - v3)) * (v1 / v2)
        MRENDVI = (v1 - v2) / (v1 + v2 - c1 * v3)
        MRESR = (v1 - v2) / (v1 + v2)
        MSAVI = c1 * (c2 * v1 + 1 - np.sqrt(((c2 * v1 + 1) ** 2) - c3 * (v1 - v2)))
        MSR = ((v1 / v2) - 1) / (np.sqrt(v1 / v2 + 1))
        MSI = v1 / v2
        MTVI = c1 * (c1 * (v1 - v2) - c2 * (v3 - v2))
        NDI1 = (v1 - v2) / (v1 - v3)
        NDI2 = (v1 - v2) / (v1 - v3)
        NDI3 = (v1 - v2) / (v3 - v4)
        NDNI = (np.log(1 / v1) - np.log(1 / v2)) / (np.log(1 / v1) + np.log(1 / v2))
        NDLI = (np.log(1 / v1) - np.log(1 / v2)) / (np.log(1 / v1) + np.log(1 / v2))
        NDVI = (v1 - v2) / (v1 + v2)
        NMDI = (v1 - (v2 - v3)) / (v1 + (v2 - v3))
        OSAVI = ((1 + c1) * (v1 - v2)) / (v1 + v2 + c2)
        PRI = (v1 - v2) / (v1 + v2)
        PSRI = (v1 - v2) / v3
        PSNDc = (v1 - v2) / (v1 + v2)
        PSSRa = v1 / v2
        PSSRb = v1 / v2
        PSSRc = v1 / v2
        RARS = v1 / v2
        RDVI = (v1 - v2) / (np.sqrt(v1 + v2))
        RENDVI = (v1 - v2) / (v1 + v2)
        SIPI = (v1 - v2) / (v1 - v3)
        SR = v1 / v2
        SR2 = v1 / v2
        SR3 = v1 / v2
        SR4 = v1 / v2
        SR5 = v1 / (v2 * v3)
        SR6 = v1 / (v2 * v3)
        SR7 = v1 / (v2 * v3)
        TCARI = c1 * ((v1 - v2) - c2 * (v1 - v3) * (v1 / v2))
        VREI1 = v1 / v2
        VREI2 = (v1 - v2) / (v3 - v4)
        WBI = v1 / v2
        TGI = c1 * ((v1 - v2) * (v3 - v4) - (v1 - v4) * (v1 - v2))
        ZM = v1 / v2
        LMVI1 = (v1 - v2) / (v1 + v2)
        RVI = v1 / v2
        all_ind = [ARI, ARI2, BG1, BG2, BRI1, BRI2, CAI, CRI550, CRI700, DI1, GM1, GM2, Greenness_index, LIC3, HI,
                   MCARI, MRENDVI, MRESR, MSAVI, MSR, MSI, MTVI, NDI1, NDI2, NDI3, NDNI, NDLI, NDVI, NMDI, OSAVI, PRI,
                   PSRI, PSNDc, PSSRa, PSSRb, PSSRc, RARS, RDVI, RENDVI, SIPI, SR, SR2, SR3, SR4, SR5, SR6, SR7, TCARI,
                   TGI, VREI1, VREI2, WBI, ZM, LMVI1, RVI]
        name_ind = ["ARI", "ARI2", "BG1", "BG2", "BRI1", "BRI2", "CAI", "CRI550", "CRI700", "DI1", "GM1", "GM2",
                    "Greenness index(G)", "LIC3", "HI", "MCARI", "MRENDVI", "MRESR", "MSAVI", "MSR", "MSI", "MTVI",
                    "NDI1", "NDI2", "NDI3", "NDNI", "NDLI", "NDVI", "NMDI", "OSAVI", "PRI", "PSRI", "PSNDc", "PSSRa",
                    "PSSRb", "PSSRc", "RARS", "RDVI", "RENDVI", "SIPI", "SR", "SR2", "SR3", "SR4", "SR5", "SR6", "SR7",
                    "TCARI", "TGI", "VREI1", "VREI2", "WBI", "ZM", "LMVI1", "RVI"]
        # dict = {'RVI':RVI, 'ARI': ARI,"ARI2":ARI2}
        # df = pd.DataFrame({'Spectral index': name_ind, 'Spectral index value': all_ind})
        return all_ind
