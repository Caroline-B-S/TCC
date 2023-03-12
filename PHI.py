import numpy as np
import pandas as pd
from mendeleev import element
from itertools import combinations
from scipy import constants
import math
import xlsxwriter

df = pd.read_csv('parameters.csv', delimiter=';')
arrAtomicSize = np.array(df['atomic_size']) * 100
arrMeltingT = np.array(df['Tm'])
elements = np.array(df['elements'])

dfHmix = pd.read_excel(r"Hmix.xlsx", index_col=0)

def normalizer(alloys):
    total = np.sum(alloys, axis=1).reshape((-1, 1))
    norm_alloys = alloys / total
    return norm_alloys

allVEC = {element(i).symbol: element(i).nvalence() for i in elements}
arrVEC = np.array(list(allVEC.values()))


def parVEC(alloys):
    compNorm = normalizer(alloys)
    VEC = compNorm * arrVEC
    VECFinal = np.sum(VEC, axis=1)
    return VECFinal


def Smix(compNorm):
    x = np.sum(np.nan_to_num((compNorm) * np.log(compNorm)), axis=1)
    Smix = -constants.R * 10 ** -3 * x
    return Smix


def Tm(compNorm):
    Tm = np.sum(compNorm * arrMeltingT, axis=1)
    return Tm


def Hmix(compNorm):
    elements_present = compNorm.sum(axis=0).astype(bool)
    compNorm = compNorm[:, elements_present]
    element_names = elements[elements_present]
    Hmix = np.zeros(compNorm.shape[0])
    for i, j in combinations(range(len(element_names)), 2):
        Hmix = (
                Hmix
                + 4
                * dfHmix[element_names[i]][element_names[j]]
                * compNorm[:, i]
                * compNorm[:, j]
        )
    return Hmix


def Sh(compNorm):
    Sh = abs(Hmix(compNorm)) / Tm(compNorm)
    return Sh


def csi_i(compNorm, AP):
    supportValue = np.sum((1 / 6) * math.pi * (arrAtomicSize * 2) ** 3 * compNorm, axis=1)
    rho = AP / supportValue
    csi_i = (1 / 6) * math.pi * rho[:, None] * (arrAtomicSize * 2) ** 3 * compNorm
    return csi_i


def deltaij(i, j, newCompNorm, newArrAtomicSize, csi_i_newCompNorm, AP):
    element1Size = newArrAtomicSize[i] * 2
    element2Size = newArrAtomicSize[j] * 2
    deltaij = ((csi_i_newCompNorm[:, i] * csi_i_newCompNorm[:, j]) ** (1 / 2) / AP) * (
                ((element1Size - element2Size) ** 2) / (element1Size * element2Size)) * (
                          newCompNorm[:, i] * newCompNorm[:, j]) ** (1 / 2)
    return deltaij


def y1_y2(compNorm, AP):
    csi_i_compNorm = csi_i(compNorm, AP)
    elements_present = compNorm.sum(axis=0).astype(bool)
    newCompNorm = compNorm[:, elements_present]
    newCsi_i_compNorm = csi_i_compNorm[:, elements_present]
    newArrAtomicSize = arrAtomicSize[elements_present]
    y1 = np.zeros(newCompNorm.shape[0])
    y2 = np.zeros(newCompNorm.shape[0])
    for i, j in combinations(range(len(newCompNorm[0])), 2):
        deltaijValue = deltaij(i, j, newCompNorm, newArrAtomicSize, newCsi_i_compNorm, AP)
        y1 += deltaijValue * (newArrAtomicSize[i] * 2 + newArrAtomicSize[j] * 2) * (
                    newArrAtomicSize[i] * 2 * newArrAtomicSize[j] * 2) ** (-1 / 2)
        y2_ = np.sum((newCsi_i_compNorm / AP) * (
                    ((newArrAtomicSize[i] * 2 * newArrAtomicSize[j] * 2) ** (1 / 2)) / (newArrAtomicSize * 2)), axis=1)
        y2 += deltaijValue * y2_
    return y1, y2


def y3(compNorm, AP):
    csi_i_compNorm = csi_i(compNorm, AP)
    x = (csi_i_compNorm / AP) ** (2 / 3) * compNorm ** (1 / 3)
    y3 = (np.sum(x, axis=1)) ** 3
    return y3


def Z(compNorm, AP):
    y1Values, y2Values = y1_y2(compNorm, AP)
    y3Values = y3(compNorm, AP)
    Z = ((1 + AP + AP ** 2) - 3 * AP * (y1Values + y2Values * AP) - AP ** 3 * y3Values) * (1 - AP) ** (-3)
    return Z


def eq4B(compNorm, AP):
    y1Values, y2Values = y1_y2(compNorm, AP)
    y3Values = y3(compNorm, AP)
    eq4B = -(3 / 2) * (1 - y1Values + y2Values + y3Values) + (3 * y2Values + 2 * y3Values) * (1 - AP) ** -1 + (
                3 / 2) * (1 - y1Values - y2Values - (1 / 3) * y3Values) * (1 - AP) ** -2 + (y3Values - 1) * np.log(
        1 - AP)
    return eq4B


def Se(compNorm, AP):
    Se = (eq4B(compNorm, AP) - np.log(Z(compNorm, AP)) - (3 - 2 * AP) * (1 - AP) ** -2 + 3 + np.log(
        (1 + AP + AP ** 2 - AP ** 3) * (1 - AP) ** -3)) * constants.R * 10 ** -3
    return Se


def parPhi(alloys):
    compNorm = normalizer(alloys)
    SeBCC = Se(compNorm, 0.68)
    SeFCC = Se(compNorm, 0.74)
    SeMean = (abs(SeBCC) + abs(SeFCC)) / 2
    phi = (Smix(compNorm) - Sh(compNorm)) / SeMean
    return phi


if __name__ == '__main__':

    BCC = pd.read_csv('BCC.csv', delimiter=';')
    alloysBCC = np.asarray(BCC) / 100
    phiBCC = parPhi(alloysBCC)
    dfBCC = pd.DataFrame(phiBCC)

    LC14 = pd.read_csv('LC14.csv', delimiter=';')
    alloysLC14 = np.asarray(LC14) / 100
    phiLC14 = parPhi(alloysLC14)
    dfLC14 = pd.DataFrame(phiLC14)

    BCC_LC14 = pd.read_csv('BCC_LC14.csv', delimiter=';')
    alloysBCC_LC14 = np.asarray(BCC_LC14) / 100
    phiBCC_LC14 = parPhi(alloysBCC_LC14)
    dfBCC_LC4 = pd.DataFrame(phiBCC_LC14)

    writer = pd.ExcelWriter('phi.xlsx', engine='xlsxwriter')

    dfBCC.to_excel(writer, sheet_name='BCC')
    dfLC14.to_excel(writer, sheet_name='LC14')
    dfBCC_LC4.to_excel(writer, sheet_name='BCC_LC14')

    writer.close()