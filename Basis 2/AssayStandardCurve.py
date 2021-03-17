import math
import cv2
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
from scipy import signal, misc
import scipy.optimize
import peakutils
import json
from statistics import mean
import sys
import pandas as pd
import pprint
from sklearn import linear_model
import statsmodels.api as sm

# Assays dictionary:
# + assay
# ++ concentration
# +++ concentration range (hue, saturation, value)
# ++ units

def powerFuncFitEq(x, A, B, C):
    # invX = 1 / x
    # return A + (B * (x ^ C))
    return (A + (B * (x ** C)))

def fourPtLogisticFitEq(x, A, B, C, D,E):
    # invX = 1 / x
    # return A + (B * (x ^ C))
    return (A + ((B - A) / (E + ((C * x) ** D))))

def fourPtLogisticFitPlusLinear(x,A,B,C,D,E,F,G):

    # invX = 1 / x
    # return A + (B * (x ^ C))
    return (( A / (B + (C * np.exp(-x * D)))) + (E * x) + F)

def cubicFit(x,A,B,C,D,E):

    # invX = 1 / x
    # return A + (B * (x ^ C))
    return ((A*(x**4)) + (B*(x**3)) + (C*(x**2)) + (D*x) + E)

def arccot(x,A,B,C,D,E,F):

    # invX = 1 / x
    # return A + (B * (x ^ C))
    return (((A * (np.arctan((B/x) + C))) + D) + (E*x) + F)

def arcsinhFit(x,A,B,C,D,E,F):

    # invX = 1 / x
    # return A + (B * (x ^ C))
    return (((A * (np.arcsinh((B*x) + C))) + D))

def arcctanFit(x,A,B,C,D,E,F):

    # invX = 1 / x
    # return A + (B * (x ^ C))
    return (((A * (np.arctan((B*x) + C))) + D) + (E*x) + F)

# def cubicFitEq(x, A, B, C, D):
#     # invX = 1 / x
#     # return A + (B * (x ^ C))
#     return ((A*x**3) + (B*x**2) + (C*x) + D)

def plotDataAndFit(xData,yData,xFit,yFit,titleString,fitParamsString,plotShow):

    plt.figure()
    plt.plot(xData, yData, marker='o', color='b', linestyle='-', label='Data')
    plt.plot(xFit, yFit, marker='.', color='r', linestyle='--', label='Fit')
    xLabelString = f'Concentration [{unitString}]'
    plt.xlabel(xLabelString)
    plt.ylabel('Scaled Signal [-]')
    plt.title(titleString)
    ax = plt.gca()
    plt.text(x=0.1,y=0.1,s=fitParamsString,transform=ax.transAxes)
    plt.legend()
    if plotShow:
        plt.show()
    return

# Init assays dictionray:
keys = ['conc','units','notes']
urobilinogen = {}
glucose = {}
bilirubin = {}
ketones = {}
specificGravity = {}
blood = {}
pH = {}
protein = {}
nitrite = {}
leukocytes = {}
ascorbicAcid = {}

urobilinogen['conc'] = {}
glucose['conc'] = {}
bilirubin['conc'] = {}
ketones['conc'] = {}
specificGravity['conc'] = {}
blood['conc'] = {}
pH['conc'] = {}
protein['conc'] = {}
nitrite['conc'] = {}
leukocytes['conc'] = {}
ascorbicAcid['conc'] = {}


urobilinogen['conc'][0.1] = (29,83,242)
urobilinogen['conc'][1.0] = (17,76,229)
urobilinogen['conc'][2.0] = (12,101,220)
urobilinogen['conc'][4.0] = (10,132,232)
urobilinogen['conc'][6.0] = (np.mean([10,3]),np.mean([132, 161]), np.mean([232,209])) # AUGMENTED DATA
urobilinogen['conc'][8.0] = (3,161,209)
urobilinogen['units'] = 'mg/dL'
urobilinogen['notes'] = '0.1 <-> 1.0 is normal'

glucose['conc'][0] = (141,102,212)
glucose['conc'][50] = (np.mean([141,58]),np.mean([102, 82]), np.mean([212,177])) # AUGMENTED DATA
glucose['conc'][100] = (58,82,177)
glucose['conc'][250] = (30,140,173)
glucose['conc'][500] = (16,152,161)
glucose['conc'][750] = (np.mean([16,8]),np.mean([152, 181]), np.mean([161,147])) # AUGMENTED DATA
glucose['conc'][1000] = (8,181,147)
glucose['units'] = 'mg/dL'
glucose['notes'] = ''

bilirubin['conc'][0] = (26,91,239)
bilirubin['conc'][1.0] = (22,99,231)
bilirubin['conc'][2.0] = (17,118,212)
bilirubin['conc'][3.0] = (10,119,200)
bilirubin['units'] = '-'
bilirubin['notes'] = '+, ++, +++'

ketones['conc'][0] = (23,87,239)
ketones['conc'][5] = (18,104,226)
ketones['conc'][15] = (10,110,233)
ketones['conc'][40] = (0,125,190)
ketones['conc'][100] = (0.9, 148,160)
ketones['units'] = 'mg/dL'
ketones['notes'] = ''

specificGravity['conc'][1.0] = (135,185,116)
specificGravity['conc'][1.005] = (104,63,102)
specificGravity['conc'][1.0075] = (np.mean([104,43]),np.mean([63, 70]), np.mean([102,133])) # AUGMENTED DATA
specificGravity['conc'][1.010] = (43,70,133)
specificGravity['conc'][1.015] = (35,125,136)
specificGravity['conc'][1.020] = (30,147,166)
specificGravity['conc'][1.025] = (26,180,163)
specificGravity['conc'][1.030] = (22,185,200)
specificGravity['units'] = '-'
specificGravity['notes'] = ''

blood['conc'][0] = (35,170,238)
blood['conc'][10] = (43,125,183)
blood['conc'][50] = (70,79,134)
blood['conc'][250] = (134,112,101)
blood['units'] = 'RBC/uL'
blood['notes'] = '*Need to add in non-hemolysis colors*'

pH['conc'][5.0] = (17,159,240)
pH['conc'][6.0] = (24,162,229)
pH['conc'][6.5] = (24,156,186)
pH['conc'][7.0] = (35,138,174)
pH['conc'][7.5] = (np.mean([35,86]),np.mean([138, 68]), np.mean([174,129])) # AUGMENTED DATA
pH['conc'][8.0] = (86,68,129)
pH['conc'][9.0] = (134,101,92)
pH['units'] = '-'
pH['notes'] = ''

protein['conc'][0] = (35,156,231)
protein['conc'][15] = (35,132,207)
protein['conc'][30] = (36,132,177)
protein['conc'][100] = (42,100,162)
protein['conc'][300] = (49,75,177)
protein['conc'][1300/2] = (np.mean([49,62]),np.mean([75, 61]), np.mean([177,152])) # AUGMENTED DATA
protein['conc'][1000] = (62,61,152)
protein['units'] = 'mg/dL'
protein['notes'] = '15 is considered trace'

nitrite['conc'][0.0] = (27,86,239)
nitrite['conc'][1.0] = (11,82,228)
nitrite['conc'][2.0] = (248,162,187)
nitrite['units'] = '-'
nitrite['notes'] = '0 = Neg; 1.0 = Trace; 2.0 = Pos.'

leukocytes['conc'][0.0] = (22,87,237)
leukocytes['conc'][25] = (19,107,225)
leukocytes['conc'][np.mean([25,75])] = (np.mean([19,10]),np.mean([107, 94]), np.mean([225,204])) # AUGMENTED DATA
leukocytes['conc'][75] = (10,94,204)
leukocytes['conc'][np.mean([75,500])] = (np.mean([10,249]),np.mean([94, 94]), np.mean([204,175])) # AUGMENTED DATA
leukocytes['conc'][500] = (249,94,175)
leukocytes['units'] = 'WBC/uL'
leukocytes['notes'] = ''

ascorbicAcid['conc'][0.0] = (122,216,139)
ascorbicAcid['conc'][20] = (67,125,158)
ascorbicAcid['conc'][40] = (36,202,179)
ascorbicAcid['units'] = 'mg/dL'
ascorbicAcid['notes'] = ''

# + Creating list of assays to add to dictionary
assayList = ['urobilinogen', 'glucose', 'bilirubin', 'ketones', 'specificGravity', 'blood', 'pH', 'protein', 'nitrite',
             'leukocytes','ascorbicAcid']

assaysDict = {}

# - Looping over list
for assay in assayList:

    # + Adding individual assay dictionary to compiled dictionary
    assaysDict[assay] = eval(assay)


# pprint.pprint(assaysDict)

# assayDF = pd.DataFrame.from_dict(assaysDict,orient='index')
# print(assayDF.to_string())


for assay in assaysDict:

    # print(f"***\n{assay}\n\t{assaysDict[assay]['notes']}")
    # print(assay)
    assayConcDict = assaysDict[assay]['conc']

    concList = []
    hueList = []
    satList = []
    valList = []
    unitString = assaysDict[assay]['units']

    for conc in assayConcDict:
        concList.append(conc)
        hueList.append(assayConcDict[conc][0])
        satList.append(assayConcDict[conc][1])
        valList.append(assayConcDict[conc][2])

    # Re-work Hue as different Pure Red
    refColor = (0,0,0) # pure red hue, zero sat and val
    hueScaledList = []
    satScaledList = []
    valScaledList = []

    for iHue, iSat, iVal in zip(hueList, satList, valList):
        dHue = min(abs(iHue - refColor[0]), 255.0 - abs(iHue - refColor[0])) / 128 # / 180.0
        dSat = abs(iSat - refColor[1]) / 255.0
        dVal = abs(iVal - refColor[2]) / 255.0
        # print(f'Hue: {iHue}\tdHue: {dHue}')
        # print(f'Sat: {iSat}\tdSat: {dSat}')
        # print(f'Val: {iVal}\tdVal: {dVal}')

        hueScaledList.append(dHue)
        satScaledList.append(dSat)
        valScaledList.append(dVal)

    assaysDict[assay]['scaledHue'] = hueScaledList
    assaysDict[assay]['scaledSat'] = satScaledList
    assaysDict[assay]['scaledVal'] = valScaledList
    # print(concList)
    # print(hueScaledList)
    # print(satScaledList)
    # print(valScaledList)






# More accurate? Multi-variate regression

# Find distance between Hue and Reference, say 0 (red)
# Normalize this back to between 0 -> 255 (or normalize all between 0 -> 1)
# https://stackoverflow.com/questions/35113979/calculate-distance-between-colors-in-hsv-space

# Crude: Standalone image processing target to use for

# urobilinogen -> Sat
# glucose -> Hue
# bilirubin -> Val, Sat
# ketones -> Val, Sat
# specificGravity -> Hue, Val
# blood -> Hue, Val
# pH -> Hue, Val
# protein -> Sat
# nitrite -> Sat
# leukocytes -> Val
# ascorbicAcid -> Val

# + Fitting to polynomials
# - Urobilinogen -> Sat
# print(list(assaysDict['urobilinogen']['conc'].keys()))
urobilinogenConc = list(assaysDict['urobilinogen']['conc'].keys())
# print(urobilinogenConc)
urobilinogenScaledSatList = assaysDict['urobilinogen']['scaledSat']
# urobilinogenParams = np.polyfit(urobilinogenConc, urobilinogenSatList, 8)
# urobilinogenFit = np.poly1d(urobilinogenParams)

uroFitParams, uroFitCov = sp.optimize.curve_fit(fourPtLogisticFitEq, urobilinogenConc, urobilinogenScaledSatList, maxfev=100000)
urobilinogenFitParamsString = f"4 Pt Logistic: y = A + ((B - A) / (E + ((C * x) ** D)))\nUrobilinogen Fit Params: \n" + str([f"{x:.8f}" for x in uroFitParams])
urobilinogenXFit = np.linspace(urobilinogenConc[0],urobilinogenConc[-1],100)
print(urobilinogenFitParamsString)
plotDataAndFit(urobilinogenConc,urobilinogenScaledSatList,urobilinogenXFit,fourPtLogisticFitEq(urobilinogenXFit, *uroFitParams),'Urobilinogen (Saturation)',urobilinogenFitParamsString,True)

# - Glucose -> Hue
glucoseConc = list(assaysDict['glucose']['conc'].keys())
glucoseScaledHueList = assaysDict['glucose']['scaledHue']
glucloseFitParams, glucoseFitCov = sp.optimize.curve_fit(arccot, glucoseConc, glucoseScaledHueList, maxfev=100000)
glucloseFitParamsString = f"ArcCot Func: y = ((A * (np.arctan((B/x) + C))) + D) + (E*x) + F)\nGlucose Fit Params: \n" + str([f"{x:.8f}" for x in glucloseFitParams])
glucloseXFit = np.linspace(glucoseConc[0],glucoseConc[-1],100)
print(glucloseFitParamsString)
plotDataAndFit(glucoseConc,glucoseScaledHueList,glucloseXFit,arccot(glucloseXFit, *glucloseFitParams),'Glucose (Hue)',glucloseFitParamsString,True)

# Bilirubin -> Val, Sat
bilirubinConc = list(assaysDict['bilirubin']['conc'].keys())
bilirubinScaledValList = assaysDict['bilirubin']['scaledVal']
bilirubinFitParams, bilirubinFitCov = sp.optimize.curve_fit(powerFuncFitEq, bilirubinConc, bilirubinScaledValList, maxfev=100000)
bilirubinFitParamsString = f"Power Func: y = A + (B * (x ** C))\nBilirubin Fit Params: \n" + str([f"{x:.8f}" for x in bilirubinFitParams])
bilirubinXFit = np.linspace(bilirubinConc[0],bilirubinConc[-1],100)
print(bilirubinFitParamsString)
plotDataAndFit(bilirubinConc,bilirubinScaledValList,bilirubinXFit,powerFuncFitEq(bilirubinXFit, *bilirubinFitParams),'Bilirubin (Val)',bilirubinFitParamsString,True)

# Ketones -> Val, Sat
ketonesConc = list(assaysDict['ketones']['conc'].keys())
ketonesScaledSatList = assaysDict['ketones']['scaledSat']
ketonesFitParams, ketonesFitCov = sp.optimize.curve_fit(powerFuncFitEq, ketonesConc, ketonesScaledSatList, maxfev=100000)
ketonesFitParamsString = f"Power Func: y = A + (B * (x ** C))\nKetones Fit Params: \n" + str([f"{x:.8f}" for x in ketonesFitParams])
ketonesXFit = np.linspace(ketonesConc[0],ketonesConc[-1],100)
print(ketonesFitParamsString)
plotDataAndFit(ketonesConc,ketonesScaledSatList,ketonesXFit,powerFuncFitEq(ketonesXFit, *ketonesFitParams),'Ketones (Sat)',ketonesFitParamsString,True)

# SpecificGravity -> Hue, Val
specificGravityConc = list(assaysDict['specificGravity']['conc'].keys())
specificGravityScaledHueList = assaysDict['specificGravity']['scaledHue']
specificGravityFitParams, specificGravityFitCov = sp.optimize.curve_fit(arccot, specificGravityConc, specificGravityScaledHueList, maxfev=100000)#, p0=[1,1,1,1,1])
specificGravityFitParamsString = f"ArcCot Func: y = ((A * (np.arctan((B/x) + C))) + D) + (E*x) + F)\nSpecific Gravity Fit Params: \n" + str([f"{x:.8f}" for x in specificGravityFitParams])
specificGravityXFit = np.linspace(specificGravityConc[0],specificGravityConc[-1],100)
print(specificGravityFitParamsString)
plotDataAndFit(specificGravityConc,specificGravityScaledHueList,specificGravityXFit,arccot(specificGravityXFit, *specificGravityFitParams),'Specific Gravity (Hue)',specificGravityFitParamsString,True)


# blood -> Hue, Val
bloodConc = list(assaysDict['blood']['conc'].keys())
bloodScaledHueList = assaysDict['blood']['scaledHue']
bloodFitParams, bloodFitCov = sp.optimize.curve_fit(powerFuncFitEq, bloodConc, bloodScaledHueList, maxfev=100000)
bloodFitParamsString = f"Power Func: y = A + (B * (x ** C))\nBlood Fit Params: \n" + str([f"{x:.8f}" for x in bloodFitParams])
bloodXFit = np.linspace(bloodConc[0],bloodConc[-1],100)
print(bloodFitParamsString)
plotDataAndFit(bloodConc,bloodScaledHueList,bloodXFit,powerFuncFitEq(bloodXFit, *bloodFitParams),'Blood (Hue)',bloodFitParamsString,True)


# pH -> Hue, Val
pHConc = list(assaysDict['pH']['conc'].keys())
pHScaledHueList = assaysDict['pH']['scaledHue']
pHFitParams, pHFitCov = sp.optimize.curve_fit(fourPtLogisticFitEq, pHConc, pHScaledHueList, maxfev=100000)
pHFitParamsString = f"Power Func: y = A + (B * (x ** C))\npH Fit Params: \n" + str([f"{x:.8f}" for x in pHFitParams])
pHXFit = np.linspace(pHConc[0],pHConc[-1],100)
print(pHFitParamsString)
plotDataAndFit(pHConc,pHScaledHueList,pHXFit,fourPtLogisticFitEq(pHXFit, *pHFitParams),'pH (Hue)',pHFitParamsString,True)


# protein -> Sat
proteinConc = list(assaysDict['protein']['conc'].keys())
proteinScaledSatList = assaysDict['protein']['scaledSat']
proteinFitParams, proteinFitCov = sp.optimize.curve_fit(arccot, proteinConc, proteinScaledSatList, maxfev=100000)
proteinFitParamsString = f"ArcCot Func: y = ((A * (np.arctan((B/x) + C))) + D) + (E*x) + F)\nProtein Fit Params: \n" + str([f"{x:.8f}" for x in proteinFitParams])
proteinXFit = np.linspace(proteinConc[0],proteinConc[-1],100)
print(proteinFitParamsString)
plotDataAndFit(proteinConc,proteinScaledSatList,proteinXFit,arccot(proteinXFit, *proteinFitParams),'protein (Sat)',proteinFitParamsString,True)


# nitrite -> Sat
nitriteConc = list(assaysDict['nitrite']['conc'].keys())
nitriteScaledSatList = assaysDict['nitrite']['scaledSat']
nitriteFitParams, nitriteFitCov = sp.optimize.curve_fit(powerFuncFitEq, nitriteConc, nitriteScaledSatList, maxfev=100000)
nitriteFitParamsString = f"Power Func: y = A + (B * (x ** C))\nNitrite Fit Params: \n" + str([f"{x:.8f}" for x in nitriteFitParams])
nitriteXFit = np.linspace(nitriteConc[0],nitriteConc[-1],100)
print(nitriteFitParamsString)
plotDataAndFit(nitriteConc,nitriteScaledSatList,nitriteXFit,powerFuncFitEq(nitriteXFit, *nitriteFitParams),'nitrite (Sat)',nitriteFitParamsString,True)


# leukocytes -> Val
leukocytesConc = list(assaysDict['leukocytes']['conc'].keys())
leukocytesScaledValList = assaysDict['leukocytes']['scaledVal']
leukocytesFitParams, leukocytesFitCov = sp.optimize.curve_fit(arccot, leukocytesConc, leukocytesScaledValList, maxfev=100000)
leukocytesFitParamsString = f"ArcCot Func: y = ((A * (np.arctan((B/x) + C))) + D) + (E*x) + F)\nLeukocytes Fit Params: \n" + str([f"{x:.8f}" for x in leukocytesFitParams])
leukocytesXFit = np.linspace(leukocytesConc[0],leukocytesConc[-1],100)
print(leukocytesFitParamsString)
plotDataAndFit(leukocytesConc,leukocytesScaledValList,leukocytesXFit,arccot(leukocytesXFit, *leukocytesFitParams),'leukocytes (Sat)',leukocytesFitParamsString,True)


# ascorbicAcid -> Val
ascorbicAcidConc = list(assaysDict['ascorbicAcid']['conc'].keys())
ascorbicAcidScaledValList = assaysDict['ascorbicAcid']['scaledVal']
ascorbicAcidFitParams, ascorbicAcidFitCov = sp.optimize.curve_fit(powerFuncFitEq, ascorbicAcidConc, ascorbicAcidScaledValList, maxfev=100000)
ascorbicAcidFitParamsString = f"Power Func: y = A + (B * (x ** C))\nAscorbic Acid Fit Params: \n" + str([f"{x:.8f}" for x in ascorbicAcidFitParams])
print(ascorbicAcidFitParamsString)
ascorbicAcidXFit = np.linspace(ascorbicAcidConc[0],ascorbicAcidConc[-1],100)
plotDataAndFit(ascorbicAcidConc,ascorbicAcidScaledValList,ascorbicAcidXFit,powerFuncFitEq(ascorbicAcidXFit, *ascorbicAcidFitParams),'ascorbicAcid (Sat)',ascorbicAcidFitParamsString,True)


# 'urobilinogen': {
# 'param': 'saturation',
# 'fitEq': ((0.6640 * (np.arctan((0.1623/x) + 0.3322))) + 2.3982) + (1.4109*x)
# 'concRange': list(np.linspace(0,1000,100))
# }

# 'glucose': {
# 'param': hue',
# 'fitEq': (((2705.6435 * (np.arctan((427290.0633/x) + 1966.8450))) + -38771.1625) + (0.00028822*x) + 34522.0699)
# 'concRange': list(np.linspace(0,1000,100))
# }

# 'bilirubin': {
# 'param': 'val',
# 'fitEq': (0.9397 + (-0.0426 * (x ** 1.2008)),
# 'concRange': list(np.linspace(0,3,100))
# }

# 'ketones': {
# 'param': 'sat',
# 'fitEq': 0.3435 + (0.0252 * (x ** 0.4839)),
# 'concRange': list(np.linspace (0, 100, 100))
# }

# 'specificGravity': {
# 'param': 'hue',
# 'fitEq': ((0.2188 * (np.arctan((684.0730/x) + -679.1174))) + 50987.0780) + (-4.0613*x) + -50982.3893)
# 'concRange': list(np.linspace(1,1.03,100))
# }

# 'blood': {
# 'param': 'hue',
# 'fitEq': 0.2629 + (0.0255 * (x ** 0.5965))
# 'concRange': list(np.linspace(0, 250, 100))
# }

# 'pH': {
# 'param': 'hue',
# 'fitEq': (1.0264' + ((-2.8194 - 1.0264') / (4.3717 + ((0.1410 * x) ** 15.7145)))),
# 'concRange': list(np.linspace(5,9,100))
# }

# 'protein': {
# 'param': 'sat',
# 'fitEq': ((1952.8780 * (np.arctan((404271.9767/x) + 5184.5437))) + -2163.3303) + (-0.00001234*x) + -903.6394)
# 'concRange': list(np.linspace(0, 1000,100)),
# }

# 'nitrite': {
# 'param': 'sat',
# 'fitEq':  0.3294 + (0.00000005 * (x ** 22.4571)),
# 'concRange': list(np.linspace(0, 2, 100))
# }

# 'leukocytes': {
# 'param': 'sat',
# 'fitEq': ((22.1788 * (np.arctan((6849.0873/x) + 118.4425))) + 16850.7943) + (-0.00014901*x) + -16884.7002),
# 'concRange': list(np.linspace(0, 500, 100)
# }

# 'ascorbicAcid': {
# 'param': 'sat',
# 'fitEq': 0.54509804 + (0.00298474 * (x ** 1.07400058)),
# 'concRange': list(np.linspace(0, 40,100))
# }