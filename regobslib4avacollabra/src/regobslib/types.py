"""Enums and types used in Regobs
"""
from enum import IntEnum, Enum

from .misc import FloatEnum

__author__ = 'arwi'


class Observation:
    OBSERVATION_TYPE = None


class SnowObservation(Observation):
    pass


class SnowProfile(SnowObservation):
    class Hardness(IntEnum):
        FIST_MINUS = 1
        FIST = 2
        FIST_PLUS = 3
        FIST_TO_FOUR_FINGERS = 4
        FOUR_FINGERS_MINUS = 5
        FOUR_FINGERS = 6
        FOUR_FINGERS_PLUS = 7
        FOUR_FINGERS_TO_ONE_FINGER = 8
        ONE_FINGER_MINUS = 9
        ONE_FINGER = 10
        ONE_FINGER_PLUS = 11
        ONE_FINGER_TO_PEN = 12
        PEN_MINUS = 13
        PEN = 14
        PEN_PLUS = 15
        PEN_TO_KNIFE = 16
        KNIFE_MINUS = 17
        KNIFE = 18
        KNIFE_PLUS = 19
        KNIFE_TO_ICE = 20
        ICE = 21

    class GrainForm(IntEnum):
        PP = 1
        PP_CO = 2
        PP_ND = 3
        PP_PL = 4
        PP_SD = 5
        PP_IR = 6
        PP_GP = 7
        PP_HL = 8
        PP_IP = 9
        PP_RM = 10
        MM = 11
        MM_RP = 12
        MM_CI = 13
        DF = 14
        DF_DC = 15
        DF_BK = 16
        RG = 17
        RG_SR = 18
        RG_LR = 19
        RG_WP = 20
        RG_XF = 21
        FC = 22
        FC_SO = 23
        FC_SF = 24
        FC_XR = 25
        DH = 26
        DH_CP = 27
        DH_PR = 28
        DH_CH = 29
        DH_LA = 30
        DH_XR = 31
        SH = 32
        SH_SU = 33
        SH_CV = 34
        SH_XR = 35
        MF = 36
        MF_CL = 37
        MF_PC = 38
        MF_SL = 29
        MF_CR = 40
        IF = 41
        IF_IL = 42
        IF_IC = 43
        IF_BI = 44
        IF_RC = 45
        IF_SC = 46

    class GrainSize(FloatEnum):
        ZERO_POINT_ONE = 0.1
        ZERO_POINT_THREE = 0.3
        ZERO_POINT_FIVE = 0.3
        ZERO_POINT_SEVEN = 0.3
        ONE = 1.0
        ONE_POINT_FIVE = 1.5
        TWO = 2.0
        TWO_POINT_FIVE = 2.5
        THREE = 3.0
        THREE_POINT_FIVE = 3.5
        FIVE = 5
        FIVE_POINT_FIVE = 5.5
        SIX = 6.0
        EIGHT = 8.0
        TEN = 10.0

    class Wetness(IntEnum):
        D = 1
        D_M = 2
        M = 3
        M_W = 4
        W = 5
        W_V = 6
        V = 7
        V_S = 8
        S = 9

    class CriticalLayer(IntEnum):
        UPPER = 11
        LOWER = 12
        WHOLE = 13
