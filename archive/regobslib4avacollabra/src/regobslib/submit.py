"""Classes representing the submission forms in Regobs v5
"""

# To enable postponed evaluation of annotations (default for 3.10)
from __future__ import annotations

import pprint
from typing import Optional, List, Union, Dict, TypeVar, Type, Callable

from .misc import NoObservationError
import src.regobslib.types as types


__author__ = 'arwi'

# PyCharm can not parse this type, but it is correctly defined.
ObsJson = Dict[str,
               Optional[Union[str,
                              int,
                              float,
                              'ObsJson',
                              List['ObsJson']]]]

ObsDict = Dict[str, Optional[Union[str, int, float, List[Union[str, int, float, 'ObsDict']]]]]

T = TypeVar('T')
U = TypeVar('U')


class Serializable:
    def serialize(self) -> ObsJson:
        raise NotImplementedError()

    @staticmethod
    def _clean(json: ObsJson) -> ObsJson:
        return {k: v for k, v in json.items() if v is not None and v != []}


class Deserializable:
    @classmethod
    def deserialize(cls, json: ObsJson) -> Deserializable:
        raise NotImplementedError()

    @staticmethod
    def _convert(json: ObsJson, idx: str, target: Type[T], target_sec: Optional[Type[U]] = int) -> Union[T, U]:
        elem = json[idx] if idx in json else None
        try:
            return target(elem) if elem is not None else None
        except ValueError:
            return target_sec(elem)

    @staticmethod
    def _apply(json: ObsJson, idx: str, apply: Callable[[Union[str, int, float]], T] = None) -> Optional[T]:
        elem = json[idx] if idx in json else None
        return apply(elem) if elem is not None else None

    @staticmethod
    def _deserialize_to(json: ObsJson, idx: str, target: Type[Deserializable]) -> Deserializable:
        elem = json[idx] if idx in json else None
        return target.deserialize(elem) if elem is not None else None


class Dictable:
    def to_dict(self) -> ObsDict:
        raise NotImplementedError()

    def __str__(self) -> str:
        return pprint.pformat(self.to_dict())


class Observation(types.Observation, Serializable, Deserializable, Dictable):
    OBSERVATION_TYPE = None

    def to_dict(self) -> ObsDict:
        raise NotImplementedError()

    def serialize(self) -> ObsJson:
        raise NotImplementedError()

    @classmethod
    def deserialize(cls, json: ObsJson) -> SnowObservation:
        raise NotImplementedError()


class SnowObservation(types.SnowObservation, Observation):
    def to_dict(self) -> ObsDict:
        raise NotImplementedError()

    def serialize(self) -> ObsJson:
        raise NotImplementedError()

    @classmethod
    def deserialize(cls, json: ObsJson) -> SnowObservation:
        raise NotImplementedError()


class SnowProfile(types.SnowProfile, SnowObservation):
    class Layer(Serializable, Deserializable, Dictable):
        def __init__(self,
                     thickness_cm: float,
                     hardness: SnowProfile.Hardness,
                     grain_form_primary: Optional[SnowProfile.GrainForm] = None,
                     grain_size_mm: Optional[SnowProfile.GrainSize] = None,
                     wetness: Optional[SnowProfile.Wetness] = None,
                     hardness_bottom: Optional[SnowProfile.Hardness] = None,
                     grain_form_sec: Optional[SnowProfile.GrainForm] = None,
                     grain_size_max_mm: Optional[SnowProfile.GrainSize] = None,
                     critical_layer: Optional[SnowProfile.CriticalLayer] = None,
                     comment: Optional[str] = None):
            """A snow layer of a snow profile.

            @param thickness_cm: Layer thickness (in cm).
            @param hardness: Layer hardness (F, 4F, 1F, P, K, I).
            @param grain_form_primary: Primary grain form.
            @param grain_size_mm: Grain size (in mm).
            @param wetness: Moisture content of the layer.
            @param hardness_bottom: Hardness at the bottom of the layer (F, 4F, 1F, P, K, I).
            @param grain_form_sec: Secondary grain form.
            @param grain_size_max_mm: Maximum grain size (in mm).
            @param critical_layer: Is this layer critical, and what part of it?
            @param comment: Comment regarding the snow layer.
            """
            if thickness_cm is not None and thickness_cm < 0:
                raise ValueError("Thickness must be larger than or equal to 0.")

            self.thickness_cm = thickness_cm
            self.hardness = hardness
            self.grain_form_primary = grain_form_primary
            self.grain_size_mm = grain_size_mm
            self.wetness = wetness
            self.hardness_bottom = hardness_bottom
            self.grain_form_sec = grain_form_sec
            self.grain_size_max_mm = grain_size_max_mm
            self.critical_layer = critical_layer
            self.comment = comment

        def to_dict(self) -> ObsDict:
            return {
                "thickness_cm": self.thickness_cm,
                "hardness": self.hardness,
                "grain_form_primary": self.grain_form_primary,
                "grain_size_mm": self.grain_size_mm,
                "wetness": self.wetness,
                "hardness_bottom": self.hardness_bottom,
                "grain_form_sec": self.grain_form_sec,
                "grain_size_max_mm": self.grain_size_max_mm,
                "critical_layer": self.critical_layer,
                "comment": self.comment,
            }

        def serialize(self) -> ObsJson:
            return self._clean({
                'Thickness': self.thickness_cm / 100 if self.thickness_cm is not None else None,
                'HardnessTID': self.hardness,
                'GrainFormPrimaryTID': self.grain_form_primary,
                'GrainSizeAvg': self.grain_size_mm / 100 if self.grain_size_mm is not None else None,
                'WetnessTID': self.wetness,
                'HardnessBottomTID': self.hardness_bottom,
                'GrainFormSecondaryTID': self.grain_form_sec,
                'GrainSizeAvgMax': self.grain_size_max_mm / 100 if self.grain_size_max_mm is not None else None,
                'CriticalLayerTID': self.critical_layer,
                'Comment': self.comment,
            })

        @classmethod
        def deserialize(cls, json: ObsJson) -> SnowProfile.Layer:
            try:
                grain_size_mm = cls._apply(json, "GrainSizeAvg", lambda x: SnowProfile.GrainSize(x * 100))
            except ValueError:
                grain_size_mm = cls._apply(json, "GrainSizeAvg", lambda x: x * 100)
            try:
                grain_size_max_mm = cls._apply(json, "GrainSizeAvgMax", lambda x: SnowProfile.GrainSize(x * 100))
            except ValueError:
                grain_size_max_mm = cls._apply(json, "GrainSizeAvgMax", lambda x: x * 100)
            layer = object.__new__(cls)
            layer.thickness_cm = cls._apply(json, "Thickness", lambda x: x * 100)
            layer.hardness = cls._convert(json, "HardnessTID", SnowProfile.Hardness)
            layer.grain_form_primary = cls._convert(json, "GrainFormPrimaryTID", SnowProfile.GrainForm)
            layer.grain_size_mm = grain_size_mm
            layer.wetness = cls._convert(json, "WetnessTID", SnowProfile.Wetness)
            layer.hardness_bottom = cls._convert(json, "HardnessBottomTID", SnowProfile.Hardness)
            layer.grain_form_sec = cls._convert(json, "GrainFormSecondaryTID", SnowProfile.GrainForm)
            layer.grain_size_max_mm = grain_size_max_mm
            layer.critical_layer = cls._convert(json, "CriticalLayerTID", SnowProfile.CriticalLayer)
            layer.comment = cls._convert(json, "Comment", str)
            return layer

    class SnowTemp(Serializable, Deserializable, Dictable):
        def __init__(self,
                     depth_cm: float,
                     temp_c: float):
            """Snow temperature at a given depth.

            @param depth_cm: The depth measured (in cm from top).
            @param temp_c: The measured temperature (in Celsius).
            """
            if temp_c > 0:
                raise ValueError("Snow temperature must be lower than or equal to 0.")

            self.depth_cm = depth_cm
            self.temp_c = temp_c

        def to_dict(self) -> ObsDict:
            return {
                "depth_cm": self.depth_cm,
                "temp_c": self.temp_c,
            }

        def serialize(self) -> ObsJson:
            return self._clean({
                'Depth': self.depth_cm / 100 if self.depth_cm is not None else None,
                'SnowTemp': self.temp_c,
            })

        @classmethod
        def deserialize(cls, json: ObsJson) -> SnowProfile.SnowTemp:
            temp = object.__new__(cls)
            temp.depth_cm = cls._apply(json, "Depth", lambda x: x * 100)
            temp.temp_c = cls._convert(json, "SnowTemp", float)
            return temp

    class Density(Serializable, Deserializable, Dictable):
        def __init__(self,
                     thickness_cm: float,
                     density_kg_per_cubic_metre: float):
            """Snow density at a given depth.

            @param thickness_cm: The thickness of the sample layer (in cm).
            @param density_kg_per_cubic_metre: The density (in kg/m³).
            """
            if thickness_cm < 0:
                raise ValueError("Thickness must be larger than or equal to 0.")

            self.thickness_cm = thickness_cm
            self.density_kg_per_cubic_metre = density_kg_per_cubic_metre

        def to_dict(self) -> ObsDict:
            return {
                "thickness_cm": self.thickness_cm,
                "density_kg_per_cubic_metre": self.density_kg_per_cubic_metre,
            }

        def serialize(self) -> ObsJson:
            return self._clean({
                'Thickness': self.thickness_cm / 100 if self.thickness_cm is not None else None,
                'Density': self.density_kg_per_cubic_metre,
            })

        @classmethod
        def deserialize(cls, json: ObsJson) -> SnowProfile.Density:
            density = object.__new__(cls)
            density.thickness_cm = cls._apply(json, "Thickness", lambda x: x * 100)
            density.density_kg_per_cubic_metre = cls._convert(json, "Density", float)
            return density

    def __init__(self,
                 layers: List[SnowProfile.Layer] = (),
                 temperatures: List[SnowProfile.SnowTemp] = (),
                 densities: List[SnowProfile.Density] = (),
                 is_profile_to_ground: Optional[bool] = None,
                 comment: Optional[str] = None):
        """A snow profile, including tests, layers, temperatures and densities.

        @param layers: Snow layers included in the profile.
        @param temperatures: Temperatures measured in the snow profile.
        @param densities: Densities measured in the snow.
        @param is_profile_to_ground: Whether the snow profile was dug to the ground.
        @param comment: Comment regarding the snow profile.
        """
        if all(e is None for e in [layers, temperatures, densities, comment]):
            raise NoObservationError("Neither layers, temperatures, densities or comment passed to snow profile.")

        self.layers = layers
        self.temperatures = temperatures
        self.densities = densities
        self.is_profile_to_ground = is_profile_to_ground
        self.comment = comment

    def to_dict(self) -> ObsDict:
        return {
            "layers": list(map(lambda x: x.to_dict(), self.layers)) if self.layers is not None else None,
            "temperatures": list(
                map(lambda x: x.to_dict(), self.temperatures)
            ) if self.temperatures is not None else None,
            "densities": list(map(lambda x: x.to_dict(), self.densities)) if self.densities is not None else None,
            "is_profile_to_ground": self.is_profile_to_ground,
            "comment": self.comment,
        }

    def serialize(self) -> ObsJson:
        return self._clean({
            'StratProfile': self._clean({
                'Layers': list(map(lambda x: x.serialize(), self.layers)),
            }),
            'SnowTemp': self._clean({
                'Layers': list(map(lambda x: x.serialize(), self.temperatures)),
            }),
            'SnowDensity': [self._clean({
                'Layers': list(map(lambda x: x.serialize(), self.densities)),
            })] if len(self.densities) else None,
            'IsProfileToGround': self.is_profile_to_ground,
            'Comment': self.comment,
        })

    @classmethod
    def deserialize(cls, json: ObsJson) -> SnowProfile:
        layers = None
        if "StratProfile" in json and json["StratProfile"] is not None:
            if "Layers" in json["StratProfile"] and json["StratProfile"]["Layers"] is not None:
                layers = list(map(lambda x: SnowProfile.Layer.deserialize(x), json["StratProfile"]["Layers"]))
        temperatures = None
        if "SnowTemp" in json and json["SnowTemp"] is not None:
            if "Layers" in json["SnowTemp"] and json["SnowTemp"]["Layers"] is not None:
                temperatures = list(map(lambda x: SnowProfile.SnowTemp.deserialize(x), json["SnowTemp"]["Layers"]))
        densities = None
        if "SnowDensity" in json and json["SnowDensity"] is not None:
            for d_elem in json["SnowDensity"]:
                if "Layers" in d_elem and d_elem["Layers"] is not None:
                    if densities is None:
                        densities = []
                    densities += list(map(lambda x: SnowProfile.Density.deserialize(x), d_elem["Layers"]))
        profile = object.__new__(cls)
        profile.layers = layers
        profile.temperatures = temperatures
        profile.densities = densities
        profile.is_profile_to_ground = cls._convert(json, "IsProfileToGround", bool)
        profile.comment = cls._convert(json, "Comment", str)
        return profile
