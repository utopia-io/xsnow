# Snow profiles in regobslib

regobslib is a client library for Norway's field observation database. It is therefore focused on making code that uses the library as legible as possible. It is constructed using ordinary Python classes and makes heavy use of enums for categorical data types.

## Initializing a snow profile

We can initialize a profile like this:
```
from src.regobslib import SnowProfile
import json

profile = SnowProfile(layers=[SnowProfile.Layer(15,
                                                SnowProfile.Hardness.ONE_FINGER,
                                                SnowProfile.GrainForm.PP,
                                                SnowProfile.GrainSize.TWO,
                                                SnowProfile.Wetness.D,
                                                SnowProfile.Hardness.FOUR_FINGERS,
                                                SnowProfile.GrainForm.DF,
                                                SnowProfile.GrainSize.ONE),
                              SnowProfile.Layer(0.5,
                                                SnowProfile.Hardness.FIST,
                                                SnowProfile.GrainForm.SH,
                                                SnowProfile.GrainSize.FIVE,
                                                critical_layer=SnowProfile.CriticalLayer.WHOLE,
                                                comment="This is what I'm worried about"),
                              SnowProfile.Layer(2,
                                                SnowProfile.Hardness.ICE,
                                                SnowProfile.GrainForm.MF_CR)
                            ],
                      temperatures=[SnowProfile.SnowTemp(10, -4)],
                      densities=[SnowProfile.Density(50, 300)],
                      is_profile_to_ground=False,
                      comment="SH above MFcr. Very PWL. Much dangerous.")
```

If we print this we will get something that looks like a dict. It's not however, we have just defined the method `SnowProfile.__str__` (or rather `Dictable.__str__` if we're pedantic):
```
{'comment': 'SH above MFcr. Very PWL. Much dangerous.',
 'densities': [{'density_kg_per_cubic_metre': 300, 'thickness_cm': 50}],
 'is_profile_to_ground': False,
 'layers': [{'comment': None,
             'critical_layer': None,
             'grain_form_primary': <GrainForm.PP: 1>,
             'grain_form_sec': <GrainForm.DF: 14>,
             'grain_size_max_mm': <GrainSize.ONE: 1.0>,
             'grain_size_mm': <GrainSize.TWO: 2.0>,
             'hardness': <Hardness.ONE_FINGER: 10>,
             'hardness_bottom': <Hardness.FOUR_FINGERS: 6>,
             'thickness_cm': 15,
             'wetness': <Wetness.D: 1>},
            {'comment': "This is what I'm worried about",
             'critical_layer': <CriticalLayer.WHOLE: 13>,
             'grain_form_primary': <GrainForm.SH: 32>,
             'grain_form_sec': None,
             'grain_size_max_mm': None,
             'grain_size_mm': <GrainSize.FIVE: 5.0>,
             'hardness': <Hardness.FIST: 2>,
             'hardness_bottom': None,
             'thickness_cm': 0.5,
             'wetness': None},
            {'comment': None,
             'critical_layer': None,
             'grain_form_primary': <GrainForm.MF_CR: 40>,
             'grain_form_sec': None,
             'grain_size_max_mm': None,
             'grain_size_mm': None,
             'hardness': <Hardness.ICE: 21>,
             'hardness_bottom': None,
             'thickness_cm': 2,
             'wetness': None}],
 'temperatures': [{'depth_cm': 10, 'temp_c': -4}]}
```

As it is a class we find properties as attributes. We can of course convert it to a proper dict if we want to, and index it instead:
```
profile.is_profile_to_ground
profile.to_dict()["is_profile_to_ground"]
```

Since this is a client library for a web API things are serialized/deserialized to the JSON-schema that API uses. This is usually done behind the scenes, but the `SnowProfile.deserialize` and `SnowProfile.serialize` is available if someone would like to do it manually:
```
SnowProfile.deserialize(json.loads("""{
    "StratProfile": {
        "Layers": [
            {
                "Thickness": 0.15,
                "HardnessTID": 10,
                "GrainFormPrimaryTID": 1,
                "GrainSizeAvg": 0.02,
                "WetnessTID": 1,
                "HardnessBottomTID": 6,
                "GrainFormSecondaryTID": 14,
                "GrainSizeAvgMax": 0.01
            },
            {
                "Thickness": 0.005,
                "HardnessTID": 2,
                "GrainFormPrimaryTID": 32,
                "GrainSizeAvg": 0.05,
                "CriticalLayerTID": 13,
                "Comment": "This is what I\'m worried about"
            },
            {
                "Thickness": 0.02,
                "HardnessTID": 21,
                "GrainFormPrimaryTID": 40
            }
        ]
    },
    "SnowTemp": {
        "Layers": [
            {
                "Depth": 0.1,
                "SnowTemp": -4
            }
        ]
    },
    "SnowDensity": [
        {
            "Layers": [
                {
                    "Thickness": 0.5,
                    "Density": 300
                }
            ]
        }
    ],
    "IsProfileToGround": false,
    "Comment": "SH above MFcr. Very PWL. Much dangerous."
}"""))
```