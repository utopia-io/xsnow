"""
Sample data utilities for xsnow testing and tutorials.

This module uses `pooch` to manage a registry of small SNOWPACK
test datasets hosted on GitLab. These datasets are downloaded on first access,
cached locally in a project-specific directory, and then reused in future runs.

The provided helper functions are intended to support unit tests, tutorials, and development:

* `single_profile()`: A single, simple profile wrapped into an `xsnowDataset`.

* `single_profile_timeseries()`: An `xsnowDataset` from a single location containing many timestamps.

* `snp_gridded_ds()`: A sample `xsnowDataset` of gridded SNOWPACK simulations

For more complex tutorials or integration tests:

* `snp_gridded_dir()`: A locally cached directory tree containing gridded simulations  

* `snp_snowobs_dir()`: A locally cached directory tree of manual & modeled profiles
  
Notes
-----
- All files are verified by SHA256 hash on download to avoid corrupt files.
- Cached files are never automatically deleted; users can clear the cache
  manually by removing the relevant cache directories (`~/.cache/xsnow-*`)
"""

import pooch
from xsnow import read, xsnowDataset

URL_SNP_TESTDATA = "https://gitlab.com/avacollabra/postprocessing/sample-data/-/raw/main/"
REGISTRY_SNP_GRIDDED = {
    "pros/gridded/VIR1A.pro": "sha256:f1322f04f4e3250b377bdab3e17695396378df1c5bc66a94dc31af322ec56fb0",
    "pros/gridded/VIR1A.smet": "sha256:958f6c24dff650fa866303f8b2e6f75d37f13585893b4e99c3371434544d136d",
    "pros/gridded/VIR1A1.pro": "sha256:554e2b3120c8c0270aee1e8164326d36f8dab87688bdc4f682a1a7196d2ed6e0",
    "pros/gridded/VIR1A1.smet": "sha256:7a59456895ec0a486cc39916e80a2a44ce6ae987247969975976f17c0278f3ac",
    "pros/gridded/VIR1A2.pro": "sha256:cebeaab904cadc2a179f78f7492b93c415719aa8f2e9dfb4c56dd071ccb8c9ef",
    "pros/gridded/VIR1A2.smet": "sha256:00ed6b0497b6cf2f0df18cf2b3c6c6c265733d03cb246d1154656401c47a70b2",
    "pros/gridded/VIR1A3.pro": "sha256:430199607fb0d1daf69f357b9de664a3cb959dfc7b3c9cb76f19a7b48256f41a",
    "pros/gridded/VIR1A3.smet": "sha256:badb32cbd48b375cebe01f227d5c4e305fda0674d45641e47e7015265ac5d672",
    "pros/gridded/VIR1A4.pro": "sha256:36103603f1eb6da93e6658c1dc37e258de7a2d389cd1ac5fce5616d0de741cf9",
    "pros/gridded/VIR1A4.smet": "sha256:0de605cf26f906ea3c18c2aee24bda1b36bc49bf2c4b783af6ee0e42e34de336",
    "pros/gridded/VIR2A.pro": "sha256:c4b841fd02351c613317565684d8cedef9861def53a51ae60bb483d4f4333e28",
    "pros/gridded/VIR2A.smet": "sha256:6594c6e89e3c32b0d93b9a9dd67094a0388e7ef730ca569d8b34cbeffb2f6efc",
    "pros/gridded/VIR2A1.pro": "sha256:4e146b2a7fa34b7a84f97bd963b088d387a2fbc6e652384c7c72ff01388138c3",
    "pros/gridded/VIR2A1.smet": "sha256:71ef82663023a04871be92639f04871dcd3bd081c8ab2656503b30dc3f9082a4",
    "pros/gridded/VIR2A2.pro": "sha256:05f83e1b8eba9585a083e291edb66564649c1db3df53616e8f9efcb6f719b215",
    "pros/gridded/VIR2A2.smet": "sha256:74e59d6ed0d8725ba957b0aefafe9bce834bf7d87c8a403e990e17d09aa8e67e",
    "pros/gridded/VIR2A3.pro": "sha256:773f1bcf3fe219fd8fd7cf4e019f09986eac07e527070482bb1b0e3ef09b1f0d",
    "pros/gridded/VIR2A3.smet": "sha256:f9f120b0a0e6dc57e2a114ead6c0e9dbfb87211cf2ff12561ef3583a6dd04cea",
    "pros/gridded/VIR2A4.pro": "sha256:387b384b08607e04f8105f71dacd139aaadc4a7804c6209b78441b95d0caf7c4",
    "pros/gridded/VIR2A4.smet": "sha256:a503d6cdbc8380fd454bcef6b3e02b8d8c1176eed4b57270c8d286396a076c52",
    "pros/gridded/VIR3A.pro": "sha256:656a87720da9fe7ce8559f1e4e89e51d99294a366041d0fcca89ea83cd4a2dfd",
    "pros/gridded/VIR3A.smet": "sha256:2c64c8ee0e3dffa510f218ef040816db4330ed47e3b4dd7b9d7d7880f5b5a0ff",
    "pros/gridded/VIR3A1.pro": "sha256:725ec149d590296e87322f945cc8a14f264d147d171536a649999953e795b4c0",
    "pros/gridded/VIR3A1.smet": "sha256:61f8fbe8c04585a91885da0547444e147aac70ffe8333c459083c962dcdc2c03",
    "pros/gridded/VIR3A2.pro": "sha256:7d124b01167d94e0eeed9660561cd321782623a28c3aff2bf70cb1b0cd566ca9",
    "pros/gridded/VIR3A2.smet": "sha256:6ece4bc665fcac3ac866b9829f31f8a678d6abaa0e1f4981eb7998d70b2f120f",
    "pros/gridded/VIR3A3.pro": "sha256:1bad035e435a76b6560129f5ec9c8be717a26e1b5064588f393a70fdb164a8ad",
    "pros/gridded/VIR3A3.smet": "sha256:74d5ba378b43fd8b5af38f858f7ac1dd500020b98c9e63b765f17951c3525c84",
    "pros/gridded/VIR3A4.pro": "sha256:f0c662f91d01f3516796f5207976de92189b61466918c0da47386b1c031732de",
    "pros/gridded/VIR3A4.smet": "sha256:82b029bdc6f9fb2a0a67918bace37ba9bf72adb3534eea71fde82231fe3adc86",
    "pros/gridded/VIR4A.pro": "sha256:bedfffb9b50b45e43a493950dd10988ded8caf5a6143f0452f50a520f3501cfc",
    "pros/gridded/VIR4A.smet": "sha256:d636c9560f61328ed090536b4a4d3682cf6d720cb4db7879ffd7fde0dac87f79",
    "pros/gridded/VIR4A1.pro": "sha256:1a639075f2f7f59700d753a61c35d4ea0fd36f15ffafe89df527987d2474c246",
    "pros/gridded/VIR4A1.smet": "sha256:295598d8c900ff7a0e86633c3ab5e07730f8f6c927dfa37f749d06eabc72f551",
    "pros/gridded/VIR4A2.pro": "sha256:0765c70b19826e4c1e0abd9090ea19c828baf770808e32a0aad52d828c0d4510",
    "pros/gridded/VIR4A2.smet": "sha256:51f6f67e3813b047add6105fd980d5db7a952015d59384a7e14c741ea2bc1ed8",
    "pros/gridded/VIR4A3.pro": "sha256:f1a2707d9baeea35085e5651e2705ee0494e4601ef41a434b1f029e5d276f256",
    "pros/gridded/VIR4A3.smet": "sha256:2d871280f20b650af317df1f8fe466d83d86289c766ac1ca5daf84afd0f94282",
    "pros/gridded/VIR4A4.pro": "sha256:060694918abd487eb7b6fc54015430d9ba84814da8492c1e1ea6554221e4db82",
    "pros/gridded/VIR4A4.smet": "sha256:f5cd13346e6958477f4e72092c5a44ac56abae48a46997d354fba6ecce98c4e9",
    "pros/gridded/VIR5A.pro": "sha256:eaea3c79c82f8400311804d4858180129da6f1fdd1a3d3c824c9e67dde886c2f",
    "pros/gridded/VIR5A.smet": "sha256:d0666fd5ff2cd361188caf997fe8c1bb22beedcb1e3e5f5958aa6b81915c119a",
    "pros/gridded/VIR5A1.pro": "sha256:4b0e1aa3ff1589c94221e172555c5f38c7cdfe519f77fad451e66917a1d595d1",
    "pros/gridded/VIR5A1.smet": "sha256:dd2e99a08fec337fd33c603b00c6f344e37e647cbed7fdbf1d06fe5854d498e9",
    "pros/gridded/VIR5A2.pro": "sha256:02c266bd6b4e1702f6d1ae6a8550f1a13d1d1ac90ecb16133c1a9b8810f7e32d",
    "pros/gridded/VIR5A2.smet": "sha256:d274eace58bbf52350d11a6ecc71d23cf3edafd323acef2046ddf5fcfdd24a03",
    "pros/gridded/VIR5A3.pro": "sha256:499ea426bc0975bedd14ce97f77c49b14ff8e27d5848c1f633bb142b54a49e2a",
    "pros/gridded/VIR5A3.smet": "sha256:b3a80152069ff10e65d806fec0b395d545752808ea74ecf04dea57c8ccc3b8db",
    "pros/gridded/VIR5A4.pro": "sha256:b875ee95d29cb3bfb1e4a4ce29ded4c1b05c96d887ab748ff9ccec28c5496bfd",
    "pros/gridded/VIR5A4.smet": "sha256:a57139e2ae0f0869f0fc7321c3c2000845bdb2b6fba25a500886d339a7626925",
    "smets/gridded/forecast/VIR1A.smet": "sha256:a4eca3aabc606b79edb25b4060b60189a1b0e5ff5d08eef5369e7b5a54a9c923",
    "smets/gridded/forecast/VIR2A.smet": "sha256:1de7a1e2abe204537c03053c658a072c5c31fc4ec48154d7d5c2988dca028647",
    "smets/gridded/forecast/VIR3A.smet": "sha256:6bb011556d6cef148df6d60b9c70e320bf2b0405fb2c52833ba386d377f99911",
    "smets/gridded/forecast/VIR4A.smet": "sha256:d9f001889c604f6e5c1234ce429dcad9d0942fd7ebc50124da0eeeb918594802",
    "smets/gridded/forecast/VIR5A.smet": "sha256:90d8ceca2cbc90fa5ce4d515b177063dbad8547cff1c8150e70f1031fd23c89f",
    "smets/gridded/nowcast/VIR1A.smet": "sha256:e7a05ec0f8ecaaf1c8fbff8edbcc782063b0c5307fba2273e3e3d5d96a250483",
    "smets/gridded/nowcast/VIR2A.smet": "sha256:d697cdb52050e038d366492a8cf14fc87e546f5cdbb40ff355011c1939c9b589",
    "smets/gridded/nowcast/VIR3A.smet": "sha256:d05add25b736a459ee017c52be90a787c1a0d73e340769f25ec862a8f531622a",
    "smets/gridded/nowcast/VIR4A.smet": "sha256:5d6d52ab8602c15f944b1283f2d990e2a93d6a094f9b94bcf8493ccbea919f4d",
    "smets/gridded/nowcast/VIR5A.smet": "sha256:81e05e0bbb3c3c2d3a9fea86b955112057d690e621f26f7de577d7d68c350a89",
}

SNP_GRIDDED = pooch.create(
    path=pooch.os_cache("xsnow-snp-gridded"),  # cache dir, e.g. ~/.cache/xsnow-snp-gridded
    base_url=URL_SNP_TESTDATA, 
    registry=REGISTRY_SNP_GRIDDED,
)

def snp_gridded_dir() -> str:
    """
    Get the local directory containing all SNOWPACK files from a sample gridded simulation.

    The function ensures that all registered SNOWPACK files (input `.smet` drivers and
    output `.pro` & `.smet` profiles) are downloaded, and return the absolute cache
    directory path.
    
    Returns:
    --------
        A path to a locally cached directory, where:
          - dir/smets/gridded/[nowcast|forecast]/: contains the .smet files used to drive SNOWPACK
          - dir/pros/gridded/: contains the SNOWPACK output files (.pro and .smet)

    Examples:
    ---------
      >>> import os
      >>> from xsnow.sample_data import snp_gridded_dir
      >>> gridded_dir = snp_gridded_dir()
      >>> os.listdir(f"{gridded_dir}/pros/gridded")
    """
    for fname in REGISTRY_SNP_GRIDDED:
        SNP_GRIDDED.fetch(fname)

    return str(SNP_GRIDDED.path)


def snp_gridded_ds() -> xsnowDataset:
    """
    Get a sample dataset representing a gridded SNOWPACK simulation
    
    Returns:
    --------
        A sample xsnowDataset

    Examples:
    ---------
      >>> from xsnow.sample_data import snp_gridded_ds
      >>> ds = snp_gridded_ds()
      >>> ds
    """
    snp_dir = snp_gridded_dir()
    ds = read(f"{snp_dir}/pros/gridded")

    return ds


REGISTRY_SNP_SNOWOBS = {
    "pits/LawisProfile20908.caaml": "sha256:b0a68d690985c72c6d1d3ce903d420fe8ee8c9f8b31d256ca67ca93c3bb12d67",
    "pits/LawisProfile20908.pro": "edcdd5fa16a8d953b207012c878efbf126d070057db91349812686f37fab1f58",
    "pits/LawisProfile20915.caaml": "sha256:4df0e9179911f0be0bcf7ce787e568a7820ffb7fb51f781730a9aa5938fa043b",
    "pits/LawisProfile20915.pro": "4582a6a973b54389a8e8aca6473b2c18e136adc340edb37d677e3a2467fafa4c",
    "pros/snowobs/LawisProfile20908.pro": "sha256:8646bdc87094ffc989aa9ed198d385aa8f765ce0ab0a9dc8227d840667537a11",
    "pros/snowobs/LawisProfile20908.smet": "sha256:e62f683c3129ed58ae8b9944a3108ad768570d761f6f3b89d86cba5db5308f32",
    "pros/snowobs/LawisProfile20915.pro": "sha256:4aaa6abd0837c95d1c8674c155ea2b3d342c965297d2ea716201f0ec8efbb36e",
    "pros/snowobs/LawisProfile20915.smet": "sha256:9d4987937bee60478c35f9f468f7a60cc8de0f89f7f862070a553f0e42a56d87",
    "smets/snowobs/forecast/LawisProfile20908.smet": "sha256:444aba0615ef6066042d8113eb56f5200aea7a241f0da77c193db7a621ad493a",
    "smets/snowobs/forecast/LawisProfile20915.smet": "sha256:07f1215a5bc86c1636ae5a5cc854c9e8cea9090ea3007d3391927235d2bfc60d",
    "smets/snowobs/nowcast/LawisProfile20908.smet": "sha256:44b213fef6e639f044af8e1ae884a0315e2a7b7cbad7abd5d8fd6de122e596da",
    "smets/snowobs/nowcast/LawisProfile20915.smet": "sha256:58caf597116621b659ae7d4439857226ad3d84a73ffa9d0e6ede0568e2a67e48",
}

SNP_SNOWOBS = pooch.create(
    path=pooch.os_cache("xsnow-snp-snowobs"),
    base_url=URL_SNP_TESTDATA,
    registry=REGISTRY_SNP_SNOWOBS,
)

def snp_snowobs_dir(minimal: bool=False) -> str:
    """
    Get the local directory containing all SNOWPACK files from a sample simulation that
    was started off of manual snow profiles.

    Parameters:
    -----------
      minimal (bool): Fetch all files related to this experiment, or only one specific file pattern?
    
    Returns:
    --------
        A path to a locally cached directory, where:
          - dir/pits/: contains the manual profiles
          - dir/smets/snowobs/[nowcast|forecast]/: contains the .smet files used to drive SNOWPACK
          - dir/pros/snowobs/: contains the SNOWPACK output files (.pro and .smet)

    See Also:
    ---------
    single_profile : Get a single profile dataset
    single_profile_timeseries : Get a single profile timeseries dataset

    Notes:
    ------
        - This sample dataset is well suited to test the `realization` capability of `xsnow` (manual and modeled profiles)
        - Until we have a .caaml parser, we can use the converted .pro files of the manuals.
          The .caaml versions contain a temperature profile on a different vertical spacing than the layers. 
          The .pro versions are missing this offset-temperature profile.
        - LawisProfile20908.caaml contains at least one weak layer that is getting loaded by a storm
        - LawisProfile20915.caaml develops a wetting front over complex stratigraphy
    """
    for fname in REGISTRY_SNP_SNOWOBS:
        if not minimal or "LawisProfile20908.pro" in fname:
            SNP_SNOWOBS.fetch(fname)

    return str(SNP_SNOWOBS.path)

def single_profile() -> xsnowDataset:
    """
    Get a single, simple `xsnowDataset` containing one (manual) profile from one location & time.

    Examples:
    ---------
      >>> import xsnow
      >>> ds = xsnow.single_profile()
      >>> ds
    """
    snowobs_dir = snp_snowobs_dir()
    ds = read(f"{snowobs_dir}/pits/LawisProfile20908.pro")
    return ds

def single_profile_timeseries() -> xsnowDataset:
    """
    Get an `xsnowDataset` from one location with many timestamps.

    In fact, this is a SNOWPACK simulation that was started off of the (manual) `single_profile()`.

    Examples:
    ---------
      >>> import xsnow
      >>> ds = xsnow.single_profile_timeseries()
      >>> ds
    """
    snowobs_dir = snp_snowobs_dir()
    ds = read(f"{snowobs_dir}/pros/snowobs/")
    
    first_location = ds.data.location.values[0]
    ds = ds.sel(location=[first_location])
    
    return ds


REGISTRY_SNP_ENSFX = {
    "smets/ens-fx/2024-01-17T00Z/det/VIR1A.smet": "sha256:baa3ae8667d0c4b746828e59d9fd846110e22c18bb9207311270604cd92be38f",
    "smets/ens-fx/2024-01-17T00Z/det/VIR5A.smet": "sha256:1a1e6ad7a1960f883cc65a1d7eff940bd114ac491938c023bb8997a1c16bec4f",
    "smets/ens-fx/2024-01-17T00Z/p01/VIR1A.smet": "sha256:c5533824928f5a31e82e056249dc4060a18c6a72e89870a3ea0e26d80e79c249",
    "smets/ens-fx/2024-01-17T00Z/p01/VIR5A.smet": "sha256:43fc619570701f62c0e828a7ed8d140c84b4a8dc9a661c3234e4c229889ce7d7",
    "smets/ens-fx/2024-01-19T10Z/det/VIR1A.smet": "sha256:edceb13fcad925cffa8f12327587e9e7f14339423d9a0a5b9fe9cce097422161",
    "smets/ens-fx/2024-01-19T10Z/det/VIR5A.smet": "sha256:aefaa14ef79f655fb86a82ac7075145365cfc5f1c85fb008c9ea8f6307a64b49",
    "smets/ens-fx/2024-01-19T10Z/p01/VIR1A.smet": "sha256:68c76f4c09c025af885c1e029fcbdcd30d1d9075547aa701f7f2082b584f75e7",
    "smets/ens-fx/2024-01-19T10Z/p01/VIR5A.smet": "sha256:f22e52d715ce4144e3afa12d66faf776d81b45dec37826d4a12e0249b4de0f53",
    "smets/ens-fx/2024-01-25T12Z/det/VIR1A.smet": "sha256:bfa74c97c4c98c0c22930764d1126b821b939d065987d01d7272694ee34746aa",
    "smets/ens-fx/2024-01-25T12Z/det/VIR5A.smet": "sha256:5d6d0c279ed2e2bed734d2172ff20fb9e44011e506a7c4748ef52a3b137c8410",
    "smets/ens-fx/2024-01-25T12Z/p01/VIR1A.smet": "sha256:b97f80deb1319c64008b710152c8ad79a60fe71271b9c05067f077dbac840b41",
    "smets/ens-fx/2024-01-25T12Z/p01/VIR5A.smet": "sha256:fe43e731eda05bd210e4b1b1bc55fec0e72da38efdd06ebeea7091fa2d263056",
    "smets/ens-fx/analysis/det/VIR1A.smet": "sha256:bf46bb5da8ae9d037097d136b8b5eaf676f618270d107c85e000580c19a475c1",
    "smets/ens-fx/analysis/det/VIR5A.smet": "sha256:3759eb31a7d168efbf7ca11370d2d096337d35814bc8bda888aaf2bae2917660",
}

SNP_ENSFX = pooch.create(
    path=pooch.os_cache("xsnow-snp-ensfx"),
    base_url=URL_SNP_TESTDATA,
    registry=REGISTRY_SNP_ENSFX,
)

def snp_ensfx_dir() -> str:
    """
    Get the local directory containing all SNOWPACK files from a toy dataset that corresponds
    to the ensemble_forecasts extension (i.e., comparing different model runs (lead times) of
    deterministic or ensemble members)

    
    Returns:
    --------
        A path to a locally cached directory, where:
          - dir/smets/ens-fx/{run}/{member}/: contains .smet files with weather inputs

    See Also:
    ---------
    read_ensemble_fx : Read this directory with the ensemble_forecasts extension
    """
    for fname in REGISTRY_SNP_ENSFX:
        SNP_ENSFX.fetch(fname)

    return str(SNP_ENSFX.path)
