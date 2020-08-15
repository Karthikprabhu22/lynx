#! /bin/sh
rm data/interim/cmbs4_hm.h5
rm data/interim/mean_maps.h5
python scripts/convert_mask_to_h5.py -d configs/masking/cmbs4.yaml
python scripts/make_wsp00.py -d configs/masking/cmbs4.yaml
python scripts/make_cmbs4_data_hm.py -d configs/simulation/cmbs4_hm.yaml --verbose
python scripts/separate.py -d configs/simulation/cmbs4_hm.yaml -p configs/masking/cmbs4.yaml -m configs/model/validation.yaml --verbose
python scripts/powerspectra.py -d configs/simulation/cmbs4_hm.yaml -m configs/model/validation.yaml -p configs/masking/cmbs4.yaml --verbose
python scripts/likelihood.py -d configs/simulation/cmbs4_hm.yaml -m configs/model/validation.yaml -p configs/masking/cmbs4.yaml -l configs/likelihood/validation.yaml --verbose
python scripts/make_mean_maps.py -d configs/simulation/cmbs4_hm.yaml --verbose

