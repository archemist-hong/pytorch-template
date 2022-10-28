#!/bin/bash

# slack 채팅방에 남기기
ARRAY=('ColorJitter' 'RandomAffine' 'RandomHorizontalFlip' 'RandomPerspective' 'RandomRotation' 'RandomVerticalFlip' 'GaussianBlur' 'RandomAdjustSharpness' 'RandomAutocontrast' )

for arg in "${ARRAY[@]}"
do
    # change config.json
    python ./utils/change_config.py "$arg"

    # exec train.py
    python train.py
done