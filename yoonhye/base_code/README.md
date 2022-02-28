# pstage_01_image_classification

## Getting Started    
### Dependencies
- torch==1.6.0
- torchvision==0.7.0                                                              

### Install Requirements
- `pip install -r requirements.txt`

### Training
- `SM_CHANNEL_TRAIN=[train image dir] SM_MODEL_DIR=[model saving dir] python train.py`
> ex)
SM_CHANNEL_TRAIN='/opt/ml/input/data/train/images' SM_MODEL_DIR='/opt/ml/baseline/saved_models' python train.py

### cal_val
- `SM_CHANNEL_EVAL='/opt/ml/input/data/train/images' SM_CHANNEL_MODEL='/opt/ml/baseline/saved_models/resnet_e30_msbp_centerCrop' SM_OUTPUT_DATA_DIR='/opt/ml/baseline/train_output' python cal_val.py`
> ex)
SM_CHANNEL_EVAL='/opt/ml/input/data/train/images' SM_CHANNEL_MODEL='/opt/ml/baseline/saved_models/resnet_e30_msbp_centerCrop' SM_OUTPUT_DATA_DIR='/opt/ml/baseline/train_output' python cal_val.py --model ResNet18 --name resnet_e30_msbp_centerCrop --dataset MaskSplitByProfileDataset --augmentation MyAugmentation

### Inference
- `SM_CHANNEL_EVAL=[eval image dir] SM_CHANNEL_MODEL=[model saved dir] SM_OUTPUT_DATA_DIR=[inference output dir] python inference.py`
> ex)
SM_CHANNEL_EVAL='/opt/ml/input/data/eval' SM_CHANNEL_MODEL='/opt/ml/baseline/saved_models/densenet_e30_msbp' SM_OUTPUT_DATA_DIR='/opt/ml/baseline/output' python inference.py --model densenet --name densenet_e30_msbp

### Evaluation
- `SM_GROUND_TRUTH_DIR=[GT dir] SM_OUTPUT_DATA_DIR=[inference output dir] python evaluation.py`
