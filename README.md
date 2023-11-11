<div align="center">
    <h1>EmoTxMLSM Enable Machines to Understand Human Emotions and Mental States<br>
    <a href="https://img.shields.io/badge/python-3.6-blue"><img src="https://img.shields.io/badge/python-3.6-blue"></a>
    <a href="https://img.shields.io/badge/made_with-pytorch-red"><img src="https://img.shields.io/badge/made_with-pytorch-red"></a>
    <a href="https://img.shields.io/badge/dataset-MovieGraphs-orange"><img src="https://img.shields.io/badge/dataset-MovieGraphs-orange"></a>
</div>

## :bookmark: Contents
1. [About](#robot-about)
2. [Setting up the repository](#toolbox-setting-up-the-repository)
    1. [Create a virtual environment](#earth_asia-create-a-python-virtual-environment)
    2. [Setup the data directory](#stars-download-the-moviegraphs-features)
    3. [Update the config template](#book-create-the-configyaml)
3. [Feature Extraction](#bomb-feature-extraction)
4. [Train EmoTx with different configurations!](#weight_lifting-train)

## :robot: About
This is the code repository for EmoTxMLSM, my project for CS577 in IIT on 2023 Autumn. Contained within this repository is the implementation of EmoTxMLSM, a Transformer-driven model crafted to forecast emotions and mental states across both scene and character scopes. My model harnesses a variety of modalities such as visual, facial, and language features to encompass a holistic comprehension of emotions within intricate cinematic settings. Furthermore, I furnish the extracted features encompassing full-frame scenes, character facial expressions, and subtitle data sourced from the MovieGraphs dataset.
<br>

## :toolbox: Setting up the repository
### :earth_asia: Create a python-virtual environment
1. Clone the repository and change the working directory to be project's root.
```
$ git clone https://github.com/lelour/EmoTxMLSM.git
$ cd EmoTxMLSM
```
2. This project strictly requires `python==3.6`.

Create a virtual environment using Conda-
```
$ conda create -n emotxmlsm python=3.6
$ conda activate emotxmlsm
(emotxmlsm) $ pip install -r requirements.txt
```
OR

Create a virtual environment using pip (make sure you have Python3.6 installed)
```
$ python3.6 -m pip install virtualenv
$ python3.6 -m virtualenv emotxmlsm
$ source emotxmlsm/bin/activate
(emotxmlsm) $ pip install -r requirements.txt
```

### :stars: Download the MovieGraphs features
You can also use `wget` to download these files-
```
$ wget -O <FILENAME> <LINK>
```

|File name | Contents | Comments |
|----------|---------------|----------|
| [EmoTx_min_feats.tar.gz](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/EdbbcQvEaBlIg6Sktxw60lQBiOUyDWdbKf3GhF88mrEhaA?download=1) | <ul><li>Extended character tracks</li><li>`emotic_mapping.json`</li><li>MovieGraphs pickle</li><li>Scene (full frame) features extracted from MViT_v1 model pre-trained on _Kinetics400 dataset</li><li>Character face features extracted from ResNet50 pre-trained on VGGFace, FER13 and SFEW datasets</li><li>Subtitle features (from both pre-trained and fine-tuned RoBERTa)</li><li>All pre-trained backbones used in EmoTx</li></ul> | contains `data/` directory which will occupy 167GB of disk space. |
| [InceptionResNetV1_VGGface_face_feats.tar.gz](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/EaTYYi2G2DNCpUaO7-Hl9PgBaXgCedq3QMxbuVos3Sfa7A?download=1) | Character face features extracted from InceptionResNet_v1 model pre-trained on VGGface2 dataset. | Contains `generic_face_features/` directory. To use these features with EmoTx, move this directory inside `data/` extracted from `EmoTx_min_feats.tar.gz`. After extraction, `generic_face_features/` will occupy 32GB of disk space. |
| [VGG-vm_FER13_face_feats.tar.gz](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/Ecx72Es1AdBGnT1Zr0U2R0cBjsx5WXP1nNAHjW2-3CdtbA?download=1) | Character face features  extracted from VGG-vm model pretrained on VGGFace and FER13 datasets | Contains `emo_face_features/` directory. TO use these features with EmoTx, move this directory inside `data/` extracted with `EmoTx_min_feats.tar.gz`. After extraction, `emo_face_features/` will occupy 254GB of disk sace.|
| [ResNet152_ImgNet_scene_feats.tar.gz](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/EaPRbRO5hmVFtOXgkb-mLCkBX11y2T4dzwdXXSLbX0eAtw?download=1) | Scene (full frame) features extracted from ResNet152 model pre-trained on ImageNet dataset | Contains `generic_scene_features/` directory. To use these features with EmoTx, move this directory inside `data/` extracted from `EmoTx_min_feats.tar.gz`. After extraction, `generic_scene_features/` will occupy 72GB of disk space.  |
| [ResNet50_PL365_scene_feats.tar.gz](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhruv_srivastava_research_iiit_ac_in/EaSkvXnjw6tBsXsRYf41yTgBbafy63-Nen_MRXulx0ycQA?download=1) | Scene (full frame) features extracted from ResNet50 model pre-trained on Places365 dataset. | Contains `resnet50_places_scene_features/` directory. To use these features with EmoTx, move this directory inside `data/` extracted from `EmoTx_min_feats.tar.gz`. After extraction, `resnet50_places_scene_features/` will occupy 143GB of disk space. |
<br>

### :book: Create the `config.yaml`
1. Create a copy of the given config template
```
(emotxmlsm) $ cp config_base.yaml config.yaml
```
2. Edit the lines `2-9` in config as directed in the comments. If you have extracted the `EmoTx_min_feats.tar.gz` in `/home/user/data`, then the path variables in `config.yaml` would be-
```
# Path variables
data_path: /home/user/data
resource_path: /home/user/data/MovieGraph/resources/
clip_srts_path: /home/user/data/MovieGraph/srt/clip_srt/
emotic_mapping_path: /home/user/data/emotic_mapping.json
pkl_path: /home/user/data/MovieGraph/mg/py3loader/
save_path: /home/user/checkpoints/
saved_model_path: /home/user/data/pretrained_models/
hugging_face_cache_path: /home/user/.cache/
dumps_path: "./dumps"

# Directory names
...
```
Refer the full `config_base.yaml` for the default parameter configuration.

## :bomb: Feature Extraction
Follow the instructions in [feature_extractors/README.md](feature_extractors/README.md) to extract required features from MovieGraphs dataset. Note that we have already provided the pre-extracted features above and therefore you need not extract the features again.

## :weight_lifting: Train
After extracting the features and creating the config, you can train EmoTxMLSM on a NVIDIA V100 GPU!<br>
You can also use the pre-trained weights provided in the [Download](#mag-download) section.<br>
Note: the `Eval_mAP: [[A,B], C]` in log line (printed during training) represents the char_mAP, scene_mAP and average of both respectively.<br>
Note: it is recommended to use [wandb](https://wandb.ai)<br>

Using the default values given in the `config_base.yaml`
1. To train EmoTx for MovieGraphs-top10 emotion label set, use the default config (no argument required)
```
(emotxmlsm) $ python trainer.py
```
2. To train EmoTxMLSM with MovieGraphs-top25 emotion label set-
```
(emotxmlsm) $ python trainer.py top_k=25
```
3. To use EmoticMapping label set-
```
(emotxmlsm) $ python trainer.py use_emotic_mapping=True
```
4. To use different scene features (valid keywords- `mvit_v1`, `resnet50_places`, `generic`) [generic=ResNet150_ImageNet]
```
(emotxmlsm) $ python trainer.py scene_feat_type="mvit_v1"
```
5. To use different character face features (valid keywords- `resnet50_fer`, `emo`, `generic`) [emo=VGG-vm_FER13, generic=InceptionResNetV1_VGGface]
```
(emotxmlsm) $ python trainer.py face_feat_type="resnet50_fer"
```
6. To use fine-tuned/pre-trained subtitle features (valid choices- `False` (to use fine-tuned RoBERTa) | `True` (to use pre-trained RoBERTa))
```
(emotxmlsm) $ python trainer.py srt_feat_pretrained=False
```
7. Train with only scene features
```
(emotxmlsm) $ python trainer.py use_char_feats=False use_srt_feats=False get_char_targets=False
```
8. To train with only character face features
```
(emotxmlsm) $ python trainer.py use_scene_feats=False use_srt_feats=False get_scene_targets=False
```
9. To train with scene and subtitle features
```
(emotxmlsm) $ python trainer.py use_char_feats=False get_char_targets=False
```
10. Enable wandb logging (recommended)
```
(emotxmlsm) $ python trainer.py wandb.logging=True wandb.project=<PROJECT_NAME> wandb.entity=<WANDB_USERNAME>
```
All the above arguments can be combined to train with different configurations.



