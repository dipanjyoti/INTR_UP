# UniPrompt-CAM

## Clone the repository
```sh
git clone https://github.com/dipanjyoti/UniPrompt-CAM.git
cd UniPrompt-CAM
```

## Environment Setup
Create python environment
```sh
conda create -n uni_prompt python=3.7
conda activate uni_prompt
source env_setup.sh
```

## Data Preparation
Follow the below format for data.
```
datasets
├── dataset_name
│   ├── train
│   │   ├── class1
│   │   │   ├── img1.jpeg
│   │   │   ├── img2.jpeg
│   │   │   └── ...
│   │   ├── class2
│   │   │   ├── img3.jpeg
│   │   │   └── ...
│   │   └── ...
│   └── val
│       ├── class1
│       │   ├── img4.jpeg
│       │   ├── img5.jpeg
│       │   └── ...
│       ├── class2
│       │   ├── img6.jpeg
│       │   └── ...
│       └── ...
```


##  Evaluation and Interpretation
Follow Prompt-CAM GitHub link for mode details.
```sh
CUDA_VISIBLE_DEVICES=0  python visualize.py --config ./experiment/config/prompt_cam/dino/cub/args.yaml --checkpoint ./checkpoints/dino/cub/model.pt --vis_cls 23
```

## INTR Training
Follow Prompt-CAM GitHub link for mode details.
```sh
CUDA_VISIBLE_DEVICES=0  python main.py --config ./experiment/config/prompt_cam/dino/cub/args.yaml
```

