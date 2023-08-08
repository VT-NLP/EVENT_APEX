<h1 align="center">The Art of Prompting: Event Detection based on Type Specific Prompts
</h1>

## Table of Contents
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Trigger Detection](#trigger-detection)
- [Citation](#citation)
- [License](#license)

## Installation
To install the dependency packages, run
```
conda create --name query_extract_EE python=3.8
conda activate query_extract_EE
pip install -r requirements.txt
export PROJECT_ROOT=$project_root_path
```

## Data Preparation 

1. Follow https://github.com/wilburOne/ACE_ERE_Scripts to process the raw data and save to ./data/ace_en/processed_data 
and ./data/ere_en/processed_data respectively
2. Save the event data into .txt files, process the .txt file and save as Torch TensorDataset
```
cd preprocess
./preprocess_data.sh
```
## Trigger Detection
To train the trigger detection model for ACE under supervised learning setting, run
```
export DATA_PATH=$data_folder
python run_trigger_detection.py main --do_train True --ace True
```
To train the trigger detection model for ERE under few-shot learning setting, run
```
python run_trigger_detection.py main --do_train True --ere True --few_shot True
```






## Citation
If you find this repo useful, please cite the following paper:
```
@inproceedings{wang-etal-2023-art,
    title = "The Art of Prompting: Event Detection based on Type Specific Prompts",
    author = "Wang, Sijia  and
      Yu, Mo  and
      Huang, Lifu",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-short.111",
    doi = "10.18653/v1/2023.acl-short.111",
    pages = "1286--1299",
    abstract = "We compare various forms of prompts to represent event types and develop a unified framework to incorporate the event type specific prompts for supervised, few-shot, and zero-shot event detection. The experimental results demonstrate that a well-defined and comprehensive event type prompt can significantly improve event detection performance, especially when the annotated data is scarce (few-shot event detection) or not available (zero-shot event detection). By leveraging the semantics of event types, our unified framework shows up to 22.2{\%} F-score gain over the previous state-of-the-art baselines.",
}
```


<!-- LICENSE -->
## License

Distributed under the MIT License.

