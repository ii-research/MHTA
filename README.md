# Multi-dimensional Hierarchical Temporal Alignment for Improved Temporal Commonsense Reasoning in Large Language Models
## Setup Instructions

To run this project, please follow the steps below:
### 1. Install Requirements

First, install the necessary dependencies for the project by running:
```bash
pip install -r requirements.txt
```
### 2. Download The Test Dataset

Go to [MCTACO](https://github.com/CogComp/MCTACO/tree/master) and download the dataset. Place the downloaded dev data in the `data` folder at the project root.

### 3. Configure

Open the `configs/config.yaml` file and modify the `test` field to point to the path of the dev data you just downloaded. For example:

```yaml
test: "data/your_dataset_folder"
```
### 4. Set Model Name

In `conf/config.yaml`, locate the `base_model_name` field and set the name of the model you want to use. For example:

```yaml
base_model_name: "mistralai/Mistral-7B-Instruct-v0.1"
```
### 5. Run the Main Program

Once the above steps are complete, you can run the main program by executing:

```bash
python main.py
# MHTA
