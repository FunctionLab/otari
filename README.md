<p align="center">
  <img height="200" src="images/logo.png">
</p>


Welcome to the Otari framework repository! Otari is a comprehensive and interpretable graph-based framework of transcript isoform regulation, powering the characterization of transcriptomic diversity and isoform-level variant effects at scale.

This repository can be used to run the Otari model and get the Otari regulatory profiles, isoform abundance predictions, and variant effect predictions for input sequences or variants.

We also provide information and instructions for [how to train the Otari graph neural network model](#training). 

### Requirements

Please create a new Anaconda environment specifically for running Otari: 

```
conda env create -n otari -f requirements.yml
conda activate otari
```

### Setup

Please download and extract the trained Otari model, ConvSplice, Sei, and Seqweaver weights, as well as `resources` (containing hg38 FASTA files, GENCODE annotations, pickle files, node sequence attributes, transcript datasets) before proceeding:

```
sh ./download_data.sh
```

- [Otari model](https://doi.org/10.5281/zenodo.4906996)
- [Otari framework `resources` directory](https://doi.org/10.5281/zenodo.4906961)
- Model weights for `predictors` including ConvSplice, Sei, and Seqweaver should be downloaded as well(link)


### Variant effect prediction

1. The following scripts can be used to obtain Otari variant effects at the isoform level (must run on a GPU node):
(1) `variant_effect_prediction.py` (and corresponding bash script, `variant_effect_prediction.sh`): Accepts a .tsv variant file as input and makes variant effect predictions.

Example usage:
```
sh variant_effect_prediction.sh <input-file> <output-dir> --annotate --visualize
```

Arguments:
- `<input-file>`: .tsv input file with variants. Format must be `chr \t pos \t ref \t alt`
- `<output-dir>`: Path to output directory (will be created if does not exist)
- `--annotate`: boolean True or False (default is True). Annotate should only be set to false if variants are already annotated to genes and strands (make sure the genes column is called `genes`).
- `--visualize`: boolean True or False (default is False). Visualize tissue-specific variant effects, transcript splice structures, and most affected nodes. .png files saved to `<output-dir>/figures`. 

Expected outputs:
-  `variant_effects_comprehensive.tsv`: variant effect prediction for every isoform and tissue. Includes `max_effect` and `mean_effect` across tissues. 
- `interpretability_analysis.tsv`: interpretability metrics including most impacted node and features.
- `variant_to_most_affected_node_embedding.pkl`: node sequence attributes for the most impacted node for each variant and transcript.
- `figures/` containing variant effects and transcript structures.

### Example variant effect prediction run

We provide `test.tsv` (hg38 coordinates) so you can try running this command once you have installed all the requirements. Additionally, `example_slurm_scripts` contains example scripts with the same expected input arguments if you need to submit your job to a compute cluster. 

Example command run on GPU:
```
sh variant_effect_prediction.sh test.tsv ./test_outputs --annotate --visualize
```

## Training

The configuration file and script for running train is under the `train` directory. To run Otari model training, you will need GPU computing capability (we ran training on 1x Nvidia A100 GPU). 

The training data is available [here](https://doi.org/10.5281/zenodo.4907037) and should be downloaded and extracted into the `resources` directory. 

```
cd ./train
sh ./download_data.sh  # in the train directory
```

The Otari training configuration YAML file is provided as the `train/configs.yml` file. Please update the dataset location (same as `<output_dir>` below) in `train/configs.yml`, as well as any other hyperparameters that you would like to modify for training.

We provide an example SLURM script `train/train.sh` for submitting a training job to a cluster. To preprocess the data and train the model from scratch, run the following scripts in order:
```
sh preprocess/preprocess_data.sh resources/espresso.txt 'espresso' <output_dir>
sh train/train.sh
```

You can use the same conda environment to train Otari.

## Help 
Please post in the Github issues or e-mail Aviya Litman (aviya@princeton.edu) with any questions. 

