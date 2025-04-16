from collections import defaultdict

import yaml
import numpy as np
import pandas as pd


def load_config(yaml_path):
    with open(yaml_path, 'r') as file:
        configs = yaml.safe_load(file)
    return DotDict(configs)


class DotDict(dict):
    def __getattr__(self, attr):
        value = self[attr]
        if isinstance(value, dict):
            return DotDict(value)
        return value


def compute_tissue_cutoffs(dataset='espresso'):
    """
    Compute the percentile cutoffs for each tissue to binarize expression values 
    based on the specified dataset. The function supports three datasets: 
    'espresso', 'gtex', and 'ctx'. For each dataset, it processes the expression 
    data, normalizes it using log2 transformation with a pseudocount, and computes 
    the 30th and 70th percentiles for each tissue.
    Args:
        dataset (str): The dataset to process. Options are:
            - 'espresso': Uses ESPRESSO isoform data.
            - 'gtex': Uses GTEx long-read isoform data.
            - 'ctx': Uses HumanCTX transcript data.
          Default is 'espresso'.
    Returns:
        dict: A dictionary where keys are tissue names and values are tuples 
        containing the 30th and 70th percentile cutoffs for the respective tissue.
    Notes:
        - For 'espresso', the function reads data from 
          '../resources/isoform_data_cpm_ESPRESSO_add_features.tsv.gz'.
        - For 'gtex', the function reads data from 
          '../resources/isoform_data_tpm_GTEx_long_reads_add_features.tsv.gz', 
          merges samples from the same tissues, and excludes certain cell lines.
        - For 'ctx', the function reads data from 
          '../resources/HumanCTX_transcripts.csv', filters out novel transcripts, 
          and computes averages across samples for adult and fetal cortex.
    """

    tissue_to_percentiles = {}

    if dataset == 'espresso':
        abundance_df = pd.read_csv('../resources/isoform_data_cpm_ESPRESSO_add_features.tsv.gz', sep='\t', header=0)
        
        for tissue in list(abundance_df.columns)[5:-4]:
            abundances = list(abundance_df[tissue])
            abundances = [np.log2(abundance + 0.01) for abundance in abundances] # normalize + pseudocount
            percentile_lower = np.percentile(abundances, 30)
            percentile_upper = np.percentile(abundances, 70)
            percentile_lower = round(percentile_lower, 6)
            percentile_upper = round(percentile_upper, 6)
            tissue_to_percentiles[tissue] = (percentile_lower, percentile_upper)
    
    elif dataset == 'gtex':
        abundance_df = pd.read_csv('../resources/isoform_data_tpm_GTEx_long_reads_add_features.tsv.gz', sep='\t', header=0)

        # merge samples from the same tissues
        tissue_names = list(abundance_df.loc[:,'GTEX-1192X-0011-R10a-SM-4RXXZ|Brain - Frontal Cortex (BA9)':'GTEX-WY7C-0008-SM-3NZB5_exp|Cells - Cultured fibroblasts'].columns)
        tissue_name_groups = defaultdict(list)
        for tissue_name in tissue_names:
            tissue_group = tissue_name.split('|')[1]
            if tissue_group not in tissue_name_groups:
                tissue_name_groups[tissue_group] = []
            tissue_name_groups[tissue_group].append(tissue_name)
        
        # filter cell lines
        exclude = ['Cells - Cultured fibroblasts', 'K562']
        filtered_tissue_names = []
        filtered_tissue_name_groups = {}
        for key, value in tissue_name_groups.items():
            if key not in exclude:
                filtered_tissue_names += value
                filtered_tissue_name_groups[key] = value
        
        abundance_df = abundance_df[['transcript_ID', 'gene_ID'] + filtered_tissue_names]
        for key, value in filtered_tissue_name_groups.items():
            abundance_df[key] = abundance_df[value].mean(axis=1)
        abundance_df = abundance_df.drop(columns=filtered_tissue_names)

        for tissue in list(abundance_df.columns)[2:]:
            abundances = list(abundance_df[tissue])
            abundances = [np.log2(abundance + 0.01) for abundance in abundances] # normalize + pseudocount
            percentile_lower = np.percentile(abundances, 30)
            percentile_upper = np.percentile(abundances, 70)
            percentile_lower = round(percentile_lower, 6)
            percentile_upper = round(percentile_upper, 6)
            tissue_to_percentiles[tissue] = (percentile_lower, percentile_upper)

    elif dataset == 'ctx':
        abundance_df = pd.read_csv('../resources/HumanCTX_transcripts.csv')
        abundance_df = abundance_df[abundance_df['associated_transcript'] != 'novel']

        # take average across samples
        abundance_df['FL.AdultCTX'] = abundance_df[['FL.AdultCTX1', 'FL.AdultCTX2', 'FL.AdultCTX3', 'FL.AdultCTX4', 'FL.AdultCTX5']].mean(axis=1)
        abundance_df['FL.FetalCTX'] = abundance_df[['FL.FetalCTX1', 'FL.FetalCTX2', 'FL.FetalCTX3', 'FL.FetalCTX4', 'FL.FetalCTX5']].mean(axis=1)
        abundance_df = abundance_df[['associated_transcript', 'FL.AdultCTX', 'FL.FetalCTX', 'chrom']]
        abundance_df['transcript_ID'] = abundance_df['associated_transcript'].str.split('.').str[0]
        abundance_df = abundance_df.drop(columns=['associated_transcript'])

        for tissue in list(abundance_df.columns)[:2]:
            abundances = list(abundance_df[tissue])
            abundances = [np.log2(abundance + 0.01) for abundance in abundances] # normalize + pseudocount
            percentile_lower = np.percentile(abundances, 30)
            percentile_upper = np.percentile(abundances, 70)
            percentile_lower = round(percentile_lower, 6)
            percentile_upper = round(percentile_upper, 6)
            tissue_to_percentiles[tissue] = (percentile_lower, percentile_upper)

    return tissue_to_percentiles


def binarize(x, tissue, tissue_cutoffs):
    if   x <= tissue_cutoffs[tissue][0]:   return 0
    elif x >= tissue_cutoffs[tissue][1]:   return 1
    else:           return np.nan


def assign_to_genes(variants, genes, window=2000):
    """
    Annotates genetic variants by assigning them to nearby genes based on their genomic positions.
    This function identifies genes that are within a specified window (in base pairs) 
    around the transcription start site (TSS) of each variant. It returns a DataFrame 
    containing the variants annotated with the corresponding gene information.
    Args:
        variants (pd.DataFrame): A DataFrame containing variant information. 
            Expected columns include:
                - 'chr': Chromosome of the variant.
                - 'pos': Position of the variant on the chromosome.
        genes (pd.DataFrame): A DataFrame containing gene information. 
            Expected columns include:
                - 'chr': Chromosome of the gene.
                - 'start': Start position of the gene.
                - 'end': End position of the gene.
                - 'name': Name of the gene.
                - 'strand': Strand of the gene ('+' or '-').
                - 'feature': Feature type (e.g., 'gene').
        window (int, optional): The number of base pairs around the TSS to consider 
            for assigning variants to genes. Defaults to 2000.
    Returns:
        pd.DataFrame: A DataFrame containing the annotated variants. 
            Includes all original columns from the `variants` DataFrame, 
            with additional columns:
                - 'gene': Name of the assigned gene.
                - 'strand': Strand of the assigned gene.
    """

    assigned_variants = []
    
    for _, row in variants.iterrows():
        matching_genes = genes[(genes['start']-window <= row['pos']) & (genes['end'] >= row['pos']) & (genes['chr'] == str(row['chr'])) & (genes['feature'] == 'gene')]
        
        if matching_genes.empty: 
            continue
        
        for _, gene_row in matching_genes.iterrows():
            new_row = row.copy() 
            new_row['gene'] = gene_row['name'] 
            new_row['strand'] = gene_row['strand']
            assigned_variants.append(new_row)
    
    assigned_variants_df = pd.DataFrame(assigned_variants)
    
    return assigned_variants_df
