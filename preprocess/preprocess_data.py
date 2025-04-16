import os
import pathlib
import argparse
from collections import defaultdict

import pandas as pd
import numpy as np
import pickle as rick
import torch
from torch_geometric.data import Data
import h5py


def convert_edges(edges):
    """
    Converts edge connections into a PyTorch tensor for graph representation.

    Args:
        edges (pd.DataFrame): A DataFrame with two columns ('Node1', 'Node2'), 
                              where each row represents an edge between two nodes. 
                              Node IDs are assumed to be 0-based integers.

    Returns:
        torch.Tensor: A tensor of shape (2, num_edges) representing the edge index 
                      in a format compatible with PyTorch Geometric. The tensor is 
                      transposed and contiguous for efficient processing.
    """
    edges = edges.astype(int)
    edge_index = torch.tensor(edges.values, dtype=torch.long)
    edge_index = edge_index.t().contiguous()
    return edge_index


def process_espresso_abundance_data(abundance_data, embedding_data, output_path):
    """
    Processes ESPRESSO abundance data and generates graph representations for transcripts.
    This function reads transcript abundance data and corresponding node embeddings, 
    constructs graph representations for each transcript, and saves the resulting graphs 
    to a specified output path.
    Args:
        abundance_data (str): Path to the input file containing transcript abundance data 
                              in tab-separated format.
        embedding_data (dict): A dictionary containing node embeddings for each transcript. 
                               The embeddings should be accessible via transcript IDs.
        output_path (str): Directory path where the processed graph data will be saved.
    Returns:
        None: The function saves the processed graph data as a PyTorch file at the specified 
              output path.
    Notes:
        - The abundance data file is expected to have a column named 'transcript_ID' and 
          other columns representing abundance values.
        - The function uses preloaded resources ('transcripts.pkl' and 'gene2chrom.pkl') 
          to map gene IDs to chromosome numbers.
        - Graphs are constructed with node embeddings as features, directed edges between 
          consecutive nodes, and abundance values as targets.
        - Chromosomes 'X', 'Y', and 'M' are excluded from processing.
        - The output is saved as a PyTorch file named 'all_transcript_graphs_directed_espresso.pt'.
    Raises:
        KeyError: If a transcript ID is not found in the embedding data.
        FileNotFoundError: If the required resource files ('transcripts.pkl' or 'gene2chrom.pkl') 
                           are not found.
    """
    
    abundance_df = pd.read_csv(abundance_data, sep='\t', header=0)

    with open(f'../resources/transcripts.pkl', 'rb') as f:
        transcripts = rick.load(f)
    print(len(transcripts))
    with open(f'../resources/gene2chrom.pkl', 'rb') as f:
        gene2chrom = rick.load(f)

    transcript_ids = abundance_df['transcript_ID'].tolist()

    print("Number of transcripts in ESPRESSO:")
    print(len(transcript_ids))

    all_graphs = []
    for transcript_id in transcript_ids:
        gene_id = abundance_df[abundance_df['transcript_ID'] == transcript_id]['gene_ID'].values[0]
        chrom = gene2chrom[gene_id].replace('chr', '')
        if chrom in ['X', 'Y', 'M']:
            continue

        row = abundance_df[abundance_df['transcript_ID'] == transcript_id].iloc[0]
        abundances = list(row[5:-4])
        abundances = [float(abundance) for abundance in abundances]
        abundances = [np.log2(abundance + 0.01) for abundance in abundances] # normalize + pseudocount

        edges = []
        target = abundances

        # get node attributes
        try:
            node_embeddings = np.array(embedding_data[transcript_id]['reference'])
        except KeyError:
            continue
        
        # get edges
        for j in range(node_embeddings.shape[0]-1):
            seg1 = j
            seg2 = j+1
            edges.append((seg1, seg2))

        # create data object with x, edge_index, and y
        x = torch.tensor(node_embeddings, dtype=torch.float)
        df = pd.DataFrame(edges, columns=['Node1', 'Node2'])
        df = df.drop_duplicates()
        edge_idx = convert_edges(df)

        graph_data = Data(x=x, edge_index=edge_idx, y=torch.tensor(target, dtype=torch.float), transcript_id=transcript_id, chrom=int(chrom))
        all_graphs.append(graph_data)

    torch.save(all_graphs, os.path.join(output_path, 'all_transcript_graphs_directed_espresso.pt'))


def process_GTEx_abundance_data(abundance_data, embedding_data, output_path):
    """
    Processes GTEx abundance data and generates graph representations for transcripts.
    This function reads transcript abundance data, filters and aggregates it by tissue types, 
    normalizes the data, and combines it with transcript embeddings to create graph data objects. 
    The resulting graph data objects are saved to a specified output path.
    Args:
        abundance_data (str): Path to the input abundance data file in tab-separated format.
        embedding_data (dict): A dictionary containing transcript embeddings. The keys are 
            transcript IDs, and the values are dictionaries with embedding information.
        output_path (str): Directory path where the processed graph data objects will be saved.
    Notes:
        - The function excludes specific tissue types ('Cells - Cultured fibroblasts', 'K562') 
          from the analysis.
        - Transcripts associated with chromosomes 'X', 'Y', and 'M' are excluded.
        - Abundance values are normalized using log2 transformation with a pseudocount of 0.01.
        - Graph data objects are created using PyTorch Geometric's `Data` class, with node 
          attributes, edge indices, and target values.
    """
    
    
    abundance_df = pd.read_csv(abundance_data, sep='\t', header=0)

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

    with open(f'../resources/gene2chrom.pkl', 'rb') as f:
        gene2chrom = rick.load(f)

    transcript_ids = abundance_df['transcript_ID'].tolist()

    all_graphs = []
    for transcript_id in transcript_ids:
        gene_id = abundance_df[abundance_df['transcript_ID'] == transcript_id]['gene_ID'].values[0]
        
        chrom = gene2chrom[gene_id].replace('chr', '')
        if chrom in ['X', 'Y', 'M']:
            continue

        # get abundances
        row = abundance_df[abundance_df['transcript_ID'] == transcript_id].iloc[0]
        abundances = list(row[2:])
        abundances = [float(abundance) for abundance in abundances]
        abundances = [np.log2(abundance + 0.01) for abundance in abundances] # normalize + pseudocount

        edges = [] 
        target = abundances 

        # get node attributes
        try:
            node_embeddings = np.array(embedding_data[transcript_id]['reference']) 
        except KeyError:
            continue
        
        # get edges
        for j in range(node_embeddings.shape[0]-1):
            seg1 = j
            seg2 = j+1
            edges.append((seg1, seg2))

        # create data object with x, edge_index, and y
        x = torch.tensor(node_embeddings, dtype=torch.float)
        df = pd.DataFrame(edges, columns=['Node1', 'Node2'])
        df = df.drop_duplicates()
        edge_idx = convert_edges(df)

        graph_data = Data(x=x, edge_index=edge_idx, y=torch.tensor(target, dtype=torch.float), transcript_id=transcript_id, chrom=int(chrom))
        all_graphs.append(graph_data)

    torch.save(all_graphs, os.path.join(output_path, 'all_transcript_graphs_directed_GTEx.pt'))


def process_CTX_abundance_data(abundance_data, embedding_data, output_path):
    """
    Processes Human and Fetal CTX abundance data and generates graph data objects for each transcript.
    This function reads abundance data from a CSV file, filters out novel transcripts, computes average 
    abundance values for adult and fetal CTX samples, and constructs graph data objects for each transcript 
    using node embeddings and edge information. The resulting graph data objects are saved to a specified 
    output path.
    Args:
        abundance_data (str): Path to the CSV file containing abundance data. The file should include columns 
                              for transcript information, sample-specific abundance values, and chromosome data.
        embedding_data (dict): A dictionary containing node embeddings for each transcript. The keys are 
                               transcript IDs, and the values are dictionaries with a 'reference' key pointing 
                               to the embedding array.
        output_path (str): Path to the directory where the processed graph data objects will be saved.
    Notes:
        - The function excludes transcripts located on chromosomes X, Y, and M.
        - Abundance values are normalized using log2 transformation with a pseudocount of 0.01.
        - Node embeddings are used to construct graph nodes, and edges are created sequentially between 
          adjacent segments.
        - The resulting graph data objects include node features (`x`), edge indices (`edge_index`), 
          target abundance values (`y`), transcript ID (`transcript_id`), and chromosome number (`chrom`).
    """

    abundance_df = pd.read_csv(abundance_data, header=0)

    # subset to known transcripts
    abundance_df = abundance_df[abundance_df['associated_transcript'] != 'novel']

    # average across samples
    abundance_df['FL.AdultCTX'] = abundance_df[['FL.AdultCTX1', 'FL.AdultCTX2', 'FL.AdultCTX3', 'FL.AdultCTX4', 'FL.AdultCTX5']].mean(axis=1)
    abundance_df['FL.FetalCTX'] = abundance_df[['FL.FetalCTX1', 'FL.FetalCTX2', 'FL.FetalCTX3', 'FL.FetalCTX4', 'FL.FetalCTX5']].mean(axis=1)
    abundance_df = abundance_df[['associated_transcript', 'FL.AdultCTX', 'FL.FetalCTX', 'chrom']]
    abundance_df['transcript_ID'] = abundance_df['associated_transcript'].str.split('.').str[0]
    abundance_df = abundance_df.drop(columns=['associated_transcript'])

    transcript_ids = abundance_df['transcript_ID'].tolist()
    
    all_graphs = []
    for transcript_id in transcript_ids:
        chrom = abundance_df[abundance_df['transcript_ID'] == transcript_id]['chrom'].values[0].replace('chr', '')

        if chrom in ['X', 'Y', 'M']:
            continue

        # get abundances
        row = abundance_df[abundance_df['transcript_ID'] == transcript_id].iloc[0]
        abundances = list(row[:2])
        abundances = [float(abundance) for abundance in abundances]
        abundances = [np.log2(abundance + 0.01) for abundance in abundances] # normalize + pseudocount

        edges = []
        target = abundances

        # get node attributes
        try:
            node_embeddings = np.array(embedding_data[transcript_id]['reference']) 
        except KeyError:
            continue
        
        # get edges
        for j in range(node_embeddings.shape[0]-1):
            seg1 = j
            seg2 = j+1
            edges.append((seg1, seg2))

        # create data object with x, edge_index, and y
        x = torch.tensor(node_embeddings, dtype=torch.float)
        df = pd.DataFrame(edges, columns=['Node1', 'Node2'])
        df = df.drop_duplicates()
        edge_idx = convert_edges(df)

        graph_data = Data(x=x, edge_index=edge_idx, y=torch.tensor(target, dtype=torch.float), transcript_id=transcript_id, chrom=int(chrom))
        all_graphs.append(graph_data)

    torch.save(all_graphs, os.path.join(output_path, 'all_transcript_graphs_directed_CTX.pt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--abundance_data', 
        type=str,
        help='path to transcript abundance data (GTEx, Espresso, or CTX data)', 
        required=True
    )

    parser.add_argument(
        '--dataset', 
        type=str,
        help='one of: espresso, gtex, ctx', 
        required=True
    )

    parser.add_argument(
        '--output', 
        type=str,
        help='output path to save data', 
        required=True
    )

    args = parser.parse_args()
    
    for path in [args.output]:
        path = pathlib.Path(path)
        path.mkdir(exist_ok=True)

    # load node embedding vectors
    node_embedding_path = '../resources/reference_ConvSplice.h5'
    embedding_data = h5py.File(node_embedding_path, 'r')
    
    if args.dataset == 'espresso':
        process_espresso_abundance_data(args.abundance_data, embedding_data, args.output)
    elif args.dataset == 'gtex':
        process_GTEx_abundance_data(args.abundance_data, embedding_data, args.output)
    elif args.dataset == 'ctx':
        process_CTX_abundance_data(args.abundance_data, embedding_data, args.output)
