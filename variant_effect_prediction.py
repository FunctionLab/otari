import os
import random
import argparse
from collections import defaultdict

import torch
import pandas as pd
import numpy as np
import pickle as rick
from torch_geometric.data import Data
import pytorch_lightning as pl
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.lines as mlines

from utils.utils import assign_to_genes
from get_variant_node_embeddings import main, read_gtf
from preprocess.preprocess_data import convert_edges
from utils.genome_utils import Genome
from model.otari import Otari
from structure_visualization import plot_transcript_structures
   

def QC_variants(variants):
    """
    Perform quality control on a DataFrame of genetic variants.
    This function filters out duplicate variants, indels (insertions and deletions),
    and variants located on sex chromosomes (X, Y) or mitochondrial DNA (M).
    Args:
        variants (pd.DataFrame): A DataFrame containing genetic variant information.
            It is expected to have the following columns:
            - 'ref': Reference allele as a string.
            - 'alt': Alternate allele as a string.
            - 'chr': Chromosome identifier as a string.
    Returns:
        pd.DataFrame: A filtered DataFrame containing only unique single nucleotide
        variants (SNVs) located on autosomal chromosomes.
    """

    variants = variants.drop_duplicates(keep='first')
    variants = variants[variants['ref'].str.len() == 1]
    variants = variants[variants['alt'].str.len() == 1]
    variants = variants[~variants['chr'].isin(['X', 'Y', 'M'])]
    variants = variants[~variants['chr'].isin(['chrX', 'chrY', 'chrM'])]
    
    return variants
    

def reformat_graph(embed, transcript_id, gene_id, transcript_variant_identifiers):
    """
    Converts node embeddings into a directed graph object.
    This function processes node embeddings and constructs a directed graph 
    representation using PyTorch Geometric's `Data` object. It identifies 
    edges between consecutive segments that are exons and creates a graph 
    with node features, edge indices, and additional metadata.
    Args:
        embed (list or numpy.ndarray): Node embeddings representing features 
            for each segment in the transcript.
        transcript_id (str): Identifier for the transcript associated with 
            the graph.
        gene_id (str): Identifier for the gene associated with the transcript.
        transcript_variant_identifiers (dict): Dictionary containing metadata 
            about the transcript.
    Returns:
        torch_geometric.data.Data: A PyTorch Geometric `Data` object containing:
            - `x` (torch.Tensor): Node feature matrix.
            - `edge_index` (torch.Tensor): Edge indices defining the graph structure.
            - `transcript_id` (str): Transcript identifier.
            - `identifiers` (list): Variant identifiers.
            - `gene_id` (str): Gene identifier.
            - `transcripts` (str): Transcript identifier (duplicated for metadata).
    Example:
        >>> embed = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        >>> transcript_id = "ENST00000367770"
        >>> gene_id = "ENSG00000157764"
        >>> transcript_variant_identifiers = {
        ...     "identifiers": ["var1", "var2", "var3"],
        ... }
        >>> graph = reformat_graph(embed, transcript_id, gene_id, transcript_variant_identifiers)
        >>> print(graph)
        Data(x=[3, 2], edge_index=[2, 1], transcript_id='ENST00000367770', ...)
    """

    edges = [] 
    for j in range(len(embed)-1):
        seg1 = j
        seg2 = j+1
        edges.append((seg1, seg2))

    # create data object with x, edge_index, and y
    x = torch.tensor(embed, dtype=torch.float)
    df = pd.DataFrame(edges, columns=['Node1', 'Node2'])
    df = df.drop_duplicates()
    edge_idx = convert_edges(df)
    batch_idx = torch.zeros(x.shape[0], dtype=torch.long)

    # send to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    edge_idx = edge_idx.to(device)
    batch_idx = batch_idx.to(device)

    graph_data = Data(
        x=x, 
        edge_index=edge_idx, 
        batch_idx=batch_idx,
        transcript_id=transcript_id, 
        identifiers=transcript_variant_identifiers['identifiers'],
        gene_id=gene_id,
        transcripts=transcript_id
        )

    return graph_data


def predict_variant_effects(model_path, variant_path, output_path, annotate):
    """
    Predict the effects of genetic variants on isoform usage and graph structure.
    This function processes a set of genetic variants, evaluates their effects on 
    transcript isoforms, and computes various metrics to quantify the impact of 
    these variants. The results include variant effect scores, interpretability 
    analysis, and node embeddings for the most impacted nodes.
    Args:
        model_path (str): Path to the pre-trained model file for prediction.
        variant_path (str): Path to the input file containing variant information 
            in TSV format with columns: 'chr', 'pos', 'ref', 'alt'.
        output_path (str): Directory where the output files will be saved.
        annotate (bool): Whether to annotate variants with gene information.
    Outputs:
        - 'variant_effects_comprehensive.tsv': A comprehensive table of variant 
          effects across all transcripts and tissues.
        - 'max_variant_effects_across_transcripts.tsv': A summary table of the 
          maximum absolute effects per variant across transcripts.
        - 'interpretability_analysis.tsv': A table containing interpretability 
          analysis results, including the most affected node and top features.
        - 'variant_to_most_affected_node_embedding.pkl': A pickle file containing 
          node embeddings for the most impacted nodes for each variant.
    Notes:
        - The function performs quality control (QC) on the input variants.
        - If `annotate` is True, variants are assigned to genes using a GTF file.
        - The function uses a pre-trained model to predict the effects of variants 
          on transcript graphs.
        - Z-score normalization is applied to graph features before computing 
          interpretability metrics.
        - Tissue-specific scores are calculated as log2 fold changes between 
          alternative and reference predictions.
    """
    pl.seed_everything(42, workers=True)

    variants = pd.read_csv(variant_path, sep='\t') # .tsv with columns: chr, pos, ref, alt
    variants = QC_variants(variants)
    print(f'Variant count after QC: {variants.shape[0]}')

    if annotate:
        genes = pd.read_csv('../ceph/otari/resources/gencode.v47.basic.annotation.clean.gtf.gz', sep='\t')
        variants = assign_to_genes(variants, genes)
        print(f'Variant count after annotation: {variants.shape[0]}')
    
    with open('../ceph/otari/resources/gene2transcripts.pkl', 'rb') as file:
        gene2transcripts = rick.load(file)
    with open('../ceph/otari/resources/transcript2gene.pkl', 'rb') as file:
        transcript2gene = rick.load(file)

    # load model
    model = Otari()  # Same model architecture
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # load gtf reader and Genome object
    gtf_reader = read_gtf()
    genome = Genome('../ceph/otari/resources/hg38.fa')

    tissue_names = ['Brain', 'Caudate_Nucleus', 'Cerebellum', 'Cerebral_Cortex', 'Corpus_Callosum', 'Fetal_Brain', 'Fetal_Spinal_Cord', 'Frontal_Lobe', 'Hippocampus', 'Medulla_Oblongata', 'Pons', 'Spinal_Cord', 'Temporal_Lobe', 'Thalamus', 'bladder', 'blood', 'colon', 'heart', 'kidney', 'liver', 'lung', 'ovary', 'pancreas', 'prostate', 'skeletal_muscle', 'small_intestine', 'spleen', 'stomach', 'testis', 'thyroid']
    
    # store variant effects
    variant_effects_df = []
    interpretability_df = []
    node_embed_dictionary = defaultdict(dict)
    for i, variant in variants.iterrows():  
        print(f'Processing variant {i+1}/{variants.shape[0]}')   
        transcript_ids = []
        transcript_ids.extend(gene2transcripts[str(variant['gene'])])
        
        # update node attributes
        transcript_variant_embeddings, transcript_variant_identifiers = main(transcript_ids, variant, gtf_reader, genome)
        
        for i, tid in enumerate(transcript_ids): # predict for every variant-transcript pair
            if len(transcript_variant_embeddings[tid]) != 2:
                continue
            gene = transcript2gene[tid]
            keys = transcript_variant_embeddings[tid].keys()
            ref_key = [key for key in keys if 'reference' in key][0]
            alt_key = [key for key in keys if 'reference' not in key][0]
            
            ref_embed = transcript_variant_embeddings[tid][ref_key]
            alt_embed = transcript_variant_embeddings[tid][alt_key]
            
            variant = alt_key

            # convert to graphs
            ref_graph = reformat_graph(ref_embed, tid, gene, transcript_variant_identifiers[tid][ref_key])
            alt_graph = reformat_graph(alt_embed, tid, gene, transcript_variant_identifiers[tid][alt_key])

            # z-score normalization across features and graphs
            means = torch.mean(ref_graph.x, dim=0)
            stes = torch.std(ref_graph.x, dim=0)
            epsilon = 1e-8 
            alt_norm = (alt_graph.x - means) / (stes + epsilon)
            ref_norm = (ref_graph.x - means) / (stes + epsilon)
            alt_norm = alt_norm.cpu().detach().numpy()
            ref_norm = ref_norm.cpu().detach().numpy()

            # compute most impacted node and features
            most_impacted_node = int(np.argmax(np.sum(np.abs(alt_norm - ref_norm), axis=1)))
            top_features = np.argsort(np.abs(alt_norm[most_impacted_node] - ref_norm[most_impacted_node]))[::-1][:10]
            top_features = list(top_features)
            
            interpretability_vec = [variant, gene, tid, most_impacted_node, top_features]
            interpretability_df.append(interpretability_vec)

            # save node embeddings of most impacted node
            node_embed_dictionary[variant][tid] = alt_norm[most_impacted_node] - ref_norm[most_impacted_node]

            with torch.no_grad():
                pred_ref = model(ref_graph)
                pred_alt = model(alt_graph)
                
                pred_ref = pred_ref.squeeze(0).cpu().detach().numpy()
                pred_alt = pred_alt.squeeze(0).cpu().detach().numpy()                

                # compute absolute max and mean scores across tissues
                max_score = np.max(np.abs(np.log2((2**(pred_alt)+1)/(2**(pred_ref)+1))))
                mean_score = np.mean(np.abs(np.log2((2**(pred_alt)+1)/(2**(pred_ref)+1))))
                
                tissue_scores = np.log2((2**(pred_alt)+1)/(2**(pred_ref)+1))

                score_vec = [variant, tid, max_score, mean_score]
                score_vec.extend(tissue_scores)
                variant_effects_df.append(score_vec)
    
    # save variant_effects_df to file
    cols = ['variant_id', 'transcript_id', 'max_effect', 'mean_effect']
    cols.extend(tissue_names)
    variant_effects_df = pd.DataFrame(variant_effects_df, columns=cols)
    variant_effects_df.to_csv(os.path.join(output_path, 'variant_effects_comprehensive.tsv'), sep='\t', index=False)
    
    # interpretability df
    interpretability_df = pd.DataFrame(interpretability_df, columns=['variant_id', 'gene_id', 'transcript_id', 'most_affected_node', 'top_features'])
    interpretability_df.to_csv(os.path.join(output_path, 'interpretability_analysis.tsv'), sep='\t', index=False)

    with open(os.path.join(output_path, f'variant_to_most_affected_node_embedding.pkl'), 'wb') as f:
        rick.dump(node_embed_dictionary, f)

    
def visualize_results(output_path):
    """
    Visualizes the results of variant effect predictions and interpretability analysis.
    This function generates and saves plots for tissue-specific variant effects and 
    transcript structures for each variant. It reads data from the specified output 
    directory, processes it, and creates visualizations for better understanding of 
    the variant effects across tissues and the most affected nodes in transcript structures.
    Args:
        output_path (str): The path to the directory containing the output files 
                           ('variant_effects_comprehensive.tsv' and 'interpretability_analysis.tsv') 
                           and where the generated figures will be saved.
    Output:
        - Tissue-specific variant effect scatter plots saved as PNG files in the 
          'figures' subdirectory of the output path.
        - Transcript structure plots highlighting the most affected nodes saved as PNG files 
          in the 'figures' subdirectory of the output path.
    """
    
    variant_effects_df = pd.read_csv(os.path.join(output_path, 'variant_effects_comprehensive.tsv'), sep='\t').drop_duplicates(keep='first')
    interpretability_df = pd.read_csv(os.path.join(output_path, 'interpretability_analysis.tsv'), sep='\t').drop_duplicates(keep='first')
    variant_ids = variant_effects_df['variant_id'].unique()

    tissue_names = ['Brain', 'Caudate_Nucleus', 'Cerebellum', 'Cerebral_Cortex', 'Corpus_Callosum', 'Fetal_Brain', 'Fetal_Spinal_Cord', 'Frontal_Lobe', 'Hippocampus', 'Medulla_Oblongata', 'Pons', 'Spinal_Cord', 'Temporal_Lobe', 'Thalamus', 'bladder', 'blood', 'colon', 'heart', 'kidney', 'liver', 'lung', 'ovary', 'pancreas', 'prostate', 'skeletal_muscle', 'small_intestine', 'spleen', 'stomach', 'testis', 'thyroid']

    def get_distinct_colors(n, colormap=cm.tab20):  # or cm.rainbow
        return [colormap(i / max(n - 1, 1)) for i in range(n)]
    
    for variant_name in variant_ids:
        vep = variant_effects_df.loc[variant_effects_df['variant_id'] == variant_name]
        num_transcripts = len(vep)
        # colors = [cm.rainbow(random.random()) for _ in range(num_transcripts)]
        colors = get_distinct_colors(num_transcripts, colormap=cm.rainbow)

        # plot tissue-specific variant effects
        _, ax = plt.subplots(1, 1, figsize=(9.5, 5))
        # for i, row in vep.iterrows():
        #     for j, tissue in enumerate(tissue_names):
        #         ax.scatter(j, row[tissue], color=colors[i], s=80, alpha=0.8)
        #     sns.lineplot(x=tissue_names, y=row[tissue_names], color=colors[i])
        for i, row in enumerate(vep.itertuples(index=False)):
            y_vals = [getattr(row, tissue) for tissue in tissue_names]
            ax.scatter(range(len(tissue_names)), y_vals, color=colors[i], s=80, alpha=0.8)
            ax.plot(range(len(tissue_names)), y_vals, color=colors[i], alpha=0.6)
        ax.set_ticks(range(len(tissue_names)))
        ax.set_xticklabels(tissue_names, rotation=90)
        ax.set_ylabel('log2 fold change', fontsize=13.5)
        ax.set_xlabel('Tissues', fontsize=12)
        ax.set_title(f'Variant effects for {variant_name}', fontsize=14, pad=10, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        transcript_ids = vep['transcript_id'].values
        handles = [
            mlines.Line2D([], [], marker='o', linestyle='-', color=colors[i], markersize=8, label=transcript_ids[i])
            for i in range(len(transcript_ids))
        ]
        ax.legend(handles=handles, title='Transcript ID', fontsize=10, title_fontsize=12, loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        plt.tight_layout()
        plt.savefig(f'{output_path}/figures/variant_effects_{variant_name}.png', dpi=600, bbox_inches='tight')
        plt.close()

        # plot structure and most affected nodes
        # most affected nodes will be highlighted in yellow
        variant_interpret = interpretability_df.loc[interpretability_df['variant_id'] == variant_name]
        most_affected_nodes = dict(zip(variant_interpret['transcript_id'], variant_interpret['most_affected_node']))
        most_affected_nodes = {k: int(v) for k, v in most_affected_nodes.items()}
        gene_id = variant_interpret['gene_id'].mode()
        if len(gene_id) > 1:
            gene_id = random.choice(gene_id)
        else:
            gene_id = gene_id[0]
        plot_transcript_structures(
                                   gene_id, 
                                   colors, 
                                   save_path = f'{output_path}/figures/variant_structure_{variant_name}.png',
                                   most_affected_nodes=most_affected_nodes
                                   )


if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Predict variant effects on isoform expression.')
    parser.add_argument('--variant_path', type=str, required=True, help='Path to the variant file (tsv).')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output directory.')
    parser.add_argument('--annotate', action='store_true', default=True, help='Whether to annotate variants to genes.')
    parser.add_argument('--visualize', action='store_true', default=False, help='Whether to visualize results.')
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if not os.path.exists(os.path.join(args.output_path, 'figures')):
        os.makedirs(os.path.join(args.output_path, 'figures'))

    model_path = '../ceph/otari/resources/otari.pth'
    
    predict_variant_effects(model_path, args.variant_path, args.output_path, annotate=args.annotate)

    if args.visualize:
        visualize_results(args.output_path)

    print('Otari variant effect prediction completed! Results saved to:', args.output_path)
