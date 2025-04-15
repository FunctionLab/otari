import os
import sys
from collections import defaultdict

import numpy as np
from tqdm import tqdm
import pandas as pd

from predictors.ConvSplice_predictor import ConvSplice_predictor
from predictors.sei_predictor import SeiPredictor
from predictors.seqweaver_predictor import SeqweaverPredictor
from cyvcf2 import VCF
from utils.genome_utils import GTFReader


CHR_INDEX_DICT = {'1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, '10':10, '11':11, '12':12, 
  '13':13, '14':14, '15':15, '16':16, '17':17, '18':18, '19':19, '20':20, '21':21, '22':22, 'X':23, 'Y':24}


def read_gtf():
    gtf_path = 'resources/gencode.v47.basic.annotation.gtf'
    genome_path = 'resources/hg38.fa'
    gtf_reader = GTFReader(gtf_path, genome_path = genome_path, add_splice_site = True)
    return gtf_reader


def read_vcf(vcf_path):
    vcf = VCF(vcf_path, strict_gt = True)
    return vcf


def read_vcf_tsv(vcf_path):
    vcf = pd.read_csv(vcf_path, sep = '\t') # columns: chr, pos, ref, alt
    return vcf


def check_vcf_file(vcf_path):
    if os.path.exists(vcf_path + '.tbi'):
        return True
    sys.stderr.write("\nCould not retrieve index file for '{}'".format(vcf_path))
    sys.stderr.write("\n- VCF needs to be sorted and index first. \n- Example code: tabix -p vcf example.vcf.gz\n")
    sys.exit(1)


def get_variant_in_region(chrom, region, variant_series):
    _chrom = chrom.replace('chr', '')

    if _chrom not in CHR_INDEX_DICT:
        return []

    region_start, region_end = region
    _pos = variant_series['pos'] - 1  # 0-based position
    
    # Check if the variant is within the region
    if _pos < region_start or _pos >= region_end:
        return []
    
    _ref = variant_series['ref']
    _alt = variant_series['alt']
    _rs_id = '{}_{}_{}_{}_{}'.format(_chrom, _pos + 1, _ref, _alt, 'hg38')
    
    return [(chrom, _pos, _ref, _alt, _rs_id)]


def get_predictor_region_of_interest(ref_pos, predictor_name):
    if predictor_name == 'sei':
        return [ref_pos - 2048, ref_pos + 2048]
    
    elif predictor_name == 'seqweaver':
        return [ref_pos - 1000, ref_pos + 1000]
    
    elif predictor_name == 'ConvSplice':
        return [ref_pos - 10000, ref_pos + 10000]
    
    else:
        raise ValueError('predictor name not recognized')


def reorder_embed(input_emb):
    input_emb = np.array(input_emb)
    feature_idx = np.array([0, 1] + list(range(4, 1736 + 4)) + list(range(4 + 1736 * 2, 4 + 1736 * 2 + 10062)) + [2, 3] + 
                           list(range(4 + 1736, 4 + 1736 * 2)) + list(range(4 + 1736 * 2 + 10062, 4 + 1736 * 2 + 10062 * 2)))
    return input_emb[feature_idx]
    

def get_transcript_embedding(transcript, predictor_names, predictors, vcf, verbose = 1):
    '''
    This function takes a transcript object and returns the embedding of the transcript and its variants.
    transcript : Transcript object
    predictor_names : list of strings
    predictors : dict of predictor objects
    vcf : cyvcf2.VCF or .tsv vcf pandas object
    '''

    chrom = transcript.chrom 
    strand = transcript.strand
    exons = transcript.exons

    reference_coords = []
    for exon in exons:
        exon_start, exon_end = exon[0], exon[1]
        reference_coords.append((chrom, exon_start, exon_start + 1, strand))
        reference_coords.append((chrom, exon_end, exon_end + 1, strand))

    reference_embedding = {}
    for predictor_name in predictor_names:
        reference_embedding[predictor_name] = {}
        if predictor_name == 'ConvSplice':
            preds = predictors[predictor_name].predict_batch(reference_coords, transcript, verbose = verbose)
        else:
            preds = predictors[predictor_name].predict_batch(reference_coords, verbose = verbose)
        reference_embedding[predictor_name] = preds

    dt = np.dtype([('chr', 'S2'), ('start', 'i4'), ('end', 'i4'), ('strand', 'S1')])
    # add reference to transcript_variant_embeddings
    transcript_variant_embeddings = {}
    transcript_variant_identifiers = defaultdict(lambda: defaultdict(list))
    embedding = []
    identifiers = []
    segment_id = []
    edge_embeddings = []
    for idx, exon in enumerate(exons):
        _embedding = []
        for predictor_name in predictor_names:
            if strand == '+':
                _embedding.extend(reference_embedding[predictor_name][2 * idx])
                _embedding.extend(reference_embedding[predictor_name][2 * idx + 1])
            else:
                _embedding.extend(reference_embedding[predictor_name][2 * idx + 1])
                _embedding.extend(reference_embedding[predictor_name][2 * idx])
        embedding.append(reorder_embed(_embedding))
        identifiers.append((str(chrom.replace('chr', '')), int(exon[0]), int(exon[1]), str(strand)))
        segment_id.append('e')
        # compute distance between exons for edge embeddings
        if idx < len(exons) - 1:
            next_exon = exons[idx+1]
            if strand == '+':
                edge_embeddings.append(abs(next_exon[0] - exon[1]))
            else:
                edge_embeddings.append(abs(exon[0] - next_exon[1]))
                
    transcript_variant_embeddings['reference'] = np.array(embedding)
    transcript_variant_identifiers['reference']['identifiers'] = np.array(identifiers, dtype=dt)
    transcript_variant_identifiers['reference']['segment_id'] = np.array(segment_id, dtype='S1')
    transcript_variant_identifiers['reference']['edge_embeddings'] = np.array(edge_embeddings)

    if vcf is None:
        return transcript_variant_embeddings, transcript_variant_identifiers

    variant_embedding = {}
    variant_id2mapped_exons = {}
    for predictor_name in predictor_names:
        variant_embedding[predictor_name] = {}
        variant_id2mapped_exons[predictor_name] = {}
        exon_variant_coords = []
        for idx, exon in enumerate(exons):
            exon_start, exon_end = exon[0], exon[1]
            variants_in_exon_start = get_variant_in_region(chrom, get_predictor_region_of_interest(exon_start, predictor_name), vcf)
            variants_in_exon_end = get_variant_in_region(chrom, get_predictor_region_of_interest(exon_end, predictor_name), vcf)

            for variant in variants_in_exon_start:
                # variant format (chrom, pos, ref, alt, rs_id)
                variant_id = variant[-1]
                if variant_id not in variant_id2mapped_exons[predictor_name]:
                    variant_id2mapped_exons[predictor_name][variant_id] = []
                # map variant to exon start, each exon start is mapped to 2 * idx
                variant_id2mapped_exons[predictor_name][variant_id].append(2 * idx)
                exon_variant_coord = (chrom, exon_start, exon_start + 1, strand, variant[1], variant[3])
                exon_variant_coords.append(exon_variant_coord)

            for variant in variants_in_exon_end:
                # variant format (chrom, pos, ref, alt, rs_id)
                variant_id = variant[-1]
                if variant_id not in variant_id2mapped_exons[predictor_name]:
                    variant_id2mapped_exons[predictor_name][variant_id] = []
                # map variant to exon end, each exon end is mapped to 2 * idx + 1
                variant_id2mapped_exons[predictor_name][variant_id].append(2 * idx + 1)
                exon_variant_coord = (chrom, exon_end, exon_end + 1, strand, variant[1], variant[3])
                exon_variant_coords.append(exon_variant_coord)

        exon_variant_coords = list(set(exon_variant_coords))
        if len(exon_variant_coords) == 0:
            continue
    
        if predictor_name == 'ConvSplice':
            exon_variant_embedding = predictors[predictor_name].predict_batch(exon_variant_coords, transcript, verbose = verbose)
        else:
            exon_variant_embedding = predictors[predictor_name].predict_batch(exon_variant_coords, verbose = verbose)

        for i, coord in enumerate(exon_variant_coords):
            variant_embedding[predictor_name][coord] = exon_variant_embedding[i]

    # get transcript variant embeddings
    all_variant_ids = set()
    for predictor_name in predictor_names:
        all_variant_ids.update(variant_id2mapped_exons[predictor_name].keys())

    for variant_id in all_variant_ids:
        embedding = []
        identifiers = []
        segment_id = []
        sp = variant_id.split('_')
        variant_pos, _, variant_alt = int(sp[1]) - 1, sp[2], sp[3]
        for idx, exon in enumerate(exons):
            _embedding = []
            for predictor_name in predictor_names:
                if strand == '+':
                    if variant_id not in variant_id2mapped_exons[predictor_name]:
                        _embedding.extend(reference_embedding[predictor_name][2 * idx])
                        _embedding.extend(reference_embedding[predictor_name][2 * idx + 1])
                    else:
                        if 2 * idx in variant_id2mapped_exons[predictor_name][variant_id]:
                            _embedding.extend(variant_embedding[predictor_name][(chrom, exon[0], exon[0] + 1, strand, variant_pos, variant_alt)])
                        else:
                            _embedding.extend(reference_embedding[predictor_name][2 * idx])
                        if 2 * idx + 1 in variant_id2mapped_exons[predictor_name][variant_id]:
                            _embedding.extend(variant_embedding[predictor_name][(chrom, exon[1], exon[1] + 1, strand, variant_pos, variant_alt)])
                        else:
                            _embedding.extend(reference_embedding[predictor_name][2 * idx + 1])
                else:
                    if variant_id not in variant_id2mapped_exons[predictor_name]:
                        _embedding.extend(reference_embedding[predictor_name][2 * idx + 1])
                        _embedding.extend(reference_embedding[predictor_name][2 * idx])
                    else:
                        if 2 * idx + 1 in variant_id2mapped_exons[predictor_name][variant_id]:
                            _embedding.extend(variant_embedding[predictor_name][(chrom, exon[1], exon[1] + 1, strand, variant_pos, variant_alt)])
                        else:
                            _embedding.extend(reference_embedding[predictor_name][2 * idx + 1])
                        if 2 * idx in variant_id2mapped_exons[predictor_name][variant_id]:
                            _embedding.extend(variant_embedding[predictor_name][(chrom, exon[0], exon[0] + 1, strand, variant_pos, variant_alt)])
                        else:
                            _embedding.extend(reference_embedding[predictor_name][2 * idx])  
                            
            embedding.append(reorder_embed(_embedding)) # append embedding for a single exon
            identifiers.append((str(chrom.replace('chr', '')), int(exon[0]), int(exon[1]), str(strand)))
            segment_id.append('e')
                
        transcript_variant_embeddings[variant_id] = np.array(embedding)
        transcript_variant_identifiers[variant_id]['identifiers'] = np.array(identifiers, dtype=dt)
        transcript_variant_identifiers[variant_id]['segment_id'] = np.array(segment_id, dtype='S1')
        transcript_variant_identifiers[variant_id]['edge_embeddings'] = np.array(edge_embeddings)
        
    return transcript_variant_embeddings, transcript_variant_identifiers


def main(transcript_ids, variants, gtf_reader, genome):    
    predictor_names = ['ConvSplice', 'seqweaver', 'sei']

    print('Making predictions using the following predictors:')
    predictors = {}
    for predictor_name in predictor_names:
        print('- {}'.format(predictor_name))
        if predictor_name == 'ConvSplice':
            predictors['ConvSplice'] = ConvSplice_predictor(genome, use_cuda=True)
        if predictor_name == 'seqweaver':
            predictors['seqweaver'] = SeqweaverPredictor(genome, use_cuda=True)
        if predictor_name == 'sei':
            predictors['sei'] = SeiPredictor(genome, use_cuda=True)
    
    composite_embeddings = {}
    composite_identifiers = {}
    for transcript_id in tqdm(transcript_ids, total = len(transcript_ids), desc = 'Processing transcripts'):
        transcript = gtf_reader.transcripts[transcript_id]
        transcript_variant_embeddings, transcript_variant_identifiers = get_transcript_embedding(transcript, predictor_names, predictors, variants, verbose=0)
        composite_embeddings[transcript_id] = transcript_variant_embeddings
        composite_identifiers[transcript_id] = transcript_variant_identifiers

    return composite_embeddings, composite_identifiers


if __name__ == '__main__':
    main()
