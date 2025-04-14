from selene_sdk.predict import AnalyzeSequences
from selene_sdk.utils import NonStrandSpecific
from selene_sdk.predict._common import predict as sdk_predict
from selene_sdk.utils import load_features_list
import numpy as np
from tqdm import tqdm
import pandas as pd

from selene_sdk.sequences.genome import Genome
from .models.sei import Sei
import os

CWD_DIR = os.path.dirname(os.path.realpath(__file__))

class SeiPredictor:
    def __init__(self, genome, strand_specific = False, use_cuda = False):
        sei_model_path = os.path.join(CWD_DIR, 'models/model_weights/sei.pth')
        distinct_features_path = os.path.join(CWD_DIR, 'models/model_weights/sei.target.names')
        histone_features_path = os.path.join(CWD_DIR, 'models/model_weights/histone_features.csv')

        distinct_features = load_features_list(distinct_features_path)

        if strand_specific:
            sei_model = Sei(len(distinct_features))
        else:
            sei_model = NonStrandSpecific(Sei(len(distinct_features)), mode='mean')

        sdk_genome = Genome(genome.genome_path)
        self.genome = genome
        self.use_cuda = use_cuda
        self.strand_specific = strand_specific

        histone_idx = []
        histone_features = pd.read_csv(histone_features_path, sep = ',')
        distinct_feature2idx = {distinct_features[i]: i for i in range(0, len(distinct_features))}
        for idx in range(0, histone_features.shape[0]):
            histone_idx.append(distinct_feature2idx[histone_features['Cell Line'][idx]])
        self.histone_idx = np.array(histone_idx)
        
        self.model = AnalyzeSequences(
            sei_model,
            sei_model_path,
            sequence_length = 4096,
            features = distinct_features,
            reference_sequence = sdk_genome,
            batch_size = 128,
            use_cuda = self.use_cuda
        )
        self.distinct_features = distinct_features

    def predict_batch(self, coords, verbose = 0):
        # coords is a list of tuples (chrom, start, end, strand, mutate_pos, mutate_base)
        # if mutate_pos and mutate_base are not provided, they are set to empty lists
        preds = []
        # split coords into batches
        batch_size = self.model.batch_size

        if verbose:
            progress_bar = tqdm(total=len(range(0, len(coords), batch_size)), position=0, leave=True, desc='Sei Predicting')

        for i in range(0, len(coords), batch_size):
            batch_coords = coords[i:i+batch_size]
            batch_encodings = []
            for coord in batch_coords:
                if len(coord) == 4:
                    chrom, start, end, strand = coord
                    mutate_pos = []
                    mutate_base = []
                else:
                    chrom, start, end, strand, mutate_pos, mutate_base = coord
                mid_pos = start + ((end - start) // 2)
                seq_start = mid_pos - self.model._start_radius
                seq_end = mid_pos + self.model._end_radius
                # sei model is non-strand specific
                if self.strand_specific:
                    sequence = self.genome.get_sequence_from_coords(chrom, seq_start, seq_end, strand, mutate_pos, mutate_base, pad = True)
                else:
                    sequence = self.genome.get_sequence_from_coords(chrom, seq_start, seq_end, '+', mutate_pos, mutate_base, pad = True)
                encoding = self.genome.one_hot_encoding(sequence, N_fill_value = 0.25)
                batch_encodings.append(encoding)
            batch_encodings = np.array(batch_encodings)
            _preds = sdk_predict(self.model.model, batch_encodings, use_cuda=self.use_cuda)
            preds.append(_preds)
            # update progress bar
            if verbose:
                progress_bar.update(1)
                
        preds = np.concatenate(preds)
        out_preds = preds[:, self.histone_idx]

        return out_preds
