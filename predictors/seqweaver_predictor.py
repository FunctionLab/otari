from selene_sdk.predict import AnalyzeSequences
from selene_sdk.predict._common import predict as sdk_predict
from selene_sdk.utils import load_features_list
import numpy as np
from tqdm import tqdm


from selene_sdk.sequences.genome import Genome
from .models.seqweaver import Seqweaver
import os

CWD_DIR = os.path.dirname(os.path.realpath(__file__))

class SeqweaverPredictor:
    def __init__(self, genome, use_cuda = False):
        seqweaver_model_path = os.path.join(CWD_DIR, '../../ceph/otari/resources/model_weights/human_seqweaver.pth')
        distinct_features_path = os.path.join(CWD_DIR, '../../ceph/otari/resources/model_weights/seqweaver.colnames')
        distinct_features = load_features_list(distinct_features_path)
        seqweaver_model = Seqweaver(len(distinct_features))
        sdk_genome = Genome(genome.genome_path)
        self.genome = genome
        self.use_cuda = use_cuda
        
        self.model = AnalyzeSequences(
            seqweaver_model,
            seqweaver_model_path,
            sequence_length = 1000,
            features = distinct_features,
            reference_sequence = sdk_genome,
            batch_size = 1000,
            use_cuda = self.use_cuda
        )
        self.distinct_features = distinct_features

        self.BASE_TO_INDEX = {'A': 0, 'C': 2, 'G': 1, 'T': 3}

    def expand_coords_boundary(self, coords, boundary = 200):
        expanded_coords = []
        for coord in coords:
            if len(coord) == 4:
                chrom, start, end, strand = coord
            else:
                chrom, start, end, strand, mutate_pos, mutate_base = coord
            _start = start - boundary
            _end = start + boundary
            for i in range(_start, _end):
                if len(coord) == 4:
                    expanded_coords.append((chrom, i, i + 1, strand))
                else:
                    expanded_coords.append((chrom, i, i + 1, strand, mutate_pos, mutate_base))
        return expanded_coords

    def predict_batch(self, coords, verbose = 0):
        # coords is a list of tuples (chrom, start, end, strand, mutate_pos, mutate_base)
        # if mutate_pos and mutate_base are not provided, they are set to empty lists
        preds = []
        # split coords into batches
        batch_size = self.model.batch_size
        expanded_coords = self.expand_coords_boundary(coords)

        if verbose:
            progress_bar = tqdm(total=len(range(0, len(expanded_coords), batch_size)), position=0, leave=True, desc='Seqweaver Predicting')

        for i in range(0, len(expanded_coords), batch_size):
            batch_coords = expanded_coords[i:i+batch_size]
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
                sequence = self.genome.get_sequence_from_coords(chrom, seq_start, seq_end, strand, mutate_pos, mutate_base, pad = True)
                encoding = self.genome.one_hot_encoding(sequence, BASE_TO_INDEX = self.BASE_TO_INDEX, N_fill_value = 0.25)
                batch_encodings.append(encoding)
            batch_encodings = np.array(batch_encodings)
            _preds = sdk_predict(self.model.model, batch_encodings, use_cuda=self.use_cuda)
            preds.append(_preds)
            # update progress bar
            if verbose:
                progress_bar.update(1)

        preds = np.concatenate(preds)
        preds = preds.reshape(-1, 400, 217)

        out_preds = []
        for i in range(0, preds.shape[0]):
            _mean_pred = []
            _preds = preds[i]
            strand = coords[i][3]
            for j in range(0, _preds.shape[0], 50):
                _mean_pred.append(np.mean(_preds[j:j+50], axis = 0))
            _mean_pred = np.array(_mean_pred)
            if strand == '-':
                _mean_pred = _mean_pred[::-1]
            out_preds.append(_mean_pred)
        out_preds = np.array(out_preds)
        out_preds = out_preds.transpose(0, 2, 1).reshape(out_preds.shape[0], -1)

        return out_preds