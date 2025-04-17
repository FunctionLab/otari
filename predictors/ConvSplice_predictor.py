import os
import numpy as np
from tqdm import tqdm
import torch
from .models.ConvSpliceModel import ConvSplice

CWD_DIR = os.path.dirname(os.path.realpath(__file__))

class ConvSplice_predictor:
    def __init__(self, genome, use_cuda = False):
        self.genome = genome
        self.use_cuda = use_cuda
        self.batch_size = 64

        self.model_weights = [os.path.join(CWD_DIR, '../../ceph/otari/resources/model_weights/ConvSplice_model_{}.pt'.format(i + 1)) for i in range(5)]
        self.models = [self.load_model(model_path) for model_path in self.model_weights]

    def load_model(self, model_path):   
        kernel_sizes = [11, 11, 11, 11, 11, 11, 11, 11, 21, 21, 21, 21, 41, 41, 41, 41, 51, 51, 51, 51]
        dilation_rates = [1, 1, 1, 1, 4, 4, 4, 4, 10, 10, 10, 10, 25, 25, 25, 25, 25, 25, 25, 25]
        CL = 20000
        feat_dim = 32
        model = ConvSplice(feat_dim, CL, kernel_sizes, dilation_rates)
        model_weight = torch.load(model_path, map_location = 'cpu')
        model.load_state_dict(model_weight['model'])
        if self.use_cuda:
            model = model.cuda()
        return model

    def get_mutate_seq(self, reference_seq, start, end, strand, mutation_pos, mutation_base, padding_window = 10000):
        if isinstance(mutation_pos, int) and isinstance(mutation_base, str):
            mutation_pos = [mutation_pos]
            mutation_base = [mutation_base]
        if len(mutation_pos) == 0:
            return reference_seq
        mutate_seq = list(reference_seq)
        if strand == '+':
            for pos, nc in zip(mutation_pos, mutation_base):
                mutate_seq[pos - start + padding_window] = nc
        else:
            for pos, nc in zip(mutation_pos, mutation_base):
                mutate_seq[end -1 - pos + padding_window] = nc.translate(self.genome.reverse_complement)
        mutate_seq = ''.join(mutate_seq)
        return mutate_seq
    
    def predict(self, sequence_encodings):
        inputs = torch.Tensor(sequence_encodings)
        if self.use_cuda:
            inputs = inputs.cuda()
        with torch.no_grad():
            _preds = []
            for model in self.models:
                model.eval()
                outputs = model(inputs)
                _preds.append(outputs)
            _preds = torch.mean(torch.stack(_preds), axis = 0)
            return _preds.cpu().numpy()

    def get_splice_site_ann(self, pos, ss5_pos, ss3_pos):
        if pos in ss5_pos:
            splice_ann = '5ss'
        elif pos in ss3_pos:
            splice_ann = '3ss'
        else:
            splice_ann = None
        return splice_ann
    
    def predict_batch(self, coords, transcript, verbose = 0):
        preds = []

        ss5_pos, ss3_pos = [], []
        for info in transcript.ss5:
            ss5_pos.append(info[2])
        for info in transcript.ss3:
            ss3_pos.append(info[2])
        ss5_pos = set(ss5_pos)
        ss3_pos = set(ss3_pos)

        batch_size = self.batch_size
        window_size = 10000

        if verbose:
            progress_bar = tqdm(total=len(range(0, len(coords), batch_size)), position=0, leave=True, desc='ConvSplice Predicting')

        reference_seq = self.genome.get_sequence_from_coords(transcript.chrom, transcript.start, transcript.end, transcript.strand, pad = True)
        reference_seq = 'N' * window_size + reference_seq + 'N' * window_size

        for i in range(0, len(coords), batch_size):
            batch_coords = coords[i:i+batch_size]
            batch_encodings = []
            splice_ann_none_idx = []
            splice_ann_list = []
            for idx, coord in enumerate(batch_coords):
                if len(coord) == 4:
                    chrom, start, end, strand = coord
                    mutate_pos = []
                    mutate_base = []
                else:
                    chrom, start, end, strand, mutate_pos, mutate_base = coord
                mid_pos = start + ((end - start) // 2)
                splice_ann = self.get_splice_site_ann(mid_pos, ss5_pos, ss3_pos)
                if splice_ann is None:
                    splice_ann_none_idx.append(idx)
                    continue
                splice_ann_list.append(splice_ann)

                transcript_seq = self.get_mutate_seq(reference_seq, transcript.start, transcript.end, transcript.strand, mutate_pos, mutate_base, padding_window = window_size)

                if strand == '+':
                    if splice_ann == '5ss':
                        context_seq = transcript_seq[mid_pos - transcript.start: mid_pos - transcript.start + 2 * window_size + 2]
                    else:
                        context_seq = transcript_seq[mid_pos - transcript.start - 2: mid_pos - transcript.start + 2 * window_size]
                else:
                    if splice_ann == '5ss':
                        context_seq = transcript_seq[transcript.end - mid_pos: transcript.end - mid_pos + 2 * window_size + 2]
                    else:
                        context_seq = transcript_seq[transcript.end - mid_pos - 2: transcript.end - mid_pos + 2 * window_size]
                
                encoding = self.genome.one_hot_encoding(context_seq)
                batch_encodings.append(encoding)

            batch_encodings = np.array(batch_encodings)
            _preds = []
            if batch_encodings.shape[0] != 0:
                _preds = self.predict(batch_encodings)
            splice_preds = []

            pred_idx = 0
            for idx in range(0, len(batch_coords)):
                if idx in splice_ann_none_idx:
                    splice_preds.append([0, 0])
                else:
                    splice_ann = splice_ann_list[pred_idx]
                    '''
                    if splice_ann == '5ss':
                        splice_preds.append([_preds[pred_idx][0][1], _preds[pred_idx][0][2]])
                    else:
                        splice_preds.append([_preds[pred_idx][0][1], _preds[pred_idx][0][2]])
                    '''
                    if splice_ann == '5ss':
                        splice_preds.append([_preds[pred_idx][0][1], 0])
                    else:
                        splice_preds.append([0, _preds[pred_idx][0][2]])
                    pred_idx += 1
            preds.extend(splice_preds)

            if verbose:
                progress_bar.update(1)

        out_preds = np.array(preds)
        return out_preds