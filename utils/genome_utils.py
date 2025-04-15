from tqdm import tqdm
from pysam import FastaFile
import pandas as pd
import numpy as np


FASTA_FILE_OBJ = FastaFile('../resources/hg38.fa')
REVERSE_COMPLEMENT = str.maketrans('ATCG', 'TAGC')


def get_genomic_seq(chrom, strand, start, end, mutate_pos = [], mutate_base = []):
    # mutation position is 0-based
    seq = FASTA_FILE_OBJ.fetch(chrom, start, end).upper()
    if len(mutate_pos) > 0:
        seq = list(seq)
        for pos, base in zip(mutate_pos, mutate_base):
            seq[pos - start] = base
        seq = ''.join(seq)
    if strand == '-':
        seq = seq.translate(REVERSE_COMPLEMENT)[::-1]
    return seq
    

class Genome:
    def __init__(self, genome_path):
        self.genome_path = genome_path
        self.genome = FastaFile(self.genome_path)
        self.reverse_complement = str.maketrans('ATCG', 'TAGC')
        self.map = np.asarray([[0, 0, 0, 0],
                               [1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])
        self.INDEX_TO_LATIN = {0: '\x01', 1: '\x02', 2: '\x03', 3: '\x04'}

    def get_sequence_from_coords(self, chrom, start, end, strand, mutate_pos = [], mutate_base = [], pad = False):
        # get genomic sequence from start to end (0-based)
        start_pad, end_pad = 0, 0
        if pad:
            chr_len = self.genome.get_reference_length(chrom)
            if start < 0:
                start_pad = -1 * start
                start = 0
            if end > chr_len:
                end_pad = end - chr_len
                end = chr_len
            seq = 'N' * start_pad + self.genome.fetch(chrom, start, end).upper() + 'N' * end_pad
        else:
            seq = self.genome.fetch(chrom, start, end).upper()

        if isinstance(mutate_pos, int) and isinstance(mutate_base, str):
            mutate_pos = [mutate_pos]
            mutate_base = [mutate_base]

        # mutation position is 0-based
        if mutate_pos is not None and len(mutate_pos) > 0:
            seq = list(seq)
            for pos, base in zip(mutate_pos, mutate_base):
                if pos - start < 0 or pos - start >= len(seq):
                    continue
                seq[pos - start] = base
            seq = ''.join(seq)

        if strand == '-':
            seq = seq.translate(self.reverse_complement)[::-1]

        return seq

    def one_hot_encoding(self, seq, BASE_TO_INDEX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}, N_fill_value = 0):
        seq = seq.upper().replace('A', self.INDEX_TO_LATIN[BASE_TO_INDEX['A']]).replace('C', self.INDEX_TO_LATIN[BASE_TO_INDEX['C']])
        seq = seq.replace('G', self.INDEX_TO_LATIN[BASE_TO_INDEX['G']]).replace('T', self.INDEX_TO_LATIN[BASE_TO_INDEX['T']]).replace('N', '\x00')
        seq_array = self.map[np.frombuffer(seq.encode('latin-1'), np.int8) % 5]
        is_N = seq_array.sum(axis = 1) == 0
        seq_array = seq_array.astype('float32')
        seq_array[is_N] = N_fill_value
        return seq_array


class Transcript:
    def __init__(self, transcript_id, gene_id, chrom, strand, start, end):
        self.transcript_id = transcript_id
        self.gene_id = gene_id
        self.chrom = chrom
        self.strand = strand
        self.start = start
        self.end = end
        self.transcript_type = None
        self.exons = []
        self.ss5 = set()
        self.ss3 = set()

    def __str__(self):
        return f'{self.transcript_id} {self.gene_id} {self.chrom} {self.strand} {self.start} {self.end}'
    
    def __repr__(self):
        return self.__str__()
    
    def add_exon(self, start, end):
        self.exons.append((start, end))

    def add_ss5(self, ss5):
        self.ss5.add(ss5)
    
    def add_ss3(self, ss3):
        self.ss3.add(ss3)

    def get_exon_rank(self, ss_type, pos):
        if ss_type == '5ss':
            ss = self.ss5
        else:
            ss = self.ss3
        ss_sorted = sorted(ss, key = lambda x: x[2])
        rank = None
        for i, ss_pos in enumerate(ss_sorted):
            if ss_pos[2] == pos:
                rank = i
        if self.strand == '-':
            rank = len(ss_sorted) - rank - 1
        return rank


class Gene:
    def __init__(self, gene_id, gene_name, chrom, start, end, strand):
        self.gene_id = gene_id
        self.gene_name = gene_name
        self.chrom = chrom
        self.start = start 
        self.end = end
        self.transcripts = {}
        self.strand = strand
        self.gene_type = None

    def __str__(self):
        return f'{self.gene_id} {self.transcripts}'
    
    def __repr__(self):
        return self.__str__()
    
    def __len__(self):
        return len(self.transcripts)
    
    def add_transcript(self, transcript):
        self.transcripts[transcript.transcript_id] = transcript
        if self.strand is None:
            self.strand = transcript.strand

    def __getitem__(self, transcript_id):
        return self.transcripts[transcript_id]
    

class GTFReader:
    def __init__(self, gtf_path, genome_path = None, add_splice_site = False):
        self.gtf_path = gtf_path
        self.genes = {}
        self.read_gtf()
        self.gene_id2name, self.gene_name2id = {}, {}
        self.gene2gene_type = {}
        self.transcripts = {}
        self.get_id_map()
        self.add_transcript_info()
        self.sort_transcript_exons()
        if genome_path:
            self.genome = Genome(genome_path)
        else:
            self.genome = None
        if add_splice_site:
            if not genome_path:
                raise ValueError('Genome path is required to add splice site annotation')
            self.add_splice_site_annotation()

    def __len__(self):
        return len(self.genes)
    
    def __getitem__(self, gene_id):
        return self.genes[gene_id]

    def count_file_line(self, f_path):
        with open(f_path) as f:
            return sum(1 for _ in f)
    
    def get_id_map(self):
        for gene_id in self.genes:
            self.gene_id2name[gene_id] = self.genes[gene_id].gene_name
            self.gene_name2id[self.genes[gene_id].gene_name] = gene_id

    def get_gene(self, gene_id):
        if gene_id in self.genes:
            return self.genes[gene_id]
        elif gene_id in self.gene_name2id:
            return self.genes[self.gene_name2id[gene_id]]
        else:
            raise ValueError(f'Gene {gene_id} not found')

    def add_transcript_info(self):
        for gene_id in self.genes:
            gene = self.genes[gene_id]
            for transcript_id in gene.transcripts:
                self.transcripts[transcript_id] = gene.transcripts[transcript_id]

    def sort_transcript_exons(self):
        for transcript_id in self.transcripts:
            transcript = self.transcripts[transcript_id]
            if transcript.strand == '+':
                transcript.exons = sorted(transcript.exons, key = lambda x: x[0])
            else:
                transcript.exons = sorted(transcript.exons, key = lambda x: x[0], reverse = True)
                
    def read_gtf(self):
        gtf_file_line = self.count_file_line(self.gtf_path)
        fp = open(self.gtf_path)
        for line in tqdm(fp, total = gtf_file_line, desc = 'Reading GTF file'):
            if line.startswith('#'):
                continue
            sp = line.strip().split('\t')
            annotate_info = sp[-1].split(';')
            
            chrom = sp[0]
            strand = sp[6]
            start, end = int(sp[3]) - 1, int(sp[4])

            if sp[2] == 'gene':
                gene_id, gene_name, gene_type = None, None, None
                
                for info in annotate_info:
                    if 'gene_id' in info:
                        gene_id = info.split('"')[1].split('.')[0]
                    if 'gene_name' in info:
                        gene_name = info.split('"')[1]
                    if 'gene_type' in info:
                        gene_type = info.split('"')[1]
                
                if gene_id is None:
                    raise ValueError('Gene ID not found')
                
                gene = Gene(gene_id, gene_name, chrom, start, end, strand)
                gene.gene_type = gene_type
                self.genes[gene_id] = gene

            elif sp[2] == 'transcript':
                transcript_id, transcript_type = None, None
                gene_id, gene_name = None, None
                for info in annotate_info:
                    if 'transcript_id' in info:
                        transcript_id = info.split('"')[1].split('.')[0]
                    if 'transcript_type' in info:
                        transcript_type = info.split('"')[1]
                    if 'gene_id' in info:
                        gene_id = info.split('"')[1].split('.')[0]
                    if 'gene_name' in info:
                        gene_name = info.split('"')[1]

                if transcript_id is None:
                    raise ValueError('Transcript ID not found')
                
                transcript = Transcript(transcript_id, gene_id, chrom, strand, start, end)
                transcript.transcript_type = transcript_type
                self.genes[gene_id].add_transcript(transcript)

            elif sp[2] == 'exon':
                gene_id, transcript_id = None, None
                for info in annotate_info:
                    if 'gene_id' in info:
                        gene_id = info.split('"')[1].split('.')[0]
                    if 'transcript_id' in info:
                        transcript_id = info.split('"')[1].split('.')[0]
                
                self.genes[gene_id][transcript_id].add_exon(start, end)

        fp.close()

    def add_splice_site_annotation(self):
        for gene_id in tqdm(self.genes, desc = 'Adding splice site annotation'):
            gene = self.genes[gene_id]
            for transcript_id in gene.transcripts:
                transcript = gene[transcript_id]
                exons = transcript.exons
                if len(exons) < 2:
                    continue
                chrom = transcript.chrom
                strand = transcript.strand
                if strand == '+':
                    for exon in transcript.exons:
                        if exon[1] != transcript.end:
                            splice_dinucleotide = self.genome.get_sequence_from_coords(chrom, exon[1], exon[1] + 2, strand)
                            ss_info = (chrom, strand, exon[1], '5ss', splice_dinucleotide)
                            transcript.add_ss5(ss_info)
                        
                        if exon[0] != transcript.start:
                            splice_dinucleotide = self.genome.get_sequence_from_coords(chrom, exon[0] - 2, exon[0], strand)
                            ss_info = (chrom, strand, exon[0], '3ss', splice_dinucleotide)
                            transcript.add_ss3(ss_info)
                else:
                    for exon in transcript.exons:
                        if exon[0] != transcript.start:
                            splice_dinucleotide = self.genome.get_sequence_from_coords(chrom, exon[0] - 2, exon[0], strand)
                            ss_info = (chrom, strand, exon[0], '5ss', splice_dinucleotide)
                            transcript.add_ss5(ss_info)
                        
                        if exon[1] != transcript.end:
                            splice_dinucleotide = self.genome.get_sequence_from_coords(chrom, exon[1], exon[1] + 2, strand)
                            ss_info = (chrom, strand, exon[1], '3ss', splice_dinucleotide)
                            transcript.add_ss3(ss_info)

    def summary_gtf(self):
        num_genes = len(self.genes)
        num_transcripts = len(self.transcripts)
        num_exons = sum([len(transcript.exons) for transcript in self.transcripts.values()])

        mean_num_transcripts_per_gene = pd.Series([len(gene.transcripts) for gene in self.genes.values()]).mean()
        mean_num_exons_per_gene = pd.Series([sum([len(transcript.exons) for transcript in gene.transcripts.values()]) for gene in self.genes.values()]).mean()

        max_num_transcripts_per_gene = pd.Series([len(gene.transcripts) for gene in self.genes.values()]).max()
        max_num_exons_per_gene = pd.Series([sum([len(transcript.exons) for transcript in gene.transcripts.values()]) for gene in self.genes.values()]).max()

        min_num_transcripts_per_gene = pd.Series([len(gene.transcripts) for gene in self.genes.values()]).min()
        min_num_exons_per_gene = pd.Series([sum([len(transcript.exons) for transcript in gene.transcripts.values()]) for gene in self.genes.values()]).min()

        print('Number of genes:', num_genes)
        print('Number of transcripts:', num_transcripts)
        print('Number of exons:', num_exons)
        print('Mean number of transcripts per gene:', mean_num_transcripts_per_gene)
        print('Mean number of exons per gene:', mean_num_exons_per_gene)
        print('Max number of transcripts per gene:', max_num_transcripts_per_gene)
        print('Max number of exons per gene:', max_num_exons_per_gene)
        print('Min number of transcripts per gene:', min_num_transcripts_per_gene)
        print('Min number of exons per gene:', min_num_exons_per_gene)
        