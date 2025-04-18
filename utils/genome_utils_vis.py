from pysam import FastaFile
from tqdm import tqdm


FASTA_FILE_OBJ = FastaFile('../ceph/otari/resources/hg38.fa')
REVERSE_COMPLEMENT = str.maketrans('ATCG', 'TAGC')


def get_genomic_seq(chrom, strand, start, end, mutate_pos = [], mutate_base = []):
    seq = FASTA_FILE_OBJ.fetch(chrom, start, end).upper()
    if len(mutate_pos) > 0:
        seq = list(seq)
        for pos, base in zip(mutate_pos, mutate_base):
            seq[pos - start] = base
        seq = ''.join(seq)
    if strand == '-':
        seq = seq.translate(REVERSE_COMPLEMENT)[::-1]
    return seq


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
        self.principal_transcript = {}
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

    def add_principal_transcript(self, transcript_id):
        self.principal_transcript[transcript_id] = transcript_id

    def __getitem__(self, transcript_id):
        return self.transcripts[transcript_id]
    

class GTFReader:
    def __init__(self, gtf_path, add_splice_site = True, appris_path = None):
        self.gtf_path = gtf_path
        self.appris_path = appris_path
        self.genes = {}
        self.read_gtf()
        if not appris_path is None:
            self.read_appris_annotation()
        self.gene_id2name, self.gene_name2id = {}, {}
        self.gene2gene_type = {}
        self.transcripts = {}
        self.get_id_map()
        self.add_transcript_info()
        if add_splice_site:
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
        return self.genes[gene_id]

    def add_transcript_info(self):
        for gene_id in self.genes:
            gene = self.genes[gene_id]
            for transcript_id in gene.transcripts:
                self.transcripts[transcript_id] = gene.transcripts[transcript_id]

    def read_gtf(self):
        gtf_file_line = self.count_file_line(self.gtf_path)
        fp = open(self.gtf_path)
        for line in tqdm(fp, total = gtf_file_line, desc = 'Reading GTF file'):
            if line.startswith('#'):
                continue
            sp = line.strip().split('\t')
            if sp[2] == 'gene':
                gene_id = sp[-1].split('"')[1].split('.')[0]
                start, end = int(sp[3]) - 1, int(sp[4])
                chrom = sp[0]
                strand = sp[6]
                gene_name = sp[-1].split(';')[2].split('"')[1]
                if 'gene_type' in sp[-1]:
                    gene_type = sp[-1].split(';')[1].split('"')[1]
                else:
                    gene_type = None
                gene = Gene(gene_id, gene_name, chrom, start, end, strand)
                gene.gene_type = gene_type
                self.genes[gene_id] = gene
            elif sp[2] == 'transcript':
                transcript_id = sp[-1].split('"')[3].split('.')[0]
                start, end = int(sp[3]) - 1, int(sp[4])
                chrom = sp[0]
                strand = sp[6]
                gene_id = sp[-1].split('"')[1].split('.')[0]
                gene_name = sp[-1].split(';')[3].split('"')[1]
                if 'transcript_type' in sp[-1]:
                    transcript_type = sp[-1].split(';')[4].split('"')[1]
                else:
                    transcript_type = None
                transcript = Transcript(transcript_id, gene_id, chrom, strand, start, end)
                transcript.transcript_type = transcript_type
                self.genes[gene_id].add_transcript(transcript)
            elif sp[2] == 'exon':
                gene_id = sp[-1].split('"')[1].split('.')[0]
                transcript_id = sp[-1].split('"')[3].split('.')[0]
                start, end = int(sp[3]) - 1, int(sp[4])
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
                    for exon in transcript.exons[:-1]:
                        splice_dinucleotide = get_genomic_seq(chrom, strand, exon[1], exon[1] + 2)
                        ss_info = (chrom, strand, exon[1], '5ss', splice_dinucleotide)
                        transcript.add_ss5(ss_info)
                    for exon in transcript.exons[1:]:
                        splice_dinucleotide = get_genomic_seq(chrom, strand, exon[0] - 2, exon[0])
                        ss_info = (chrom, strand, exon[0], '3ss', splice_dinucleotide)
                        transcript.add_ss3(ss_info)
                else:
                    for exon in transcript.exons[:-1]:
                        splice_dinucleotide = get_genomic_seq(chrom, strand, exon[1], exon[1] + 2)
                        ss_info = (chrom, strand, exon[1], '3ss', splice_dinucleotide)
                        transcript.add_ss3(ss_info)
                    for exon in transcript.exons[1:]:
                        splice_dinucleotide = get_genomic_seq(chrom, strand, exon[0] - 2, exon[0])
                        ss_info = (chrom, strand, exon[0], '5ss', splice_dinucleotide)
                        transcript.add_ss5(ss_info)

    def read_appris_annotation(self):
        fp = open(self.appris_path)
        fp.readline()
        for line in fp:
            sp = line.strip().split('\t')
            gene_id = sp[1]
            transcript_id = sp[2]
            appris_class = sp[4]
            if appris_class == 'PRINCIPAL:1':
                self.genes[gene_id].add_principal_transcript(transcript_id)
                