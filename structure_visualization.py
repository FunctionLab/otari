import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from utils.genome_utils_vis import GTFReader


def _plot_gene_structure(gene_id, transcript_ids, gtf_reader, colors, most_affected_nodes, var_pos):
    """
    Plots the transcript structures, including exons and optional annotations for variants and affected nodes.

    Parameters:
    -----------
    gene_id : str
        The identifier of the gene to be visualized.
    transcript_ids : list of str
        A list of transcript IDs to include in the visualization.
    gtf_reader : object
        An object that provides access to gene and transcript information, such as start, end, strand, and exons.
    colors : list of str
        A list of colors to use for each transcript's exons. The order corresponds to the `transcript_ids` list.
    most_affected_nodes : dict or None
        A dictionary mapping transcript IDs to the index of the most affected exon. If None, no special highlighting is applied.
    var_pos : int or None
        The genomic position of a variant to be highlighted with a vertical dashed red line. If None, no variant is highlighted.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The matplotlib figure object containing the plot.
    ax : matplotlib.axes._subplots.AxesSubplot
        The matplotlib axes object containing the plot.

    Notes:
    ------
    - Exons are represented as colored rectangles, and introns are represented as arrows indicating the strand direction.
    - If the strand is negative, the exons and introns are reversed in order.
    - The plot is scaled based on the gene's start and end positions, with additional spacing for better visualization.
    """
    gene = gtf_reader.get_gene(gene_id)
    transcripts = gene.transcripts
    transcripts = {tid: transcripts[tid] for tid in transcript_ids if tid in transcripts}
    fig, ax = plt.subplots(1,1,figsize=(9.5, 4.5))

    x_space = (gene.end - gene.start) / 200
    x_axis_size = gene.end - gene.start + 2 * x_space

    transcript_ids = transcript_ids[::-1]
    colors = colors[::-1]

    if var_pos:
        ax.axvline(x=var_pos, color='red', linestyle='--', linewidth=0.9)

    for i, transcript_id in enumerate(transcript_ids):
        transcript = transcripts[transcript_id]
        if transcript.strand or transcript.strand == b'+':
            exons = transcript.exons
        else:
            exons = transcript.exons[::-1]

        line_start = (exons[0][0] + exons[0][1]) / 2
        line_end = (exons[-1][0] + exons[-1][1]) / 2
        ax.plot([line_start, line_end], [i, i], color='dimgray')

        for idx, exon in enumerate(exons):
            color = colors[i]
            if most_affected_nodes and idx == int(most_affected_nodes[transcript_id]):
                color = 'red'
            ax.add_patch(Rectangle((exon[0], i - 0.35), exon[1] - exon[0], 0.7, color=color, linewidth=4, zorder=2))
            if idx < len(exons) - 1:
                intron_start = exon[1]
                intron_end = exons[idx + 1][0]
                intron_mid_point = (intron_start + intron_end) / 2

                if (intron_end - intron_start) / x_axis_size < 0.01:
                    continue

                if transcript.strand == '+' or transcript.strand == b'+':
                    ax.annotate('', xy=(intron_mid_point + 0.005 * x_axis_size, i), xytext=(intron_mid_point, i),
                                arrowprops=dict(arrowstyle='->', color='dimgray'), ha = 'center')
                else:
                    ax.annotate('', xy=(intron_mid_point - 0.005 * x_axis_size, i), xytext=(intron_mid_point, i),
                                arrowprops=dict(arrowstyle='->', color='dimgray'), ha = 'center')   
    return fig, ax


def plot_transcript_structures(gene_id, transcript_ids, colors, save_path, most_affected_nodes=None, var_pos=None):
    """
    This function visualizes the structure of a gene and highlights the most 
    affected nodes for the specified transcripts. The plot is saved to the 
    specified file path.
    Args:
        gene_id (str): The ID of the gene to be plotted.
        transcript_ids (list of str): A list of transcript IDs associated with the gene.
        colors (dict): A dictionary mapping transcript IDs to their respective colors for visualization.
        save_path (str): The file path where the plot will be saved.
        most_affected_nodes (dict, optional): A dictionary mapping transcript IDs to their most affected nodes. 
            Defaults to None.
        var_pos (int, optional): The position of a variant to be highlighted on the plot. Defaults to None.
    """

    gtf_path = '/mnt/home/alitman/ceph/Genome_Annotation_Files_hg38/gencode.v47.basic.annotation.gtf'
    gtf_reader = GTFReader(gtf_path, True)
    
    fig, ax = _plot_gene_structure(gene_id, transcript_ids, gtf_reader, colors, most_affected_nodes, var_pos)
    ax.set_xlim(gtf_reader.get_gene(gene_id).start - 1000, gtf_reader.get_gene(gene_id).end + 1000)
    ax.ticklabel_format(useOffset=False, style='plain')
    ax.set_facecolor('white')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.yaxis.set_visible(False)
    plt.savefig(save_path, bbox_inches='tight', dpi=900)
    plt.close()
