from pickle import STRING
import sys

from pymsa import MSA, Entropy, PercentageOfNonGaps, PercentageOfTotallyConservedColumns, Star, SumOfPairs
from pymsa import PAM250, Blosum62, FileMatrix
from pymsa.util.fasta import read_fasta_file_as_list_of_pairs


def write_red(f, str_):
    f.write('<span style="background-color:#f54021; font-size:16px">%s</span>' % str_)

def write_blue(f, str_):
    f.write('<span style="background-color:#5564eb; font-size:16px">%s</span>' % str_)
def write_white(f, str_):
    f.write('<span style="background-color:#ffffff; font-size:16px">%s</span>' % str_)



def print_alignment(msa: MSA, cx_point: int = 100, output_file = open('output_file_alignment', 'w')):
    sub_sequences = [[]] * msa.number_of_sequences
    for i, sequence in enumerate(msa.sequences):
        sub_sequences[i] = [sequence[i: i + cx_point] for i in range(0, len(sequence), cx_point)]

    for k in range(len(sub_sequences[0])):
        sequences = [item[k] for item in sub_sequences]
        colour_scheme = [0] * len(sequences[0])

        for i, column in enumerate(zip(*sequences)):
            if len(set(column)) <= 1:
                colour_scheme[i] = 'red' if len(set(column)) <= 1 else 0
            else:
                if set(column).issubset(['I', 'L', 'V']):
                    colour_scheme[i] = 'blue'
                elif set(column).issubset(['F', 'W', 'Y']):
                    colour_scheme[i] = 'blue'
                elif set(column).issubset(['K', 'R', 'H']):
                    colour_scheme[i] = 'blue'
                elif set(column).issubset(['D', 'E']):
                    colour_scheme[i] = 'blue'
                elif set(column).issubset(['G', 'A', 'S']):
                    colour_scheme[i] = 'blue'
                elif set(column).issubset(['T', 'N', 'Q', 'M']):
                    colour_scheme[i] = 'blue'

        longest_id = len(max(msa.ids, key=len))
        for sequence, id in zip(sequences, msa.ids):
            output_file.write(id + ' ' * (longest_id - len(id)))
            for i in range(len(colour_scheme)):
                if colour_scheme[i] == 'red':
                    write_red(output_file, sequence[i])
                elif colour_scheme[i] == 'blue':
                    write_blue(output_file, sequence[i])
                else:
                    write_white(output_file, sequence[i])
            output_file.write('<br>')
        output_file.write('<br>')

def run_all_scores(sequences: list, output_file_name) -> None:
    aligned_sequences = list(pair[1] for pair in sequences)
    sequences_id = list(pair[0] for pair in sequences)

    output_file = open(output_file_name, 'w')
    output_file.write('<pre style= "font-size:16px">')
    msa = MSA(aligned_sequences, sequences_id)
    print_alignment(msa, output_file=output_file)
    
    # Percentage of non-gaps and totally conserved columns
    non_gaps = PercentageOfNonGaps(msa)
    totally_conserved_columns = PercentageOfTotallyConservedColumns(msa)

    percentage = non_gaps.compute()
    output_file.write("Percentage of non-gaps: {0} %".format(percentage))
    output_file.write("<br>")
    conserved = totally_conserved_columns.compute()
    output_file.write("Percentage of totally conserved columns: {0}".format(conserved))
    output_file.write("<br>")

    # Entropy
    value = Entropy(msa).compute()
    output_file.write("Entropy score: {0}".format(value))
    output_file.write("<br>")

    # Sum of pairs
    value = SumOfPairs(msa, Blosum62()).compute()
    output_file.write("Sum of Pairs score (Blosum62): {0}".format(value))
    output_file.write("<br>")

    value = SumOfPairs(msa, PAM250()).compute()
    output_file.write("Sum of Pairs score (PAM250): {0}".format(value))
    output_file.write("<br>")

    value = SumOfPairs(msa, FileMatrix('PAM380.txt')).compute()
    output_file.write("Sum of Pairs score (PAM380): {0}".format(value))
    output_file.write("<br>")

    # Star
    value = Star(msa, Blosum62()).compute()
    output_file.write("Star score (Blosum62): {0}".format(value))
    output_file.write("<br>")

    value = Star(msa, PAM250()).compute()
    output_file.write("Star score (PAM250): {0}".format(value))
    output_file.write("<br>")
    output_file.write('<pre>')
    


if __name__ == '__main__':
    """
    sequences = [("1g41",
                  "S-EMTPREIVSELDQHIIGQADAKRAVAIALRNRWRRMQLQEPLRHE--------VTP-KNILMIGPTGVGKTEIARRLAKLANAPFIKVEATKFT----"
                  "VGKEVDSIIRDLTDSAMKLVRQQEIAKNR---------------------------------------------------------------------LI"
                  "DDEAAKLINPEELKQKAIDAVE--QNGIVFIDEIDKICKKGEYSGADVSREGVQRDLLPLVEGSTVSTKHGMVKTDHILFIASGAFQVARPSDL------"
                  "-----------IPELQGRLPIR-VEL---TALSAADFERILTEPHASLTEQYKALMATEGVNIAFTTDAVKKIAEAAFRVNEKTENIGARRLHTVMERLM"
                  "DKISFSASDMNGQTVNIDAAYVADALGEVVENEDLSRFIL"),
                 ("1e94",
                  "HSEMTPREIVSELDKHIIGQDNAKRSVAIALRNRWRRMQLNEELRHE--------VTP-KNILMIGPTGVGKTEIARRLAKLANAPFIKVEATKFTEVGY"
                  "VGKEVDSIIRDLTDAAVKMVRVQAIEKNRYRAEELAEERILDVLIPPAKNNWGQTEQQQEPSAARQAFRKKLREGQLDDKEIEKQKARKLKIKDAMKLLI"
                  "EEEAAKLVNPEELKQDAIDAVE--QHGIVFIDEIDKICKRGESSGPDVSREGVQRDLLPLVEGCTVSTKHGMVKTDHILFIASGAFQIAKPSDL------"
                  "-----------IPELQGRLPIR-VEL---QALTTSDFERILTEPNASITVQYKALMATEGVNIEFTDSGIKRIAEAAWQVNESTENIGARRLHTVLERLM"
                  "EEISYDASDLSGQNITIDADYVSKHLDALVADEDLSRFIL"),
                 ("1e32",
                  "R-ED-EEESLNEVGYDDVGG--CRKQLAQ-----I-KEMVELPLRHPALFKAIGVKPP-RGILLYGPPGTGKTLIARAVANETGAFFFLINGPEIM-SKL"
                  "A-GESESN--------------------------------------------------------------------------------------------"
                  "-------------LRKAFEEAEKNAPAIIFIDELDAIAPKREKTHGEVERRIVSQ-LLTLMDGL--------KQRAHVIVMAATN----RPNSIDPALRR"
                  "FGRFDREVDIGIPDATGRLEILQIHTKNMKLADDVDLEQVANETHGH---------------------------------------VGADLAALCSEAAL"
                  "QAIRKKMDLIDLEDETIDAEVM-NSL-AVTMDDFRWALSQ"),
                 ("1d2n",
                  "------EDYASYIMNGIIKWGDP---VTRVLD--DGELLVQQTKNSD--------RTPLVSVLLEGPPHSGKTALAAKIAEESNFPFIKICSPDKM-IGF"
                  "SETAKCQA--------------------------------------------------------------------------------------------"
                  "-------------MKKIFDDAYKSQLSCVVVDDIERLLDYV-PIGPRFSNLVLQA-LLVLLKKA-------PPQGRKLLIIGTTS----R-KDVLQEMEM"
                  "LNA---------------------------------FSTTIHVPNIATGEQL--LEALEL-LGNFKDKE---RTTIAQQVKGKKVWIGIKKLLMLIEM--"
                  "-------------SLQMDPEYRVRKFLALLREEGAS-PLD")]
    """
    if len(sys.argv) != 3:
        print("Wrong number of arguments")
    else:
        sequences = read_fasta_file_as_list_of_pairs(sys.argv[1])
        run_all_scores(sequences, sys.argv[2])
