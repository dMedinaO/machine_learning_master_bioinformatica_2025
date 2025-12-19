from Bio import SeqIO
import pandas as pd

class CommonsFunctions:

    @classmethod
    def read_fasta_docs(cls, path_file):
        list_data = []

        for record in SeqIO.parse(path_file, "fasta"):
            row = {
                "id_seq" : record.id,
                "sequence" : str(record.seq)
            }

            list_data.append(row)

        df = pd.DataFrame(list_data)
        return df