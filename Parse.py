import os, sys
from pathlib import Path
import pandas as pd

def dequote(s):
    """
    If a string has single or double quotes around it, remove them.
    Make sure the pair of quotes match.
    If a matching pair of quotes is not found, return the string unchanged.
    """
    if s == '' or s == " ":
        return s
    if (s[0] == s[-1]) and s.startswith(("'", '"')):
        return s[1:-1]
    return s

def dequote_list(ls):
    """
    dequote for a list
    """
    ns = []
    for s in ls:
        if (s[0] == s[-1]) and s.startswith(("'", '"')):
            ns.append(s[1:-1])
        else:
            ns.append(s)

    return ns

def parsefile(file):

    with open(file, 'r') as f:
        lines = f.readlines()

    IN = lines[0].split('=')[1].strip().strip("\n")
    Infile = lines[1].split('=')[1].strip().strip("\n")
    title = lines[5].split('=')[1].strip().strip("\n")
    out = lines[6].split('=')[1].strip().strip("\n")

    if not os.path.exists(out):
        os.mkdir(out)

    A = open(Path(IN, Infile)).readlines()

    Cell_Lines = []
    for col in A[0].split('\t'):
        Cell_Lines.append(dequote(col.strip().strip('\n')))

    Nodes = []
    for i in range(1,len(A)):
        Nodes.append(dequote(A[i].split('\t')[0].strip()))

    data = pd.DataFrame(columns = Cell_Lines, index = Nodes)

    for i in range(1,len(A)):
        data.loc[dequote(A[i].split('\t')[0].strip()),dequote_list(A[0].strip('\n').split('\t'))] = A[i].strip('\n').split('\t')[1:]

    if lines[2].split('=')[1].strip().strip("\n") == 'True':
        return data, title, out
    else:
        Data = pd.DataFrame(index = Nodes)

        if lines[3].split('=')[1].strip().strip("\n") != '':

            string = lines[3].split('=')[1].strip().strip("\n")
            select = []
            for col in string.split(','):
                if ':' in col:
                    Data = pd.concat(Data, data[col.split(':')[0]:col.split(':')[1]])
                else:
                    select.append(col)

            Data = pd.concat(Data, data[select])

            return Data, title, out

        else:
            string = lines[4].split('=')[1].strip().strip("\n")
            deselect = []
            for col in string.split(','):
                if ':' in col:
                    data = data.drop(data.ix[:, col.split(':')[0].strip():col.split(':')[1].strip()].columns, axis = 1)
                else:
                    deselect.append(col.strip())
            data = data.drop(deselect, axis=1)

            return data, title, out
