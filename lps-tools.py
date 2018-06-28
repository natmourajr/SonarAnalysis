#!/usr/bin/env python3
"""
Converts pandas DataFrames to LaTeX {tabular}.
Does not require any external LaTeX packages
Version 1.1: Python3 read
"""
import pandas
import io 
__author__  = "Vinicius dos Santos Mello"
__license__ = "Apache license v2.0"

def convertToLaTeX(df, alignment="c"):
    """
    Convert a pandas dataframe to a LaTeX tabular.
    Prints labels in bold, does not use math mode
    """
    numColumns = df.shape[1]
    numRows = df.shape[0]
    output = io.StringIO()
    colFormat = ("%s|%s" % (alignment, alignment * numColumns))
    #Write header
    output.write("\\begin{tabular}{%s}\n" % colFormat)
    columnLabels = ["\\textbf{%s}" % label for label in df.columns]
    output.write("& %s\\\\\\hline\n" % " & ".join(columnLabels))
    #Write data lines
    for i in range(numRows):
        output.write("\\textbf{%s} & %s\\\\\n"
                     % (df.index[i], " & ".join([str(val) for val in df.ix[i]])))
    #Write footer
    output.write("\\end{tabular}")
    return output.getvalue()

if __name__ == "__main__":
    import numpy
    #Example code
    array = numpy.zeros((5,6))
    df = pandas.DataFrame(array, index=list("abcde"), columns=list("ABCDEF"))
    print(convertToLaTeX(df))