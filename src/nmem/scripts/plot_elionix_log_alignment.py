# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:32:21 2023

@author: omedeiro
"""
from nmem.analysis.alignment_plots import plot_alignment_histogram


def parse_elionix_log(logfile):
    """
    Parses the Elionix log file and returns a list of alignment differences in nm.
    """
    with open(logfile) as f:
        lines = f.readlines()
    diff_list = []
    for idx, line in enumerate(lines):
        if ".car\n" in line:
            pos1A = lines[idx + 17].split("\t")[5:7]
            pos1B = lines[idx + 24].split("\t")[5:7]
            pos1C = lines[idx + 31].split("\t")[5:7]
            pos1D = lines[idx + 38].split("\t")[5:7]
            pos2A = lines[idx + 45].split("\t")[5:7]
            pos2B = lines[idx + 52].split("\t")[5:7]
            pos2C = lines[idx + 59].split("\t")[5:7]
            pos2D = lines[idx + 66].split("\t")[5:7]
            if pos1A[0] != "0\n":
                diff1Ax = float(pos1A[0]) - float(pos2A[0])
                diff1Ay = float(pos1A[1]) - float(pos2A[1])
                diff1Bx = float(pos1B[0]) - float(pos2B[0])
                diff1By = float(pos1B[1]) - float(pos2B[1])
                diff1Cx = float(pos1C[0]) - float(pos2C[0])
                diff1Cy = float(pos1C[1]) - float(pos2C[1])
                diff1Dx = float(pos1D[0]) - float(pos2D[0])
                diff1Dy = float(pos1D[1]) - float(pos2D[1])
                diff_list.extend(
                    [
                        diff1Ax * 1e6,
                        diff1Ay * 1e6,
                        diff1Bx * 1e6,
                        diff1By * 1e6,
                        diff1Cx * 1e6,
                        diff1Cy * 1e6,
                        diff1Dx * 1e6,
                        diff1Dy * 1e6,
                    ]
                )
    return diff_list


def main(
    logfile="../data/elionix_schedule/New schedule.log",
    binwidth=1,
    save_fig=False,
    output_path="alignment_histogram.pdf",
):
    """
    Main function to parse Elionix log and plot alignment histogram.
    """
    diff_list = parse_elionix_log(logfile)
    plot_alignment_histogram(
        diff_list, binwidth=binwidth, save_fig=save_fig, output_path=output_path
    )


if __name__ == "__main__":
    main()
