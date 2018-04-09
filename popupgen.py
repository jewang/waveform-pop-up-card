#!/usr/bin/env python

"""A pop-up pattern generator

TODO
- Test patterns to verify they are physically possible
- Thinner plot lines
- Look into pipenv requirements.txt

"""

from __future__ import print_function

import argparse
import sys

import matplotlib
import numpy as np
from pydub import AudioSegment

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

def sample_audio(filepath):
    print("Reading " + filepath)
    audio = AudioSegment.from_file(filepath)
    arr = np.array(audio.get_array_of_samples())
    arr = arr[arr > 0]
    step = int(len(arr)/60)
    avg = moving_average(arr, step)
    avg = max(avg) - avg
    ret = avg[::step]
    return ret

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def generate_pdf(samples, outfile):
    print("Generating " + outfile)
    if (len(samples) ** 2 == 1):
        samples = samples[:-1]

    half_index = int(len(samples)/2)
    arr_half1 = samples[:half_index]
    arr_half2 = samples[half_index:] * -1

    with plt.rc_context({"axes.edgecolor":"r"}):
        fig = plt.figure(figsize=[4.5, 7])
        plt.xticks([])
        plt.yticks([])
        plotarr(arr_half1)
        plotarr(arr_half2)
        plt.hlines(0, 0, len(arr_half1) - 1, colors="g", zorder=100)
        fig.tight_layout()
        plt.xlim(0, len(arr_half1) - 1)
        plt.savefig(outfile)

def plotarr(ret):
    for l in plt.step(range(len(ret)), ret):
        l.set_color("g")

    plt.stem(ret, markerfmt=" ", linefmt="r-")
    plt.stem(ret[1:], markerfmt=" ", linefmt="r-")

def run(infile, outfile):
    generate_pdf(sample_audio(infile), outfile)

def main(arguments):

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--infile", help="Input file",
                        type=argparse.FileType("r"))
    parser.add_argument("--outfile", help="Output file",
                        default=sys.stdout, type=argparse.FileType("w"))

    args = parser.parse_args(arguments)

    run(args.infile.name, args.outfile.name)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))