import numpy as np
from astropy.io import fits
import argparse
from riptide import TimeSeries, ffa_search, find_peaks
import matplotlib.pyplot as plt


def read_fits_file(fname):
    f = fits.open(fname)
    header = f[0].header
    data = f[0].data

    return header, data

def search_data(data, header):
    assert data.shape[:-1] == (header.npix, header.npix)

    nsamples = data.shape[-1]
    data_len = nsamples * header['TSAMP']

    bins_min = 3
    bins_max = 256

    Pmin = bins_min * header['TSAMP']
    Pmax = data_len / 2

    ducy_max = 0.5
    rmed_width = 4 #seconds

    all_cands = []
    cand_locations = []
    for lpix in range(data.shape[0]):
        for mpix in range(data.shape[1]):
            print(f"Searching pix ({lpix}, {mpix})")
            pixel_data = data[lpix, mpix]
            pixel_time_series = TimeSeries(pixel_data, 
                                           tsamp = header['TSAMP'])
            ts_ffa, pgram = ffa_search(pixel_time_series,
                                            period_min=Pmin,
                                            period_max=Pmax,
                                            fpmin=2,
                                            bins_min=bins_min,
                                            bins_max=bins_max,
                                            ducy_max=ducy_max,
                                            deredden=True,
                                            rmed_width=rmed_width,
                                            )
            peaks, polycos = find_peaks(pgram)
            if len(peaks) > 0:
                print(f"Found {len(peaks)} cands in pixel ({lpix}, {mpix}).")
                all_cands.append(peaks)
                cand_locations.append([lpix, mpix])
                plt.figure()
                pgram.display()
            

    return all_cands, cand_locations

def main():
    header, data = read_fits_file(args.fname)
    all_cands, cand_locations = search_data(data, header)
    print(f"Len of all_cands = {len(all_cands)}.\n Done!")

if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("fname", type=str, help="Path to the fits file")
    a.add_argument("-t", type=float, help="Threshold (def:6)", default=6)
    a.add_argument("-noise_estimator", type=str, help="i(individual pixel) or g (global).. def: i", default='i')
    a.add_argument("-Pmin", type=float, help="Min period (s) - default = 3xtsamp", default=None)
    a.add_argument("-Pmax", type=float, help="Max period (s) - default = tobs / 2", default=None)
    a.add_argument("-bins_min", type=int, help="Min no of bins after folding", default=None)
    a.add_argument("-bins_max", type=int, help="Max no of bins after folding", default=None)
    a.add_argument("-ducy_max", type=float, help="Maximum duty cycle to optimally search. Limits the maximum width of boxcar filters", default = 0.25)
    a.add_argument("-rmed_width", type=float, help="Width of running median filter (in sec) to subtract from input")

    args = a.parse_args()
    main()
