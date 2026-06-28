#!/usr/bin/env python3
"""
nclim2codas.py  —  convert a NANSEN Climate ADCP HDF5 file to CODAS short-form netCDF.

Usage
-----
    nclim2codas.py  <src.adcp.h5>  <trg.nc>  [--platform NAME]
                    [--cruise_id ID]  [--sonar SONAR]
"""

import argparse
import os
import sys

from .ensembles_class import Ensembles


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="nclim2codas.py",
        description="Convert a NANSEN Climate ADCP HDF5 file to CODAS short-form netCDF.",
    )
    parser.add_argument("src",
                        help="Source ADCP HDF5 file (e.g. cruise.adcp.h5)")
    parser.add_argument("trg",
                        help="Target netCDF output path (e.g. cruise_codas.nc)")
    parser.add_argument("--platform", default="UNSPECIFIED",
                        help="Ship / platform name (default: UNSPECIFIED)")
    parser.add_argument("--cruise_id", default=None,
                        help="Cruise identifier; derived from source filename if omitted")
    parser.add_argument("--sonar", default=None,
                        help="Sonar / instrument identifier; read from file if omitted")

    args = parser.parse_args()

    # --- pre-flight validation -----------------------------------------------
    if not os.path.isfile(args.src):
        print(f"ERROR: source file not found: {args.src}", file=sys.stderr)
        sys.exit(1)

    trg_dir = os.path.dirname(os.path.abspath(args.trg)) or "."
    if not os.path.isdir(trg_dir):
        print(f"ERROR: target directory not found: {trg_dir}", file=sys.stderr)
        sys.exit(1)

    # --- load ----------------------------------------------------------------
    print(f"Reading {args.src} ...")
    ens = Ensembles()
    ens.load(args.src)
    print(f"Data shape: {ens.dims()}")

    # --- convert -------------------------------------------------------------
    ens.save_as_codas_nc(
        args.trg,
        platform=args.platform,
        cruise_id=args.cruise_id,
        sonar=args.sonar,
    )
    print("Done.")


if __name__ == "__main__":
    main()
