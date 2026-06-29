# nclim-utils

Python utilities for working with data from the **NCLIM** processing system. The package
is intended to grow over time to cover various NCLIM-related data types; the current
functionality is focused on shipboard ADCP data.

## Current functionality

- **Reader** for processed shipboard ADCP data stored in the native **NCLIM HDF5 format**
  (`.adcp.h5` files), via the `Ensembles` class.
- **Command-line converter** (`nclim2codas`) from NCLIM HDF5 to a **CODAS-style short-form
  netCDF** file — the standard end-user product compatible with pycurrents and most
  oceanographic analysis tools.

## What it does

The core class `Ensembles` represents a processed shipboard ADCP ensemble dataset in the
NCLIM ADCP processing chain. It can:

- **Load** a processed ADCP dataset from an NCLIM `.adcp.h5` file (`Ensembles.load()`)
- **Export** a CODAS-compatible short-form netCDF file containing ocean velocity profiles
  (u, v), ship velocity (uship, vship), position, depth, amplitude, percent-good, and
  instrument configuration (`Ensembles.save_as_codas_nc()`)
- **Store / restore** the full internal object in HDF5 for
  downstream processing (`store()`, `load()`);
  netCDF export is handled via `save_as_nc()`.

## Installation

Requires Python ≥ 3.9. Dependencies (`numpy`, `h5py`, `netCDF4`) are installed
automatically.

```bash
pip install git+https://github.com/marek-os/nclim-utils.git
```

On systems with a managed/externally-controlled Python (e.g. Homebrew, Debian/Ubuntu),
add `--break-system-packages`.

## Updating

Since this is installed via git, `pip` won't pick up new commits on a
plain re-run of `pip install`. To force an update:

```bash
pip install --force-reinstall --no-deps git+https://github.com/marek-os/nclim-utils.git
```

**Verify the install:**

```bash
python -c "import nclimadcp; print(nclimadcp.__file__)"
nclim2codas --help
```

## Command-line tool — `nclim2codas`

After installation the command `nclim2codas` is available on the terminal (Linux, macOS,
Windows).

```
usage: nclim2codas [-h] [--platform NAME] [--cruise_id ID] [--sonar SONAR] src trg

Convert an NCLIM ADCP HDF5 file to CODAS short-form netCDF.

positional arguments:
  src               Source ADCP HDF5 file  (e.g. cruise.adcp.h5)
  trg               Target netCDF output path  (e.g. cruise_codas.nc)

options:
  -h, --help        show this help message and exit
  --platform NAME   Ship / platform name (default: UNSPECIFIED)
  --cruise_id ID    Cruise identifier; derived from source filename if omitted
  --sonar SONAR     Sonar / instrument identifier; read from file if omitted
```

### Examples

Minimal — platform name taken from the file, cruise ID from the filename:

```bash
nclim2codas  /data/ADCP/cruise.adcp.h5  /data/netcdf/cruise_codas.nc
```

With metadata:

```bash
nclim2codas  /data/ADCP/cruise.adcp.h5  /data/netcdf/cruise_codas.nc \
             --platform  "RV Nansen" \
             --cruise_id  2023401 \
             --sonar      os150nb
```

The tool validates that the source file exists and that the target directory is present
before reading any data, and exits with an informative error message if either check fails.

## Python API

```python
from nclimadcp import Ensembles

ens = Ensembles()
ens.load("cruise.adcp.h5")
print(ens.dims())          # (nprofs, nbins)

ens.save_as_codas_nc(
    "cruise_codas.nc",
    platform="RV Nansen",
    cruise_id="2023401",
    sonar="os150nb",
)
```

## Output variables (CODAS short form)

| Variable | Description |
|---|---|
| `time` | Decimal day from start of year |
| `lon`, `lat` | GPS position at end of ensemble |
| `u`, `v` | Ocean eastward / northward velocity (m s⁻¹) |
| `uship`, `vship` | Ship eastward / northward velocity (m s⁻¹) |
| `depth` | Bin-centre depths (m, positive down) |
| `amp` | Received signal strength (ADCP counts) |
| `pg` | Percent-good pings after editing |
| `pflag` | Profile editing flags |
| `heading` | Mean ship heading during ensemble (°) |
| `tr_temp` | ADCP transducer temperature (°C) |
| `num_pings` | Number of pings averaged per ensemble |
| `detected_bottom` | Edited bottom depth along track (m) |

## Roadmap

Additional NCLIM data types and utilities are planned for future releases.