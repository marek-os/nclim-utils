"""
ensembles_class.py
-------------------
Python class mirroring the IDL MO_RawEnsembles structure.

Provides
--------
    read_codas(dbpathname)        populate from a CODAS ADCP database
    store(h5path, root=None)      write to HDF5 in the IDL-compatible layout
    load(h5path, root=None)       read back HDF5 written by store() (or IDL)
    store_netcdf(ncpath)          write to dimensioned netCDF4
    save_as_nc(ncpath)            public 'save as netCDF' entry point

HDF5 layout
-----------
All datasets live under  /MO_RAWEnsambles.v01/  (IDL spelling preserved).
Optional root= prepends an extra group.
Sub-group  PROPS/  holds scalar control parameters.

Design rule: NO array field is ever None after read_codas().
Missing data -> NaN (floats) or copy of nearest equivalent (uint8).

Requires: numpy, h5py  (pycurrents optional — only for CODAS read/write)
"""


from __future__ import annotations
import os
import glob
import subprocess
import tempfile
import numpy as np
import h5py
from datetime import datetime, timezone, timedelta
import platform
import shutil
import netCDF4 as nc4



_H5_PREFIX = "/MO_RAWEnsambles.v01/"   # IDL spelling preserved exactly




# techData[i] : (name, units, description).  units='' = dimensionless/flag,
# units=None = unit not established in source (confirm against CODAS).
_TECHDATA_FIELDS = [
    ("frequency",      "kHz",  "Transducer frequency"),
    ("beam_angle",     "deg",  "Beam angle from vertical"),
    ("xmt_pulse",      None,   "Transmit pulse length"),
    ("bin_length",     "m",    "Depth bin length"),
    ("blank_distance", None,   "Blank distance after transmit"),
    ("heading_align",  "deg",  "Heading alignment offset"),
    ("narrowband",     "",     "Narrowband flag (1=narrowband, 0=broadband)"),
    ("xdcr_facing",    "",     "Transducer facing (1=downward, 0=upward)"),
    ("earth_coords",   "",     "Earth-coordinates flag (1=earth, 0=beam)"),
    ("tr_depth",       "m",    "Transducer depth below surface"),
    ("version",        "",     "CONFIGURATION_1 version"),
    ("nbins",          "",     "Number of depth bins"),
]

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _yearday_to_jd(dday, yearbase):
    jan1 = datetime(yearbase, 1, 1, tzinfo=timezone.utc)
    jd_jan1 = 2440587.5 + jan1.timestamp() / 86400.0
    return jd_jan1 + np.asarray(dday, dtype=np.float64)

def _jd_to_year(jd):
    """Calendar year for a Julian Day — inverse of _yearday_to_jd's epoch.
    Uses timedelta (not fromtimestamp) to stay safe on Windows for any
    pre-1970 dates."""
    base = datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (base + timedelta(seconds=(float(jd) - 2440587.5) * 86400.0)).year

def _clean(arr):
    out = np.ma.filled(arr, np.nan).astype(np.float32)
    out[np.abs(out) > 1e37] = np.nan
    return out

def _nan_matrix(n, b):
    a = np.empty((n, b), dtype=np.float32); a[:] = np.nan; return a

def _nan_vector(n):
    a = np.empty(n, dtype=np.float32); a[:] = np.nan; return a



def _haversine_nm(lon, lat):
    R = 3440.065
    lo, la = np.deg2rad(lon), np.deg2rad(lat)
    dlo, dla = np.diff(lo), np.diff(lat)
    a = np.sin(dla/2)**2 + np.cos(la[:-1])*np.cos(la[1:])*np.sin(dlo/2)**2
    c = 2*np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    dist = np.empty(len(lon), dtype=np.float64)
    dist[0] = 0.0; dist[1:] = np.cumsum(R*c)
    return dist


 #--- constants, identical to the sample CODAS short-form file -------------

_FILL_F64 = np.float64(1e38)
_FILL_F32 = np.float32(1e38)
_FILL_U8 = np.uint8(255)
_FILL_I8 = np.int8(-1)

_CODAS_VARIABLES_TEXT = """
Variables in this CODAS short-form Netcdf file are intended for most end-user
scientific analysis and display purposes. For additional information see
the CODAS_processing_note global attribute and the attributes of each
of the variables.


============= =================================================================
time          Time at the end of the ensemble, days from start of year.
lon, lat      Longitude, Latitude from GPS at the end of the ensemble.
u,v           Ocean eastward and northward velocity component profiles.
uship, vship  Eastward and northward velocity components of the ship.
heading       Mean ship heading during the ensemble.
depth         Bin centers in nominal meters (no sound speed profile correction).
tr_temp       ADCP transducer temperature.
pg            Percent Good pings after editing in vector-averaged velocities.
pflag         Profile Flags based on editing, used to mask velocities.
amp           Received signal strength in ADCP-specific units; no correction
              for spreading or attenuation.
============= =================================================================

"""

_CODAS_PROCESSING_NOTE_TEXT = """
CODAS processing note:
======================
See pycurrents adcp_nc.py documentation for full text. Profile flags:

binary    decimal    below    Percent
value     value      bottom   Good       bin
-------+----------+--------+----------+-------+
000         0
001         1                            bad
010         2                  bad
011         3                  bad       bad
100         4         bad
101         5         bad                bad
110         6         bad      bad
111         7         bad      bad       bad
-------+----------+--------+----------+-------+
"""


def _fillcast(arr, dtype, fill):
    """Cast to dtype, replacing NaN with fill (works for float and int dtypes)."""
    arr = np.asarray(arr)
    return np.ma.filled(np.ma.masked_invalid(arr.astype(np.float64)), float(fill)).astype(dtype)



# ---------------------------------------------------------------------------
# class
# ---------------------------------------------------------------------------

class Ensembles:
    """
    Python mirror of IDL MO_RawEnsembles.

    Internally all matrices are (nprofs, nbins) — natural for numpy.
    On store() all matrices are transposed to (nbins, nprofs) to match
    the IDL convention [nbins, nprofs] in the HDF5 file.
    """

    def __init__(self):

        self.__h5_prefix = _H5_PREFIX

        # scalar metadata
        self.ID             = ""
        self.yearbase       = 0
        self.nprofs         = 0
        self.nbins          = 0
        self.bin_length_m   = 0.0
        self.tr_depth_m     = 0.0
        self.freq_hz        = 0.0
        self.avg_interval_s = 0.0
        self.bIsDirty       = 0

        # techData[12] - IDL index convention
        self.techData = np.zeros(12, dtype=np.float64)

        # IDL control scalars
        self.v_threshold = 50.0  # mm/s
        self.bt_threshold = 50.0  # mm/s
        self.pg_threshold = 25.0  # %
        self.th_amin = 0.0
        self.th_amax = 350.0
        self.th_sog = 0.0
        self.th_maxsog = -1.0
        self.maxDepth = -1.0  # internally used; -1 = not set
        self.vref = 1  # 1=NAV
        self.vref_medf = 0
        self.vref_smooth = 0
        self.depth_medf = 0
        self.topMaskOffset = 0
        self.pgNO = 4
        self.ampNo = 0
        self.vbsnum = 1
        self.missalign = 0.0
        self.mTvgLikePower = 2.0
        self.mInstrumentID = 'OSxxx'
        self.mFileType = -1
        self.mNoGoodLinePos = -1
        self.bMagNoMask = 0
        self.bDispRawAmpl = 0
        self.bUse_3Beam = 0
        self.bNoBTline = 0
        self.bHasRawDepths = 0
        self.bBottomManually = 0
        self.bShowExtMask = 1
        self.bVbs_pending = 0
        self.vbsSizeRatio = 8.0

        self.sm_vrange  = np.zeros(2, dtype=np.int32)
        self.sm_arange  = np.zeros(2, dtype=np.int32)
        self.wtrange    = np.zeros(2, dtype=np.int32)
        self.mOrigin    = np.zeros(2, dtype=np.float64)
        self.mSpacing   = np.zeros(2, dtype=np.float64)
        self.th_navdir = np.array([999.0, 999.0], dtype=np.float64)
        self.th_heddir = np.array([999.0, 999.0], dtype=np.float64)
        self.techData = np.zeros(12, dtype=np.float64)
        self.techData[8] = 150.0 # frequency

        self.techData[1] = 30.0  # BEAM_ANGLE
        self.techData[2] = 8.0   # TRANSMIT_PULSE_LENGTH
        self.techData[4] = 10.0  # blank after transmit
        self.techData[5] = 45.0  # Heading align
        self.techData[6] = 1.0  #  IS_NARROWBAND
        self.techData[7] = 1.0  #  is_DOWNWARDFACING
        self.techData[8] = 1.0  # IS_EARTHCOORDINATES
        self.techData[9] = 5.0  # TRANDUCER DEPTH
        # --- array fields (None until read_codas) ---

        # velocities [nprofs x nbins] float32
        self.VX   = None; self.VY   = None
        self.VZ   = None; self.VERR = None

        # percent good [nprofs x nbins] uint8
        self.PGOOD4 = None; self.PGOOD3 = None
        self.PGOOD2 = None; self.PGOOD1 = None
        self.PGOOD_ = None

        # amplitude [nprofs x nbins] uint8
        self.Ampl = None; self.Amp_ = None
        self.Amp1 = None; self.Amp2 = None
        self.Amp3 = None; self.Amp4 = None

        # correlations [nprofs x nbins] float32  NaN when unavailable
        self.Cor1 = None; self.Cor2 = None
        self.Cor3 = None; self.Cor4 = None

        # bottom tracking [nprofs,] float32  NaN when unavailable
        self.BTX  = None; self.BTY  = None
        self.BTZ  = None; self.BERR = None

        # navigation / attitude [nprofs,]
        self.NAVX    = None; self.NAVY    = None
        self.PITCH   = None; self.ROLL    = None
        self.HEADING = None; self.TRACK   = None
        self.TEMP    = None; self.DIST    = None

        # position / time [nprofs,]
        self.lon  = None; self.lat  = None; self.JDAY = None

        # depths
        self.DEPTH      = None   # ocean bottom [nprofs,] float32
        self.depth_bins = None   # bin centres  [nbins,]  float32

        # quality
        self.lgb        = None; self.lgb_edited = None
        self.mab        = None; self.pflag      = None
        self.ev_std     = None; self.nping      = None

        # external (manual) bottom mask [nprofs x nbins] uint8  0=unmasked
        self.ext_mask = None

        # sound speed [nprofs x nbins] float32
        self.SSPEED = None

        # volume backscattering [nprofs x nbins] float32  None = not computed
        self.vbs0 = None; self.vbs1 = None; self.vbs2 = None
        self.vbs3 = None; self.vbs4 = None

        # GPS / time fix arrays (raw RDI only; in CODAS = copies of main vars)
        self.mTrackFix1  = None; self.mTrackFix2  = None
        self.mJdayFix1   = None; self.mJdayFix2   = None
        self.mEnsembleNo = None

        # surface reference fields (optional, from ship log)
        self.mSurfaceDrift = None
        self.mWind         = None

        self.SNDSPD = None  # sound speed used by CODAS (nprofs,) float32


    # -----------------------------------------------------------------------
    # isValid  (mirrors IDL MO_RawEnsembles::isValid)
    # -----------------------------------------------------------------------

    def isValid(self):
        """Return True if data has been loaded (VY is populated)."""
        return self.VY is not None


    # -----------------------------------------------------------------------
    # dims  (mirrors IDL MO_RawEnsembles::dims)
    # -----------------------------------------------------------------------

    def dims(self):
        """Return [nprofs, nbins] — IDL dims() equivalent."""
        return np.array([self.nprofs, self.nbins], dtype=np.int32)

    # -----------------------------------------------------------------------
    # _m_external  (mirrors IDL MO_RawEnsembles::_m_external)
    # -----------------------------------------------------------------------

    def _m_external(self):
        """
        Return the external (manual) bottom mask as uint8 [nbins x nprofs]
        — IDL layout matches Array[nbins, nprofs].
        Creates a zero-filled mask if none has been set.
        0 = unmasked.
        """
        if self.ext_mask is None:
            self.ext_mask = np.zeros((self.nprofs, self.nbins), dtype=np.uint8)
        return self.ext_mask



    # -----------------------------------------------------------------------
    # store  (translated from IDL MO_RawEnsembles::store)
    # -----------------------------------------------------------------------

    def store(self, h5path: str, root: str = None,
              compress: int = 4, overwrite: bool = True) -> None:
        """
        Write the object to an HDF5 file in the IDL-compatible layout.

        Parameters
        ----------
        h5path   : output file path, e.g. "cruise.h5"
        root     : optional extra group prepended to the prefix,
                   e.g. root='/leg1'  ->  /leg1/MO_RAWEnsambles.v01/...
        compress : gzip level 0-9 for matrix datasets (default 4)
        overwrite: if True (default) remove any pre-existing prefix group
        """
        if not self.isValid():
            raise RuntimeError("store(): object has no data — call read_codas() first.")

        prefix = self.__h5_prefix          # '/MO_RAWEnsambles.v01/'
        if root is not None:
            prefix = root.rstrip("/") + prefix

        opts = dict(compression="gzip", compression_opts=compress)

        mode = "a" if os.path.exists(h5path) else "w"

        with h5py.File(h5path, mode) as h:

            # remove any previously stored group at this prefix
            if prefix.strip("/") in h:
                del h[prefix.strip("/")]

            # helpers
            def _set(name, arr, **kw):
                h.create_dataset(prefix + name, data=np.asarray(arr), **kw)

            def _setT(name, arr, **kw):
                """Write matrix transposed: Python (nprofs,nbins) -> IDL (nbins,nprofs)."""
                h.create_dataset(prefix + name, data=np.asarray(arr).T, **kw)

            # ---- geometry --------------------------------------------------
            _set("ORIGIN",  self.mOrigin)
            _set("SPACING", self.mSpacing)

            # ---- velocities  (nprofs,nbins) -> stored as (nbins,nprofs) ----
            _setT("VX",   self.VX,   **opts)
            _setT("VY",   self.VY,   **opts)
            _setT("VZ",   self.VZ,   **opts)
            _setT("VERR", self.VERR, **opts)

            # ---- percent good ----------------------------------------------
            _setT("PGOOD4", self.PGOOD4, **opts)
            _setT("PGOOD1", self.PGOOD1, **opts)
            _setT("PGOOD2", self.PGOOD2, **opts)
            _setT("PGOOD3", self.PGOOD3, **opts)

            # ---- amplitude -------------------------------------------------
            _setT("Ampl", self.Ampl, **opts)
            _setT("AMP1", self.Amp1, **opts)
            _setT("AMP2", self.Amp2, **opts)
            _setT("AMP3", self.Amp3, **opts)
            _setT("AMP4", self.Amp4, **opts)

            # ---- bottom tracking (vectors — no transpose) ------------------
            _set("BTX",  self.BTX)
            _set("BTY",  self.BTY)
            _set("BTZ",  self.BTZ)
            _set("BERR", self.BERR)

            # ---- navigation / attitude (vectors) ---------------------------
            _set("NAVX",    self.NAVX)
            _set("NAVY",    self.NAVY)
            _set("PITCH",   self.PITCH)
            _set("ROLL",    self.ROLL)
            _set("HEADING", self.HEADING)
            _setT("TRACK",   self.TRACK, **opts)
            _set("TEMP",    self.TEMP)
            _set("DEPTH",   self.DEPTH)
            _set("JDAY",    self.JDAY)

            # ---- external bottom mask — _m_external() already returns .T --
            _setT("EMASK", self._m_external(), **opts)

            # ---- instrument ID ---------------------------------------------
            _set("INSTRUMENTID", np.array([self.mInstrumentID], dtype=h5py.string_dtype()))
            _set("ID", np.array([self.ID], dtype=h5py.string_dtype()))

            # ---- manual bottom depth (nbins,) — depth axis, no transpose --
            #_set("MAN_BOTTOM", self.depth_bins)
            #_set("MAN_BOTTOM", self.nprofs)
            if self.MAN_BOTTOM is not None:
                _set("MAN_BOTTOM", self.MAN_BOTTOM)
            else:
                _set("MAN_BOTTOM", _nan_vector(self.nprofs))

            # ---- correlations & sound speed --------------------------------
            _setT("COR1",   self.Cor1,  **opts)
            _setT("COR2",   self.Cor2,  **opts)
            _setT("COR3",   self.Cor3,  **opts)
            _setT("COR4",   self.Cor4,  **opts)

            _set("SSPEED", self.SSPEED)

            # ---- volume backscattering (conditional) -----------------------
            # for tag, arr in (("VBS0", self.vbs0), ("VBS1", self.vbs1),
            #                  ("VBS2", self.vbs2), ("VBS3", self.vbs3),
            #                  ("VBS4", self.vbs4)):
            #     if arr is not None:
            #         _setT(tag, arr, **opts)

            # ---- GPS / time fix arrays (conditional, vectors) -------------
            if self.mTrackFix1 is not None:
                _set("TRACK_FIX1", self.mTrackFix1)
            if self.mTrackFix2 is not None:
                _set("TRACK_FIX2", self.mTrackFix2)
            if self.mJdayFix1 is not None:
                _set("JDAY_FIX1", self.mJdayFix1)
            if self.mJdayFix2 is not None:
                _set("JDAY_FIX2", self.mJdayFix2)
            if self.mEnsembleNo is not None:
                _set("ENSEMBLE_NO", self.mEnsembleNo)

            # ---- surface reference fields (conditional, matrices) ----------
            if self.mSurfaceDrift is not None:
                _setT("SURFACE_DRIFT", self.mSurfaceDrift)
            if self.mWind is not None:
                _setT("SURFACE_WIND", self.mWind)

            # ---- NOGOOD_LINE -----------------------------------------------
            _set("NOGOOD_LINE", np.array([self.mNoGoodLinePos], dtype=np.int32))

            # ---- BEDIT_BYHAND ----------------------------------------------
            _set("BEDIT_BYHAND", np.array([self.bBottomManually], dtype=np.int16))

            # ---- TVGLIKE ---------------------------------------------------
            _set("TVGLIKE", np.array([self.mTvgLikePower], dtype=np.float64))

            # ---- PROPS sub-group -------------------------------------------
            props = prefix + "PROPS/"

            _set("PROPS/sm_vrange", self.sm_vrange)
            _set("PROPS/sm_arange", self.sm_arange)
            _set("PROPS/wtrange",   self.wtrange)

            # dblProps[6]: th_amin, th_amax, th_sog, maxDepth, th_maxsog, missalign
            dblProps = np.array([
                self.th_amin, self.th_amax, self.th_sog,
                self.maxDepth, self.th_maxsog, self.missalign
            ], dtype=np.float64)
            _set("PROPS/dblProps", dblProps)

            # intProps[11]: v_thr, bt_thr, pg_thr, vref, vref_medf,
            #               vref_smooth, depth_medf, pgNO, ampNo,
            #               topMaskOffset, bNoBTline
            intProps = np.array([
                self.v_threshold, self.bt_threshold, self.pg_threshold,
                self.vref, self.vref_medf, self.vref_smooth,
                self.depth_medf, self.pgNO, self.ampNo,
                self.topMaskOffset, self.bNoBTline
            ], dtype=np.int16)
            _set("PROPS/intProps", intProps)

            _set("PROPS/techData", self.techData)

            # VERSION_int[4]: NX, NY, hasManualDepths, version_counter
            # hasManualDepths() and dims() are IDL methods; approximate here:
            has_manual = int(self.bBottomManually)
            version_int = np.array(
                [self.nprofs, self.nbins, has_manual, 100],
                dtype=np.int16)
            _set("PROPS/VERSION_int", version_int)

        print(f"store(): wrote {self.nprofs}x{self.nbins}  ->  {h5path}  "
              f"[prefix: {prefix}]")

    def _save_as_netcdf_native(self, ncpath: str, root: str = None,
                               compress: int = 4, overwrite: bool = True) -> None:
        """
        Write the object to netCDF4 in a layout parallel to store() (HDF5).

        DEPRECATED: use save_as_codas_nc() to export data for external use. This
        method writes the full native (all-variable) layout, mirroring the
        internal object structure, rather than the CODAS-style short form
        produced by save_as_nc().

        Group  MO_RAWEnsambles.v01/  holds the arrays; sub-group PROPS/ holds
        the packed control vectors. Unlike store(), matrices are written
        natural (profile, bin) — netCDF keys on dimension names, and CF
        convention puts the record dimension first.

        Parameters
        ----------
        ncpath    : output file path, e.g. "cruise.nc"
        root      : optional parent group, e.g. root='/leg1'
        compress  : zlib level 0-9 for 2-D variables (default 4)
        overwrite : if True, clobber the whole file; else append (mode 'a')
        """

        if not self.isValid():
            raise RuntimeError("store_netcdf(): no data — call read_codas() or load() first.")

        N, B = self.nprofs, self.nbins
        mode = "w" if (overwrite or not os.path.exists(ncpath)) else "a"

        bin_depth = self.depth_bins if self.depth_bins is not None else self.yaxis()
        man_bottom = getattr(self, "MAN_BOTTOM", None)

        with nc4.Dataset(ncpath, mode, format="NETCDF4") as ncfile:

            # ---- group nesting --------------------------------------------
            grp = ncfile
            if root is not None:
                grp = grp.createGroup(root.strip("/"))
            grp = grp.createGroup("MO_RAWEnsambles.v01")

            # ---- dimensions -----------------------------------------------
            grp.createDimension("profile", N)  # set to None for unlimited
            grp.createDimension("bin", B)
            grp.createDimension("two", 2)
            grp.createDimension("tech", len(self.techData))

            # ---- helper ---------------------------------------------------
            def V(g, name, data, dims, dtype=None, comp=False, **attrs):
                data = np.asarray(data)
                dtype = dtype or data.dtype
                v = g.createVariable(name, dtype, dims,
                                     zlib=bool(comp),
                                     complevel=compress if comp else 0)
                v[...] = data
                for k, val in attrs.items():
                    setattr(v, k, val)
                return v

            PB = ("profile", "bin")

            # ---- coordinates ----------------------------------------------
            V(grp, "JDAY", self.JDAY, ("profile",),
              units="days", long_name="Julian day")
            V(grp, "depth_bins", bin_depth, ("bin",),
              units="m", positive="down", long_name="bin centre depth")
            V(grp, "lon", self.lon, ("profile",),
              units="degrees_east", standard_name="longitude")
            V(grp, "lat", self.lat, ("profile",),
              units="degrees_north", standard_name="latitude")

            # ---- geometry -------------------------------------------------
            V(grp, "ORIGIN", self.mOrigin, ("two",))
            V(grp, "SPACING", self.mSpacing, ("two",))

            # ---- velocities (mm/s) ----------------------------------------
            for nm, arr in (("VX", self.VX), ("VY", self.VY),
                            ("VZ", self.VZ), ("VERR", self.VERR)):
                V(grp, nm, arr, PB, np.float32, comp=True, units="mm s-1")

            # ---- percent good ---------------------------------------------
            for nm, arr in (("PGOOD4", self.PGOOD4), ("PGOOD1", self.PGOOD1),
                            ("PGOOD2", self.PGOOD2), ("PGOOD3", self.PGOOD3)):
                V(grp, nm, arr, PB, np.float64, comp=True, units="percent")

            # ---- amplitude ------------------------------------------------
            V(grp, "Ampl", self.Ampl, PB, np.uint8, comp=True)
            for nm, arr in (("AMP1", self.Amp1), ("AMP2", self.Amp2),
                            ("AMP3", self.Amp3), ("AMP4", self.Amp4)):
                V(grp, nm, arr, PB, comp=True)

            # ---- correlations ---------------------------------------------
            for nm, arr in (("COR1", self.Cor1), ("COR2", self.Cor2),
                            ("COR3", self.Cor3), ("COR4", self.Cor4)):
                V(grp, nm, arr, PB, np.float32, comp=True)

            # ---- external bottom mask -------------------------------------
            V(grp, "EMASK", self._m_external(), PB, np.uint8, comp=True,
              long_name="manual bottom mask (0=unmasked)")

            # ---- bottom tracking ------------------------------------------
            for nm, arr in (("BTX", self.BTX), ("BTY", self.BTY),
                            ("BTZ", self.BTZ), ("BERR", self.BERR)):
                V(grp, nm, arr, ("profile",), np.float32)

            # ---- navigation / attitude ------------------------------------
            for nm, arr in (("NAVX", self.NAVX), ("NAVY", self.NAVY),
                            ("PITCH", self.PITCH), ("ROLL", self.ROLL),
                            ("HEADING", self.HEADING), ("TEMP", self.TEMP),
                            ("DEPTH", self.DEPTH)):
                V(grp, nm, arr, ("profile",), np.float32)

            V(grp, "TRACK", self.TRACK, ("profile", "two"), np.float64, comp=True)
            V(grp, "SSPEED", self.SSPEED, ("profile",), np.float32, units="m s-1")
            if self.DIST is not None:
                V(grp, "DIST", self.DIST, ("profile",), np.float64, units="nmi")

            # ---- manual bottom depth --------------------------------------
            V(grp, "MAN_BOTTOM",
              man_bottom if man_bottom is not None else _nan_vector(N),
              ("profile",), np.float32, units="m")

            # ---- optional matrices ----------------------------------------
            for nm, arr in (("VBS0", self.vbs0), ("VBS1", self.vbs1),
                            ("VBS2", self.vbs2), ("VBS3", self.vbs3),
                            ("VBS4", self.vbs4), ("SURFACE_DRIFT", self.mSurfaceDrift),
                            ("SURFACE_WIND", self.mWind)):
                if arr is not None:
                    V(grp, nm, arr, PB, comp=True)

            # ---- optional GPS / time fix vectors --------------------------
            for nm, arr in (("TRACK_FIX1", self.mTrackFix1),
                            ("TRACK_FIX2", self.mTrackFix2),
                            ("JDAY_FIX1", self.mJdayFix1),
                            ("JDAY_FIX2", self.mJdayFix2),
                            ("ENSEMBLE_NO", self.mEnsembleNo)):
                if arr is not None:
                    V(grp, nm, arr, ("profile",), np.float64)

            # ---- PROPS sub-group ------------------------------------------
            pr = grp.createGroup("PROPS")
            pr.createDimension("n6", 6)
            pr.createDimension("n11", 11)
            pr.createDimension("n4", 4)

            V(pr, "sm_vrange", self.sm_vrange, ("two",), np.int32)
            V(pr, "sm_arange", self.sm_arange, ("two",), np.int32)
            V(pr, "wtrange", self.wtrange, ("two",), np.int32)

            dblProps = np.array([self.th_amin, self.th_amax, self.th_sog,
                                 self.maxDepth, self.th_maxsog, self.missalign],
                                dtype=np.float64)
            V(pr, "dblProps", dblProps, ("n6",), np.float64)

            intProps = np.array([self.v_threshold, self.bt_threshold, self.pg_threshold,
                                 self.vref, self.vref_medf, self.vref_smooth,
                                 self.depth_medf, self.pgNO, self.ampNo,
                                 self.topMaskOffset, self.bNoBTline], dtype=np.int16)
            V(pr, "intProps", intProps, ("n11",), np.int16)

            V(pr, "techData", self.techData, ("tech",), np.float64)

            version_int = np.array([N, B, int(self.bBottomManually), 100], dtype=np.int16)
            V(pr, "VERSION_int", version_int, ("n4",), np.int16)

            # ---- techData as global descriptors (NC_GLOBAL) ---------------
            for i, (fname, units, desc) in enumerate(_TECHDATA_FIELDS):
                if i >= len(self.techData):
                    break
                aname = f"techData_{i:02d}_{fname}"
                ncfile.setncattr(aname, float(self.techData[i]))
                if units is None:
                    label = desc + " [units: confirm against CODAS]"
                elif units:
                    label = f"{desc} [{units}]"
                else:
                    label = desc
                ncfile.setncattr(aname + "_desc", label)

            # ---- scalar metadata as group attributes ----------------------
            grp.ID = self.ID
            grp.INSTRUMENTID = self.mInstrumentID
            grp.yearbase = int(self.yearbase)
            grp.nprofs = int(N)
            grp.nbins = int(B)
            grp.bin_length_m = float(self.bin_length_m)
            grp.tr_depth_m = float(self.tr_depth_m)
            grp.freq_hz = float(self.freq_hz)
            grp.avg_interval_s = float(self.avg_interval_s)
            grp.NOGOOD_LINE = int(self.mNoGoodLinePos)
            grp.BEDIT_BYHAND = int(self.bBottomManually)
            grp.TVGLIKE = float(self.mTvgLikePower)
            grp.Conventions = "CF-1.8 (partial)"

        print(f"_save_as_netcdf_native(): wrote {N}x{B}  ->  {ncpath}")



    # -----------------------------------------------------------------------
    # convenience
    # -----------------------------------------------------------------------

    @property
    def shape(self):
        return (self.nprofs, self.nbins)

    def __repr__(self):
        if self.isValid():
            return (f"RawEnsembles(ID='{self.ID}', "
                    f"nprofs={self.nprofs}, nbins={self.nbins}, "
                    f"yearbase={self.yearbase})")
        return "RawEnsembles(empty — call read_codas() to load data)"





    # -----------------------------------------------------------------------
    # load  — read a RawEnsembles object from HDF5 file
    # -----------------------------------------------------------------------

    def load(self, h5path: str, root: str = None) -> None:
        """
        Read a RawEnsembles object from an HDF5 file written by store().

        Mirrors IDL MO_RawEnsembles::load().

        Parameters
        ----------
        h5path : str
            Path to HDF5 file, e.g. "cruise.h5"
        root : str, optional
            Extra group prepended to the prefix, e.g. root='/leg1'
        """

        prefix = self.__h5_prefix          # '/MO_RAWEnsambles.v01/'
        if root is not None:
            prefix = root.rstrip("/") + prefix

        self.ID = os.path.splitext(os.path.basename(h5path))[0]

        # helper: read required dataset — raises if missing
        def _get(h, name):
            return h[prefix + name][:]

        # helper: read optional dataset — returns None if missing
        def _get_opt(h, name):
            try:
                ds = h[prefix + name]
                if ds.shape == ():
                    return np.atleast_1d(ds[()])
                return ds[:]
            except KeyError:
                return None

        # helper: detect file type from filename suffix
        def _force_file_type(fname):
            if fname.endswith('ENX.raw.adcp.h5'):  return 2
            if fname.endswith('STA.raw.adcp.h5'):  return 0
            if fname.endswith('LTA.raw.adcp.h5'):  return 1
            if fname.endswith('ENS.raw.adcp.h5'):  return 3
            if fname.endswith('ENR.raw.adcp.h5'):  return 4
            return -1

        self.mFileType = _force_file_type(h5path)

        with h5py.File(h5path, "r") as h:

            # ---- geometry --------------------------------------------------
            self.mOrigin  = _get(h, 'ORIGIN').astype(np.float64)
            self.mSpacing = _get(h, 'SPACING').astype(np.float64)

            # ---- velocities — stored as (nbins, nprofs), transpose back ----
            self.VX   = _get(h, 'VX').T.astype(np.float32)
            self.VY   = _get(h, 'VY').T.astype(np.float32)
            self.VZ   = _get(h, 'VZ').T.astype(np.float32)
            self.VERR = _get(h, 'VERR').T.astype(np.float32)

            # derive nprofs and nbins from VX shape (nprofs, nbins)
            self.nprofs, self.nbins = self.VX.shape

            # ---- percent good ----------------------------------------------
            self.PGOOD4 = _get(h, 'PGOOD4').T.astype(np.float64)

            # ---- amplitude -------------------------------------------------
            self.Ampl = _get(h, 'Ampl').T.astype(np.float64)

            # ---- bottom tracking -------------------------------------------
            self.BTX  = _get(h, 'BTX').astype(np.float32)
            self.BTY  = _get(h, 'BTY').astype(np.float32)
            self.BTZ  = _get(h, 'BTZ').astype(np.float32)
            self.BERR = _get(h, 'BERR').astype(np.float32)

            # ---- navigation / attitude -------------------------------------
            self.NAVX    = _get(h, 'NAVX').astype(np.float32)
            self.NAVY    = _get(h, 'NAVY').astype(np.float32)
            self.PITCH   = _get(h, 'PITCH').astype(np.float32)
            self.ROLL    = _get(h, 'ROLL').astype(np.float32)
            self.HEADING = _get(h, 'HEADING').astype(np.float32)
            self.TRACK   = _get(h, 'TRACK').T.astype(np.float64)  # (nprofs,2)
            self.TEMP    = _get(h, 'TEMP').astype(np.float32)
            self.DEPTH   = _get(h, 'DEPTH').astype(np.float32)
            self.JDAY    = _get(h, 'JDAY').astype(np.float64)

            # restore lon / lat from TRACK[:,1] and TRACK[:,0]
            self.lon = self.TRACK[:, 0].copy()
            self.lat = self.TRACK[:, 1].copy()

            # ---- external mask — stored as (nbins, nprofs), transpose back -
            emask = _get(h, 'EMASK')
            self.ext_mask = emask.T.astype(np.uint8)   # (nprofs, nbins)

            # ---- instrument ID --------------------------------------------
            arr = _get_opt(h, 'INSTRUMENTID')
            if arr is not None:
                self.mInstrumentID = str(arr[0]) if arr.ndim > 0 else str(arr)

            # ---- PROPS sub-group -------------------------------------------
            props = prefix + "PROPS/"

            def _getp(name, default=None):
                try:    return h[props + name][:]
                except: return default

            self.sm_vrange = _getp('sm_vrange', np.zeros(2, np.int32))
            self.sm_arange = _getp('sm_arange', np.zeros(2, np.int32))
            wtr = _getp('wtrange')
            self.wtrange = wtr if wtr is not None else np.array([2, 5], np.int32)

            dblProps = _getp('dblProps')
            if dblProps is not None:
                self.th_amin    = float(dblProps[0])
                self.th_amax    = float(dblProps[1])
                self.th_sog     = float(dblProps[2])
                self.maxDepth   = float(dblProps[3])
                self.th_maxsog  = float(dblProps[4]) if len(dblProps) >= 5 else -1.0
                self.missalign  = float(dblProps[5]) if len(dblProps) >= 6 else 0.0

            intProps = _getp('intProps')
            if intProps is not None:
                self.v_threshold   = float(intProps[0])
                self.bt_threshold  = float(intProps[1])
                self.pg_threshold  = float(intProps[2])
                self.vref          = int(intProps[3])
                self.vref_medf     = int(intProps[4])
                self.vref_smooth   = int(intProps[5])
                self.depth_medf    = int(intProps[6])
                if len(intProps) >= 9:
                    self._set_pgno(int(intProps[7]))
                    self._set_ampno(int(intProps[8]))
                if len(intProps) >= 10:
                    self.topMaskOffset = int(intProps[9])
                if len(intProps) >= 11:
                    self.bNoBTline = int(intProps[10])

            # ---- manual bottom depth ---------------------------------------
            mandepth = _get_opt(h, 'MAN_BOTTOM')
            if mandepth is not None:
                self.MAN_BOTTOM = mandepth.astype(np.float32)
                self.bIsDirty = 0

            # ---- per-beam PGOOD (new format) or NaN fill (legacy) ----------
            pg1 = _get_opt(h, 'PGOOD1')
            if pg1 is not None:
                self.PGOOD1 = pg1.T.astype(np.float64)
                self.PGOOD2 = _get(h, 'PGOOD2').T.astype(np.float64)
                self.PGOOD3 = _get(h, 'PGOOD3').T.astype(np.float64)
            else:
                # legacy: fill with NaN
                pg_nan = np.full_like(self.PGOOD4, np.nan, dtype=np.float64)
                self.PGOOD1 = pg_nan.copy()
                self.PGOOD2 = pg_nan.copy()
                self.PGOOD3 = pg_nan.copy()
                self._set_pgno(4)

            # ---- per-beam amplitude (new format) or NaN fill (legacy) ------
            amp1 = _get_opt(h, 'AMP1')
            if amp1 is not None:
                self.Amp1 = amp1.T.astype(np.float64)
                self.Amp2 = _get(h, 'AMP2').T.astype(np.float64)
                self.Amp3 = _get(h, 'AMP3').T.astype(np.float64)
                self.Amp4 = _get(h, 'AMP4').T.astype(np.float64)
            else:
                # legacy: fill with NaN
                amp_nan = np.full_like(self.Ampl, np.nan, dtype=np.float64)
                self.Amp1 = amp_nan.copy()
                self.Amp2 = amp_nan.copy()
                self.Amp3 = amp_nan.copy()
                self.Amp4 = amp_nan.copy()
                self._set_ampno(0)
            self.Amp_ = self.Ampl.copy()

            # ---- techData + correlations + sound speed ---------------------
            tech = _getp('techData')
            if tech is not None:
                # techData in file may be shorter than our 12-element array
                self.techData[:len(tech)] = tech

                self.Cor1   = _get(h, 'COR1').T.astype(np.float32)
                self.Cor2   = _get(h, 'COR2').T.astype(np.float32)
                self.Cor3   = _get(h, 'COR3').T.astype(np.float32)
                self.Cor4   = _get(h, 'COR4').T.astype(np.float32)


                self.SSPEED = _get(h, 'SSPEED').astype(np.float32)

                # legacy correction: if tech had 9 elements transducer depth
                # was not added to origin — add 5.5 m (Nansen draft)
                if len(tech) == 9:
                    self.mOrigin[1] += 5.50
            else:
                # legacy: no techData — mark frequency invalid, fill NaN
                self.techData[0] = -999.0
                nan_mat = _nan_matrix(self.nprofs, self.nbins)
                self.Cor1 = nan_mat.copy(); self.Cor2 = nan_mat.copy()
                self.Cor3 = nan_mat.copy(); self.Cor4 = nan_mat.copy()
                self.SSPEED = _nan_matrix(self.nprofs, self.nbins)

            # ---- volume backscattering (conditional) -----------------------
            for tag in ('VBS0', 'VBS1', 'VBS2', 'VBS3', 'VBS4'):
                arr = _get_opt(h, tag)
                if arr is not None:
                    setattr(self, tag.lower(), arr.T.astype(np.float32))

            # ---- GPS / time fix arrays (conditional) ----------------------
            for tag, attr in (('TRACK_FIX1', 'mTrackFix1'),
                              ('TRACK_FIX2', 'mTrackFix2'),
                              ('JDAY_FIX1',  'mJdayFix1'),
                              ('JDAY_FIX2',  'mJdayFix2'),
                              ('ENSEMBLE_NO','mEnsembleNo')):
                arr = _get_opt(h, tag)
                if arr is not None:
                    setattr(self, attr, arr.astype(np.float64))

            # ---- surface reference (conditional) ---------------------------
            arr = _get_opt(h, 'SURFACE_DRIFT')
            if arr is not None:
                self.mSurfaceDrift = arr.T.astype(np.float32)
            arr = _get_opt(h, 'SURFACE_WIND')
            if arr is not None:
                self.mWind = arr.T.astype(np.float32)

            # ---- scalar flags ----------------------------------------------
            arr = _get_opt(h, 'NOGOOD_LINE')
            self.mNoGoodLinePos = int(arr[0]) if arr is not None \
                                  else self.nbins - 1

            arr = _get_opt(h, 'BEDIT_BYHAND')
            if arr is not None:
                self.bBottomManually = int(arr[0])
            else:
                self.bBottomManually = 1 if self._has_manual_depths() else 0

            arr = _get_opt(h, 'TVGLIKE')
            self.mTvgLikePower = float(arr[0]) if arr is not None else 2.0

        # derive bin_length and spacing from mSpacing
        self.bin_length_m   = float(self.mSpacing[1])
        self.avg_interval_s = float(self.mSpacing[0]) * 86400.0

        # fields not stored on the IDL side — regenerate on load
        self.depth_bins = self.yaxis()   # from mOrigin[1] + mSpacing[1]
        self.yearbase   = _jd_to_year(self.JDAY[0]) if (
            self.JDAY is not None and len(self.JDAY)) else 0

        print(f"load(): {self.nprofs} profiles x {self.nbins} bins  [{h5path}]")

    # -----------------------------------------------------------------------
    # _set_pgno / _set_ampno  (mirrors IDL __setPGNO / __setAmpNO)
    # -----------------------------------------------------------------------

    def _set_pgno(self, no: int) -> None:
        self.pgNO = no
        self.PGOOD_ = {1: self.PGOOD1, 2: self.PGOOD2,
                       3: self.PGOOD3}.get(no, self.PGOOD4)

    def _set_ampno(self, no: int) -> None:
        self.ampNo = no
        self.Amp_ = {1: self.Amp1, 2: self.Amp2,
                     3: self.Amp3, 4: self.Amp4}.get(no, self.Ampl)

    # -----------------------------------------------------------------------
    # _has_manual_depths  (mirrors IDL hasManualDepths)
    # -----------------------------------------------------------------------

    def _has_manual_depths(self) -> bool:
        if not hasattr(self, 'MAN_BOTTOM') or self.MAN_BOTTOM is None:
            return False
        return bool(np.any(np.isfinite(self.MAN_BOTTOM)))

    def _vel1(self):
        return self.VX

    def _vel2(self):
        return self.VY

    def _vel3(self):
        return self.VZ

    def _vel4(self):
        return self.VERR

    def origin(self):
        return self.mOrigin

    def spacing(self):
        return self.mSpacing

    def _amp1(self):
        return self.Amp1

    def _amp2(self):
        return self.Amp2

    def _amp3(self):
        return self.Amp3

    def _amp4(self):
        return self.Amp4

    def _amp0(self):
        return self.Amp_

    def depths(self):
        if not self.isValid():
            raise RuntimeError("depths(): data not loaded — call read_codas() or load() first.")
        return self.MAN_BOTTOM

    def is_valid(self):
        return self.VY is not None

    def dims(self):
        if not self.isValid():
            return 0, 0
        else:
            return self.VY.shape


    def _v_mask(self):
        return self._m_external()

    def _apply_mask(self, arr):
        """Return copy of arr with ext_mask applied (masked bins → NaN)."""
        if self.ext_mask is None or not np.any(self.ext_mask):
            return arr.copy()
        out = arr.copy()
        out[self.ext_mask.astype(bool)] = np.nan
        return out

    def v_east(self):
        """True ocean eastward velocity = VX + NAVX (removes the apparent
        opposite-signed current induced by vessel motion). vref selects
        the velocity reference: 1=NAV (default), else bottom-track BTX
        if available — bottom-track reference only applies to raw RDI
        ENX-origin data, not CODAS-origin data."""
        if self.vref == 1 or self.BTX is None:
            tout = self.NAVX
        else:
            tout = self.BTX
        return self._apply_mask(self.VX + tout[:, np.newaxis])

    def v_north(self):
        """True ocean northward velocity = VY + NAVY. See v_east()."""
        if self.vref == 1 or self.BTY is None:
            tout = self.NAVY
        else:
            tout = self.BTY
        return self._apply_mask(self.VY + tout[:, np.newaxis])




    def v_vertical(self):
        return self._apply_mask(self.VZ)

    def v_error(self):
        return self._apply_mask(self.VERR)

    def v_magnitude(self):
        return np.sqrt(self.v_east() ** 2 + self.v_north() ** 2)

    def yaxis(self, center: bool = False) -> np.ndarray:
        """Return bin depth axis (metres). center=True returns bin centres."""
        if not self.isValid():
            raise RuntimeError("YAxis(): data not loaded.")
        cent = self.mSpacing[1] * 0.5 if center else 0.0
        return (self.mOrigin[1] + self.mSpacing[1] *
                np.arange(self.nbins) + cent).astype(np.float64)

    def sound_speed(self):
        if not self.isValid():
            raise RuntimeError("Access denied : data not loaded.")
        return  self.SSPEED

    def save_as_codas_nc(self, ncpath: str,
                         platform: str = "",
                         cruise_id: str = None,
                         sonar: str = None) -> None:
        """
        Write this object to a CODAS "short form" netCDF file — the same
        end-user format produced by pycurrents' adcp_nc.py (COARDS / CF-1.8).

        Unlike store_netcdf()/save_as_nc() (which dump the full internal
        object layout), this produces the minimal, ready-to-use product:
        time, lon, lat, depth, u, v, amp, pg, pflag, heading, tr_temp,
        num_pings, uship, vship, plus one config block.

        Everything is pulled from self — no required parameters beyond the
        output path.

        Parameters
        ----------
        ncpath : str
            Output file path.
        platform : str, optional
            Ship/platform name. Not tracked on this object, so defaults to "".
        cruise_id : str, optional
            Overrides self.ID.
        sonar : str, optional
            Overrides self.mInstrumentID.
        """
        #import netCDF4 as nc4

        if not self.isValid():
            raise RuntimeError(
                "save_as_codas_nc(): no data — call read_codas() or load() first.")

        N, B = self.nprofs, self.nbins
        cruise_id = cruise_id or self.ID
        sonar = sonar or self.mInstrumentID
        yearbase = int(self.yearbase)

        # --- decimal day, yearbase-relative, from JDAY -------------------------
        jan1_jd = 2440587.5 + datetime(
            yearbase, 1, 1, tzinfo=timezone.utc).timestamp() / 86400.0
        dday = (self.JDAY - jan1_jd).astype(np.float64)

        # --- ocean velocities: mm/s -> m/s -------------------------------------
        # v_east()/v_north() add vessel velocity back (VX/VY are earth-relative
        # but contain an apparent opposite-signed current from ship motion;
        # true current = VX + NAVX, VY + NAVY) and apply ext_mask.
        u = self.v_east() / 1000.0
        v = self.v_north() / 1000.0

        # --- ship velocities: mm/s -> m/s ---------------------------------------
        uship = self.NAVX / 1000.0
        vship = self.NAVY / 1000.0

        # --- depth: broadcast the single bin-center axis to (time, depth_cell) -
        depth_axis = self.yaxis(center=True).astype(np.float32)  # (B,)
        depth_2d = np.tile(depth_axis, (N, 1))  # (N, B)

        # --- pg, amp, pflag, num_pings -------------------------------------------
        pg = np.clip(_fillcast(self.PGOOD4, np.float64, np.nan), 0, 100)
        amp = self.Ampl
        pflag = self.pflag if self.pflag is not None else np.zeros((N, B), dtype=np.int8)
        num_pings = self.nping if self.nping is not None else np.zeros(N, dtype=np.int16)

        base = datetime(yearbase, 1, 1)
        t_start = base + timedelta(days=float(dday.min()))
        t_end = base + timedelta(days=float(dday.max()))

        lon_valid = self.lon[np.isfinite(self.lon)]
        lat_valid = self.lat[np.isfinite(self.lat)]
        depth_valid = depth_2d[np.isfinite(depth_2d)]

        with nc4.Dataset(ncpath, "w", format="NETCDF4") as ds:
            ds.createDimension("time", N)
            ds.createDimension("depth_cell", B)
            ds.createDimension("num_configs", 1)

            ds.history = f"Created: {datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC"
            ds.Conventions = "COARDS CF-1.8"
            ds.software = "pycurrents"
            ds.changeset = ""
            ds.title = "Shipboard ADCP velocity profiles"
            ds.summary = (f"Shipboard ADCP velocity profiles from {cruise_id} "
                          f"using instrument {sonar} - Short Version.")
            ds.cruise_id = cruise_id
            ds.sonar = sonar
            ds.yearbase = np.int32(yearbase)
            ds.platform = platform
            ds.time_coverage_start = t_start.strftime("%Y-%m-%dT%H:%M:%SZ")
            ds.time_coverage_end = t_end.strftime("%Y-%m-%dT%H:%M:%SZ")
            if lat_valid.size:
                ds.geospatial_lat_min = np.float32(lat_valid.min())
                ds.geospatial_lat_max = np.float32(lat_valid.max())
            ds.geospatial_lat_units = "degrees_north"
            if lon_valid.size:
                ds.geospatial_lon_min = np.float32(lon_valid.min())
                ds.geospatial_lon_max = np.float32(lon_valid.max())
            ds.geospatial_lon_units = "degrees_east"
            if depth_valid.size:
                ds.geospatial_vertical_min = np.float32(depth_valid.min())
                ds.geospatial_vertical_max = np.float32(depth_valid.max())
            ds.geospatial_vertical_units = "m"
            ds.geospatial_vertical_positive = "down"
            ds.CODAS_variables = _CODAS_VARIABLES_TEXT
            ds.CODAS_processing_note = _CODAS_PROCESSING_NOTE_TEXT

            def write_var(name, data, dims, dtype, fill, long_name, units, fmt,
                          standard_name=None, extra=None):
                kwargs = {} if fill is None else dict(fill_value=fill)
                v = ds.createVariable(name, dtype, dims, **kwargs)
                v.long_name = long_name
                v.units = units
                v.C_format = fmt
                if standard_name:
                    v.standard_name = standard_name
                if extra:
                    for k, val in extra.items():
                        setattr(v, k, val)
                cast = data if fill is None else _fillcast(data, dtype, fill)
                valid = cast[cast != fill] if fill is not None else cast[np.isfinite(cast.astype(np.float64))]
                if valid.size:
                    v.data_min = dtype(valid.min())
                    v.data_max = dtype(valid.max())
                v[...] = cast
                return v

            write_var("time", dday, ("time",), np.float64, None,
                      "Decimal day", f"days since {yearbase}-01-01 00:00:00",
                      "%12.5f", standard_name="time",
                      extra={"calendar": "proleptic_gregorian", "axis": "T"})

            write_var("lon", self.lon, ("time",), np.float64, _FILL_F64,
                      "Longitude", "degrees_east", "%9.4f", standard_name="longitude")
            write_var("lat", self.lat, ("time",), np.float64, _FILL_F64,
                      "Latitude", "degrees_north", "%9.4f", standard_name="latitude")

            write_var("depth", depth_2d, ("time", "depth_cell"), np.float32, _FILL_F32,
                      "Depth", "meter", "%8.2f")

            man_bottom = self.MAN_BOTTOM if self.MAN_BOTTOM is not None \
                else np.full(N, np.nan, dtype=np.float32)
            write_var("detected_bottom", man_bottom, ("time",), np.float32, _FILL_F32,
                      "Edited bottom depth along survey track", "meter", "%8.2f")

            write_var("u", u, ("time", "depth_cell"), np.float32, _FILL_F32,
                      "Zonal velocity component", "meter second-1", "%7.2f")
            write_var("v", v, ("time", "depth_cell"), np.float32, _FILL_F32,
                      "Meridional velocity component", "meter second-1", "%7.2f")

            write_var("amp", amp, ("time", "depth_cell"), np.uint8, _FILL_U8,
                      "Received signal strength", "", "%d")
            write_var("pg", pg, ("time", "depth_cell"), np.int8, _FILL_I8,
                      "Percent good pings", "", "%d")

            # pflag: no fill value in CODAS short form
            v = ds.createVariable("pflag", "i1", ("time", "depth_cell"))
            v.long_name = "Editing flags"
            v.units = ""
            v.C_format = "%d"
            pflag_i8 = pflag.astype(np.int8)
            v.data_min, v.data_max = np.int8(pflag_i8.min()), np.int8(pflag_i8.max())
            v[...] = pflag_i8

            write_var("heading", self.HEADING, ("time",), np.float32, _FILL_F32,
                      "Ship heading", "degrees", "%6.1f")
            write_var("tr_temp", self.TEMP, ("time",), np.float32, _FILL_F32,
                      "ADCP transducer temperature", "degree_Celsius", "%4.1f")

            # num_pings: no fill value in CODAS short form
            v = ds.createVariable("num_pings", "i2", ("time",))
            v.long_name = "Number of pings averaged per ensemble"
            v.units = ""
            v.C_format = "%d"
            np_i2 = num_pings.astype(np.int16)
            v.data_min, v.data_max = np.int16(np_i2.min()), np.int16(np_i2.max())
            v[...] = np_i2

            write_var("uship", uship, ("time",), np.float32, _FILL_F32,
                      "Ship zonal velocity component", "meter second-1", "%9.4f")
            write_var("vship", vship, ("time",), np.float32, _FILL_F32,
                      "Ship meridional velocity component", "meter second-1", "%9.4f")

            # --- single config block, derived from techData / scalars ----------
            v = ds.createVariable("index_config_start", "i4", ("num_configs",))
            v.long_name = "First zero-based time index of each configuration"
            v.units = ""
            v[:] = np.array([0], dtype=np.int32)

            config_vars = [
                ("ensemble_seconds", "f4", self.avg_interval_s,
                 "Ensemble average duration", "s"),
                ("num_depth_bins", "i2", B, "Number of depth bins", ""),
                ("transducer_depth", "f4", self.tr_depth_m,
                 "Transducer depth", "m"),
                ("depth_bin_length", "f4", self.bin_length_m,
                 "Vertical averaging length", "m"),
                ("pulse_length", "f4", self.techData[2],
                 "Vertical span of sonar ping", "m"),
                ("blank_length", "f4", self.techData[4],
                 "Vertical delay between ping and first reception", "m"),
                ("ping_interval", "f4", self.avg_interval_s,
                 "Typical time between pings", "s"),
                ("transducer_orientation", "f4", self.techData[5],
                 "Approximate transducer orientation, clockwise rotation of beam 3 from forward",
                 "degrees"),
            ]
            for key, dtype, value, long_name, units in config_vars:
                v = ds.createVariable(key, dtype, ("num_configs",))
                v.long_name = long_name
                v.units = units
                v[:] = np.array([value], dtype=dtype)

        print(f"save_as_codas_nc(): wrote {N}x{B}  ->  {ncpath}  "
              f"[cruise_id={cruise_id}, sonar={sonar}]")