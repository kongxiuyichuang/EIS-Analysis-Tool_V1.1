# Generic data import for EIS and chronopotentiometry data
# ----------------------------------------------------------------------
# Reads EIS (impedance) and chrono (time, current, voltage) data from
# generic CSV, TXT, and Excel files. Column names are auto-detected
# from common aliases; explicit column names may be provided instead.

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union

from .datatypes import ZData, ChronoData

FilePath = Union[str, Path]

# Common column name aliases (compared case-insensitively after stripping
# whitespace and non-alphanumeric characters)
FREQ_ALIASES = ['freq', 'frequency', 'f']
ZRE_ALIASES = ['z_re', 'real', 're', 'zreal', 'z1']
ZIM_ALIASES = ['z_im', 'imag', 'im', 'zimag', 'z2', 'zimg']
ZMOD_ALIASES = ['z_mod', 'modulus', 'mod', 'zmod']
ZPHASE_ALIASES = ['z_phase', 'phase', 'phz', 'zphz']
Z_ALIASES = ['z', 'impedance']
TIME_ALIASES = ['time', 't', 'times']
I_ALIASES = ['i', 'i_signal', 'current', 'im', 'idc', 'i_sig']
V_ALIASES = ['v', 'v_signal', 'voltage', 'vf', 'vdc', 'v_sig']


def _norm(name):
    return ''.join(ch for ch in str(name).strip().lower() if ch.isalnum())


def _col_map(df):
    return {_norm(c): c for c in df.columns}


def _find_col(df, aliases, exclude=None):
    """Return the original column name matching one of the aliases, or None."""
    cmap = _col_map(df)
    if exclude is not None:
        excluded = {_norm(e) for e in exclude}
    else:
        excluded = set()
    for alias in aliases:
        key = _norm(alias)
        if key in cmap and key not in excluded:
            return cmap[key]
    return None


def _read_table(file: FilePath, **read_kw) -> pd.DataFrame:
    """Read a CSV/TXT/Excel file into a DataFrame with sensible defaults."""
    file = Path(file)
    if not file.exists():
        raise FileNotFoundError(f'File not found: {file}')

    ext = file.suffix.lower()
    if ext in ('.xlsx', '.xls'):
        return pd.read_excel(file, **read_kw)

    # Set defaults for text files
    defaults = dict(encoding='utf-8-sig', comment='#')
    defaults.update(read_kw)
    sep = defaults.pop('sep', None)

    if sep is not None:
        return pd.read_csv(file, sep=sep, **defaults)

    # Try comma-separated, then tab-separated
    for sep in [',', r'\t', ';', r'\s+']:
        try:
            kwargs = dict(defaults)
            if sep in (r'\t', r'\s+'):
                kwargs['engine'] = 'python'
            df = pd.read_csv(file, sep=sep, **kwargs)
            if df.shape[1] > 1:
                return df
        except (pd.errors.ParserError, UnicodeDecodeError):
            continue
    raise ValueError(f'Could not parse file {file}. Specify sep= manually.')


def _standardize_z(df, freq_col=None, real_col=None, imag_col=None,
                   mod_col=None, phase_col=None, z_col=None):
    """Rename columns to the standard names used by ZData.from_dataframe."""
    freq_col = freq_col or _find_col(df, FREQ_ALIASES)
    if freq_col is None:
        raise ValueError('Could not detect frequency column. Provide freq_col= explicitly.')
    df = df.rename(columns={freq_col: 'freq'})

    # Complex-valued column (e.g. "1.23+4.5j")
    z_col = z_col or _find_col(df, Z_ALIASES)
    real_col = real_col or _find_col(df, ZRE_ALIASES)
    imag_col = imag_col or _find_col(df, ZIM_ALIASES)
    mod_col = mod_col or _find_col(df, ZMOD_ALIASES)
    phase_col = phase_col or _find_col(df, ZPHASE_ALIASES)

    if real_col is not None and imag_col is not None:
        df = df.rename(columns={real_col: 'z_re', imag_col: 'z_im'})
    elif mod_col is not None and phase_col is not None:
        df = df.rename(columns={mod_col: 'z_mod', phase_col: 'z_phase'})
    elif z_col is not None:
        df['z_re'] = df[z_col].apply(lambda x: complex(x).real)
        df['z_im'] = df[z_col].apply(lambda x: complex(x).imag)
        df = df.drop(columns=[z_col])
    else:
        # Last resort: treat remaining numeric columns as [real, imag]
        remaining = [c for c in df.columns if c != 'freq']
        if len(remaining) >= 2:
            df = df.rename(columns={remaining[0]: 'z_re', remaining[1]: 'z_im'})
        else:
            raise ValueError('Could not detect impedance columns. Provide real_col= and imag_col= explicitly.')

    return df


def _standardize_chrono(df, time_col=None, i_col=None, v_col=None):
    time_col = time_col or _find_col(df, TIME_ALIASES)
    i_col = i_col or _find_col(df, I_ALIASES)
    v_col = v_col or _find_col(df, V_ALIASES)

    if time_col is None or i_col is None or v_col is None:
        raise ValueError('Could not detect time/current/voltage columns. '
                         'Provide time_col=, i_col=, and v_col= explicitly.')
    return df.rename(columns={time_col: 'time', i_col: 'i', v_col: 'v'})


def read_eis(file: FilePath, freq_col: Optional[str] = None,
             real_col: Optional[str] = None, imag_col: Optional[str] = None,
             mod_col: Optional[str] = None, phase_col: Optional[str] = None,
             z_col: Optional[str] = None,
             as_dataframe: bool = False, **read_kw):
    """Read EIS data from a generic CSV/TXT/Excel file.

    Auto-detects columns from common names (freq, real/imag or
    modulus/phase, or a single complex-valued 'z' column). Returns a
    :class:`ZData` object, or a DataFrame if ``as_dataframe=True``.

    Parameters
    ----------
    file : str or Path
        Path to the data file.
    freq_col, real_col, imag_col, mod_col, phase_col, z_col : str, optional
        Explicit column names. Override auto-detection when provided.
    as_dataframe : bool, optional
        If True, return the standardized DataFrame instead of a ZData.
    **read_kw
        Additional keyword arguments passed to pandas.read_csv/read_excel
        (e.g. ``sep``, ``skiprows``, ``encoding``, ``header``).
    """
    df = _read_table(file, **read_kw)
    df = _standardize_z(df, freq_col=freq_col, real_col=real_col,
                        imag_col=imag_col, mod_col=mod_col,
                        phase_col=phase_col, z_col=z_col)
    if as_dataframe:
        return df
    return ZData.from_dataframe(df)


def read_chrono(file: FilePath, time_col: Optional[str] = None,
                i_col: Optional[str] = None, v_col: Optional[str] = None,
                as_dataframe: bool = False, **read_kw):
    """Read chronopotentiometry data from a generic CSV/TXT/Excel file.

    Auto-detects time, current, and voltage columns. Returns a
    :class:`ChronoData` object, or a DataFrame if ``as_dataframe=True``.

    Parameters
    ----------
    file : str or Path
        Path to the data file.
    time_col, i_col, v_col : str, optional
        Explicit column names. Override auto-detection when provided.
    as_dataframe : bool, optional
        If True, return the standardized DataFrame instead of a ChronoData.
    **read_kw
        Additional keyword arguments passed to pandas.read_csv/read_excel.
    """
    df = _read_table(file, **read_kw)
    df = _standardize_chrono(df, time_col=time_col, i_col=i_col, v_col=v_col)
    if as_dataframe:
        return df
    return ChronoData.from_dataframe(df)
