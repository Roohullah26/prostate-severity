import os
from glob import glob
try:
    import pydicom
except Exception:  # don't fail import-time if pydicom isn't installed; raise helpful error during use
    pydicom = None
import numpy as np

def build_uid_map(dicom_root):
    if pydicom is None:
        raise ModuleNotFoundError("pydicom is required for build_uid_map; please install with 'pip install pydicom'")
    uid_map = {}
    # the PROSTATEx DICOM dataset organizes files into nested folders; match up to three levels
    # try to match series directories at multiple depths (some datasets are nested)
    for series in glob(os.path.join(dicom_root, "*", "*", "*")) + glob(os.path.join(dicom_root, "*", "*")):
        dcm_list = glob(os.path.join(series, "*.dcm"))
        if not dcm_list: continue
        try:
            ds = pydicom.dcmread(dcm_list[0], stop_before_pixels=True)
            uid_map[str(ds.SeriesInstanceUID)] = os.path.dirname(dcm_list[0])
        except Exception:
            continue
    return uid_map

def parse_pos(pos):
    if pos is None: return None
    if isinstance(pos, str):
        parts = pos.strip().split()
        if len(parts) != 3: return None
        try:
            return [float(x) for x in parts]
        except:
            return None
    if isinstance(pos, (list, tuple, np.ndarray)) and len(pos) == 3:
        return [float(x) for x in pos]
    return None
