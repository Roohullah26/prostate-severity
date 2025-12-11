import json
from src.utils_dicom import build_uid_map
from src.config import DICOM_ROOT
uid_map = build_uid_map(str(DICOM_ROOT))
with open(""uid_map.json"",""w"") as f:
    json.dump(uid_map, f)
print(""saved uid_map.json:"", len(uid_map))
