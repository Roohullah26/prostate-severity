import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

from src import config
from src.utils_dicom import build_uid_map, parse_pos


class ProstateLesionDataset(Dataset):
    """Dataset for lesion-level classification.

    By default this reads `merged_data.csv` (auto-generated from the existing
    scripts) and tries to locate DICOM series via the uid-map helpers.

    For quick local testing there's a `toy=True` mode that generates
    random images and balanced labels so CI / smoke checks can run quickly.
    """

    def __init__(self, csv_path=None, img_size=config.IMG_SIZE, toy=False, toy_len=64, sequences=None, num_slices=1, transform=None):
        self.img_size = img_size
        self.toy = toy
        # set sequence list early so toy-mode has the attribute
        self.sequences = None if sequences is None else [s.strip().lower() for s in sequences]
        # number of axial slices to stack per lesion (must be odd). If >1, we'll stack slices around lesion z.
        self.num_slices = int(num_slices) if num_slices and int(num_slices) >= 1 else 1
        if self.num_slices % 2 == 0:
            raise ValueError("num_slices must be odd (e.g. 1,3,5)")

        # optional torchvision transforms (callable that accepts PIL image or torch tensor)
        self.transform = transform

        if toy:
            self.df = pd.DataFrame({"image_id": list(range(toy_len)), "label": [random.randint(0, 1) for _ in range(toy_len)]})
            return

        if csv_path is None:
            csv_path = Path(config.ROOT) / "merged_data.csv"

        if not Path(csv_path).exists():
            raise FileNotFoundError(f"merged csv not found: {csv_path}. Run scripts/00_merge_csvs.py first or pass csv_path")

        self.df = pd.read_csv(csv_path)

        # Build a mapping SeriesUID -> directory of DICOM files
        # build_uid_map expects the DICOM root; config.DICOM_ROOT should point to the PROSTATEx/ folder
        self.uid_map = build_uid_map(config.DICOM_ROOT)

        # build per-series information (cached): subject / series description
        self.series_info = {}  # series_uid -> {path, subject, description}
        for uid, path in self.uid_map.items():
            # try to extract subject id and series description from the first DICOM
            try:
                from glob import glob
                import pydicom

                dcm_files = sorted(glob(os.path.join(path, "*.dcm")))
                if not dcm_files:
                    continue
                ds = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
                desc = str(getattr(ds, "SeriesDescription", "")).lower()
                # patient/subject id may be stored in PatientID or inferable from path
                subject = str(getattr(ds, "PatientID", ""))
                if not subject:
                    # extract folder name like ProstateX-0001 from path
                    parts = Path(path).parts
                    subject = next((p for p in parts if p.lower().startswith("prostatex")), "")

                self.series_info[uid] = {"path": path, "description": desc, "subject": subject}
            except Exception:
                # best-effort, ignore problematic series
                continue

        # we'll keep only rows that have a Series UID available in the uid_map
        if "Series UID" in self.df.columns:
            self.df = self.df[self.df["Series UID"].isin(list(self.uid_map.keys()))].reset_index(drop=True)

        # target column: ClinSig (boolean) -> convert to int
        if "ClinSig" in self.df.columns:
            self.df["label"] = self.df["ClinSig"].astype(int)
        elif "label" not in self.df.columns:
            raise ValueError("No label column found (need 'ClinSig' or 'label') in merged csv")

    def __len__(self):
        return len(self.df)

    @property
    def num_channels(self):
        # number of channels returned by __getitem__:
        # - legacy single-sequence single-slice -> 3 (RGB)
        # - single-sequence multi-slice -> num_slices (S channels)
        # - multi-sequence -> len(sequences) * num_slices
        if self.sequences is None or len(self.sequences) <= 1:
            return 3 if self.num_slices == 1 else self.num_slices
        return len(self.sequences) * max(1, self.num_slices)

    def _get_series_slices(self, series_path):
        """Return list of tuples (z, filepath) sorted by increasing z for a series folder."""
        from glob import glob
        import pydicom

        dcm_files = sorted(glob(os.path.join(series_path, "*.dcm")))
        items = []
        for f in dcm_files:
            try:
                ds = pydicom.dcmread(f, stop_before_pixels=True)
                ipp = getattr(ds, "ImagePositionPatient", None)
                # ipp may be a list-like [x, y, z]
                if ipp is not None and len(ipp) >= 3:
                    z = float(ipp[2])
                else:
                    # fallback to SliceLocation
                    z = float(getattr(ds, "SliceLocation", 0.0))
                items.append((z, f))
            except Exception:
                continue

        # sort by z and return
        items.sort(key=lambda x: x[0])
        return items


    def _load_series_closest_slice(self, series_path, lesion_pos=None):
        """Load the slice whose z is closest to lesion_pos[2]. If lesion_pos is None, return middle slice."""
        items = self._get_series_slices(series_path)
        if not items:
            raise FileNotFoundError(f"no dcm files under {series_path}")

        if lesion_pos is None:
            mid_index = len(items) // 2
        else:
            try:
                target_z = float(lesion_pos[2])
            except Exception:
                mid_index = len(items) // 2
            else:
                zs = [z for z, _ in items]
                mid_index = int(np.argmin(np.abs(np.array(zs) - target_z)))

        # pick range of indices for multi-slice
        half = (self.num_slices - 1) // 2

        selected_files = []
        for i in range(mid_index - half, mid_index + half + 1):
            # clamp index
            idx = max(0, min(i, len(items) - 1))
            selected_files.append(items[idx][1])

        # read pixel arrays for all selected files and return as list
        import pydicom

        arrays = []
        for fp in selected_files:
            ds = pydicom.dcmread(fp)
            arr = ds.pixel_array
            arr = arr.astype(np.float32)
            arr -= arr.min()
            if arr.max() > 0:
                arr /= arr.max()
            arr = (arr * 255.0).astype(np.uint8)
            arrays.append(arr)
        # if only one slice, return that array directly
        if len(arrays) == 1:
            return arrays[0]
        return arrays

    def _normalize_channel(self, arr):
        # arr expected uint8; convert to float32 0..1
        a = arr.astype(np.float32) / 255.0
        return a
        

    def _prepare_image(self, arr):
        # convert grayscale to RGB and resize (legacy single-sequence path)
        img = Image.fromarray(arr)
        img = img.resize(self.img_size)
        img = img.convert("RGB")
        return img

    def __getitem__(self, idx):
        if self.toy:
            row = self.df.iloc[idx]
            seq_count = 1 if not self.sequences else len(self.sequences)
            channels = seq_count * max(1, self.num_slices)
            # if channels == 3 and single slice, return a PIL image for legacy path
            if channels == 3 and self.num_slices == 1:
                img = Image.fromarray((np.random.rand(*self.img_size, 3) * 255).astype(np.uint8))
                label = int(row["label"])
                # apply transform if available
                if self.transform:
                    try:
                        img = self.transform(img)
                    except Exception:
                        pass
                return img, label

            # otherwise produce a random tensor CxHxW
            tarr = (np.random.rand(channels, self.img_size[0], self.img_size[1]).astype(np.float32))
            t = torch.tensor(tarr, dtype=torch.float32)
            # if transform exists (e.g. normalization), apply
            if self.transform:
                try:
                    t = self.transform(t)
                except Exception:
                    pass
            label = int(row["label"])
            return t, label

        row = self.df.iloc[idx]

        # try to parse lesion position if present
        lesion_pos = None
        if "pos" in row and not pd.isna(row["pos"]):
            lesion_pos = parse_pos(row["pos"])

        series_uid = str(row.get("Series UID", ""))
        series_path = self.uid_map.get(series_uid)
        if series_path is None:
            # fallback: try Series UID in other columns
            series_path = self.uid_map.get(series_uid)
            if series_path is None:
                raise KeyError(f"series uid {series_uid} not found in DICOM root")

        # if sequences requested -> build stacked multi-channel Tensor
        if self.sequences and len(self.sequences) > 1:
            # attempt to find sequence-specific series for the same subject
            subj = str(row.get("Subject ID", ""))
            channels = []
            for seq in self.sequences:
                chosen_path = None
                # search series_info for same subject with matching keyword
                for uid, info in self.series_info.items():
                    # check subject match
                    if subj and info.get("subject", "") and subj.lower() not in info.get("subject", "").lower():
                        continue
                    if seq in info.get("description", ""):
                        chosen_path = info.get("path")
                        break

                # fallback to peptide: use the requested row's series when available
                if chosen_path is None:
                    chosen_path = series_path

                if chosen_path is None:
                    raise KeyError(f"no series found for sequence {seq}")

                arr = self._load_series_closest_slice(chosen_path, lesion_pos=lesion_pos)
                # arr may be a single array or list of arrays depending on num_slices
                if isinstance(arr, list):
                    ch_slices = [self._normalize_channel(a) for a in arr]
                    # resize each slice then stack them into a SxHxW block for this sequence
                    from PIL import Image as _PILImg
                    resized = []
                    for sarr in ch_slices:
                        p = _PILImg.fromarray((sarr * 255).astype(np.uint8)).resize(self.img_size)
                        s = np.array(p).astype(np.float32) / 255.0
                        if s.ndim == 3:
                            s = s[..., 0]
                        resized.append(s)
                    # resized is list of HxW -> stack into num_slices x H x W
                    seq_stack = np.stack(resized, axis=0)
                    ch = seq_stack
                else:
                    # single slice HxW
                    ch = self._normalize_channel(arr)
                # resize to img_size for single-slice data; for multi-slice sequences
                # we've already resized each slice above so `ch` will be SxHxW and
                # shouldn't be directly passed to PIL.fromarray (PIL expects HxW or HxWxC).
                from PIL import Image as _PILImg

                if isinstance(ch, np.ndarray) and ch.ndim == 2:
                    p = _PILImg.fromarray((ch * 255).astype(np.uint8)).resize(self.img_size)
                    ch = np.array(p).astype(np.float32) / 255.0
                elif isinstance(ch, np.ndarray) and ch.ndim == 3:
                    # ch is already SxHxW where each slice was individually resized
                    # ensure values are float32 in 0..1 (they already should be)
                    ch = ch.astype(np.float32)
                else:
                    # fallback to best-effort conversion
                    p = _PILImg.fromarray((ch * 255).astype(np.uint8)).resize(self.img_size)
                    ch = np.array(p).astype(np.float32) / 255.0
                # ensure single channel (if resizing produced 3 channels convert to grayscale)
                # ch may be HxW (single slice) or (num_slices,H,W) for multi-slice
                channels.append(ch)

            # stack -> shape (C,H,W)
            # channels is a list of either arrays HxW (single) or arrays (S,H,W) for each sequence
            # convert into unified shape C x H x W where C = sum(seq_slices)
            channel_blocks = []
            for c in channels:
                if c.ndim == 2:
                    channel_blocks.append(c)
                elif c.ndim == 3:
                    # c is (S,H,W) -> split into S channels
                    for s in range(c.shape[0]):
                        channel_blocks.append(c[s])
            stacked = np.stack(channel_blocks, axis=0).astype(np.float32)
            # convert to torch.Tensor
            # apply transform (if provided) expects PIL or torch tensor; we currently have a numpy CxHxW
            t = torch.tensor(stacked, dtype=torch.float32)
            if self.transform:
                # torchvision transforms generally operate on HxW or PIL Image; we will call transform per-channel
                try:
                    # convert back to HxWxC for PIL convenience
                    import torchvision.transforms.functional as TF
                    # stack to HxWxC
                    h, w = t.shape[1], t.shape[2]
                    img_c = torch.permute(t, (1, 2, 0)).numpy().astype('uint8')
                    from PIL import Image as _PILImg
                    pil = _PILImg.fromarray(img_c)
                    out = self.transform(pil)
                    # ensure out is tensor shaped CxHxW
                    if isinstance(out, torch.Tensor):
                        t = out
                except Exception:
                    # ignore transform errors and keep raw tensor
                    pass
            return t, int(row["label"])

        # single-sequence path (legacy)
        arr = self._load_series_closest_slice(series_path, lesion_pos=lesion_pos)
        img = self._prepare_image(arr)
        # apply transform if provided - transform should return a tensor
        if self.transform:
            try:
                img = self.transform(img)
            except Exception:
                # transform failed; keep original PIL
                pass
        label = int(row["label"])
        return img, label
