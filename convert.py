from pathlib import Path

import numpy as np
import nibabel as nib
import pydicom
from PIL import Image

# ==========================================================
# Configuration
# ==========================================================

INPUT_FOLDER = Path(
    r"D:\Programming\PerX2CT\sample_data\GHOWHER SULTANA\ST0\SE1"
)

OUTPUT_FILE = Path("chest_ct.nii.gz")
PREVIEW_FOLDER = Path("preview")

# ==========================================================
# Read DICOM files
# ==========================================================

files = sorted(INPUT_FOLDER.glob("IM*"))

if len(files) == 0:
    raise FileNotFoundError("No IM files found.")

print(f"Found {len(files)} files")

datasets = []

for file in files:
    ds = pydicom.dcmread(file)

    z = None
    if hasattr(ds, "ImagePositionPatient"):
        z = float(ds.ImagePositionPatient[2])

    instance = int(getattr(ds, "InstanceNumber", 0))

    datasets.append((ds, z, instance))

# ==========================================================
# Sort slices
# ==========================================================

if all(z is not None for _, z, _ in datasets):
    datasets.sort(key=lambda x: x[1])
    print("Sorting using ImagePositionPatient")
else:
    datasets.sort(key=lambda x: x[2])
    print("Sorting using InstanceNumber")

# ==========================================================
# Build HU volume
# ==========================================================

volume_slices = []

for ds, _, _ in datasets:

    pixels = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))

    hu = pixels * slope + intercept

    volume_slices.append(hu.astype(np.int16))

volume = np.stack(volume_slices, axis=-1)

print("\n========== VOLUME ==========")
print("Shape:", volume.shape)
print("Datatype:", volume.dtype)
print("HU Range:", volume.min(), "to", volume.max())

# ==========================================================
# Metadata
# ==========================================================

first = datasets[0][0]

spacing_x = float(first.PixelSpacing[0])
spacing_y = float(first.PixelSpacing[1])
spacing_z = float(first.SliceThickness)

origin = np.array(first.ImagePositionPatient, dtype=np.float32)

print("\n========== METADATA ==========")
print("Patient:", getattr(first, "PatientName", "Unknown"))
print("Study:", getattr(first, "StudyDescription", "Unknown"))
print("Series:", getattr(first, "SeriesDescription", "Unknown"))
print("Slices:", volume.shape[2])
print("Voxel spacing:", spacing_x, spacing_y, spacing_z)
print("Origin:", origin)

# ==========================================================
# Create affine
# ==========================================================

affine = np.array(
    [
        [spacing_x, 0, 0, origin[0]],
        [0, spacing_y, 0, origin[1]],
        [0, 0, spacing_z, origin[2]],
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)

nii = nib.Nifti1Image(volume, affine)

nii.header.set_zooms((spacing_x, spacing_y, spacing_z))

nib.save(nii, OUTPUT_FILE)

print(f"\nSaved NIfTI:")
print(OUTPUT_FILE.resolve())

# ==========================================================
# Preview images
# ==========================================================

PREVIEW_FOLDER.mkdir(exist_ok=True)

indices = {
    "first": 0,
    "middle": volume.shape[2] // 2,
    "last": volume.shape[2] - 1,
}

for name, idx in indices.items():

    img = volume[:, :, idx].astype(np.float32)

    # Lung window
    level = -600
    width = 1500

    low = level - width / 2
    high = level + width / 2

    img = np.clip(img, low, high)
    img = (img - low) / (high - low)
    img = (img * 255).astype(np.uint8)

    Image.fromarray(img).save(PREVIEW_FOLDER / f"{name}.png")

print("Preview images saved.")

# ==========================================================
# Final summary
# ==========================================================

print("\n========== SUMMARY ==========")

print("Volume shape :", volume.shape)
print("Voxel spacing:", nii.header.get_zooms())
print("HU minimum   :", volume.min())
print("HU maximum   :", volume.max())
print("Affine:\n")
print(nii.affine)

print("\nDone.")