# Poly Haven Source Audit

**Audited:** 2026-07-30

NeuSky Synthetic v1 used 167 Poly Haven HDRIs as 16K EXR files. The accepted
local files were compared with the current 16K EXR entries returned by the
official Poly Haven API.

## Result

- 131 files have the same MD5 and byte size as the current upstream file.
- 36 files have a different MD5 and byte size.
- Checksums for all accepted files are in `hdris_16k_generation_md5.txt`.

The downloader is resolving the correct asset identifiers, resolution and EXR
format. The disagreement is therefore not caused by selecting 4K files, adding
`_16k` to filenames, or downloading a preview.

## Pixel-Level Check

`autumn_park.exr` was downloaded again and compared with the accepted local
file using OpenImageIO:

| Property | Accepted file | Current upstream file |
|---|---|---|
| Resolution | 16384 x 8192 | 16384 x 8192 |
| Channels | RGB, 32-bit float | RGB, 32-bit float |
| Compression | PIZ | PIZ |
| Capture-date metadata | 2021-05-08 | 2024-03-14 |

The image comparison gave:

- mean absolute error: `7.33328e-06`;
- RMS error: `0.00176769`;
- maximum absolute error: `1.2109375`;
- 7,903 changed pixels out of approximately 134 million (`0.00589%`).

This establishes that at least this file changed in image content, not merely
in metadata or EXR encoding.

## Interpretation

Poly Haven appears to have retouched or reprocessed some HDRIs in place while
retaining their asset identifiers. This is an inference from the changed
pixels, file sizes and metadata; the API does not expose an immutable version
history that identifies the exact upstream operation.

These sparse upstream revisions are accepted for source re-rendering and no
legacy HDRI bundle is required for the release. The prepared NeuSky dataset
remains the canonical record for reproducing the reported experiments and
figures.
