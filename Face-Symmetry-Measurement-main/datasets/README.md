# Datasets

No images are stored in this repository. All benchmark images remain with their providers.

## LFPW (subset of 300-W), used for every experiment

* Official 300-W page (download + terms, research use only): https://ibug.doc.ic.ac.uk/resources/300-W/
* 68-point markup description: https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
* Mirror of the LFPW images with 300-W annotations used here (Menpo project):
  https://github.com/menpo/lfpw-train (811 images) and https://github.com/menpo/lfpw-test (224 images)

Place the data as:

```
datasets/lfpw/trainset/image_XXXX.png + .pts      # used ONLY for mapping fit + gate calibration
datasets/lfpw/testset/image_XXXX.png  + .pts      # used ONLY for evaluation
```

For example:

```bash
git clone --depth 1 https://github.com/menpo/lfpw-train /tmp/lfpw-train
git clone --depth 1 https://github.com/menpo/lfpw-test  /tmp/lfpw-test
mkdir -p datasets/lfpw
ln -s /tmp/lfpw-train/lfpw-train datasets/lfpw/trainset
ln -s /tmp/lfpw-test/lfpw-test   datasets/lfpw/testset
```

Citations: Belhumeur et al., "Localizing parts of faces using a consensus of exemplars", CVPR 2011 / TPAMI 2013 (LFPW);
Sagonas et al., "300 Faces in-the-Wild Challenge", ICCV-W 2013 and IMAVIS 2016 (300-W annotations).

## Considered but not used

* WFLW (https://github.com/wywu/LAB): its images are distributed via Google Drive / Baidu, which were
  unreachable from our compute environment. The code accepts any 68-point `.pts` dataset, so 300-W
  (Helen/AFW/iBUG) can be added the same way.
* CelebA (https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html): only 5 landmarks, too sparse for regional symmetry.

## Privacy

Paper and slide figures draw landmark geometry on a blank canvas; dataset photographs are not reproduced.
Demo images must be your own or used with the person's consent.
