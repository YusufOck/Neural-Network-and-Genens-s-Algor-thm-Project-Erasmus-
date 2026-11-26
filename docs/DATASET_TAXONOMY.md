# Dataset Taxonomy and Supported Interpretation

## Why two evaluation levels are used

The dataset has eight top-level folders but 26 detailed leaf folders. The eight parents are not
uniform diagnoses: they are source/organ groupings, and some contain normal or benign images.
Training directly on the 26 leaves preserves the available detail. After inference, each predicted
leaf maps to exactly one parent, allowing both leaf and parent metrics to be computed from one
model.

## Complete mapping

| Parent | Leaf | Supported meaning | Modality |
|---|---|---|---|
| ALL | `all_benign` | Benign hematogone reference | Blood-smear microscopy |
| ALL | `all_early` | Early Pre-B ALL subtype, not a progression stage | Blood-smear microscopy |
| ALL | `all_pre` | Pre-B ALL subtype, not a progression stage | Blood-smear microscopy |
| ALL | `all_pro` | Pro-B ALL subtype, not a progression stage | Blood-smear microscopy |
| Brain Cancer | `brain_glioma` | Glioma among represented tumor types | Contrast-enhanced MRI |
| Brain Cancer | `brain_menin` | Meningioma among represented tumor types | Contrast-enhanced MRI |
| Brain Cancer | `brain_tumor` | Pituitary tumor among represented tumor types | Contrast-enhanced MRI |
| Breast Cancer | `breast_benign` | Benign breast lesion | H&E histopathology |
| Breast Cancer | `breast_malignant` | Malignant breast lesion | H&E histopathology |
| Cervical Cancer | `cervix_dyk` | Dyskeratotic cell; abnormal morphology, not a stage | Pap-smear microscopy |
| Cervical Cancer | `cervix_koc` | Koilocytotic cell; abnormal morphology, not a stage | Pap-smear microscopy |
| Cervical Cancer | `cervix_mep` | Metaplastic cell; benign morphology, not a stage | Pap-smear microscopy |
| Cervical Cancer | `cervix_pab` | Parabasal cell; normal morphology, not a stage | Pap-smear microscopy |
| Cervical Cancer | `cervix_sfi` | Superficial-intermediate cell; normal morphology | Pap-smear microscopy |
| Kidney Cancer | `kidney_normal` | Normal kidney class | Abdominal CT |
| Kidney Cancer | `kidney_tumor` | Kidney tumor; malignancy/stage not supplied | Abdominal CT |
| Lung and Colon Cancer | `colon_aca` | Colon adenocarcinoma | H&E histopathology |
| Lung and Colon Cancer | `colon_bnt` | Benign colon tissue | H&E histopathology |
| Lung and Colon Cancer | `lung_aca` | Lung adenocarcinoma | H&E histopathology |
| Lung and Colon Cancer | `lung_bnt` | Benign lung tissue | H&E histopathology |
| Lung and Colon Cancer | `lung_scc` | Lung squamous cell carcinoma | H&E histopathology |
| Lymphoma | `lymph_cll` | CLL/SLL class among represented malignant classes | H&E histopathology |
| Lymphoma | `lymph_fl` | Follicular lymphoma | H&E histopathology |
| Lymphoma | `lymph_mcl` | Mantle cell lymphoma | H&E histopathology |
| Oral Cancer | `oral_normal` | Normal oral epithelium | H&E histopathology |
| Oral Cancer | `oral_scc` | Oral squamous cell carcinoma | H&E histopathology |

The machine-readable authority is [`config/label_taxonomy.json`](../config/label_taxonomy.json).

## Claims that the dataset supports

- A 26-leaf dataset classification result.
- A deterministic eight-parent dataset-group result derived from the leaf prediction.
- Normal/benign/malignant or tumor meaning only for leaves whose source labels explicitly provide
  that meaning.

## Claims that the dataset does not support

- A universal healthy-versus-cancer target across all 26 leaves.
- Cancer-stage prediction.
- Reliable “no tumor” output for parent groups that contain no normal control.
- Patient-level or clinical diagnostic performance.

## Why parent accuracy is unusually high

Many remaining leaf errors occur between leaves inside the same parent. For example,
`oral_normal → oral_scc` is wrong at the 26-leaf level but still maps to the correct `Oral Cancer`
parent. The best model made 69 leaf errors, while only two crossed a parent boundary. Parent
accuracy must therefore be reported beside—not instead of—leaf accuracy.
