# VarDepth

VarDepth combines read likelihoods and reference-panel haplotype context for autosomal diploid SNV and short-indel inference. This research snapshot contains the frozen **0.41.0-rc1 caller**, the **0.44.0-dev1 coverage-adaptive filter**, model weights, and a reproducible aggregate benchmark of public HG005 data.

The reference context is probabilistic. The implementation does not replace the FASTA with an imputed genome or remap all reads to a personalised reference.

## Contents

- `caller/`: original locked Linux x86_64 / CPython 3.12 caller, own source, build scripts, configuration and weights. The prebuilt modules are platform-specific. `RELEASE_LOCK.json` preserves original byte identities; statements inside historical freeze records refer to that release event.
- `filter/`: coverage-aware filtering source, native scorer and frozen `model44.json`.
- `benchmark/`: aggregate block counts and local replay. No individual callsets, BAM/CRAM files, reference panel genotypes or raw training matrices are included.
- `docs/`: methods and reproducibility notes.

## Evidence

The matched chromosome 18/20 analysis uses HG005, previously exposed during development. Low-coverage WGS SNV F1 is 97.0151%, versus 96.4418% for QUILT2 and 96.6754% for GLIMPSE2. Low-coverage indel F1 is 92.6084%, versus 91.8321% for GLIMPSE2. In WGS, VarDepth is above the tested GATK pipeline and below DeepVariant. The expanded target analysis also remains below DeepVariant. These results do not establish independent-person generalisation, clinical utility, universal superiority, or phase accuracy.

Primary comparisons concern chr18 and chr20. The separate target-only summary uses 22 autosomes for low-coverage WGS/WES/hybrid, and 12 matched completed WGS autosomes. Missing QUILT2 WGS and indel results remain missing. Complete counts, pointwise regional intervals and all measured filter times are in `benchmark/outputs/paper-check-20260929/`.

## Replay the aggregate analysis

Use Python 3.12 and NumPy 2.3.5. From a fresh checkout:

```sh
python -m venv .venv-analysis
. .venv-analysis/bin/activate
python -m pip install -r benchmark/requirements.txt
cd benchmark
python analyze.py
```

The script validates 534 count tensors against 41,118 metadata rows and replays 10,000 paired block bootstrap repetitions, including 1/2/4-Mb sensitivities. It verifies 153 earlier aggregate rows. No cloud access or genomic files are required. The replay regenerates `ANALYSIS.json` and `SOURCE_PARITY.json` (the timestamp changes).

## Run inference

Consult `caller/README.md` and the recorded dependency files. The tested caller environment is Linux x86_64 with CPython 3.12; native and post-processing dependencies are recorded separately. Binary redistribution preserves the tested artifact but is not a portable wheel. A rebuild on another platform needs separate technical verification. Source syntax, model identity and aggregate replay have been checked; a fresh end-to-end installation was not rerun for this publication package.

Required external inputs are indexed aligned BAM/CRAM, matching reference FASTA, a phased reference panel excluding the target and close relatives, and a compatible genetic map. There is no FASTQ alignment stage. Original panel assets are not distributed with this repository.

```sh
/path/to/post-python caller/vardepth41.py \
  --native-python /path/to/native-python \
  --bam sample.bam --reference GRCh38.fa \
  --panel unrelated.chr20.bcf --genetic-map chr20.gmap.gz \
  --sample SAMPLE --chrom chr20 --threads 8 --phase-workers 4 \
  --out result-chr20
```

To follow the benchmark's sensitivity-then-v44 sequence, using a Python environment with NumPy and pysam:

```sh
python filter/select_calls42.py --source result-chr20/calls.vcf.gz \
  --output sensitivity.chr20.vcf.gz --profile sensitivity
python filter/apply44.py --source sensitivity.chr20.vcf.gz \
  --output filtered.chr20.vcf.gz --model filter/model44.json
```

The output includes both passing and filtered calls. V44P is a development-calibrated binary correctness score, not a GP vector or clinical confidence measure. Filtering preserves genotype evidence and phase; the receipt includes invariant hashes. Filter input paths must be new output files.

Optional native tree scoring:

```sh
c++ -O3 -std=c++17 -shared -fPIC filter/native44.cpp -o filter/libnative44.so
python filter/native44.py --source sensitivity.chr20.vcf.gz \
  --output native-filtered.chr20.vcf.gz --model filter/model44.json \
  --library filter/libnative44.so
```

Complete-filter runtime decreased by 4.0–5.1% in the recorded single-core comparison. This does not measure total calling speed or speed relative to other callers.

## Model and rights

The v44 model SHA-256 is `cd5d44ebdc52ac2aee97bb08113529267dbbb40a3ecc2d6c413873a083c4602b`. Original model provenance and the development sample are retained. No training or threshold changes were made for this package.

Code and weights are publicly inspectable at the author's request. An open-source licence has not yet been selected; no additional reuse licence is granted by this snapshot. Third-party dependencies retain their own licences. The manuscript and study remain research materials, not a clinically validated product.

Contact: Valery Ilinsky, Eligens SIA, valery@eligens.io. Valery Ilinsky is a shareholder of Eligens SIA.
