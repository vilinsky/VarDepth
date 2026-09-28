# VarDepth 0.41.0-rc1

Frozen research release for joint short-variant calling and reference-panel
imputation from aligned Illumina reads, followed by genotype-preserving phasing.
The tested entry point processes one diploid autosome and one sample at a time.
Run chromosomes independently and retain their separate phase sets.

## Inputs and invocation

Required: indexed, coordinate-sorted BAM/CRAM; matching indexed FASTA; an indexed
phased reference-panel BCF/VCF on that reference; matching genetic map. Read groups
must identify the requested sample. Exclude the target and its relatives from the
reference panel before running. This software does not map FASTQ reads.

```sh
/path/to/post-python vardepth41.py \
  --native-python /path/to/native-python \
  --bam sample.bam --reference GRCh38.fa \
  --panel unrelated.chr20.bcf --genetic-map chr20.gmap.gz \
  --sample SAMPLE --chrom chr20 --threads 8 --phase-workers 4 \
  --out result-chr20
```

`calls.vcf.gz` contains both read-supported and imputed records. `COMPLETE.json`
records the release digest and per-stage resources. `resources/*/RESOURCES.json`
contains CPU seconds, elapsed seconds and sampled process-tree RSS. RSS sums can
count shared pages more than once. Chromosome invocation directories are
create-only; preserve failures and use a new directory for a retry.

Use `DEPENDENCIES.json` and the two requirements files to recreate the frozen
native and post-processing environments. This artifact includes the tested
Linux x86_64 / CPython 3.12 native modules as well as their own source and build
scripts. A C++ compiler, zlib/htslib dependencies and the recorded CPU PyTorch
build are needed. Runtime modules and prepared kernel caches are covered by the
release digest. A rebuild on another platform is a different binary artifact
and requires its own technical verification. Preflight setup is recorded
separately from the four-assay benchmark.

## Frozen inference

- Native r14-numerics03 read likelihood, reference-panel and neural inference;
  4 Mb scientific windows with 200 kb padding.
- Frozen v21/v25/v32 genotype correction and v33 binary confidence estimation.
  No DeepVariant output, truth VCF or retraining is needed at inference.
- v36 coverage guard: correction is used only in fixed 4 Mb windows with at least
  100 eligible depth observations and median depth between 15.5 and 62.
  Other windows retain the native records. An entirely out-of-domain chromosome
  skips unused correction computation while preserving the same decisions.
- v39 coverage-mode phasing joins panel SNP scaffolds and read evidence, with
  a 0.01 model path-error budget, at least three supporting molecules and GQ20.
  Model error bounds are not empirical clinical error guarantees. Independent
  4 Mb windows are not joined across their boundaries. Phase reports neither
  parental origin nor a chromosome-wide phase set.
- The extended-attachment experimental phase output is retained for audit, but
  the selected release output is the frozen v39 baseline forest.

Recalibration exposes binary genotype-correctness and variant-presence scores.
These are not a joint genotype distribution. Stale GP/DS are cleared when
correction makes them incoherent; missing dosage must not be reconstructed from
these binary scores. Phase changes only phase fields and allele order, checked
by a full genotype/evidence invariant digest.

## Evidence and scope

`TECHNICAL_VALIDATION.json` records a whole-chromosome packaging regression and
an unscored BAM-to-VCF/phase smoke test. The separately stored four-assay study
uses public HG005 WGS, downsampled low-pass WGS, real IDT WES and their matched
hybrid. It is one-person assay validation: HG005 was previously used during
development, and chr22:24-30 Mb overlaps earlier training/development regions.
The study reports that region separately and an excluded-region sensitivity.

Primary supported scope is autosomal diploid SNV and short indel inference.
Sex chromosomes, mitochondrial heteroplasmy, CNV/SV, somatic calling, clinical
certification and non-Illumina platforms are outside this release's benchmark.
Extra-exonic imputation and cross-exon cis/trans assignments must be judged by
their measured coverage and errors, not presented as error-free clinical calls.

`RELEASE_LOCK.json` fixes source, weights, configuration and documentation.
`SOURCE_PROVENANCE.json` lists original own-code sources and portability edits.
Changing a frozen file invalidates the release digest. Quality benchmarks are
separate artifacts, so later interpretation does not modify the tested release.
