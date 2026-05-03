# DirectAU Algorithm Comparison: KGAU vs SimKGC

This document compares [D:\Khóa luận\KGAU\DIRECTAU_ALGORITHM.md](D:\Khóa luận\KGAU\DIRECTAU_ALGORITHM.md) and [D:\Khóa luận\SimKGC\SIMKGC_DIRECTAU_ALGORITHM.md](D:\Khóa luận\SimKGC\SIMKGC_DIRECTAU_ALGORITHM.md), focusing on what is shared, what differs, and which practices are worth transferring between the two codebases.

## Similarities

| Area | KGAU DirectAU | SimKGC DirectAU | Notes |
|---|---|---|---|
| Core objective | Alignment + uniformity | Alignment + uniformity | Both replace negative sampling with DirectAU-style optimization. |
| Encoder style | Bi-encoder with separate HR and tail towers | Bi-encoder with separate HR and tail towers | Both encode query text and entity text independently. |
| Text-based inputs | Entity/relation descriptions | Entity/relation descriptions | Both rely on semantic text rather than ID embeddings. |
| L2 normalization | Explicit | Explicit | Both project embeddings to the unit hypersphere before scoring. |
| Inference style | Rank against all entities | Rank against all entities | Both use full-candidate ranking for link prediction. |
| Uniformity computation | Separate query/entity terms | Separate query/entity terms | Both compute uniformity on unique batch elements. |
| Memory controls | AMP, checkpointing, chunking, subsampling | AMP, chunked inference, gradient clipping | Both recognize that text encoders are memory-heavy. |

## Differences

| Area | KGAU DirectAU | SimKGC DirectAU | Practical effect |
|---|---|---|---|
| Base model framing | DirectAU-KG | SimKGC + DirectAU | KGAU is a cleaner DirectAU-centric writeup; SimKGC retains broader SimKGC infrastructure. |
| Sequence length | `max_length: 64` | `max_length: 128` | SimKGC keeps longer text context; KGAU is more memory efficient. |
| Memory strategy | Gradient checkpointing, freeze lower layers, uniformity subsampling | AMP, chunked inference, gradient accumulation, unique deduplication | KGAU emphasizes training-time memory savings more aggressively. |
| Loss implementation detail | Explicit chunked uniformity implementation with subsampling | DirectAU loss plus advanced techniques section | KGAU is more focused on the loss-side efficiency story. |
| Neighbor context | Not documented in the algorithm doc | Documented and implemented via `--use-link-graph` | SimKGC has an additional structural-text hybrid feature. |
| Leakage protection | No dedicated triplet-mask section in the doc | Triplet masking documented as an advanced technique | SimKGC is more explicit about preventing training leakage. |
| Pre-batch negatives | Not documented | Documented, but noted as InfoNCE-oriented | SimKGC carries more legacy SimKGC optimization history. |
| Configuration style | YAML-style config excerpt | CLI flags and parameter table | KGAU reads like a config-driven research setup; SimKGC reads like a runnable CLI app. |
| Inference memory | Chunked encoding and uniformity subsampling | Chunked inference + AMP + gradient clipping | SimKGC is more deployment-oriented, KGAU is more training-research oriented. |

## Practices Worth Porting

| Source model | Practice to borrow | Target model | Why it helps |
|---|---|---|---|
| KGAU | Gradient checkpointing | SimKGC | Reduces activation memory when fine-tuning longer sequences or larger batch sizes. |
| KGAU | Freeze lower transformer layers | SimKGC | Can stabilize training and lower compute cost when using long descriptions. |
| KGAU | Uniformity subsampling / chunked uniformity | SimKGC | Makes the uniformity term cheaper for larger batches. |
| KGAU | Micro-batch / chunked encoding | SimKGC | Helps if longer texts or larger backbones push GPU memory limits. |
| SimKGC | Neighbor-based context augmentation | KGAU | Adds graph structure into text-only inputs and can improve sparse-entity coverage. |
| SimKGC | Triplet masking | KGAU | Prevents training leakage when scoring in-batch negatives or known positives. |
| SimKGC | Explicit LR scheduler options | KGAU | Makes optimizer behavior easier to tune and reproduce. |
| SimKGC | AMP with clear CLI switch | KGAU | Simplifies memory/speed trade-offs for users running on smaller GPUs. |

## Recommendation Summary

If the goal is **better training efficiency**, KGAU should adopt SimKGC-style neighbor augmentation and triplet masking, while keeping its own memory-saving ideas such as checkpointing and uniformity subsampling.

If the goal is **better usability and broader experimentation**, SimKGC should keep the DirectAU core but add more of KGAU’s training-efficiency practices, especially gradient checkpointing and layer freezing for larger or longer-text settings.

## Best Cross-Model Additions

1. **For KGAU**: implement `--use-link-graph`-style neighbor augmentation first, then add triplet masking.
2. **For SimKGC**: add gradient checkpointing and optional layer freezing for very long descriptions or larger backbones.
3. **For both**: keep AMP, explicit warmup scheduling, and chunked inference as default optimization tools.
