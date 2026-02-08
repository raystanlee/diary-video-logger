Video Diary Logger (V-JEPA2)

This project explores building a personal video diary system that logs what a person is doing in their environment over time, using modern self-supervised video representation models.

The long-term goal is:

Record video → understand actions → produce a human-readable activity log (a “video diary”)

⸻

Motivation

Most video-to-text systems rely on large video-language models (e.g. Video-LLaVA), which are expensive and not ideal for learning representation-level video understanding.

Instead, this project focuses on:
	•	learning strong video representations
	•	adapting them efficiently
	•	keeping the system modular and understandable

⸻

Data Sources Considered

❌ CLIP / Zero-shot video classification
	•	Requires predefined label prompts
	•	Not suitable for free-form diary-style logging

❌ Full video-language models
	•	Heavy, slow, opaque
	•	Less control over representation learning

✅ Charades Dataset (chosen)
	•	Indoor, daily human activities
	•	Multi-label (people do multiple things at once)
	•	Strong alignment with diary-style actions (phone, laptop, sitting, opening doors, etc.)

We used:
	•	HuggingFaceM4/charades
	•	480p video variant
	•	~7,985 training videos
	•	~1,000 validation videos

A custom diary-style subset (80 actions) was created from the original 157 Charades actions.

⸻

Model Choice: V-JEPA 2

We use: facebook/vjepa2-vitl-fpc16-256-ssv2

Why V-JEPA2?
	•	Self-supervised, predictive video representations
	•	No language decoder required
	•	Strong temporal understanding
	•	Designed for action understanding, not captioning

This aligns well with the goal of building a video diary before adding language generation.

⸻

Training Strategy (Progressive Adaptation)

We intentionally did not fully finetune immediately. Instead, we progressed through increasingly powerful (and expensive) stages.

⸻

Stage 1 — Head-Only Adaptation
	•	Freeze entire V-JEPA2 backbone
	•	Train only the classification head
	•	Fast, stable, low compute

Result
	•	micro-F1 ≈ 0.12–0.14
	•	Good sanity check, limited ceiling

⸻

Stage 2 — Head + Last Transformer Block
	•	Unfreeze last encoder block
	•	Use two learning rates:
	•	Head LR: 1e-4
	•	Backbone LR: 1e-5
	•	Train for up to 4000 steps

Result
	•	micro-F1 improved to ~0.24
	•	Much better temporal + semantic alignment
	•	Still efficient on Apple MPS

⸻

Stage 2 — Head + Last Transformer Block
	•	Handpick charades dataset labels and finetune only on those label for my use case
	•	Unfreeze last encoder block
	•	Use two learning rates:
	•	Head LR: 1e-4
	•	Backbone LR: 1e-5
	•	Train for up to 4000 steps

Result
	•	Must better classification based on human evaluation

⸻


Threshold Tuning (Critical Step)

Because this is multi-label classification, raw probabilities must be thresholded.

We implemented a threshold sweep:
thr ∈ [0.05 … 0.95]

Best result
	•	Threshold = 0.10
	•	micro-F1 = 0.2397
	•	Evaluated on 1000 validation videos
	•	80 diary-style action labels

This significantly outperformed naïve thresholds (e.g. 0.5).

⸻

Key Takeaways So Far
	•	Representation learning > raw captioning for this task
	•	Head-only finetuning gives fast gains
	•	Unfreezing just one transformer block gives a large jump per compute
	•	Threshold tuning matters as much as training
	•	The model is now strong enough to begin real diary inference