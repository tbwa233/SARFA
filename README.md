# SARFA

This repository contains the implementation of SARFA.

Our proposed model trains the Segment Anything Model (SAM) in a probabilistic manner to produce multiple plausible segmentation masks for a single input sample. Using a novel training objective based on minimizing the Fréchet Radiomic Distance (FRD) between the ground truth and predicted mask distributions, followed by a ranking system based on direct preference optimization (DPO).

## Abstract

The Segment Anything Model (SAM) has demonstrated strong generalizability across a variety of segmentation tasks. However, SAM often struggles in situations where the target to be segmented is ambiguous. This poses a problem in medical imaging, where accurate delineation of targets such as tumors is vital, but even expert radiologists can disagree on the appropriate boundary for a target. Addressing this, we propose SARFA (Segment Anything with Radiomic Feature Alignment), a novel framework for improved medical image segmentation. Via probabilistic prompting, SARFA generates a diverse set of plausible masks for each input image and optimizes them with a radiomics-driven training objective based on Fréchet Radiomic Distance (FRD) and Direct Preference Optimization (DPO). By minimizing the FRD between masked predicted and ground truth regions within each image, SARFA encourages segmentation outputs whose anatomical and textural characteristics align with clinically meaningful ground truth representations, without relying solely on pixel-level overlap. Evaluated on computed tomography (CT) and magnetic resonance imaging (MRI) benchmarks, SARFA outperforms existing ambiguous segmentation methods, demonstrating the effectiveness of radiomic feature alignment and DPO-style candidate mask ranking as a training objective.

## Model

![Figure](https://github.com/tbwa233/SARFA/blob/main/images/sarfa_final.png)

A probabilistic prompt generator produces diverse prompt embeddings that enable LoRA-adapted SAM to generate K candidate segmentation masks for each input image. Masked regions from both ground truth and predicted candidates are passed through a radiomic feature extraction pipeline, and their similarity is quantified using FRD. The radiomic distances are used to rank masks, defining preferred and rejected candidates for our DPO loss, which encourages anatomically and texturally consistent medical image segmentation.

## Results

## Code

## Citation
