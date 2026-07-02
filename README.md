# SARFA

This repository contains the implementation of SARFA.

Our proposed model trains the Segment Anything Model (SAM) in a probabilistic manner to produce multiple plausible segmentation masks for a single input sample. Using a novel training objective based on minimizing the Fréchet Radiomic Distance (FRD) between the ground truth and predicted mask distributions, followed by a ranking system based on direct preference optimization (DPO).

## Abstract

The Segment Anything Model (SAM) has demonstrated strong generalizability across a variety of segmentation tasks. However, SAM often struggles in situations where the target to be segmented is ambiguous. This poses a problem in medical imaging, where accurate delineation of targets such as tumors is vital, but even expert radiologists can disagree on the appropriate boundary for a target. Addressing this, we propose SARFA (Segment Anything with Radiomic Feature Alignment), a novel framework for improved medical image segmentation. Via probabilistic prompting, SARFA generates a diverse set of plausible masks for each input image and optimizes them with a radiomics-driven training objective based on Fréchet Radiomic Distance (FRD) and Direct Preference Optimization (DPO). By minimizing the FRD between masked predicted and ground truth regions within each image, SARFA encourages segmentation outputs whose anatomical and textural characteristics align with clinically meaningful ground truth representations, without relying solely on pixel-level overlap. Evaluated on computed tomography (CT) and magnetic resonance imaging (MRI) benchmarks, SARFA outperforms existing ambiguous segmentation methods, demonstrating the effectiveness of radiomic feature alignment and DPO-style candidate mask ranking as a training objective.

## Model

![Figure](https://github.com/tbwa233/SARFA/blob/main/images/sarfa_final.png)

A probabilistic prompt generator produces diverse prompt embeddings that enable LoRA-adapted SAM to generate K candidate segmentation masks for each input image. Masked regions from both ground truth and predicted candidates are passed through a radiomic feature extraction pipeline, and their similarity is quantified using FRD. The radiomic distances are used to rank masks, defining preferred and rejected candidates for our DPO loss, which encourages anatomically and texturally consistent medical image segmentation.

## Results

A brief summary of our results are below. Our proposed SARFA methods is compared to existing baselines for probabilistic baselines. For brevity, we only include results on the LIDC-IDRI CT dataset here, although additional results on the BraTS2017 MRI dataset can be found in our paper.

| Method | GED ↓ | FRD ↓ | HM-IoU ↑ | D<sub>max</sub> ↑ |
|---|---:|---:|---:|---:|
| Probabilistic U-Net | 0.324 | -- | 0.423 | 0.370 |
| HPU-Net | 0.270 | -- | 0.530 | -- |
| PHiseg | 0.262 | -- | 0.595 | -- |
| SSN | 0.259 | -- | 0.555 | -- |
| CAR | 0.252 | -- | 0.549 | 0.732 |
| PixelSeg | 0.243 | -- | 0.614 | 0.814 |
| CIMD | 0.234 | -- | 0.587 | -- |
| Mose | 0.234 | -- | 0.623 | 0.702 |
| SAMed | 0.380 | -- | 0.357 | 0.703 |
| P<sup>2</sup>SAM | 0.353 | 3.648 | 0.654 | 0.772 |
| **SARFA (ours)** | **0.206** | **2.758** | **0.659** | **0.774** |

<p align="left">
  <img src="https://github.com/tbwa233/SARFA/blob/main/images/sarfa_results.png" width="485">
</p>

## Code

The code has been written in Python using the PyTorch framework. Training requries a GPU. We provide a Jupyter Notebook, which can be run in Google Colab, containing the algorithm in a usable version. Open SARFA.ipynb and run it through.

## Citation

If you find this repo or the paper useful, please cite:
