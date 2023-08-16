# Diffusing Coupling High-Frequency-Purifying Structure Feature Extraction for Brain Multimodal Registration

This is the code for the BIBM2023 paper Diffusing Coupling High-Frequency-Purifying Structure Feature Extraction for Brain Multimodal Registration.

## Abstract of The Paper

The core of medical image registration is the alignment of corresponding anatomical structures. However, in multimodal image registration, substantial differences in appearance (intensity distribution) of the images often compel the registration model to prioritize intensity information over structure information, resulting in low accuracy of registration. Hence, the disentangling structure information from intensity information is necessary to improve the registration effectiveness. To this end, we propose a diffusing coupling high-frequency-purifying structure feature extraction for brain multimodal registration. Specifically, the denoising diffusion probabilistic models (DDPM) is firstly utilized to extract complete feature information from images. Subsequently, the discrete cosine transform (DCT) is applied to purify high-frequency anatomical structure information from the complete feature information for registration. Moreover, the introduction of dual-consistency constraint on the purified structure information is utilized to ensure the feasibility of bidirectional registration and enhance the robustness of the registration process. Through comprehensive comparisons with traditional and learning-based methods on the multimodal brain MRI dataset, our method demonstrates superior accuracy and stability in brain multimodal registration. The code is available at https://github.com/goghfan/DDNet.

## How to use

