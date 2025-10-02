# cpu-gpu-image-enhancement

This repository contains implementations of image/video enhancement algorithms in both CPU (C++/OpenCV) and GPU (CUDA).  
It includes a minimal CUDA kernel demo for RGB → Grayscale conversion and several CPU algorithms developed as part of my undergraduate research, later published in IEEE RTEICT 2021.

## Features
- **CUDA Demo**
  - RGB → Grayscale kernel (`src/CUDA_rgbtograyscale.cu`)
  - Uses OpenCV `GpuMat` + custom CUDA kernel
  - Demonstrates kernel launch, grid/block configuration, device sync
- **CPU Algorithms**
  - BHEP (Brightness Preserving Histogram Equalization with Plateau)
  - BHEP-DD (Domain Decomposition)
  - BHEP-SO (Optimized)
  - BHEP-UO (unoptimized)
  - BHEP-Video (video processing pipeline)
## ⚡ Performance Notes

The CPU implementations make use of multi-threading to parallelize pixel-level operations across cores, significantly reducing processing time for large images and video frames.

This mirrors the same design principles seen in distributed systems — splitting workloads into independent tasks, executing them in parallel, and then aggregating results. The GPU (CUDA) kernels further extend this parallelism to thousands of threads.

## Publication
This repository is based on my undergraduate research published at IEEE RTEICT 2021:

S. S. Harakannanavar, G. K. S, S. Ramachandran, T. G. S and R. A. C, "Performance analysis of CPU & GPU for Real Time Image/Video," 2021 International Conference on Recent Trends on Electronics, Information, Communication & Technology (RTEICT), Bangalore, India, 2021, pp. 439–443. doi: 10.1109/RTEICT52294.2021.9573554


##Explanation and methodology

Details regarding methods and results explained in presentation attached - https://docs.google.com/presentation/d/1TIQFLhurJGOeGf2UzuUquMgaLjN8FZh-/edit?usp=sharing&ouid=111855158327885367913&rtpof=true&sd=true
