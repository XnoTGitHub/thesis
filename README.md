# thesis
srun --pty --partition gpu_short --gres=gpu:tesla:1 --nodes=1 -t 20:20:20 /bin/bash -i

singularity instance start --nv /scratch/proj29-shared/singularity/nschroeder/pytorch_1.10.0-v0.sif instance_floriann

singularity exec instance://instance_floriann python Autoencoder_ResNet_2VarSets.py --config configs/default_depth.yaml 


github acces token:


