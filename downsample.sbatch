#!/usr/bin/bash
#SBATCH --job-name=gen_rich
#SBATCH --output=gen_rich.out
#SBATCH --error=gen_rich.err
#SBATCH --time=40:00:00

ml python/3.9.0
source /scratch/users/yonigoz/mminfinity_env/bin/activate
python downsample_rich.py
