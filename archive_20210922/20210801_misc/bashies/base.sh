#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$1_$2
#SBATCH --time=20:00:00
#SBATCH --mem=48g
#SBATCH --ntasks-per-node=16
#SBATCH --nodes=1
ulimit -s 10240
module load openmpi python/3.6.1
cd /home/tid010/hexapod
mpirun -np 16 python run.py --seed 84 --exp $1_$2 --obstacle_type $1 $3

hostname

exit 0
EOT