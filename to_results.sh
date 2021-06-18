path=/home/jorge/Documents/PHD/Simulations/SCSI/cases/results/$1

mkdir -v $path
cp -r py_files initial_conditions program_structure domains $path
mkdir -v $path/results
mkdir -v $path/results_particles
mkdir -v $path/previous_executions
mkdir -v $path/particle_tracker
