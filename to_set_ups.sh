path=/home/jorge/Documents/PHD/Simulations/SCSI/cases/set_ups/$1

mkdir -v $path
cp -r py_files initial_conditions program_structure domains $path
cp fastDelete.sh $path
mkdir -v $path/initial_conditions
mkdir -v $path/results
mkdir -v $path/results_particles
mkdir -v $path/previous_executions
mkdir -v $path/particle_tracker
