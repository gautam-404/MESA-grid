## Test optimal OMP_NUM_THREADS for parallelization

from MESAcontroller import MesaAccess, ProjectOps
import os, time

min_exec_time = 99999
optimal_omp = 1
parallel = 1
for i in range(1, (os.cpu_count()//2)+1):
    os.environ["OMP_NUM_THREADS"] = str(i)
    name = 'test'
    proj = ProjectOps(name)     
    proj.create(overwrite=True) 
    star = MesaAccess(name)
    star.set('pgstar_flag', False)
    proj.make()
    start_time = time.time()
    proj.run()
    end_time = time.time()
    elapsed_time = end_time - start_time
    for j in range(2, -(-os.cpu_count()//i)+1):
        if elapsed_time*j < min_exec_time*j:
            min_exec_time = elapsed_time
            optimal_omp = i
            parallel = j
    print('Execution time:', elapsed_time, 'seconds for OMP_NUM_THREADS =', i)
    print("Optimal OMP_NUM_THREADS:", optimal_omp, "with", parallel, "parallel processes.")
    print("\n")