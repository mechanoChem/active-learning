2023-05-10 12:38:12.997416: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /cm/shared/apps/slurm/current/lib64/slurm:/cm/shared/apps/slurm/current/lib64
2023-05-10 12:38:12.997862: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
2023-05-10 12:38:17.978540: E tensorflow/stream_executor/cuda/cuda_driver.cc:271] failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
2023-05-10 12:38:17.978634: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (exp-2-09): /proc/driver/nvidia/version does not exist
2023-05-10 12:38:17.978996: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
sbatch: error: Project not found
sbatch: error: Batch job submission failed: Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)
sbatch: error: Project not found
sbatch: error: Batch job submission failed: Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)
sbatch: error: Project not found
sbatch: error: Batch job submission failed: Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)
/home/jholber/.local/lib/python3.8/site-packages/keras/optimizers/optimizer_v2/rmsprop.py:135: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
  super(RMSprop, self).__init__(name, **kwargs)
Traceback (most recent call last):
  File "main.py", line 21, in <module>
    workflow.main_workflow()
  File "/expanse/lustre/scratch/jholber/temp_project/active-learning/active_learning.py", line 623, in main_workflow
    self.hyperparameter_search(rnd)
  File "/expanse/lustre/scratch/jholber/temp_project/active-learning/active_learning.py", line 416, in hyperparameter_search
    self.hidden_units, self.lr = hyperparameterSearch(rnd,self.N_hp_sets,commands,training_func, self.job_manager,self.Account,self.Walltime,self.Mem,)
  File "/expanse/lustre/scratch/jholber/temp_project/active-learning/hp_search.py", line 112, in hyperparameterSearch
    os.rename('idnn_{}'.format(sortedHP[0][2]),'idnn_{}'.format(rnd))
IndexError: list index out of range
