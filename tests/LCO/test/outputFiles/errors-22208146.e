2023-05-10 12:16:20.318669: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /cm/shared/apps/slurm/current/lib64/slurm:/cm/shared/apps/slurm/current/lib64
2023-05-10 12:16:20.319033: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
2023-05-10 12:16:23.632521: E tensorflow/stream_executor/cuda/cuda_driver.cc:271] failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
2023-05-10 12:16:23.632623: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (exp-4-29): /proc/driver/nvidia/version does not exist
2023-05-10 12:16:23.633138: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
/home/jholber/.local/lib/python3.8/site-packages/keras/optimizers/optimizer_v2/rmsprop.py:135: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
  super(RMSprop, self).__init__(name, **kwargs)
Traceback (most recent call last):
  File "main.py", line 21, in <module>
    workflow.main_workflow()
  File "/expanse/lustre/scratch/jholber/temp_project/active-learning/active_learning.py", line 623, in main_workflow
    self.hyperparameter_search(rnd)
  File "/expanse/lustre/scratch/jholber/temp_project/active-learning/active_learning.py", line 416, in hyperparameter_search
    self.hidden_units, self.lr = hyperparameterSearch(rnd,self.N_hp_sets,commands,training_func, self.job_manager,self.Account,self.Walltime,self.Mem,)
  File "/expanse/lustre/scratch/jholber/temp_project/active-learning/hp_search.py", line 81, in hyperparameterSearch
    submitHPSearch(N_sets,rnd,commands,training_func, job_manager, account, walltime, memory)
  File "/expanse/lustre/scratch/jholber/temp_project/active-learning/hp_search.py", line 64, in submitHPSearch
    from mechanoChemML.workflows.active_learning.slurm_manager import numCurrentJobs, submitJob
ModuleNotFoundError: No module named 'mechanoChemML'
