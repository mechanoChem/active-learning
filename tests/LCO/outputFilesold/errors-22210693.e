2023-05-10 14:03:57.205011: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /cm/shared/apps/slurm/current/lib64/slurm:/cm/shared/apps/slurm/current/lib64
2023-05-10 14:03:57.205593: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
2023-05-10 14:03:59.393299: E tensorflow/stream_executor/cuda/cuda_driver.cc:271] failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
2023-05-10 14:03:59.393379: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (exp-2-05): /proc/driver/nvidia/version does not exist
2023-05-10 14:03:59.393592: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
Traceback (most recent call last):
  File "main.py", line 21, in <module>
    workflow.main_workflow()
  File "/expanse/lustre/scratch/jholber/temp_project/active-learning/active_learning.py", line 619, in main_workflow
    self.explore(2*rnd)
  File "/expanse/lustre/scratch/jholber/temp_project/active-learning/active_learning.py", line 273, in explore
    compileCASMOutput(rnd, self.CASM_version, self.dim)            
  File "/expanse/lustre/scratch/jholber/temp_project/active-learning/data_generation_wrapper.py", line 151, in compileCASMOutput
    dataOut = dataOut[~np.isnan(dataOut).any(axis=1)] #remove any rows with nan
  File "/home/jholber/.local/lib/python3.8/site-packages/numpy/core/_methods.py", line 57, in _any
    return umr_any(a, axis, dtype, out, keepdims)
numpy.AxisError: axis 1 is out of bounds for array of dimension 1
