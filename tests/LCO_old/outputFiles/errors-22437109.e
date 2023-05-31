2023-05-19 05:43:29.858103: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /cm/shared/apps/slurm/current/lib64/slurm:/cm/shared/apps/slurm/current/lib64
2023-05-19 05:43:29.858915: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
2023-05-19 05:43:33.418184: E tensorflow/stream_executor/cuda/cuda_driver.cc:271] failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
2023-05-19 05:43:33.418257: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (exp-5-37): /proc/driver/nvidia/version does not exist
2023-05-19 05:43:33.418633: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
/home/jholber/.local/lib/python3.8/site-packages/keras/optimizers/optimizer_v2/rmsprop.py:135: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
  super(RMSprop, self).__init__(name, **kwargs)
Traceback (most recent call last):
  File "main.py", line 22, in <module>
    workflow.main_workflow()
  File "/expanse/lustre/scratch/jholber/temp_project/active-learning/active_learning.py", line 650, in main_workflow
    self.exploit(2*rnd+1)
  File "/expanse/lustre/scratch/jholber/temp_project/active-learning/active_learning.py", line 383, in exploit
    submitCASM(self.N_jobs,self.phi,kappa_local,Temp,rnd,self.Account,self.Walltime,self.Mem,casm_project_dir=self.casm_project_dir,test=self.test,job_manager=self.job_manager,casm_version=self.CASM_version, data_generation=self.data_generation)
  File "/expanse/lustre/scratch/jholber/temp_project/active-learning/data_generation_wrapper.py", line 60, in submitCASM
    json.dump(inputF,outFile,indent=4)
  File "/cm/shared/apps/spack/gpu/opt/spack/linux-centos8-skylake_avx512/gcc-8.3.1/anaconda3-2020.11-bsn4npoxyw7jzz7fajncek3bvdoaa5wv/lib/python3.8/json/__init__.py", line 179, in dump
    for chunk in iterable:
  File "/cm/shared/apps/spack/gpu/opt/spack/linux-centos8-skylake_avx512/gcc-8.3.1/anaconda3-2020.11-bsn4npoxyw7jzz7fajncek3bvdoaa5wv/lib/python3.8/json/encoder.py", line 431, in _iterencode
    yield from _iterencode_dict(o, _current_indent_level)
  File "/cm/shared/apps/spack/gpu/opt/spack/linux-centos8-skylake_avx512/gcc-8.3.1/anaconda3-2020.11-bsn4npoxyw7jzz7fajncek3bvdoaa5wv/lib/python3.8/json/encoder.py", line 405, in _iterencode_dict
    yield from chunks
  File "/cm/shared/apps/spack/gpu/opt/spack/linux-centos8-skylake_avx512/gcc-8.3.1/anaconda3-2020.11-bsn4npoxyw7jzz7fajncek3bvdoaa5wv/lib/python3.8/json/encoder.py", line 405, in _iterencode_dict
    yield from chunks
  File "/cm/shared/apps/spack/gpu/opt/spack/linux-centos8-skylake_avx512/gcc-8.3.1/anaconda3-2020.11-bsn4npoxyw7jzz7fajncek3bvdoaa5wv/lib/python3.8/json/encoder.py", line 325, in _iterencode_list
    yield from chunks
  File "/cm/shared/apps/spack/gpu/opt/spack/linux-centos8-skylake_avx512/gcc-8.3.1/anaconda3-2020.11-bsn4npoxyw7jzz7fajncek3bvdoaa5wv/lib/python3.8/json/encoder.py", line 405, in _iterencode_dict
    yield from chunks
  File "/cm/shared/apps/spack/gpu/opt/spack/linux-centos8-skylake_avx512/gcc-8.3.1/anaconda3-2020.11-bsn4npoxyw7jzz7fajncek3bvdoaa5wv/lib/python3.8/json/encoder.py", line 438, in _iterencode
    o = _default(o)
  File "/cm/shared/apps/spack/gpu/opt/spack/linux-centos8-skylake_avx512/gcc-8.3.1/anaconda3-2020.11-bsn4npoxyw7jzz7fajncek3bvdoaa5wv/lib/python3.8/json/encoder.py", line 179, in default
    raise TypeError(f'Object of type {o.__class__.__name__} '
TypeError: Object of type float32 is not JSON serializable
