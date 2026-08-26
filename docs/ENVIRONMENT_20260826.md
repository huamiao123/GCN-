# Authority experiment environment

This repository's authority TFS/DGL CPU experiments were run on Linux with:

```text
Python 3.9
PyTorch 2.5.1+cpu
DGL 2.1.0
NumPy 2.0.2
pytest 8.3.4
Intel oneAPI compilers 2024.1.0 (module: intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp)
```

Install the Python runtime with:

```bash
python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r requirements.txt
```

`torch-geometric` is listed for legacy PyG benchmark harnesses only. The
formal TFS-vs-stock-DGL launchers require neither PyG nor CUDA.

Then build the CPU AMX extension on an AMX-capable Linux host after loading
the stated oneAPI module:

```bash
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
bash scripts/build_extension.sh
```

The exact CPU model, NUMA binding and performance environment remain recorded
per result cell in its manifest and affinity files. A dependency list alone
does not reproduce those hardware conditions.
