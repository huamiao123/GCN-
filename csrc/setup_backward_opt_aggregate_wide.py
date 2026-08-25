from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sources = [
    "experiments/products_saved_t_20260812/bindings_aggregate.cpp",
    "forward_api.cpp", "backward_api.cpp", "c2/normalized_api.cpp",
    "experiments/backward_opt_20260814/v6_wide_aggregate_probe.cpp",
    "experiments/products_saved_t_20260812/backward_v2_kernels.cpp",
    "avx/ybar_layout.cpp",
]
setup(
    name="tfs_train_v2_c0_ext",
    ext_modules=[CppExtension(
        "tfs_train_v2_c0_ext", [str(HERE / x) for x in sources],
        include_dirs=[str(ROOT / "include")],
        extra_compile_args=["-O3", "-mamx-tile", "-mamx-bf16", "-mavx512f",
                            "-mavx512bw", "-mavx512bf16", "-mavx512vl"],
    )],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
