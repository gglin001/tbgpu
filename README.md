# tbgpu

## How to start

- prepare

```bash
# large repo, `--filter=blob:none` + `sparse-checkout` or `git submodule absorb-git-dirs`
git -c submodule."third_party/linux-firmware".update='!true' \
  submodule update --init --depth 1 --filter=blob:none third_party/linux-firmware
pushd third_party/linux-firmware
git sparse-checkout init --no-cone
git sparse-checkout set '/nvidia/'
git checkout
popd
# other submodule
git submodule update --init --depth 1
```

- install

```bash
uv pip install --system --no-build-isolation -e . -v
```

- run tests

```bash
python tests/vector_add.py --kernel-input ptx
# with DEBUG info
DEBUG=10 python tests/vector_add.py --kernel-input cuda --launch-mode kernel_params
# with profiling
CUDA_PTI=1 python tests/kernel_profile.py --kernel-input ptx --iters 100
```
