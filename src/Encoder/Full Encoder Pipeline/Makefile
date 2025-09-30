# Get the directory of this Makefile
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Include the common makefile definitions from the project
include ${srcdir}/../programming_examples/makefile-common

# --- Encoder Configuration ---
# Define dimensions for a standard encoder layer.
M := 128
K_MODEL := 128
K_HEAD := $(K_MODEL)

# FIX: make cannot do math, so use shell to calculate FFN_HIDDEN. 
FFN_HIDDEN := $(shell echo $$(($(K_MODEL) * 4)))

# Define the data types used for the NPU kernels
DTYPE_IN := i16
DTYPE_OUT := i32

# Define the tile sizes for the NPU kernels
TILE_M := 32
TILE_K := 32
TILE_N := 32

# Define compiler flags for the AIE kernel
KERNEL_DEFINES := -D$(DTYPE_IN)_$(DTYPE_OUT)_ONLY -DDIM_M=$(TILE_M) -DDIM_K=$(TILE_K) -DDIM_N=$(TILE_N)

# --- Build Targets ---
all: encoder.exe build/qkv_matmul/final.xclbin build/sdpa_matmul1/final.xclbin build/sdpa_matmul2/final.xclbin \
	build/ffn_matmul1/final.xclbin build/ffn_matmul2/final.xclbin

# --- Hardware Generation Rules ---

# 1. For Q, K, V Generation
build/qkv_matmul/aie.mlir: matmul.py
	mkdir -p $(@D)
	python3 $< -M $(M) -K $(K_MODEL) -N $(K_HEAD) --dtype_in $(DTYPE_IN) --dtype_out $(DTYPE_OUT) \
	-m $(TILE_M) -k $(TILE_K) -n $(TILE_N) > $@

build/qkv_matmul/mm_$(TILE_M)x$(TILE_K)x$(TILE_N).o: ../aie_kernels/aie2/mm.cc
	mkdir -p $(@D)
	cd $(@D) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2_FLAGS) $(KERNEL_DEFINES) -c ../../../aie_kernels/aie2/mm.cc -o $(@F)

build/qkv_matmul/final.xclbin: build/qkv_matmul/aie.mlir build/qkv_matmul/mm_$(TILE_M)x$(TILE_K)x$(TILE_N).o
	mkdir -p $(@D)
	cd $(@D) && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
	--no-xchesscc --no-xbridge --xclbin-name=$(@F) --npu-insts-name=insts.bin aie.mlir

# 2. For SDPA MatMul1 (Q @ K_T)
build/sdpa_matmul1/aie.mlir: matmul.py
	mkdir -p $(@D)
	python3 $< -M $(M) -K $(K_HEAD) -N $(M) --dtype_in $(DTYPE_IN) --dtype_out $(DTYPE_OUT) \
	-m $(TILE_M) -k $(TILE_K) -n $(TILE_N) --b-col-maj 1 > $@

build/sdpa_matmul1/mm_$(TILE_M)x$(TILE_K)x$(TILE_N).o: ../aie_kernels/aie2/mm.cc
	mkdir -p $(@D)
	cd $(@D) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2_FLAGS) $(KERNEL_DEFINES) -c ../../../aie_kernels/aie2/mm.cc -o $(@F)

build/sdpa_matmul1/final.xclbin: build/sdpa_matmul1/aie.mlir build/sdpa_matmul1/mm_$(TILE_M)x$(TILE_K)x$(TILE_N).o
	mkdir -p $(@D)
	cd $(@D) && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
	--no-xchesscc --no-xbridge --xclbin-name=$(@F) --npu-insts-name=insts.bin aie.mlir

# 3. For SDPA MatMul2 (Scores @ V)
build/sdpa_matmul2/aie.mlir: matmul.py
	mkdir -p $(@D)
	python3 $< -M $(M) -K $(M) -N $(K_HEAD) --dtype_in $(DTYPE_IN) --dtype_out $(DTYPE_OUT) \
	-m $(TILE_M) -k $(TILE_K) -n $(TILE_N) > $@

build/sdpa_matmul2/mm_$(TILE_M)x$(TILE_K)x$(TILE_N).o: ../aie_kernels/aie2/mm.cc
	mkdir -p $(@D)
	cd $(@D) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2_FLAGS) $(KERNEL_DEFINES) -c ../../../aie_kernels/aie2/mm.cc -o $(@F)

build/sdpa_matmul2/final.xclbin: build/sdpa_matmul2/aie.mlir build/sdpa_matmul2/mm_$(TILE_M)x$(TILE_K)x$(TILE_N).o
	mkdir -p $(@D)
	cd $(@D) && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
	--no-xchesscc --no-xbridge --xclbin-name=$(@F) --npu-insts-name=insts.bin aie.mlir

# 4. For FFN MatMul1
build/ffn_matmul1/aie.mlir: matmul.py
	mkdir -p $(@D)
	python3 $< -M $(M) -K $(K_MODEL) -N $(FFN_HIDDEN) --dtype_in $(DTYPE_IN) --dtype_out $(DTYPE_OUT) \
	-m $(TILE_M) -k $(TILE_K) -n $(TILE_N) > $@

build/ffn_matmul1/mm_$(TILE_M)x$(TILE_K)x$(TILE_N).o: ../aie_kernels/aie2/mm.cc
	mkdir -p $(@D)
	cd $(@D) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2_FLAGS) $(KERNEL_DEFINES) -c ../../../aie_kernels/aie2/mm.cc -o $(@F)

build/ffn_matmul1/final.xclbin: build/ffn_matmul1/aie.mlir build/ffn_matmul1/mm_$(TILE_M)x$(TILE_K)x$(TILE_N).o
	mkdir -p $(@D)
	cd $(@D) && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
	--no-xchesscc --no-xbridge --xclbin-name=$(@F) --npu-insts-name=insts.bin aie.mlir

# 5. For FFN MatMul2
build/ffn_matmul2/aie.mlir: matmul.py
	mkdir -p $(@D)
	python3 $< -M $(M) -K $(FFN_HIDDEN) -N $(K_MODEL) --dtype_in $(DTYPE_IN) --dtype_out $(DTYPE_OUT) \
	-m $(TILE_M) -k $(TILE_K) -n $(TILE_N) > $@

build/ffn_matmul2/mm_$(TILE_M)x$(TILE_K)x$(TILE_N).o: ../aie_kernels/aie2/mm.cc
	mkdir -p $(@D)
	cd $(@D) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2_FLAGS) $(KERNEL_DEFINES) -c ../../../aie_kernels/aie2/mm.cc -o $(@F)

build/ffn_matmul2/final.xclbin: build/ffn_matmul2/aie.mlir build/ffn_matmul2/mm_$(TILE_M)x$(TILE_K)x$(TILE_N).o
	mkdir -p $(@D)
	cd $(@D) && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
	--no-xchesscc --no-xbridge --xclbin-name=$(@F) --npu-insts-name=insts.bin aie.mlir


# --- Host Program Compilation Rule ---
encoder.exe: test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && cmake .. -DTARGET_NAME=encoder -DM_DIM=$(M) -DK_MODEL_DIM=$(K_MODEL)
	cd _build && cmake --build . --config Release
	cp _build/encoder $@

# --- Execution Rule ---
run: encoder.exe all
	./$< \
	--xclbin_qkv build/qkv_matmul/final.xclbin --insts_qkv build/qkv_matmul/insts.bin \
	--xclbin_sdpa1 build/sdpa_matmul1/final.xclbin --insts_sdpa1 build/sdpa_matmul1/insts.bin \
	--xclbin_sdpa2 build/sdpa_matmul2/final.xclbin --insts_sdpa2 build/sdpa_matmul2/insts.bin \
	--xclbin_ffn1 build/ffn_matmul1/final.xclbin --insts_ffn1 build/ffn_matmul1/insts.bin \
	--xclbin_ffn2 build/ffn_matmul2/final.xclbin --insts_ffn2 build/ffn_matmul2/insts.bin

# --- Cleanup Rule ---
clean:
	rm -rf build _build encoder.exe


