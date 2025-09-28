# Get the directory of this Makefile
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Include the common makefile definitions from the project
include ${srcdir}/../programming_examples/makefile-common

# --- Cross-Attention Configuration ---
# MODIFICATION: M is now M_TGT (Target Sequence Length)
M_TGT := 128
# MODIFICATION: Added M_SRC for the Source (Encoder) Sequence Length
M_SRC := 128
K := 64

# Define the data types used for the NPU kernels
DTYPE_IN := i16
DTYPE_OUT := i32

# Define the tile sizes for the NPU kernels.
TILE_M := 32
TILE_K := 16

# Define compiler flags for each AIE kernel
# Kernel 1 (Q @ K_T): Output tile dim is M_SRC
# Kernel 2 (Scores @ V): Input tile dim is M_SRC, output tile dim is K
KERNEL_DEFINES_1 := -D$(DTYPE_IN)_$(DTYPE_OUT)_ONLY -DDIM_M=$(TILE_M) -DDIM_K=$(TILE_K) -DDIM_N=$(TILE_M)
KERNEL_DEFINES_2 := -D$(DTYPE_IN)_$(DTYPE_OUT)_ONLY -DDIM_M=$(TILE_M) -DDIM_K=$(TILE_M) -DDIM_N=$(TILE_K)

# Explicitly define the kernel object filenames
KERNEL_OBJ_1 := mm_$(TILE_M)x$(TILE_K)x$(TILE_M).o
KERNEL_OBJ_2 := mm_$(TILE_M)x$(TILE_M)x$(TILE_K).o

# --- Build Targets ---
all: crossattention.exe build/matmul1/final.xclbin build/matmul2/final.xclbin

# --- Hardware Generation Rules (MatMul1: Q(M_TGT, K) @ K_T(K, M_SRC)) ---
build/matmul1/aie.mlir: matmul.py
	mkdir -p $(@D)
	# MODIFICATION: Updated dimensions for MatMul1
	python3 $< -M $(M_TGT) -K $(K) -N $(M_SRC) --dtype_in $(DTYPE_IN) --dtype_out $(DTYPE_OUT) \
	-m $(TILE_M) -k $(TILE_K) -n $(TILE_M) > $@

build/matmul1/$(KERNEL_OBJ_1): ../aie_kernels/aie2/mm.cc
	mkdir -p $(@D)
	cd $(@D) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2_FLAGS) $(KERNEL_DEFINES_1) -c ../../../aie_kernels/aie2/mm.cc -o $(KERNEL_OBJ_1)

build/matmul1/final.xclbin: build/matmul1/aie.mlir build/matmul1/$(KERNEL_OBJ_1)
	mkdir -p $(@D)
	cd $(@D) && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
	--no-xchesscc --no-xbridge \
		--xclbin-name=$(@F) --npu-insts-name=insts.bin aie.mlir

# --- Hardware Generation Rules (MatMul2: Scores(M_TGT, M_SRC) @ V(M_SRC, K)) ---
build/matmul2/aie.mlir: matmul.py
	mkdir -p $(@D)
	# MODIFICATION: Updated dimensions for MatMul2
	python3 $< -M $(M_TGT) -K $(M_SRC) -N $(K) --dtype_in $(DTYPE_IN) --dtype_out $(DTYPE_OUT) \
	-m $(TILE_M) -k $(TILE_M) -n $(TILE_K) > $@

build/matmul2/$(KERNEL_OBJ_2): ../aie_kernels/aie2/mm.cc
	mkdir -p $(@D)
	cd $(@D) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2_FLAGS) $(KERNEL_DEFINES_2) -c ../../../aie_kernels/aie2/mm.cc -o $(KERNEL_OBJ_2)

build/matmul2/final.xclbin: build/matmul2/aie.mlir build/matmul2/$(KERNEL_OBJ_2)
	mkdir -p $(@D)
	cd $(@D) && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
	--no-xchesscc --no-xbridge \
		--xclbin-name=$(@F) --npu-insts-name=insts.bin aie.mlir

# --- Host Program Compilation Rule ---
crossattention.exe: test.cpp
	rm -rf _build
	mkdir -p _build
	# MODIFICATION: Pass both M_DIM (for target) and M_SRC_DIM to the compiler
	cd _build && cmake .. -DTARGET_NAME=crossattention -DM_DIM=$(M_TGT) -DM_SRC_DIM=$(M_SRC) -DK_DIM=$(K)
	cd _build && cmake --build . --config Release
	cp _build/crossattention $@

# --- Execution Rule ---
run: crossattention.exe build/matmul1/final.xclbin build/matmul2/final.xclbin
	./$< \
	--xclbin1 build/matmul1/final.xclbin --insts1 build/matmul1/insts.bin \
	--xclbin2 build/matmul2/final.xclbin --insts2 build/matmul2/insts.bin

# --- Cleanup Rule ---
.PHONY: all run clean
clean:
	rm -rf build _build crossattention.exe
