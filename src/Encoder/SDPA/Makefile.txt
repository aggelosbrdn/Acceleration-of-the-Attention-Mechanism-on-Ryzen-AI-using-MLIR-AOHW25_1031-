# Get the directory of this Makefile
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Include the common makefile definitions from the project
include ${srcdir}/../programming_examples/makefile-common

# --- SDPA Configuration ---
# Define the dimensions for a standard attention head.
M := 128
K := 64

# Define the data types used for the NPU kernels
DTYPE_IN := i16
DTYPE_OUT := i32

# Define the tile sizes for the NPU kernels. These must be chosen
# to be compatible with all matrix dimensions in the pipeline.
TILE_M := 32
TILE_K := 16

# Define compiler flags for each AIE kernel, as they have different tile shapes.
# Note: The kernel source (mm.cc) uses DIM_N for the output tile dimension.
KERNEL_DEFINES_1 := -D$(DTYPE_IN)_$(DTYPE_OUT)_ONLY -DDIM_M=$(TILE_M) -DDIM_K=$(TILE_K) -DDIM_N=$(TILE_M)
KERNEL_DEFINES_2 := -D$(DTYPE_IN)_$(DTYPE_OUT)_ONLY -DDIM_M=$(TILE_M) -DDIM_K=$(TILE_M) -DDIM_N=$(TILE_K)

# Explicitly define the kernel object filenames to avoid parsing errors
KERNEL_OBJ_1 := mm_$(TILE_M)x$(TILE_K)x$(TILE_M).o
KERNEL_OBJ_2 := mm_$(TILE_M)x$(TILE_M)x$(TILE_K).o

# --- Build Targets ---
all: sdpa.exe build/matmul1/final.xclbin build/matmul2/final.xclbin

# --- Hardware Generation Rules (MatMul1: Q @ K_T) ---
build/matmul1/aie.mlir: matmul.py
	mkdir -p $(@D)
	python3 $< -M $(M) -K $(K) -N $(M) --dtype_in $(DTYPE_IN) --dtype_out $(DTYPE_OUT) \
	-m $(TILE_M) -k $(TILE_K) -n $(TILE_M) > $@

build/matmul1/$(KERNEL_OBJ_1): ../aie_kernels/aie2/mm.cc
	mkdir -p $(@D)
	cd $(@D) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2_FLAGS) $(KERNEL_DEFINES_1) -c ../../../aie_kernels/aie2/mm.cc -o $(KERNEL_OBJ_1)

build/matmul1/final.xclbin: build/matmul1/aie.mlir build/matmul1/$(KERNEL_OBJ_1)
	mkdir -p $(@D)
	cd $(@D) && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
	--no-xchesscc --no-xbridge \
		--xclbin-name=$(@F) --npu-insts-name=insts.bin aie.mlir

# --- Hardware Generation Rules (MatMul2: Softmax_Out @ V) ---
build/matmul2/aie.mlir: matmul.py
	mkdir -p $(@D)
	python3 $< -M $(M) -K $(M) -N $(K) --dtype_in $(DTYPE_IN) --dtype_out $(DTYPE_OUT) \
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
sdpa.exe: test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && cmake .. -DTARGET_NAME=sdpa -DM_DIM=$(M) -DK_DIM=$(K)
	cd _build && cmake --build . --config Release
	cp _build/sdpa $@

# --- Execution Rules ---
EXE_ARGS := --xclbin1 build/matmul1/final.xclbin --insts1 build/matmul1/insts.bin \
            --xclbin2 build/matmul2/final.xclbin --insts2 build/matmul2/insts.bin

run: sdpa.exe build/matmul1/final.xclbin build/matmul2/final.xclbin
	@echo "\n>>> Running UNMASKED (Encoder-style) Attention Test <<<"
	./sdpa.exe $(EXE_ARGS)

run-masked: sdpa.exe build/matmul1/final.xclbin build/matmul2/final.xclbin
	@echo "\n>>> Running MASKED (Decoder-style) Attention Test <<<"
	./sdpa.exe $(EXE_ARGS) --masked

test: run run-masked

# --- FINAL FIX: Declare non-file targets as "phony" to prevent errors ---
.PHONY: all run run-masked test clean

# --- Cleanup Rule ---
clean:
	rm -rf build _build sdpa.exe
