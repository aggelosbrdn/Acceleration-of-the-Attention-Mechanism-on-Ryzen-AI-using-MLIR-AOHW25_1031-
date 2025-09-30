# Get the directory of this Makefile
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Include the common makefile definitions from the project
include ${srcdir}/../programming_examples/makefile-common

# --- FFN Configuration ---
# Using small, valid dimensions for faster debugging.
M := 128
K1 := 128
N1 := 256
K2 := 256
N2 := 128

# Define the data types used for the NPU kernels
DTYPE_IN := i16
DTYPE_OUT := i32

# Define the tile sizes for the NPU kernels
TILE_M := 32
TILE_K := 32
TILE_N := 32

# FIX: Add critical compiler definitions for the AIE kernel
KERNEL_DEFINES := -D$(DTYPE_IN)_$(DTYPE_OUT)_ONLY -DDIM_M=$(TILE_M) -DDIM_K=$(TILE_K) -DDIM_N=$(TILE_N)

# --- Build Targets ---
all: ffn.exe build/matmul1/final.xclbin build/matmul2/final.xclbin

# --- Hardware Generation Rules (matmul1) ---
build/matmul1/aie.mlir: matmul.py
	mkdir -p $(@D)
	python3 $< -M $(M) -K $(K1) -N $(N1) --dtype_in $(DTYPE_IN) --dtype_out $(DTYPE_OUT) \
	-m $(TILE_M) -k $(TILE_K) -n $(TILE_N) > $@

build/matmul1/mm_$(TILE_M)x$(TILE_K)x$(TILE_N).o: ../aie_kernels/aie2/mm.cc
	mkdir -p $(@D)
	# FIX: Pass the KERNEL_DEFINES to the AIE kernel compiler
	cd $(@D) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2_FLAGS) $(KERNEL_DEFINES) -c ../../../aie_kernels/aie2/mm.cc -o $(@F)

build/matmul1/final.xclbin: build/matmul1/aie.mlir build/matmul1/mm_$(TILE_M)x$(TILE_K)x$(TILE_N).o
	mkdir -p $(@D)
	cd $(@D) && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
	--no-xchesscc --no-xbridge \
		--xclbin-name=$(@F) --npu-insts-name=insts.bin aie.mlir

# --- Hardware Generation Rules (matmul2) ---
build/matmul2/aie.mlir: matmul.py
	mkdir -p $(@D)
	python3 $< -M $(M) -K $(K2) -N $(N2) --dtype_in $(DTYPE_IN) --dtype_out $(DTYPE_OUT) \
	-m $(TILE_M) -k $(TILE_K) -n $(TILE_N) > $@

build/matmul2/mm_$(TILE_M)x$(TILE_K)x$(TILE_N).o: ../aie_kernels/aie2/mm.cc
	mkdir -p $(@D)
	# FIX: Pass the KERNEL_DEFINES to the AIE kernel compiler
	cd $(@D) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2_FLAGS) $(KERNEL_DEFINES) -c ../../../aie_kernels/aie2/mm.cc -o $(@F)

build/matmul2/final.xclbin: build/matmul2/aie.mlir build/matmul2/mm_$(TILE_M)x$(TILE_K)x$(TILE_N).o
	mkdir -p $(@D)
	cd $(@D) && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
	--no-xchesscc --no-xbridge \
		--xclbin-name=$(@F) --npu-insts-name=insts.bin aie.mlir

# --- Host Program Compilation Rule ---
ffn.exe: test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && cmake .. -DTARGET_NAME=ffn -DM_DIM=$(M) -DK1_DIM=$(K1) -DN1_DIM=$(N1) -DK2_DIM=$(K2) -DN2_DIM=$(N2)
	cd _build && cmake --build . --config Release
	cp _build/ffn $@

# --- Execution Rule ---
run: ffn.exe build/matmul1/final.xclbin build/matmul2/final.xclbin
	./$< \
	--xclbin1 build/matmul1/final.xclbin --insts1 build/matmul1/insts.bin \
	--xclbin2 build/matmul2/final.xclbin --insts2 build/matmul2/insts.bin

# --- Cleanup Rule ---
clean:
	rm -rf build _build ffn.exe

