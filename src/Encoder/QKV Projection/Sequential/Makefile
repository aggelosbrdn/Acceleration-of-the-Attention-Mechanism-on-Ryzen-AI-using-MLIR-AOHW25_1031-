# Get the directory of this Makefile
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Include the common makefile definitions from the project
include ${srcdir}/../programming_examples/makefile-common

# --- Sequential MatMul Configuration ---
# Define the dimensions for three sequential matrix multiplications:
# Step 1: M x K1
# Step 2: M x N1 (output of step 1)
# Step 3: M x N2 (output of step 2)
M := 128
K1 := 128
N1 := 128
N2 := 128

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
all: matmul3.exe build/matmul1/final.xclbin build/matmul2/final.xclbin build/matmul3/final.xclbin

# --- Hardware Generation Rules (MatMul1) ---
build/matmul1/aie.mlir: matmul.py
	mkdir -p $(@D)
	python3 $< -M $(M) -K $(K1) -N $(N1) --dtype_in $(DTYPE_IN) --dtype_out $(DTYPE_OUT) \
	-m $(TILE_M) -k $(TILE_K) -n $(TILE_N) > $@

build/matmul1/mm_$(TILE_M)x$(TILE_K)x$(TILE_N).o: ../aie_kernels/aie2/mm.cc
	mkdir -p $(@D)
	cd $(@D) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2_FLAGS) $(KERNEL_DEFINES) -c ../../../aie_kernels/aie2/mm.cc -o $(@F)

build/matmul1/final.xclbin: build/matmul1/aie.mlir build/matmul1/mm_$(TILE_M)x$(TILE_K)x$(TILE_N).o
	mkdir -p $(@D)
	cd $(@D) && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
	--no-xchesscc --no-xbridge \
		--xclbin-name=$(@F) --npu-insts-name=insts.bin aie.mlir

# --- Hardware Generation Rules (MatMul2) ---
build/matmul2/aie.mlir: matmul.py
	mkdir -p $(@D)
	python3 $< -M $(M) -K $(N1) -N $(N2) --dtype_in $(DTYPE_IN) --dtype_out $(DTYPE_OUT) \
	-m $(TILE_M) -k $(TILE_K) -n $(TILE_N) > $@

build/matmul2/mm_$(TILE_M)x$(TILE_K)x$(TILE_N).o: ../aie_kernels/aie2/mm.cc
	mkdir -p $(@D)
	cd $(@D) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2_FLAGS) $(KERNEL_DEFINES) -c ../../../aie_kernels/aie2/mm.cc -o $(@F)

build/matmul2/final.xclbin: build/matmul2/aie.mlir build/matmul2/mm_$(TILE_M)x$(TILE_K)x$(TILE_N).o
	mkdir -p $(@D)
	cd $(@D) && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
	--no-xchesscc --no-xbridge \
		--xclbin-name=$(@F) --npu-insts-name=insts.bin aie.mlir

# --- Hardware Generation Rules (MatMul3) ---
build/matmul3/aie.mlir: matmul.py
	mkdir -p $(@D)
	python3 $< -M $(M) -K $(N2) -N $(K1) --dtype_in $(DTYPE_IN) --dtype_out $(DTYPE_OUT) \
	-m $(TILE_M) -k $(TILE_K) -n $(TILE_N) > $@

build/matmul3/mm_$(TILE_M)x$(TILE_K)x$(TILE_N).o: ../aie_kernels/aie2/mm.cc
	mkdir -p $(@D)
	cd $(@D) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2_FLAGS) $(KERNEL_DEFINES) -c ../../../aie_kernels/aie2/mm.cc -o $(@F)

build/matmul3/final.xclbin: build/matmul3/aie.mlir build/matmul3/mm_$(TILE_M)x$(TILE_K)x$(TILE_N).o
	mkdir -p $(@D)
	cd $(@D) && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
	--no-xchesscc --no-xbridge \
		--xclbin-name=$(@F) --npu-insts-name=insts.bin aie.mlir

# --- Host Program Compilation Rule ---
matmul3.exe: test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && cmake .. -DTARGET_NAME=matmul3 -DM_DIM=$(M) -DK1_DIM=$(K1) -DN1_DIM=$(N1) -DN2_DIM=$(N2)
	cd _build && cmake --build . --config Release
	cp _build/matmul3 $@

# --- Execution Rule ---
run: matmul3.exe build/matmul1/final.xclbin build/matmul2/final.xclbin build/matmul3/final.xclbin
	./$< \
	--xclbin1 build/matmul1/final.xclbin --insts1 build/matmul1/insts.bin \
	--xclbin2 build/matmul2/final.xclbin --insts2 build/matmul2/insts.bin \
	--xclbin3 build/matmul3/final.xclbin --insts3 build/matmul3/insts.bin

# --- Cleanup Rule ---
clean:
	rm -rf build _build matmul3.exe
