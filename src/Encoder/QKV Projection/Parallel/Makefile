# Get the directory of this Makefile
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Include the common makefile definitions from the project
include ${srcdir}/../programming_examples/makefile-common

# --- Parallel MatMul Configuration ---
# Define the dimensions for the three parallel matrix multiplications.
# All three will have the same shape for this experiment.
M := 128
K := 128
N := 128

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
# Default "all" target builds the host program and the single hardware file.
all: matmulpara.exe build/final.xclbin

# --- Hardware Generation Rules (Only ONE set is needed) ---
build/aie.mlir: matmul.py
	mkdir -p $(@D)
	python3 $< -M $(M) -K $(K) -N $(N) --dtype_in $(DTYPE_IN) --dtype_out $(DTYPE_OUT) \
	-m $(TILE_M) -k $(TILE_K) -n $(TILE_N) > $@

build/mm_$(TILE_M)x$(TILE_K)x$(TILE_N).o: ../aie_kernels/aie2/mm.cc
	mkdir -p $(@D)
	cd $(@D) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2_FLAGS) $(KERNEL_DEFINES) -c ../../aie_kernels/aie2/mm.cc -o $(@F)

build/final.xclbin: build/aie.mlir build/mm_$(TILE_M)x$(TILE_K)x$(TILE_N).o
	mkdir -p $(@D)
	cd $(@D) && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
	--no-xchesscc --no-xbridge \
		--xclbin-name=$(@F) --npu-insts-name=insts.bin aie.mlir

# --- Host Program Compilation Rule ---
matmulpara.exe: test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && cmake .. -DTARGET_NAME=matmulpara -DM_DIM=$(M) -DK_DIM=$(K) -DN_DIM=$(N)
	cd _build && cmake --build . --config Release
	cp _build/matmulpara $@

# --- Execution Rule ---
run: matmulpara.exe build/final.xclbin
	./$< --xclbin build/final.xclbin --insts build/insts.bin

# --- Cleanup Rule ---
clean:
	rm -rf build _build matmulpara.exe

