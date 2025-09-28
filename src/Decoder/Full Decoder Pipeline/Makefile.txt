# Get the directory of this Makefile
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Include the common makefile definitions from the project
include ${srcdir}/../programming_examples/makefile-common

# --- Decoder Configuration ---
M_TGT := 128
M_SRC := 128
K_MODEL := 128
K_HEAD := $(K_MODEL)
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
KERNEL_OBJ := mm_$(TILE_M)x$(TILE_K)x$(TILE_N).o

# --- Build Targets ---
all: decoder.exe \
	build/sa_qkv/final.xclbin \
	build/sa_sdpa1/final.xclbin \
	build/sa_sdpa2/final.xclbin \
	build/ca_q/final.xclbin \
	build/ca_kv/final.xclbin \
	build/ca_sdpa1/final.xclbin \
	build/ca_sdpa2/final.xclbin \
	build/ffn1/final.xclbin \
	build/ffn2/final.xclbin

# --- Hardware Generation Rules (Identical structure to the working encoder) ---

# 1. Self-Attention (SA)
build/sa_qkv/aie.mlir: matmul.py
	mkdir -p $(@D)
	python3 $< -M $(M_TGT) -K $(K_MODEL) -N $(K_HEAD) --dtype_in $(DTYPE_IN) --dtype_out $(DTYPE_OUT) \
	-m $(TILE_M) -k $(TILE_K) -n $(TILE_N) > $@
build/sa_qkv/$(KERNEL_OBJ): ../aie_kernels/aie2/mm.cc
	mkdir -p $(@D)
	cd $(@D) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2_FLAGS) $(KERNEL_DEFINES) -c ../../../aie_kernels/aie2/mm.cc -o $(@F)
build/sa_qkv/final.xclbin: build/sa_qkv/aie.mlir build/sa_qkv/$(KERNEL_OBJ)
	cd $(@D) && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
	--no-xchesscc --no-xbridge --xclbin-name=$(@F) --npu-insts-name=insts.bin aie.mlir

build/sa_sdpa1/aie.mlir: matmul.py
	mkdir -p $(@D)
	python3 $< -M $(M_TGT) -K $(K_HEAD) -N $(M_TGT) --dtype_in $(DTYPE_IN) --dtype_out $(DTYPE_OUT) \
	-m $(TILE_M) -k $(TILE_K) -n $(TILE_N) --b-col-maj 1 > $@
build/sa_sdpa1/$(KERNEL_OBJ): ../aie_kernels/aie2/mm.cc
	mkdir -p $(@D)
	cd $(@D) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2_FLAGS) $(KERNEL_DEFINES) -DB_COL_MAJ -c ../../../aie_kernels/aie2/mm.cc -o $(@F)
build/sa_sdpa1/final.xclbin: build/sa_sdpa1/aie.mlir build/sa_sdpa1/$(KERNEL_OBJ)
	cd $(@D) && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
	--no-xchesscc --no-xbridge --xclbin-name=$(@F) --npu-insts-name=insts.bin aie.mlir

build/sa_sdpa2/aie.mlir: matmul.py
	mkdir -p $(@D)
	python3 $< -M $(M_TGT) -K $(M_TGT) -N $(K_HEAD) --dtype_in $(DTYPE_IN) --dtype_out $(DTYPE_OUT) \
	-m $(TILE_M) -k $(TILE_K) -n $(TILE_N) > $@
build/sa_sdpa2/$(KERNEL_OBJ): ../aie_kernels/aie2/mm.cc
	mkdir -p $(@D)
	cd $(@D) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2_FLAGS) $(KERNEL_DEFINES) -c ../../../aie_kernels/aie2/mm.cc -o $(@F)
build/sa_sdpa2/final.xclbin: build/sa_sdpa2/aie.mlir build/sa_sdpa2/$(KERNEL_OBJ)
	cd $(@D) && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
	--no-xchesscc --no-xbridge --xclbin-name=$(@F) --npu-insts-name=insts.bin aie.mlir

# 2. Cross-Attention (CA)
build/ca_q/aie.mlir: matmul.py
	mkdir -p $(@D)
	python3 $< -M $(M_TGT) -K $(K_MODEL) -N $(K_HEAD) --dtype_in $(DTYPE_IN) --dtype_out $(DTYPE_OUT) \
	-m $(TILE_M) -k $(TILE_K) -n $(TILE_N) > $@
build/ca_q/$(KERNEL_OBJ): ../aie_kernels/aie2/mm.cc
	mkdir -p $(@D)
	cd $(@D) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2_FLAGS) $(KERNEL_DEFINES) -c ../../../aie_kernels/aie2/mm.cc -o $(@F)
build/ca_q/final.xclbin: build/ca_q/aie.mlir build/ca_q/$(KERNEL_OBJ)
	cd $(@D) && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
	--no-xchesscc --no-xbridge --xclbin-name=$(@F) --npu-insts-name=insts.bin aie.mlir

build/ca_kv/aie.mlir: matmul.py
	mkdir -p $(@D)
	python3 $< -M $(M_SRC) -K $(K_MODEL) -N $(K_HEAD) --dtype_in $(DTYPE_IN) --dtype_out $(DTYPE_OUT) \
	-m $(TILE_M) -k $(TILE_K) -n $(TILE_N) > $@
build/ca_kv/$(KERNEL_OBJ): ../aie_kernels/aie2/mm.cc
	mkdir -p $(@D)
	cd $(@D) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2_FLAGS) $(KERNEL_DEFINES) -c ../../../aie_kernels/aie2/mm.cc -o $(@F)
build/ca_kv/final.xclbin: build/ca_kv/aie.mlir build/ca_kv/$(KERNEL_OBJ)
	cd $(@D) && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
	--no-xchesscc --no-xbridge --xclbin-name=$(@F) --npu-insts-name=insts.bin aie.mlir

build/ca_sdpa1/aie.mlir: matmul.py
	mkdir -p $(@D)
	python3 $< -M $(M_TGT) -K $(K_HEAD) -N $(M_SRC) --dtype_in $(DTYPE_IN) --dtype_out $(DTYPE_OUT) \
	-m $(TILE_M) -k $(TILE_K) -n $(TILE_N) --b-col-maj 1 > $@
build/ca_sdpa1/$(KERNEL_OBJ): ../aie_kernels/aie2/mm.cc
	mkdir -p $(@D)
	cd $(@D) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2_FLAGS) $(KERNEL_DEFINES) -DB_COL_MAJ -c ../../../aie_kernels/aie2/mm.cc -o $(@F)
build/ca_sdpa1/final.xclbin: build/ca_sdpa1/aie.mlir build/ca_sdpa1/$(KERNEL_OBJ)
	cd $(@D) && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
	--no-xchesscc --no-xbridge --xclbin-name=$(@F) --npu-insts-name=insts.bin aie.mlir

build/ca_sdpa2/aie.mlir: matmul.py
	mkdir -p $(@D)
	python3 $< -M $(M_TGT) -K $(M_SRC) -N $(K_HEAD) --dtype_in $(DTYPE_IN) --dtype_out $(DTYPE_OUT) \
	-m $(TILE_M) -k $(TILE_K) -n $(TILE_N) > $@
build/ca_sdpa2/$(KERNEL_OBJ): ../aie_kernels/aie2/mm.cc
	mkdir -p $(@D)
	cd $(@D) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2_FLAGS) $(KERNEL_DEFINES) -c ../../../aie_kernels/aie2/mm.cc -o $(@F)
build/ca_sdpa2/final.xclbin: build/ca_sdpa2/aie.mlir build/ca_sdpa2/$(KERNEL_OBJ)
	cd $(@D) && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
	--no-xchesscc --no-xbridge --xclbin-name=$(@F) --npu-insts-name=insts.bin aie.mlir

# 3. FFN
build/ffn1/aie.mlir: matmul.py
	mkdir -p $(@D)
	python3 $< -M $(M_TGT) -K $(K_MODEL) -N $(FFN_HIDDEN) --dtype_in $(DTYPE_IN) --dtype_out $(DTYPE_OUT) \
	-m $(TILE_M) -k $(TILE_K) -n $(TILE_N) > $@
build/ffn1/$(KERNEL_OBJ): ../aie_kernels/aie2/mm.cc
	mkdir -p $(@D)
	cd $(@D) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2_FLAGS) $(KERNEL_DEFINES) -c ../../../aie_kernels/aie2/mm.cc -o $(@F)
build/ffn1/final.xclbin: build/ffn1/aie.mlir build/ffn1/$(KERNEL_OBJ)
	cd $(@D) && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
	--no-xchesscc --no-xbridge --xclbin-name=$(@F) --npu-insts-name=insts.bin aie.mlir

build/ffn2/aie.mlir: matmul.py
	mkdir -p $(@D)
	python3 $< -M $(M_TGT) -K $(FFN_HIDDEN) -N $(K_MODEL) --dtype_in $(DTYPE_IN) --dtype_out $(DTYPE_OUT) \
	-m $(TILE_M) -k $(TILE_K) -n $(TILE_N) > $@
build/ffn2/$(KERNEL_OBJ): ../aie_kernels/aie2/mm.cc
	mkdir -p $(@D)
	cd $(@D) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2_FLAGS) $(KERNEL_DEFINES) -c ../../../aie_kernels/aie2/mm.cc -o $(@F)
build/ffn2/final.xclbin: build/ffn2/aie.mlir build/ffn2/$(KERNEL_OBJ)
	cd $(@D) && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
	--no-xchesscc --no-xbridge --xclbin-name=$(@F) --npu-insts-name=insts.bin aie.mlir

# --- Host Program Compilation Rule ---
decoder.exe: test.cpp
	rm -rf _build && mkdir -p _build
	cd _build && cmake .. -DTARGET_NAME=decoder -DM_TGT_DIM=$(M_TGT) -DM_SRC_DIM=$(M_SRC) -DK_MODEL_DIM=$(K_MODEL)
	cd _build && cmake --build . --config Release
	cp _build/decoder $@

# --- Execution Rule (Identical to Encoder's) ---
run: decoder.exe all
	./$< \
	--xclbin_sa_qkv build/sa_qkv/final.xclbin --insts_sa_qkv build/sa_qkv/insts.bin \
	--xclbin_sa_sdpa1 build/sa_sdpa1/final.xclbin --insts_sa_sdpa1 build/sa_sdpa1/insts.bin \
	--xclbin_sa_sdpa2 build/sa_sdpa2/final.xclbin --insts_sa_sdpa2 build/sa_sdpa2/insts.bin \
	--xclbin_ca_q build/ca_q/final.xclbin --insts_ca_q build/ca_q/insts.bin \
	--xclbin_ca_kv build/ca_kv/final.xclbin --insts_ca_kv build/ca_kv/insts.bin \
	--xclbin_ca_sdpa1 build/ca_sdpa1/final.xclbin --insts_ca_sdpa1 build/ca_sdpa1/insts.bin \
	--xclbin_ca_sdpa2 build/ca_sdpa2/final.xclbin --insts_ca_sdpa2 build/ca_sdpa2/insts.bin \
	--xclbin_ffn1 build/ffn1/final.xclbin --insts_ffn1 build/ffn1/insts.bin \
	--xclbin_ffn2 build/ffn2/final.xclbin --insts_ffn2 build/ffn2/insts.bin

# --- Cleanup Rule ---
clean:
	rm -rf build _build decoder.exe
