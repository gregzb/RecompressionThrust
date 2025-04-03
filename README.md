# Recompression - GPU Implementation using Thrust

## Background
When we think of compression, we primarily think of two forms - entropy compression and repetition compression.
* Entropy compression - letters that occur frequently should have a short representation, letters that occur rarely should have a longer representation
* Repetition compression - Phrases that reoccur are represented in terms of their previous occurrence instead of being written again from scratch.

One format for repetition compression is a context-free grammar (CFG) that only has only one possible parsing (no "OR") (otherwise known as a straight-line program (SLP)), representing runs of characters as a single node (Run-length straight line program (**RLSLP**)), and ensuring the resulting RLSLP is both height balanced and locally consistent. Recompression is a simple algorithm that outputs this format.

## Recompression

Recompression (originally introduced by Artur Jeż) is an algorithm for locally consistent run length grammar compression. [Section 3.1, TtoG, of this paper](https://arxiv.org/abs/1611.05359) (but not the original) is a good place to read a formal description of the algorithm. 

Somewhat informally, recompression consists of two phases run iteratively until the input is reduced to a single character:
* **Block Compression** - Replaces runs of consecutive identical symbols with a "block" rule
* **Pair Compression** - Replaces adjacent symbol pairs based on a selected partition with a "pair" rule

## In This Project

This project contains:
* **Naive CPU compression**
  * A naive, hash-bashed implementation of recompression with random partitioning.
* **Thrust CPU compression**
  * The thrust-ified recompression algorithm using sorting at its core instead of hashing, running on CPU.
* **Thrust GPU compression**
  * The thrust-ified recompression algorithm using sorting at its core instead of hashing, running on GPU.
* **CPU Decompression**
* **Dot graph generation**
  * Generate a dot graph from a discovered RLSLP for visualization

## Compilation and Running Prequisites
* **Git:** For cloning
* **Operating System:** Only tested on Linux, specifically Ubuntu, but I hope it works elsewhere :) (maybe CMakeLists.txt needs to be modified a liiiiittle)
* **C++ Compiler:** C++17 Compliant
* **Cuda Toolkit and nvcc:** Only tested on 12.8, probably works on 12.x
* **CMake:** Minimum version 3.10
  * argparse from github is fetched within the CMakeLists.txt
* **Graphviz:** To render generated dot graphs (optional)

## Building the project
1. Clone the repo
```
git clone <repository-url>
cd RecompressionThrust
```
2. Configure with release mode for performance
```
cmake -DCMAKE_BUILD_TYPE=Release -S . -B build
```
3. Build into the build dir
```
cmake --build build
```

## Running the project
### Compression
```
./build/Recompression compress <INPUT FILE> <OUTPUT FILE> --mode <MODE>

./build/Recompression compress einstein.en.txt.200m out.compressed --mode thrust-gpu
```
  * Available modes: `cpu`, `thrust-cpu`, `thrust-gpu`

### Decompression
```
./build/Recompression decompress <INPUT FILE> <OUTPUT FILE>

./build/Recompression decompress out.compressed out.txt
```

### Generate and visualize RLSLP dot graph
```
./build/Recompression compress <INPUT FILE> <OUTPUT FILE> --generate-debug-dot <DOT FILE>
dot -Tpng <DOT FILE> -o <PNG FILE>

./build/Recompression compress small_file_to_compress.txt out.txt --generate-debug-dot generated_graph.dot
dot -Tpng generated_graph.dot -o generated_graph.png
```

## Recommended files
Repetition compression works especially well on highly repetitive text. [Pizza&Chili](https://pizzachili.dcc.uchile.cl/repcorpus.html) has a set of highly repetitive texts.

einstein.en.txt compresses especially well

## Limitations
* As implemented in this project, it doesn't seem trivial to support streaming or external memory compression. For gpu compression, all state needs to be held in VRAM, so approximately(?) 32 to 36 times the size of the file is needed in VRAM. On an RTX 2070 Super with 8GB of RAM, I can compress 200MB files, but not 234MB files
* The compressed out could be much smaller - I represent all integers with a full 32 bits and without any entropy compression on top
* The alphabet is hardcoded to 256 chars - the chars 0 to 255, but this isn't an inherent requirement for recompression  