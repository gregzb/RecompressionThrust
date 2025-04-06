# Recompression - GPU Implementation using Thrust

This is a GPU-accelerated implementation of a grammar-based compression algorithm, Recompression, using the Thrust library. This tool is designed for quickly compressing highly repetitive texts in a format that can we can later construct indices on. The goal of this implementation is to prove that a relatively quick GPU implementation exists and not to have particularly small output sizes.

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
  * Approx ~11MB/s on my i7 6700
* **Thrust CPU compression**
  * The thrust-ified recompression algorithm using sorting at its core instead of hashing, running on CPU.
  * Approx ~10MB/s on my i7 6700
* **Thrust GPU compression**
  * The thrust-ified recompression algorithm using sorting at its core instead of hashing, running on GPU.
  * Approx ~200MB/s on my RTX 2070 Super
* **CPU Decompression**
* **Dot graph generation**
  * Generate a dot graph from a discovered RLSLP for visualization

## Compilation and Running Prerequisites
* **Git:** For cloning
* **Operating System:** Only tested on Linux, specifically Ubuntu, but I hope it works elsewhere :) (maybe CMakeLists.txt needs to be modified a little)
* **C++ Compiler:** C++17 Compliant
* **Cuda Toolkit and nvcc:** Only tested on 12.8, probably works on 12.x
* **CMake:** Minimum version 3.10
  * argparse from github is fetched within the CMakeLists.txt
* **Graphviz:** To render generated dot graphs (optional)

## Building the Project
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

## Running the Project
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

### Generate and Visualize RLSLP Dot Graph
```
./build/Recompression compress <INPUT FILE> <OUTPUT FILE> --generate-debug-dot <DOT FILE>
dot -Tpng <DOT FILE> -o <PNG FILE>

./build/Recompression compress small_file_to_compress.txt out.txt --generate-debug-dot generated_graph.dot
dot -Tpng generated_graph.dot -o generated_graph.png
```

## Recommended Files
Repetition compression works especially well on highly repetitive text. [Pizza&Chili](https://pizzachili.dcc.uchile.cl/repcorpus.html) has a set of highly repetitive texts.

einstein.en.txt compresses especially well

## Example Dot Graph
This is what the RLSLP for this text looks like:
```
abcdeabcdabcdabcd abcdeabcd
```
![Image](https://github.com/user-attachments/assets/2874008f-1780-480d-ae51-23e838c251c1)

**Pair** nodes have exactly two edges that point at different nodes. **Block** nodes have two or more edges that point at the same node.

Listing out the leaf nodes of the inorder traversal start from the root (traversing down the edges of each node from left to right) of the directed acyclic graph yields the original text (decompression).

Green nodes are nodes with in degree greater than one and are a visual indication for where "compression" is happening.

Since the alphabet size is currently hardcoded to 256, all of the internal nodes start at 256 and then count up.

## Why Are The Compressed Files So Big?
* Fixed size integers
* No entropy compression
* We store the level of each node, which are really only needed for some indices built on top of RLSLPs
  * And they could technically just be reconstructed from the RLSLP
* Recompression produces locally consistent RLSLPs - a stronger guarantee than just having some arbitrary RLSLP

None of these are algorithm limitations, just limitations of this particular implementation

## Limitations
* **Memory Usage:** GPU compression requires state to be held in VRAM. Approximately 32 to 36 times the file size in VRAM is needed.
  * It may be possible to reduce this, or make it support streaming or external memory efficiently.
  * On an RTX 2070 Super with 8GB of RAM, I can compress 200MB files, but not 234MB files
* **Compression Efficiency:** The current implementation uses full 32-bit integers without additional entropy compression, so the output is much larger than it theoretically could be
* **Alphabet:** The alphabet is hardcoded to 256 chars (0-255)
  * Not an inherent limitation of recompression 