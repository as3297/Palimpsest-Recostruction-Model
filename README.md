# Palimpsest-Recostruction-Model
A palimpsest is a historical manuscript in which the original text (termed undertext) was erased and overwritten with another script in order to recycle the parchment. Here we present our model for palimpsest text reconstruction. 
![Results of Archimedes palimpsest reconstruction](/images/generated_archi.png) Fig.1.Results of Archimedes palimpsest reconstruction.
### Prerequisite
* Python =\>3.5
* Tensorflow 1.12

## How to run the project 
- To start the inverse network for Archimedes palimpsest:
	1) Get or train weights for undertext and background generator.
	2) Set the `RESTOREDPATH_B` to background model file path and `RESTOREDPATH_UTB` to undertext model in bash script `archi/finv_subgraphs.sh`  
	3) Set `LOGDIR` to desired directory for the results
	4) Set the other parameters according to the instruction at the bottom of the file `archi/finv_subgraphs.py` 
	5) Run bash script `bash archi/finv_subgraphs.sh`.
- To train the undertext generator:
	1) Run file `archi/undertext_gen_train.py`, with `mode` set to \`utb`.