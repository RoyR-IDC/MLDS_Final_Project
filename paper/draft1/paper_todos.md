Paper todos:

You are an expert data scientist and researcher, specializing in writing professional academic papers in the fields of ML and DS and CV.

Update the paper according to the following todos that are split by sections.

for each TODO completed - mark the TODO here from "- [ ]" to "- [V]"

Abstract:

- [V] better wording for abstract beginning "Spatial structure is central to natural-image classification, but pretrained models may retain class-discriminative information even when that structure is disrupted. This report studies binary Dogs-versus-Cats classification under fixed tile-wise permutations, motivated by structured-permutation hardness questions but evaluated as an empirical computer-vision testbed."

- [V] improve wording or delete: "Across the deterministic matrix, models remain far above chance, while accuracy decreases as tile count and permutation disruption increase"

- [V] consider removing "These findings are specific to fixed deterministic permutations and frozen pretrained representations. They do not establish that the models reconstruct or invert the permutation; high accuracy may also reflect surviving local texture, object-part, background, or other class-discriminative cues"

Introduction:

- [V] improve the paper contributions section.

Related work:

- [V] make sure that the links to the references work. make sure that they also point to real existing papers

- [V] there seems to not be a lot of references - try to add more (but dont invent - add only existing ones).

- [V] in 2.3 - the wording "jigsaw tasks" is first introduced. it should be renamed to o

- [V] 2.5 should be the first one in this section

- [V] missing a section on the models from part 1 (need to write short explanations on each of them,a )


Data Overview:

- [V] no need to mention that labels are infered from filenames. also no need to mention what data is not used , and what folder is used to keep the data

- [V] it is important to state how many images are in train, how many are in validation (and the ratios between them). it is also important to state the difference in class ratios in train and validation (should be 50%-50% in both train and validation). 

- [V] regarding the 2 above tasks - consider moving section "4.1" or at least some of it into section "3". the same for section "4.2" - consider moving it or some of it into section "3".

- [V] it is important to state original image sizes

- [V] when refering to tiles, call them alsp AxB (16 tiles = 4x4) etc.

- [V]  explain why there are 3 levels of permutations easy medium hard, and explain the differences between them.

- [V] the final phrase in tis section could be updated. we only write about what we have done.


Methodology:

- [V] in the opening (before 4.1 starts) need to make some specifics more general - no need to state which folders are used for output, no need to state how many notebooks are used. we do however need to mention that the work was split into 3 parts as requested .

- [V] table 1 can move to additional information

- [V] when first introducing resnet18, add a small plot \ image with its overview of architechture

- [V] make sure all explanation are scientifically sound

- [V] need to add section about the optimizer, hyper parameters used, etc - for each part (1,2,3 . parts 2 and 3 share much of config)

- [V] section 4.6 can move outside of methodology to somewhere else.

Results:

- [V] do an intro in section 9, and then split it into 9.1, 9.2, 9.3 for the 3 different parts of the assignment

- [V] in table 9 it is unclear if "best" is refered to as the end of part 1 training (without imporvements) or "best" after all improvemens affecrs


General:

- [V] if possible - add links to tables and graphs mentioned in the paper (not only links for references)

- [V] try to sound less like an LLM agent trying to get user approbal - but like a polished data scientist researcher trying to write the best scientific paper
