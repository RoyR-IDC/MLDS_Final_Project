Paper todos:

You are an expert data scientist and researcher, specializing in writing professional academic papers in the fields of ML and DS and CV.

Update the paper according to the following todos that are split by sections.

for each TODO completed - mark the TODO here from "- [ ]" to "- [V]"

Abstract:

- [ ] do a VERY MINIMAL update (because it is currently very good as it is) - talk about the 3 models being from 3 different compared families - CNN, transformers, MLP ...

Introduction:

- [ ] do a VERY MINIMAL update (because it is currently very good as it is) - talk about the 3 models being from 3 different compared families - CNN, transformers, MLP ...


Related work:

- [ ] switch sub sections 2.5 and 2.6 for better logical reading flow
<!-- - [ ] on hold: if fixes needed for this part - make sure that 2.3, 2.4, 2.5 all have 3 parts - general, this model, expected effect in spatial disruption -->


Data Overview:

- [ ] split into sub sections in a similar manner as in section 4 (methodology)

- [ ] explain how data is resized, to what size. (in future sections, maybe talk about what the resize what affect)

- [ ] rephrase "use the same 4 × 3 grid-permutation matrix" as it is confusing with the grid sizes themselvs.

- [ ] (!) show example fo the grids and permutations - same as is the hardness section (and also there - need to put images in 3 rows) (this might require updating and re-running notebook part 3)


Methodology:

- [ ] section 4.1: 

    - start by talking about the 3 model families, and by choosing models so that all 3 have the same pretraining data

    - section 4.1: mention the 3 reviewer controoled experiments

- [ ] section 4.2: 

    - "The generator is controlled" rephrase, as generator is not used before.

    - instead of saying why gloabl reversal matters, explain what it is.

- [ ] section 4.3: 

    - move table 2 to the additional infomration section

    - model figures need to improve - figure 1, figure 2, figure 3. these are not good enough. should be more informative. using existing images from the original papers of these models if possible

    - dont talk about what os rejected , dont talk about 21k to 1k. assume that if we mention we use 1k, then we actually do as we said

- [ ] section 4.3: 
    
    - the part about the control and zero short etc. should be moved to anothe section. optimizer, loss, model HPs shpuld be discussed 

- [ ] section 4.4: 

    - move tables 3, table 4 to additional information

    - update part 2 epochs - all part 2 experimetns now use 30 epochs

    - move the sction about how images where normalized to 4.2 or to a new sction that talks about how data is processed. (if not in 4.2, then make a new section before 4.2).

    - move the section that talks about how regular augmentations are performed to 4.5 or to a dedicated new section

- [ ] section 4.5: 

    - only keep formulas where it cannot be inferred from text \ where it is easier to understand



Results:

- [ ] show all main experiment graphs. missing: 2 graphs from part 1 reviewer controlled experiment reuests  (should be referenced in 7.1). also missing, all graphs of part 3 hardness metrics

- dont show 3d graphs, not even in additional info

- the results themselves are not written in texts or analyzed.

- need to updare results


Discussion:

- make sure the general concepts of the introduction are mentioned here as well.


References:

- make sure links are valid and not broken.


General:

- [ ] make sure only latest results used


