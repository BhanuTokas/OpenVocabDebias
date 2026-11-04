# Blind concept vocabularies

Candidate phrases for naming discovered directions, written by Claude Sonnet
agents with no project context, no dataset name and no results.  Each list is
one agent's output, lowercased and de-duplicated, with nothing else changed.
The agents were told not to use tools, and their transcripts show none used
apart from returning the list.

Generated 2026-11-04, three independent runs per prompt.  Every prompt began:
"Do not use any tools and do not read any files. Answer only from your own
reasoning, in your final message."

`birds_run{1,2,3}.txt`

    An image classifier is trained on a dataset of photographs of birds to
    classify the type of bird.

    Classifiers often rely on things other than the intended task. Write a list
    of 200 short visual concepts (1 to 4 words each) that could appear in images
    like these and that such a classifier might rely on by mistake. Cover
    anything visible in an image, as broadly as you can. Do not include bird
    species names.

    Output one concept per line, lowercase, with no numbering, headings or
    commentary.

`faces_run{1,2,3}.txt`: the same prompt, with the first sentence "An image
classifier is trained on a dataset of photographs of celebrity faces to classify
whether a person's hair is blond." and "Do not include hair colour words."

`generic_run{1,2,3}.txt`

    Image classifiers often rely on things other than their intended task. Write
    a list of 400 short visual concepts (1 to 4 words each) that could bias an
    image classifier, for any kind of image dataset. Cover anything visible in
    an image, as broadly as you can: no particular domain or dataset is intended.

    Output one concept per line, lowercase, with no numbering, headings or
    commentary.

The agents may recognise a benchmark from the task description, so these lists
are blind to this project, not to the literature.
