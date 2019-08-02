# My_Attention_Seqnet
An Automatic Compiled Attention Seq2Seq Neural Machine Translation Algorithm Model.
    
Usage:

+    For training:
    
``$python train.py --corpora=[parellel_corpora_path] --maxlen=[list_of_two_integers_input and output_max_lengths]``
    
Both 'corpora' and 'maxlen' can't be empty.

+    For testing:
``$python test.py --text=[text_for_translation] --outdir=[output_path] --loadmodel=[load_compiled_models]``

Both 'text' and 'outdir' can't be empty. Lengths of text sentences should not overpass the 'maxlen'. 

For dowloading corpora file, you can try querying on this website: <http://opus.nlpl.eu/bin/opuscqp.pl>
