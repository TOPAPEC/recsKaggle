Sequential recommender (SASRec) for the HSE-26 RecSys competition: 
a Transformer over each user's chronological history, trained with full-vocabulary cross-entropy to predict the top-20 next items 
Pipeline: prep.py builds interactions with a time-based split, seqdata.py makes per-user sequences, sasrec.py trains and writes the submission to run it: python src/sasrec.py --mode full --epochs 7 --d 256 --maxlen 100 --tag sas_d256).


