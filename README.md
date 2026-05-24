# COS700-XAI-In-Cancer-Medicine-Treatment
Diagnosing XAI fragility in cancer drug response prediction: a RED XAI pipeline using SAM regularisation and evolutionary rule extraction on genomic data.

This project implements a controlled RED XAI diagnostic pipeline that uses the generalisation gap as an indicator of explanation fragility. Trains DeepCCDS on GDSC data under Adam (baseline) and SAM (intervention), extracts symbolic if-then rules via GA and NSGA-II, and evaluates rule fidelity, comprehensibility, and biological validity using PROMETHEE II and GSEA pathway enrichment.
