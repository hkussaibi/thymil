# AI-Enhanced Subtyping of Thymic Tumors: Attention-based MIL with Pathology-Specific feature extraction
# Methodology
![flowChart](https://github.com/user-attachments/assets/054f6e7d-ef1b-4b3a-984c-afde38f5b197)

## step1 Collection and Selection of WSIs from TCGA repositories:
Histopathology WSIs of 242 thymic epithelial tumors were obtained from the publicly accessible TCGA database (https://www.cancer.gov/tcga). These WSIs varied in size, ranging from 51 MB to 3.4 GB. The WSIs were categorized into six distinct subtypes/classes, with slide-level labels provided.
## step2 Patch Extraction:
A customized [Yottixel](https://github.com/KimiaLabMayo/yottixel) script, originally developed by KimiaLab, was adopted to extract patches from tissue areas in the WSIs.

## step3 Feature Extraction:
The following script has been developed to extract patches from WSIs, using the above Yottixel function, then to extract features from the produced patches on the fly (withous saving).
Three different models were employed to extract features:
### ResNet50
### [HistoEncoder](https://github.com/jopo666/HistoEncoder)
### [Phikon](https://github.com/owkin/HistoSSLscaling)
## step4 Construction of Features Bags:
## Classification Models:
### [TransMIL](https://github.com/szc19990412/TransMIL)
### [Chowder](http://arxiv.org/pdf/1802.02212)
### [AttenMIL©]()
## step5 Training:
## step6 Evaluation:

For more information, see the original study: [10.1101/2024.06.07.24308609](https://doi.org/10.1101/2024.06.07.24308609 ).

If you wish to reuse any of the codes mentioned above, please ensure to cite the original manuscript accordingly.

```bibtex
@article {Kussaibi2024.06.07.24308609,
	author = {Kussaibi, Haitham},
	title = {AI-Enhanced Subtyping of Thymic Tumors: Attention-based MIL with Pathology-Specific Feature Extraction},
	elocation-id = {2024.06.07.24308609},
	year = {2024},
	doi = {10.1101/2024.06.07.24308609},
	publisher = {Cold Spring Harbor Laboratory Press},
	URL = {https://www.medrxiv.org/content/early/2024/08/17/2024.06.07.24308609},
	eprint = {https://www.medrxiv.org/content/early/2024/08/17/2024.06.07.24308609.full.pdf},
	journal = {medRxiv}
}
```
© 2024 anapath.org This code is made available under the Apache-2 License and is available for non-commercial academic purposes.
