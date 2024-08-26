# AI-Enhanced Subtyping of Thymic Tumors: Attention-based MIL with Pathology-Specific feature extraction
# Methodology
![flowChart](https://github.com/user-attachments/assets/054f6e7d-ef1b-4b3a-984c-afde38f5b197)

## Collection and Selection of WSIs from TCGA repositories:
Histopathology WSIs of 242 thymic epithelial tumors were obtained from the publicly accessible TCGA database (https://www.cancer.gov/tcga). These WSIs varied in size, ranging from 51 MB to 3.4 GB. The WSIs were categorized into six distinct subtypes/classes, with slide-level labels provided.
## Patch Extraction:
A customized Yottixel script, originally developed by KimiaLab, was adopted to extract patches from tissue areas in the WSIs.
## Feature Extraction:
Three models were employed to extract features from the patches on the fly: ResNet50, HistoEncoder, and Phikon.
## Construction of Features Bags:
This approach involves grouping features into bags of uniform size, each containing 200 features/instances using (torch.chunk) function.
## Classification Models:
AttenMIL, TransMIL, and Chowder were trained for classification.
## Training Loop:
The training loop was implemented in PyTorch. It includes the following techniques: 
- ClassWeightedCrossEntropyLoss
- Adam Optimizer with learning rate scheduler
- Early Stopping
- Cross Validation


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
