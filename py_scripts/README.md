
## Pre-trained model comparisons

- **EczemaNet_VGG16**: Full-suite EczemaNet with pretrained [VGG16](https://arxiv.org/pdf/1409.1556.pdf) (ImageNet) as feature extractor. (# of parameters: 138,357,544)
- **EczemaNet_VGG19**: Full-suite EczemaNet with pretrained [VGG19](https://arxiv.org/pdf/1409.1556.pdf) (ImageNet) as feature extractor. (# of parameters: 143,667,240)
- **EczemaNet_ResNet50**:  Full-suite EczemaNet with pretrained [ResNet50](https://arxiv.org/abs/1512.03385) (ImageNet) as feature extractor. (# of parameters: 25,636,712)
- **EczemaNet_MobileNet**: Full-suite EczemaNet with pretrained [MobileNet](https://arxiv.org/abs/1704.04861) (ImageNet) as feature extractor. (# of parameters: 4,253,864)
- **EczemaNet_InceptionV3**: Full-suite EczemaNet with pretrained [InceptionV3](https://arxiv.org/abs/1512.00567) (ImageNet) as feature extractor. (# of parameters: 23,851,784)

---

## Eczemanet Model training and Sign dependence study

- **EczemaNet**:  Full-suite EczemaNet with pretrained <CNN> (ImageNet) as feature extractor. Feature selector decided based on performance shown above.
- **EczemaNet_SD**: Full-suite Eczemanet, but only using a  single  network (FC Block) for all disease signs instead of separate blocks. The idea is that the Sign Dependence (SD) can be learned.
- **EczemaNet_FSD**: Full-suite EczemaNet with Fully-Sign-Dependence (FSD). Additional FC layer connecting pre-existing output, to try to learn the dependencies between signs for more accurate prediction.
- **EczemaNet_ASD**: Full-suit EczemaNet with Autoregressice-Sign-Dependence (ASD). Interconnecting FC layers in a sequential manner using pre-existing medical knowledge, to map the dependencies between signs.

---

## Baseline studies (Ablations)

- **Baseline_Init**: Full-suite Eczemanet with random initialised <CNN> as feature extractor.
- **Baseline_Manual**: Full-suite Eczemanet with a smaller, manually labeled dataset only.
- **Baseline_Categorical**: Full-suite Eczemanet with categorical instead of ordinal outputs.
- **Baseline_Simple_Ensemble**: Full-suite Eczemanet with simple ensemble, instead of convolutional sum for combined score.
- **Baseline_Whole_Image**: Full-suite Eczemanet trained with whole image, instead of image crops.
- **Baseline_Univariate_EASI**: Full-suite Eczemanet with EASI as solo output, ordinal encoding.
- **Baseline_Univariate_SASSAD**: Full-suite Eczemanet with SASSAD as solo output, ordinal encoding.
- **Baseline_Univariate_SCORAD**: Full-suite Eczemanet with SCORAD as solo output, ordinal encoding.
- **Baseline_Univariate-Regression_EASI**: Full-suite Eczemanet with EASI as solo output, regression problem.
- **Baseline_Univariate-Regression_SCORAD**: Full-suite Eczemanet with SASSAD as solo output, regression problem.
- **Baseline_Univariate-Regression_SASSAD**: Full-suite Eczemanet with SCORAD as solo output, regression problem.
- **Baseline_Whole_Image-Univariate-Regression_EASI**: Full-suite Eczemanet trained on whole image, and with EASI as solo output - regression problem.
- **Baseline_Whole_Image-Univariate-Regression_SASSAD**: Full-suite Eczemanet trained on whole image, and with SASSAD as solo output - regression problem.
- **Baseline_Whole_Image-Univariate-Regression_SCORAD**: Full-suite Eczemanet trained on whole image, and with SCORAD as solo output - regression problem.















