# EczemaNet: Automating Detection and Severity Assessment of Atopic Dermatitis

Atopic dermatitis (AD), also known as eczema, is one of the most common chronic skin diseases. AD severity is primarily evaluated based on visual inspections by clinicians, but is subjective and has large inter- and intra-observer variability in many clinical study settings.

To aid the standardisation and automating the evaluation of AD severity, a CNN computer vision pipeline that first detects areas of AD from photographs, and then makes probabilistic predictions on the severity of the disease is developed. EczemaNet combines transfer and multitask learning, ordinal classification, and ensembling over crops to make its final predictions. EczemaNet was tested using a set of images acquired in a published clinical trial, and demonstrate low RMSE with well-calibrated prediction intervals. The effectiveness of using CNNs for non-neoplastic dermatological diseases with a medium-size dataset, and their potential for more efficiently and objectively evaluating AD severity, which has greater clinical relevance than classification.

## Publication

If you use EczemaNet in your research, please cite our MIML '20 paper:

``` text
@inproceedings{eczemaNet2020,
  author = {Kevin Pan, Guillem Hurault, Kai Arulkumaran, Hywel Williams and Reiko J. Tanaka},
  title = {EczemaNet: Automating Detection and Severity Assessment of Atopic Dermatitis},
  journal={MLMI: International Workshop on Machine Learning in Medical Imaging},
  year = {2020}
}
```

## License

This open source version of EczemaNet is licensed under the GPLv3 license, which can be seen in the [LICENSE](/LICENSE) file.

A **closed source** version of EczemaNet is also available without the restrictions of the GPLv3 license with a software usage agreement from Imperial College London. For more information, please contact Diana Yin <d.yin@imperial.ac.uk>.

``` text
EczemaNet: Automating Detection and Severity Assessment of Atopic Dermatitis
Copyright (C) 2020  Kevin Pan <kevin.pan18@imperial.ac.uk>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
```
