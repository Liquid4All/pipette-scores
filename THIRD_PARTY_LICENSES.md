# Third-party licenses

`pipette-scores` vendors and adapts code from the projects below. Each entry's
verbatim license text follows it.

The vendored `vendor/ifstruct` validator is Liquid AI's own
(`Liquid4All/ifstruct`) — no third-party license; it is covered by this
repository's license.

---

## allenai/IFBench — Apache-2.0

Two copies are vendored:

- `vendor/ifbench` — the upstream repository as a git submodule, used only for
  its test set. License at `vendor/ifbench/LICENSE`.
- `vendor/ifbench-checkers` — the scoring source (`evaluation_lib.py`,
  `instructions.py`, `instructions_registry.py`, `instructions_util.py`) at
  upstream commit `1a902f9`, with Liquid AI modifications to
  `instructions.py`, `instructions_util.py` and `evaluation_lib.py`. Each
  modified file carries a notice, and the changes are listed in
  `vendor/ifbench-checkers/README.md`. License at
  `vendor/ifbench-checkers/LICENSE`. At build time these files are
  bundled into the wheel under `pipette_scores/scoring/ifbench/_upstream/`
  (including `LICENSE`).

Source: https://github.com/allenai/IFBench

Licensed under the Apache License, Version 2.0 (SPDX: `Apache-2.0`); full text
in `vendor/ifbench/LICENSE` (identical in both copies). Upstream ships no `NOTICE` file. You may obtain a
copy of the License at http://www.apache.org/licenses/LICENSE-2.0.

IFBench data note: `packages/pipette-scores/scripts/build_ifbench_dataset.py`
materializes rows from `vendor/ifbench/data/IFBench_test.jsonl`, pinned by the
`vendor/ifbench` submodule commit. Upstream identifies the dataset license as
Open Data Commons Attribution License v1.0 (SPDX: `ODC-By-1.0`):
https://opendatacommons.org/licenses/by/1-0/. Upstream states that the data is
intended for research and educational use in accordance with Ai2's Responsible
Use Guidelines: https://allenai.org/responsible-use.

---

## open-compass/VLMEvalKit — Apache-2.0

Adapted, with Liquid AI modifications, from VLMEvalKit at commit `8eca8e5`
(each file names the upstream file and functions it ports, and how it
departs from them):

- `packages/pipette-scores/pipette_scores/scoring/vlm_upstream/` — answer
  matching (`matching_util.py`), yes/no extraction and the MME and POPE
  ratings (`yorn.py`), multiple-choice helpers and MMMU preprocessing
  (`multiple_choice.py`), OCRBench scoring (`ocrbench.py`). The main change:
  no LLM judge and no random choice, an answer the matcher cannot read
  scores as wrong. Listed in `PORTING_NOTES.md` there.
- `packages/pipette-scores/pipette_scores/scoring/profile_answers.py` — the
  MiniCPM-V wrapper's `extract_answer`.
- `packages/pipette-scores/pipette_scores/dataset_catalog/prompt_profiles.py`
  — the prompts VLMEvalKit's model wrappers build, where a profile cites one.
- `packages/pipette-scores/pipette_scores/dataset_catalog/_vlm_build.py`,
  `_vlm_common.py` and `packages/pipette-scores/scripts/build_*_dataset.py`
  for the seven VLM evals — the datasets' default prompts and the TSV
  sources and MD5s VLMEvalKit pins.

Source: https://github.com/open-compass/VLMEvalKit

Copyright 2023 VLMEvalKit Authors.

Licensed under the Apache License, Version 2.0 (SPDX: `Apache-2.0`); the full
text is the same as IFBench's, in `vendor/ifbench/LICENSE`. Upstream ships no
`NOTICE` file. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0.

---

## openai/prm800k — MIT

Vendored: `packages/pipette-scores/pipette_scores/scoring/math_500/prm800k/grading/grader.py`,
`packages/pipette-scores/pipette_scores/scoring/math_500/prm800k/grading/__init__.py`
Source: https://github.com/openai/prm800k

The sympy grader in
`packages/pipette-scores/pipette_scores/scoring/math_500/liquid_evals_parser.py`
(`grade_answer_sympy` and its helpers) derives from this grader and reaches this
package vendored via Liquid4All/liquid_evals.

Copyright (c) 2023 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## hendrycks/math (MATH `math_equivalence`) — MIT

`packages/pipette-scores/pipette_scores/scoring/math_500/prm800k/grading/math_normalize.py`
is, per its own header, "largely copied from the Hendrycks' MATH release
(math_equivalence)", and reaches this package vendored via prm800k.
`mathd_normalize_answer` and `_strip_string` in
`packages/pipette-scores/pipette_scores/scoring/math_500/liquid_evals_parser.py`
("Dan Hendrycks' code") reach it vendored via Liquid4All/liquid_evals.
Source: https://github.com/hendrycks/math

Copyright (c) 2021 Dan Hendrycks

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## groq/openbench — MIT

The robust-boxed numeric normalization (`normalize_numeric_answer`) in
`packages/pipette-scores/pipette_scores/scoring/math_500/math_generic.py` is copied from
openbench `robust_boxed.py`.
Source: https://github.com/groq/openbench

Copyright (c) 2025 Groq, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## huggingface/datasets — Apache-2.0

Adapted: `resample_16k_mono` in
`packages/pipette-scores/pipette_scores/dataset_catalog/_asr_audio.py` follows the
decoding sequence of `datasets` 2.19.0's `Audio.decode_example`
(`src/datasets/features/audio.py`: soundfile read, transpose, `librosa.to_mono`,
`librosa.resample` to the feature's `sampling_rate`), which the Open ASR
Leaderboard applies to every model's audio, so the ASR evals resample as it does
without depending on `datasets` 2.x or librosa.
Source: https://github.com/huggingface/datasets/blob/2.19.0/src/datasets/features/audio.py

Copyright The HuggingFace Datasets Authors. Licensed under the Apache License,
Version 2.0 (SPDX: `Apache-2.0`); full text in
[Apache License 2.0](#apache-license-20) below.

---

## librosa/librosa — ISC

Adapted: the same function follows librosa's `to_mono`, `resample` with its
default `res_type="soxr_hq"` (soxr, then `util.fix_length` to
`ceil(n * target_sr / orig_sr)` samples) and `util.fix_length`.
Source: https://github.com/librosa/librosa

ISC License

Copyright (c) 2013--2026, librosa development team.

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

---

## huggingface/open_asr_leaderboard — Apache-2.0

Vendored verbatim (byte for byte):
`packages/pipette-scores/pipette_scores/scoring/asr/open_asr_normalizer/normalizer.py`,
`packages/pipette-scores/pipette_scores/scoring/asr/open_asr_normalizer/english_abbreviations.py`
Source: https://github.com/huggingface/open_asr_leaderboard at
`67e8bd6acea240819ad67080f6f31e15d4a90da5` (`normalizer/`).

Vendored as unchanged excerpts:
`packages/pipette-scores/pipette_scores/scoring/asr/open_asr_normalizer/multilingual.py`
holds `FILLER_WORDS` and `MultilingualNormalizer` (`normalizer/data_utils.py`)
and `normalize_compound_pairs` (`normalizer/eval_utils.py`) at
`62a457c5c32d0de4df77e9470308aae7d45b055f`, with Liquid AI's header, imports and
an Armenian stand-in (`scoring/asr/NOTICE.md`).

Copyright 2022 The OpenAI team and The HuggingFace Team. Licensed under the
Apache License, Version 2.0 (SPDX: `Apache-2.0`); full text in
[Apache License 2.0](#apache-license-20) below. Upstream ships no `NOTICE` file.
Per its header, most of the code is copied from openai/whisper (MIT, see
[openai/whisper](#openaiwhisper--mit)).

---

## openai/whisper — MIT

The origin of the Whisper-derived normalizer above
(`open_asr_normalizer/normalizer.py`), whose header says most of its code is
copied from it (`whisper/normalizers/`). Source: https://github.com/openai/whisper

MIT License

Copyright (c) 2022 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Apache License 2.0

The license of huggingface/datasets and huggingface/open_asr_leaderboard above
(the same text as this repository's `LICENSE`):

```text
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```
