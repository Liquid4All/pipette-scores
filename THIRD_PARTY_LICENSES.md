# Third-party licenses

`pipette-scores` vendors and adapts code from the projects below. Each entry's
verbatim license text follows it.

The vendored `vendor/ifstruct` validator is Liquid AI's own
(`Liquid4All/ifstruct`) — no third-party license; it is covered by this
repository's license.

---

## allenai/IFBench — Apache-2.0

Vendored as the `vendor/ifbench` git submodule — source
`evaluation_lib.py`, `instructions.py`, `instructions_registry.py`,
`instructions_util.py`, with the license at `vendor/ifbench/LICENSE`. At build
time these are bundled into the wheel under
`pipette_scores/scoring/ifbench/_upstream/` (including `LICENSE`).
Source: https://github.com/allenai/IFBench

Licensed under the Apache License, Version 2.0 (SPDX: `Apache-2.0`); full text
in `vendor/ifbench/LICENSE`. Upstream ships no `NOTICE` file. You may obtain a
copy of the License at http://www.apache.org/licenses/LICENSE-2.0.

IFBench data note: `packages/pipette-scores/scripts/build_ifbench_dataset.py`
materializes rows from `vendor/ifbench/data/IFBench_test.jsonl`, pinned by the
`vendor/ifbench` submodule commit. Upstream identifies the dataset license as
Open Data Commons Attribution License v1.0 (SPDX: `ODC-By-1.0`):
https://opendatacommons.org/licenses/by/1-0/. Upstream states that the data is
intended for research and educational use in accordance with Ai2's Responsible
Use Guidelines: https://allenai.org/responsible-use.

---

## openai/prm800k — MIT

Vendored: `packages/pipette-scores/pipette_scores/scoring/math_500/prm800k/grading/grader.py`,
`packages/pipette-scores/pipette_scores/scoring/math_500/prm800k/grading/__init__.py`
Source: https://github.com/openai/prm800k

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
