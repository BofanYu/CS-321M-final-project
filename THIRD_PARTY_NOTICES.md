# Third-Party Notices and Adapted Code

This project is primarily original code for the CS 321M final project. The only
vendored third-party source code is the small local `pyrecodes` subset under:

```text
.tmp_pyrecodes_main/pyrecodes/
```

## pyrecodes

- Package: `pyrecodes`
- Version marker in local subset: `0.3.0`
- Authors listed by the package: Nikola Blagojevic and Bozidar Stojadinovic
- Source/project page: https://github.com/NikolaBlagojevic/pyrecodes
- PyPI page: https://pypi.org/project/pyrecodes/
- License: BSD-3-Clause

The local copy is not the full upstream package. It keeps only the files needed
for the household survey LLM experiment:

```text
pyrecodes/utilities.py
pyrecodes/household/convert_survey_to_household_info.py
pyrecodes/household/household_gpt_base.py
pyrecodes/household/household_survey_gpt.py
pyrecodes/household/household_survey_gpt_prompts.json
pyrecodes/household/literature_summaries.json
```

Project-specific modifications include:

- a local OpenAI-compatible Ollama backend for experiments with local models;
- context-only prompt handling for local models that do not fully support the
  OpenAI developer-message behavior;
- retry/error handling around LLM calls;
- more robust parsing of final displacement decision labels from model output;
- prompt templates and ruleset/literature context for the binary household
  displacement-duration task;
- lazy optional dependency handling for geometry utilities not used by this
  workflow.

Code files adapted from `pyrecodes` are marked with module-level comments or
docstrings at the top of the file.

## Runtime Dependencies

The project depends on third-party Python packages listed in `requirements.txt`.
Their source code is not vendored in this repository. License terms for those
packages are provided by their installed distributions and project pages.

## BSD 3-Clause License Text

The following BSD-3-Clause license text applies to the adapted `pyrecodes` code
included in `.tmp_pyrecodes_main/pyrecodes/`. The local upstream files did not
include a separate copyright header; package metadata lists Nikola Blagojevic
and Bozidar Stojadinovic as authors.

```text
BSD 3-Clause License

Copyright (c), pyrecodes contributors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
