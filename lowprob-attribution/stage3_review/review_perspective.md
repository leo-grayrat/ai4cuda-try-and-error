# Review — Perspective (cross-domain)

**Recommendation:** Accept; the generalisation matters more than the CUDA specifics.

## The finding is not about CUDA

Strip the domain and what remains is this: the authors measured conditional token probabilities inside a multi-turn agent loop and found that the variation is governed by *what the agent has just been told*, not by *which vocabulary the domain uses*. A failed tool result raises the rate by 4.6×; a successful one lowers it below the baseline; and the domain-specific vocabulary is the *most* certain part of the stream.

That is a statement about agentic RL as a context-distribution problem, and it should transfer. Every tool-using agent that generates structured artefacts — SQL, LaTeX, CAD scripts, configuration languages, HDL — presents the same structure: a domain whose surface vocabulary is templated, wrapped in a loop whose failure messages are out-of-distribution text. If the pattern holds, then "our domain's tokens are rare" is unlikely to be the binding constraint in any of them, and the productive question becomes what the environment says when it says no.

## What would falsify the generalisation

The paper's tasks are reductions, and reduction kernels are unusually templated. A domain where the surface vocabulary is *not* templated — natural-language-heavy generation, or code whose structure is chosen rather than recalled — might behave differently. The domain reviewer raises this (§task scope) and I agree it is the main boundary.

There is a sharper test available: keep the task fixed and vary the *feedback*. The paper truncates the text and finds no effect. That suggests the state (failure) rather than the string is doing the work. If a third condition gave a *different* failure message of the same length — e.g. "wrong answer on shape 2 of 3" instead of a compile error — and the rate differed, that would localise what about failure matters. The current design cannot separate "the model knows it failed" from "the model sees diagnostics".

## On the negative intervention result

The truncation result is more useful outside CUDA than inside it. Compressing tool output is a standard cost measure across agent systems, and the paper shows it does not shrink the uncertainty tail in this setting. That is a concrete cost/benefit datum for anyone building such a pipeline. It deserves more prominence than a subsection.

## On the retracted first pass

Including it is unusual. It is also the most transferable part of the paper for anyone doing measurement work: the failure was not a coding error but a *definitional* one, where a reasonable exclusion ("don't mislabel English prose as code") silently narrowed the measured class by 96%. The general lesson — vary the definition boundary and see whether the conclusion survives — is domain-independent.

## Style

The paper is dense but readable. §5.5 could be tightened; the narrative of the mistake runs longer than the technical content requires. Conversely, §5.3 (why context matters) is the most compressed section and deserves more space, since it carries the constructive half of the argument.

## Judgment

Accept. I would encourage the authors to frame the contribution in the abstract as a statement about agentic RL context distribution, with CUDA as the measured instance, rather than as a CUDA-specific result. The latter undersells it.
