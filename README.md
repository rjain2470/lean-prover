# RAG-prover :zap:

A Retrieval Augmented Generation (RAG) stack built atop DeepSeek-Prover-V2 7B, a lightweight LLM designed to produce formalized proofs of mathematical statements in Lean.

## About The Project 📝
We import the full contents of mathlib4, a massive collection of formalized proofs in Lean, and use OpenAI's <code>text-embedding-3-large</code> model to textually embed these statements into Euclidean space. Then, given a theorem statement T in Lean inputted by the user, we append T to our textual embedding and retrieve its k nearest neighors, where k is a hyperparameter. Finally, a prompt asking DeepSeek Prover v2 7B to prove T including the k nearest neighbors is automatically generated and inputted, and the output is retrieved and outputted for the user.

## Example 🚀
Suppose we want to prove that addition of natural numbers is commutative. Then, as a user, we could input the following:
```lean4
∀ n m : ℕ, n + m = m + n
```
Setting k = 5, this query is embedded using our textual embedder and transformed into the following prompt:
```
You are an expert Lean 4 code generator. Your goal is to prove the following statement:
STATEMENT:
  ∀ n m : ℕ, n + m = m + n

You may find the following list of theorems/formal statements helpful:
  • Logic.Equiv.Fin.Basic.finAddFlip : Fin (m + n) ≃ Fin (n + m)
  • Data.Nat.Init.dvd_right_iff_eq : (∀ a : ℕ, m ∣ a ↔ n ∣ a) ↔ m = n
  • Data.Nat.Init.dvd_left_iff_eq : (∀ a : ℕ, a ∣ m ↔ a ∣ n) ↔ m = n
  • Algebra.Group.Int.Even.even_sub : Even (m - n) ↔ (Even m ↔ Even n)
  • Data.ENat.Basic.forall_natCast_le_iff_le : (∀ a : ℕ, a ≤ m → a ≤ n) ↔ m ≤ n

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
```
This prompt leverages and augments DeepSeek-Prover's core capabilities, such as chain-of-thought (CoT) reasoning and structured planning, while compensating for its weaknesses, in particular retrieving relevant results from mathlib4. Upon prompting DeepSeek-Prover with the above text, it produces the following valid proof.
```lean4
theorem statement : ∀ n m : ℕ, n + m = m + n := by
  have h_main : ∀ n m : ℕ, n + m = m + n := by
    intro n
    intro m
    induction m with
    | zero =>
      -- Base case: m = 0
      simp
    | succ m ih =>
      -- Inductive step: assume the statement holds for m, prove for m + 1
      simp_all [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]
      <;> omega
  exact h_main
```

### Installation ✨
The required libraries to use <code>RAG-prover</code> are as follows:
- argparse
- asyncio
- json
- logging
- os
- pathlib
- re
- sys
- time
- typing
- \_\_future\_\_.

You must additionally install and import the following libraries using <code>!pip install</code>:
- aiofiles
- numpy
- openai
- tiktoken
- torch
- tqdm
- transformers.

## Limitations 🚩
As of August 2025, <code>lean-prover</code> is not publicly available for download as a package.

## License ⚖️
Distributed under the MIT License. See LICENSE.txt for more information.

## Contact 📞
Ritik Jain - https://www.linkedin.com/in/ritik-jain-91a201220/ - rjain92682@gmail.com
