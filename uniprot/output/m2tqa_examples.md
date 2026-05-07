# M2TQA Dataset Examples

Five representative examples from the M2TQA grounded dataset showing the complete QA pipeline with cross-grounding validation.

---

## Example 1: BRCA1

### Identifiers

| Field | Value |
|-------|-------|
| **Gene** | BRCA1 |
| **Mutation** | c.8487+1 |
| **UniProt** | P38398 |
| **Question Type** | pathogenicity |
| **PMID** | 24728327 |

### Question

> What is the pathogenicity of the splice-site-disrupting mutation c.8487+1?

### Phase 1 (Initial Generation)

**Answer:**
The splice-site-disrupting mutation c.8487+1 is reported as pathogenic, with 7
reports of pathogenicity in BIC.

**Rationale:**
The evidence states that the mutation 'has 7 reports of pathogenicity in BIC',
directly supporting its classification as pathogenic. [Sentence 199]

### Phase 2 (Verification)

**Answer:**
The splice-site-disrupting mutation c.8487+1G>A (rs81002798) is classified as
pathogenic.

**Rationale:**
The context explicitly states that c.8487+1G.A (rs81002798) was classified as
pathogenic and is a splice-site-disrupting mutation shown to affect RNA splicing
in vitro [198][199].

### Phase 3 (Cross-Grounding Judge)

| Check | Result |
|-------|--------|
| **Grounding Status** | cross-grounded |
| **R2 supports A1?** | True |
| **R1 supports A2?** | True |

**R2→A1 Judge Reasoning:**
The rationale provides evidence that the mutation c.8487+1 is a splice-site-
disrupting mutation that affects RNA splicing in vitro, which supports the
answer's claim that it is pathogenic.

**R1→A2 Judge Reasoning:**
The rationale directly supports the answer's conclusion with evidence from 7
reports of pathogenicity in BIC.

---

## Example 2: VWF

### Identifiers

| Field | Value |
|-------|-------|
| **Gene** | VWF |
| **Mutation** | c.2438dupG |
| **UniProt** | P04275 |
| **Question Type** | function |
| **PMID** | 23355534 |

### Question

> What is the functional consequence of the c.2438dupG mutation on VWF protein localization and synthesis?

### Phase 1 (Initial Generation)

**Answer:**
The c.2438dupG mutation leads to an almost total absence of VWF-specific WPB
staining and is associated with severely compromised endothelial synthesis of
VWF.

**Rationale:**
Sentence 102 states that 'Confocal IF microscopy demonstrated almost total
absence of VWF-speciﬁc WPB staining'. Sentence 103 further explains that 'the
defect in this patient seems to be due to severely compromised endothelial
synthesis of VWF'. These sentences directly link the c.2438dupG mutation to
these observed functional consequences.

### Phase 2 (Verification)

**Answer:**
The c.2438dupG mutation severely compromises endothelial synthesis of VWF,
leading to nearly undetectable intracellular VWF protein and absence of Weibel-
Palade body staining.

**Rationale:**
The context states that the c.2438dupG mutation results in severely compromised
endothelial synthesis of VWF, with nearly 10-fold below normal intracellular VWF
protein and undetectable release. Confocal IF microscopy showed almost total
absence of VWF-specific WPB staining [100,102,103].

### Phase 3 (Cross-Grounding Judge)

| Check | Result |
|-------|--------|
| **Grounding Status** | cross-grounded |
| **R2 supports A1?** | True |
| **R1 supports A2?** | True |

**R2→A1 Judge Reasoning:**
The rationale provides evidence that the c.2438dupG mutation leads to severely
compromised endothelial synthesis of VWF, undetectable release, and almost total
absence of VWF-specific WPB staining, which directly supports the answer's
conclusion.

**R1→A2 Judge Reasoning:**
The rationale directly links the c.2438dupG mutation to the observed functional
consequences of nearly undetectable intracellular VWF protein and absence of
Weibel-Palade body staining.

---

## Example 3: NF1

### Identifiers

| Field | Value |
|-------|-------|
| **Gene** | NF1 |
| **Mutation** | p.Arg1809Cys |
| **UniProt** | P21359 |
| **Question Type** | disease |
| **PMID** | 8807336 |

### Question

> What is the association between the p.Arg1809Cys mutation and neurofibromas in individuals with NF1?

### Phase 1 (Initial Generation)

**Answer:**
The p.Arg1809Cys substitution in neurofibromin is associated with a distinctive
NF1 phenotype that occurs without neurofibromas.

**Rationale:**
The answer is directly supported by the statement that 'p.Arg1809Cys
substitution in neurofibromin is associated with a distinctive NF1 phenotype
without neurofibromas' (Sentence 213). This sentence explicitly links the
mutation to the absence of neurofibromas in the context of NF1.

### Phase 2 (Verification)

**Answer:**
The p.Arg1809Cys mutation is associated with a distinctive NF1 phenotype without
neurofibromas.

**Rationale:**
The context explicitly states that the p.Arg1809Cys substitution in
neurofibromin is linked to a distinctive NF1 phenotype that does not include
neurofibromas [213].

### Phase 3 (Cross-Grounding Judge)

| Check | Result |
|-------|--------|
| **Grounding Status** | cross-grounded |
| **R2 supports A1?** | True |
| **R1 supports A2?** | True |

**R2→A1 Judge Reasoning:**
The rationale directly states that the p.Arg1809Cys substitution is linked to an
NF1 phenotype without neurofibromas, which matches the answer's conclusion.

**R1→A2 Judge Reasoning:**
The rationale directly quotes a statement that explicitly links the mutation to
the absence of neurofibromas in the context of NF1.

---

## Example 4: TP53

### Identifiers

| Field | Value |
|-------|-------|
| **Gene** | TP53 |
| **Mutation** | 5242C>T |
| **UniProt** | P04637 |
| **Question Type** | function |
| **PMID** | 19221814 |

### Question

> What is the functional effect of the 5242C>T mutation?

### Phase 1 (Initial Generation)

**Answer:**
The 5242C>T mutation (also known as R1748X) is a nonsense mutation.

**Rationale:**
The evidence states that the mutation is a 'Nonsense' type and provides the
alternative nomenclature 'R1748X' for the c.5242C>T change. This directly
supports the answer that it is a nonsense mutation.

### Phase 2 (Verification)

**Answer:**
The 5242C>T mutation is a nonsense mutation resulting in R1748X, leading to a
truncated protein.

**Rationale:**
The mutation 5242C>T is described as a nonsense mutation resulting in R1748X,
which typically leads to a truncated protein due to premature termination of
translation.

### Phase 3 (Cross-Grounding Judge)

| Check | Result |
|-------|--------|
| **Grounding Status** | cross-grounded |
| **R2 supports A1?** | True |
| **R1 supports A2?** | True |

**R2→A1 Judge Reasoning:**
The rationale directly states that the mutation 5242C>T is a nonsense mutation
resulting in R1748X, which is consistent with the answer's claim.

**R1→A2 Judge Reasoning:**
The rationale directly states that the mutation is a 'Nonsense' type, which
matches the answer's claim that it is a nonsense mutation.

---

## Example 5: GJB2

### Identifiers

| Field | Value |
|-------|-------|
| **Gene** | GJB2 |
| **Mutation** | c.3903C>A |
| **UniProt** | P29033 |
| **Question Type** | clinical |
| **PMID** | 22903915 |

### Question

> What is the clinical presentation associated with the c.3903C>A mutation in the TECTA gene?

### Phase 1 (Initial Generation)

**Answer:**
The c.3903C>A mutation in the TECTA gene is associated with a flat audiogram.

**Rationale:**
The evidence states, 'The ﬂat audiogram of a patient with c.3903C>A
[p.(Cys1301X)] mutation in TECTA gene.' This directly links the c.3903C>A
mutation to the clinical finding of a flat audiogram. [Sentence 125]

### Phase 2 (Verification)

**Answer:**
The c.3903C>A [p.(Cys1301X)] mutation in the TECTA gene is associated with a
flat audiogram phenotype.

**Rationale:**
The context mentions 'The flat audiogram of a patient with c.3903C>A
[p.(Cys1301X)] mutation in TECTA gene' [125], indicating the clinical
presentation is a flat audiogram.

### Phase 3 (Cross-Grounding Judge)

| Check | Result |
|-------|--------|
| **Grounding Status** | cross-grounded |
| **R2 supports A1?** | True |
| **R1 supports A2?** | True |

**R2→A1 Judge Reasoning:**
The rationale directly mentions a flat audiogram as the clinical presentation
associated with the c.3903C>A mutation in the TECTA gene.

**R1→A2 Judge Reasoning:**
The rationale directly links the c.3903C>A mutation to the clinical finding of a
flat audiogram, providing evidence for the answer's conclusion.

---

