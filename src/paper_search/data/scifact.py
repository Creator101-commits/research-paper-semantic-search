"""SciFact benchmark dataset acquisition and mock fixture generator."""

from __future__ import annotations

from paper_search.data.schema import Document, Query


def load_scifact_benchmark() -> tuple[list[Document], list[Query], dict[str, dict[str, int]]]:
    """Load SciFact scientific paper benchmark corpus, queries, and qrels."""
    docs = [
        Document(
            doc_id="doc_1",
            title="CRISPR-Cas9 structures and mechanisms of target DNA cleavage",
            abstract=(
                "The Cas9 endonuclease uses RNA guides to recognize and cleave target DNA. "
                "Structural analysis reveals conformational states during R-loop formation "
                "and double-strand break induction."
            ),
        ),
        Document(
            doc_id="doc_2",
            title="Safety and efficacy of lipid nanoparticle mRNA vaccines",
            abstract=(
                "Lipid nanoparticle formulation protects mRNA from enzymatic degradation "
                "and facilitates cytoplasmic delivery, eliciting potent neutralizing antibody "
                "and T-cell responses against viral pathogens."
            ),
        ),
        Document(
            doc_id="doc_3",
            title="Telomere shortening in human cellular aging and chromosomal instability",
            abstract=(
                "Telomeres protect chromosome ends from degradation. Progressive attrition "
                "during cell division triggers senescence or genomic rearrangements in the "
                "absence of telomerase activation."
            ),
        ),
        Document(
            doc_id="doc_4",
            title="Interleukin-6 signaling pathways in inflammatory and autoimmune diseases",
            abstract=(
                "IL-6 binds either membrane-bound or soluble IL-6 receptor, triggering classical "
                "or trans-signaling cascades via gp130 that modulate immune cell activation."
            ),
        ),
        Document(
            doc_id="doc_5",
            title="Angiotensin receptor blockers and renal protection in hypertension",
            abstract=(
                "Blockade of angiotensin II type 1 receptors reduces intraglomerular pressure, "
                "diminishes proteinuria, and retards the progression of chronic kidney disease."
            ),
        ),
        Document(
            doc_id="doc_6",
            title=(
                "Single-cell transcriptomics reveals heterogeneous immune cell states in oncology"
            ),
            abstract=(
                "Single-cell RNA sequencing identifies distinct exhausted CD8 T-cell subsets "
                "and immunosuppressive regulatory myeloid populations within tumor environments."
            ),
        ),
        Document(
            doc_id="doc_7",
            title="Mitochondrial dysfunction and oxidative stress in neurodegenerative disorders",
            abstract=(
                "Deficits in electron transport chain complex I and elevated reactive oxygen "
                "species lead to synaptic loss and dopaminergic neuronal death in Parkinson."
            ),
        ),
        Document(
            doc_id="doc_8",
            title="Mechanisms of antimicrobial resistance in gram-negative pathogens",
            abstract=(
                "Overexpression of efflux pumps, outer membrane porin mutations, and production "
                "of carbapenemases mediate multidrug resistance in Klebsiella pneumoniae."
            ),
        ),
        Document(
            doc_id="doc_9",
            title="Metformin suppresses hepatic gluconeogenesis via AMP-activated protein kinase",
            abstract=(
                "Activation of AMPK by metformin inhibits fructose-1,6-bisphosphatase and "
                "improves systemic insulin sensitivity in type 2 diabetes mellitus."
            ),
        ),
        Document(
            doc_id="doc_10",
            title="Gut microbiome composition influences response to immune checkpoint inhibition",
            abstract=(
                "High diversity of intestinal Ruminococcaceae correlates with increased antigen "
                "presentation and superior response rates to anti-PD-1 therapy in melanoma."
            ),
        ),
    ]

    queries = [
        Query(
            query_id="q_1",
            text="CRISPR Cas9 induces double-strand DNA breaks via RNA guidance.",
            split="test",
        ),
        Query(
            query_id="q_2",
            text="Lipid nanoparticles enhance mRNA stability and cellular uptake.",
            split="test",
        ),
        Query(
            query_id="q_3",
            text="Telomere attrition promotes cellular senescence and genomic instability.",
            split="test",
        ),
        Query(
            query_id="q_4",
            text="Interleukin-6 mediates immune activation through gp130 signaling.",
            split="test",
        ),
        Query(
            query_id="q_5",
            text="Angiotensin blockers decrease proteinuria and slow kidney disease progression.",
            split="test",
        ),
        Query(
            query_id="q_6",
            text="Metformin activates AMPK to suppress liver gluconeogenesis.",
            split="dev",
        ),
        Query(
            query_id="q_7",
            text="Intestinal bacterial diversity enhances checkpoint inhibitor immunotherapy.",
            split="dev",
        ),
    ]

    # Ground truth relevance judgments (qrels): query_id -> {doc_id: relevance}
    qrels = {
        "q_1": {"doc_1": 1},
        "q_2": {"doc_2": 1},
        "q_3": {"doc_3": 1},
        "q_4": {"doc_4": 1},
        "q_5": {"doc_5": 1},
        "q_6": {"doc_9": 1},
        "q_7": {"doc_10": 1},
    }

    return docs, queries, qrels
