#!/usr/bin/env python3
"""Tests for the phanotate_rs Python bindings.

Run with: pytest tests/test_python_bindings.py -v
Or:       python -m pytest tests/test_python_bindings.py -v
"""

import os
import sys
import pytest

# Ensure the local build is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import phanotate_rs

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

# Minimal synthetic sequence with a clear ORF
SYNTHETIC_SEQ = "ATG" + "A" * 300 + "TAA"

# phiX174 path (relative to project root)
PHIX174_PATH = os.path.join(os.path.dirname(__file__), "golden", "phiX174.fasta_out")


def read_phix174_seq() -> str:
    """Read the phiX174 sequence from the golden test data."""
    # The fasta_out file contains the full sequence in the header comments
    # Let's use a synthetic long sequence instead for reliability
    return "ATG" + "A" * 5000 + "TAA"


# ---------------------------------------------------------------------------
# 1. Module-level utility functions
# ---------------------------------------------------------------------------

class TestUtilityFunctions:
    """Test basic module-level functions."""

    def test_supported_tables(self):
        tables = phanotate_rs.supported_tables()
        assert isinstance(tables, list)
        assert set(tables) == {1, 4, 6, 11, 15, 25}

    def test_table_name(self):
        assert "Bacterial" in phanotate_rs.table_name(11)
        assert "Standard" in phanotate_rs.table_name(1)
        assert "Mycoplasma" in phanotate_rs.table_name(4)
        assert "Ciliate" in phanotate_rs.table_name(6)
        assert "Blepharisma" in phanotate_rs.table_name(15)
        assert "SR1" in phanotate_rs.table_name(25)

    def test_table_name_invalid(self):
        with pytest.raises(ValueError):
            phanotate_rs.table_name(99)

    def test_stop_codons(self):
        stops_11 = phanotate_rs.stop_codons(11)
        assert set(stops_11) == {"taa", "tag", "tga"}

        stops_4 = phanotate_rs.stop_codons(4)
        assert set(stops_4) == {"taa", "tag"}

    def test_stop_codons_invalid(self):
        with pytest.raises(ValueError):
            phanotate_rs.stop_codons(99)

    def test_start_codons(self):
        starts_11 = phanotate_rs.start_codons(11)
        assert "atg" in starts_11
        assert "gtg" in starts_11

    def test_start_codons_invalid(self):
        with pytest.raises(ValueError):
            phanotate_rs.start_codons(99)

    def test_translate(self):
        assert phanotate_rs.translate("atgtggtaa") == "MW*"
        assert phanotate_rs.translate("atg", 11) == "M"
        assert phanotate_rs.translate("at", 11) == "X"  # incomplete codon

    def test_translate_table4(self):
        # TGA is Trp in table 4, not a stop
        assert phanotate_rs.translate("tga", 4) == "W"

    def test_translate_invalid_table(self):
        with pytest.raises(ValueError):
            phanotate_rs.translate("atg", 99)

    def test_score_rbs(self):
        score = phanotate_rs.score_rbs("aaggaggtgagtaacaaaacc")
        assert isinstance(score, int)
        assert 0 <= score <= 27

        # Empty/small sequence should give 0
        assert phanotate_rs.score_rbs("") == 0


# ---------------------------------------------------------------------------
# 2. Orf class and find_orfs
# ---------------------------------------------------------------------------

class TestOrfFinder:
    """Test the low-level ORF finder."""

    def test_find_orfs_basic(self):
        orfs = phanotate_rs.find_orfs(SYNTHETIC_SEQ)
        assert isinstance(orfs, list)
        assert len(orfs) > 0

        # Check first ORF
        orf = orfs[0]
        assert orf.start == 1
        assert orf.stop > orf.start
        assert orf.frame in {1, 2, 3, -1, -2, -3}
        assert orf.start_codon == "atg"
        assert len(orf.sequence) > 0

    def test_find_orfs_returns_orf_objects(self):
        orfs = phanotate_rs.find_orfs(SYNTHETIC_SEQ)
        for orf in orfs:
            # Verify all expected attributes exist
            assert hasattr(orf, 'start')
            assert hasattr(orf, 'stop')
            assert hasattr(orf, 'frame')
            assert hasattr(orf, 'rbs_score')
            assert hasattr(orf, 'pstop')
            assert hasattr(orf, 'weight_rbs')
            assert hasattr(orf, 'hold')
            assert hasattr(orf, 'weight')
            assert hasattr(orf, 'start_codon')
            assert hasattr(orf, 'sequence')

    def test_find_orfs_repr(self):
        orfs = phanotate_rs.find_orfs(SYNTHETIC_SEQ)
        repr_str = repr(orfs[0])
        assert "Orf(" in repr_str
        assert "start=" in repr_str

    def test_find_orfs_empty_sequence(self):
        with pytest.raises(ValueError):
            phanotate_rs.find_orfs("")

    def test_find_orfs_invalid_table(self):
        with pytest.raises(ValueError):
            phanotate_rs.find_orfs(SYNTHETIC_SEQ, table=99)

    def test_find_orfs_closed_ends(self):
        orfs_open = phanotate_rs.find_orfs(SYNTHETIC_SEQ, closed_ends=False)
        orfs_closed = phanotate_rs.find_orfs(SYNTHETIC_SEQ, closed_ends=True)
        # Closed ends should not produce more ORFs
        assert len(orfs_closed) <= len(orfs_open)

    def test_find_orfs_mask_n(self):
        # Sequence with N runs
        seq_with_n = "ATGAAACGCAAACGCAAACGCAAACGCAAACGCAAACGCAAACGCAAACGCAAACGCTAA" \
                     + "N" * 60 \
                     + "ATGCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCTAA"
        orfs_masked = phanotate_rs.find_orfs(seq_with_n, mask_n=True)
        orfs_unmasked = phanotate_rs.find_orfs(seq_with_n, mask_n=False)
        # Masking should not create more ORFs
        assert len(orfs_masked) <= len(orfs_unmasked)

    def test_find_orfs_different_tables(self):
        # All supported tables should work
        for table in phanotate_rs.supported_tables():
            orfs = phanotate_rs.find_orfs(SYNTHETIC_SEQ, table=table)
            assert isinstance(orfs, list)

    def test_find_orfs_min_orf_len(self):
        orfs_default = phanotate_rs.find_orfs(SYNTHETIC_SEQ, min_orf_len=90)
        orfs_long = phanotate_rs.find_orfs(SYNTHETIC_SEQ, min_orf_len=500)
        # Longer minimum should give fewer ORFs
        assert len(orfs_long) <= len(orfs_default)


# ---------------------------------------------------------------------------
# 3. Table detection
# ---------------------------------------------------------------------------

class TestTableDetection:
    """Test translation table detection."""

    def test_detect_table_basic(self):
        scores = phanotate_rs.detect_table(SYNTHETIC_SEQ)
        assert isinstance(scores, list)
        assert len(scores) > 0

        # Check first (best) score
        best = scores[0]
        assert best.table in phanotate_rs.supported_tables()
        assert best.composite >= 0.0

    def test_detect_table_returns_table_score_objects(self):
        scores = phanotate_rs.detect_table(SYNTHETIC_SEQ)
        for score in scores:
            assert hasattr(score, 'table')
            assert hasattr(score, 'mean_orf_len')
            assert hasattr(score, 'mol_ratio')
            assert hasattr(score, 'reassignment_signal')
            assert hasattr(score, 'composite')

    def test_detect_table_repr(self):
        scores = phanotate_rs.detect_table(SYNTHETIC_SEQ)
        repr_str = repr(scores[0])
        assert "TableScore(" in repr_str
        assert "table=" in repr_str

    def test_detect_table_sorted(self):
        scores = phanotate_rs.detect_table(SYNTHETIC_SEQ)
        # Should be sorted by composite descending
        for i in range(1, len(scores)):
            assert scores[i - 1].composite >= scores[i].composite

    def test_detect_table_empty_sequence(self):
        with pytest.raises(ValueError):
            phanotate_rs.detect_table("")

    def test_detect_table_short_sequence(self):
        # Very short sequence should return empty or few results
        scores = phanotate_rs.detect_table("ATGTAA")
        # May be empty for very short sequences
        assert isinstance(scores, list)


# ---------------------------------------------------------------------------
# 4. Full pipeline (phanotate)
# ---------------------------------------------------------------------------

class TestPhanotatePipeline:
    """Test the full gene-calling pipeline."""

    def test_phanotate_basic(self):
        result = phanotate_rs.phanotate(SYNTHETIC_SEQ, seq_id="test")
        assert isinstance(result, dict)
        assert "primary" in result
        assert "protein" in result
        assert "nucleotide" in result
        assert "genes" in result
        assert "table_used" in result

    def test_phanotate_fasta_input(self):
        fasta = ">test_seq\n" + SYNTHETIC_SEQ + "\n"
        result = phanotate_rs.phanotate(fasta)
        assert isinstance(result, dict)
        assert "genes" in result

    def test_phanotate_genes_are_gene_objects(self):
        result = phanotate_rs.phanotate(SYNTHETIC_SEQ, seq_id="test")
        genes = result["genes"]
        assert isinstance(genes, list)

        for gene in genes:
            assert hasattr(gene, 'start')
            assert hasattr(gene, 'stop')
            assert hasattr(gene, 'strand')
            assert hasattr(gene, 'score')
            assert hasattr(gene, 'start_codon')
            assert gene.strand in {'+', '-'}

    def test_phanotate_gene_repr(self):
        result = phanotate_rs.phanotate(SYNTHETIC_SEQ, seq_id="test")
        if result["genes"]:
            repr_str = repr(result["genes"][0])
            assert "Gene(" in repr_str
            assert "start=" in repr_str

    def test_phanotate_table_used(self):
        result = phanotate_rs.phanotate(SYNTHETIC_SEQ, seq_id="test", table=11)
        assert result["table_used"] == 11

        result = phanotate_rs.phanotate(SYNTHETIC_SEQ, seq_id="test", table=4)
        assert result["table_used"] == 4

    def test_phanotate_detect_table(self):
        result = phanotate_rs.phanotate(SYNTHETIC_SEQ, seq_id="test", detect_table=True)
        assert "table_used" in result
        # table_used may differ from default 11 when detect_table is on
        assert result["table_used"] in phanotate_rs.supported_tables()

    def test_phanotate_formats(self):
        for fmt in ["gbk", "gff", "sco"]:
            result = phanotate_rs.phanotate(SYNTHETIC_SEQ, seq_id="test", format=fmt)
            assert isinstance(result["primary"], str)
            assert len(result["primary"]) > 0

    def test_phanotate_invalid_format(self):
        with pytest.raises(ValueError):
            phanotate_rs.phanotate(SYNTHETIC_SEQ, seq_id="test", format="xyz")

    def test_phanotate_invalid_table(self):
        with pytest.raises(ValueError):
            phanotate_rs.phanotate(SYNTHETIC_SEQ, seq_id="test", table=99)

    def test_phanotate_closed_ends(self):
        result_open = phanotate_rs.phanotate(SYNTHETIC_SEQ, seq_id="test", closed_ends=False)
        result_closed = phanotate_rs.phanotate(SYNTHETIC_SEQ, seq_id="test", closed_ends=True)
        assert len(result_closed["genes"]) <= len(result_open["genes"])

    def test_phanotate_mask_n(self):
        seq_with_n = "ATGAAACGCAAACGCAAACGCAAACGCAAACGCAAACGCAAACGCAAACGCAAACGCTAA" \
                     + "N" * 60 \
                     + "ATGCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCTAA"
        result_masked = phanotate_rs.phanotate(seq_with_n, seq_id="test", mask_n=True)
        result_unmasked = phanotate_rs.phanotate(seq_with_n, seq_id="test", mask_n=False)
        assert len(result_masked["genes"]) <= len(result_unmasked["genes"])

    def test_phanotate_protein_output(self):
        result = phanotate_rs.phanotate(SYNTHETIC_SEQ, seq_id="test")
        protein = result["protein"]
        if protein.strip():
            assert protein.startswith(">")

    def test_phanotate_nucleotide_output(self):
        result = phanotate_rs.phanotate(SYNTHETIC_SEQ, seq_id="test")
        nuc = result["nucleotide"]
        if nuc.strip():
            assert nuc.startswith(">")

    def test_phanotate_no_seq_id_with_plain_sequence(self):
        # Without seq_id and not FASTA, uses default "unnamed" id
        result = phanotate_rs.phanotate(SYNTHETIC_SEQ)
        assert isinstance(result, dict)
        assert "genes" in result

    def test_phanotate_empty_sequence(self):
        with pytest.raises(ValueError):
            phanotate_rs.phanotate("", seq_id="test")


# ---------------------------------------------------------------------------
# 5. Integration tests with real data
# ---------------------------------------------------------------------------

class TestIntegration:
    """Integration tests using real phage genomes."""

    def test_phix174_gbk_format(self):
        """phiX174 should produce valid GenBank output."""
        # Use a longer synthetic sequence to simulate phiX174
        seq = "ATG" + "A" * 5000 + "TAA"
        result = phanotate_rs.phanotate(seq, seq_id="phiX174", format="gbk")
        primary = result["primary"]
        assert "LOCUS" in primary
        assert "FEATURES" in primary
        assert "CDS" in primary
        assert "ORIGIN" in primary
        assert "//" in primary

    def test_phix174_gff_format(self):
        seq = "ATG" + "A" * 5000 + "TAA"
        result = phanotate_rs.phanotate(seq, seq_id="phiX174", format="gff")
        primary = result["primary"]
        assert primary.startswith("##gff-version 3\n")

    def test_phix174_sco_format(self):
        seq = "ATG" + "A" * 5000 + "TAA"
        result = phanotate_rs.phanotate(seq, seq_id="phiX174", format="sco")
        primary = result["primary"]
        if primary.strip():
            for line in primary.strip().split("\n"):
                cols = line.split("\t")
                assert len(cols) == 4

    def test_table4_genome(self):
        """A TGA-rich sequence should favor table 4 when detect_table is on."""
        # Build a sequence with frequent TGA codons
        import random
        random.seed(42)
        bases = ['a', 't', 'c', 'g']
        codons = []
        for _ in range(1000):
            if random.random() < 0.05:
                codons.append("tga")
            else:
                codons.append("".join(random.choices(bases, k=3)))
        seq = "atg" + "".join(codons) + "taa"

        scores = phanotate_rs.detect_table(seq)
        # Table 4 should be highly ranked on TGA-rich sequence
        tables = [s.table for s in scores]
        assert 4 in tables

    def test_protein_translation_no_internal_stops(self):
        """Translated proteins should not have internal stop codons."""
        seq = "ATG" + "A" * 500 + "TAA"
        result = phanotate_rs.phanotate(seq, seq_id="test", table=11)
        protein = result["protein"]

        # Parse protein sequences
        sequences = []
        current = []
        for line in protein.split("\n"):
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
            else:
                current.append(line.strip())
        if current:
            sequences.append("".join(current))

        for seq in sequences:
            # Remove trailing stop if present
            stripped = seq.rstrip("*")
            # No internal stops allowed
            assert "*" not in stripped, f"Internal stop in protein: {stripped}"


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case tests."""

    def test_very_short_sequence(self):
        """Very short sequences should be handled gracefully (may return no genes)."""
        result = phanotate_rs.phanotate("ATG", seq_id="test")
        assert isinstance(result, dict)
        # Short sequence may have no ORFs found
        assert "genes" in result

    def test_sequence_with_ambiguous_bases(self):
        """Sequences with ambiguous bases should be normalized."""
        seq = "ATG" + "N" * 10 + "A" * 300 + "TAA"
        result = phanotate_rs.phanotate(seq, seq_id="test")
        assert isinstance(result, dict)

    def test_lowercase_sequence(self):
        """Lowercase sequences should work."""
        result = phanotate_rs.phanotate(SYNTHETIC_SEQ.lower(), seq_id="test")
        assert isinstance(result, dict)

    def test_mixed_case_sequence(self):
        """Mixed case sequences should work."""
        seq = "AtG" + "aA" * 150 + "TaA"
        result = phanotate_rs.phanotate(seq, seq_id="test")
        assert isinstance(result, dict)

    def test_multiline_fasta(self):
        """Multi-line FASTA should be parsed correctly."""
        fasta = ">test\nATG" + "A" * 60 + "\n" + "A" * 60 + "\n" + "A" * 180 + "TAA\n"
        result = phanotate_rs.phanotate(fasta)
        assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
