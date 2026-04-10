"""
Unit tests for chronoscope.math_tools_map.

Validates that:
  - All expected categories are present.
  - Every FunctionEntry has non-empty required fields.
  - Each qualified_name follows the expected pattern.
  - Query helpers (list_categories, get_category, get_mapping, describe) work
    correctly.
  - Every referenced function actually exists in the codebase.
"""

import importlib
import inspect
import pytest

from chronoscope.math_tools_map import (
    MATH_TOOLS_MAP,
    FunctionEntry,
    describe,
    get_category,
    get_mapping,
    list_categories,
)

# ── Expected categories ────────────────────────────────────────────────────

EXPECTED_CATEGORIES = {
    "spectral_analysis_fft",
    "statistical_features",
    "autocorrelation_lag",
    "dimensionality_reduction",
    "changepoint_state_detection",
    "evaluation_validation",
}


class TestCategories:
    def test_all_expected_categories_present(self):
        assert EXPECTED_CATEGORIES.issubset(set(list_categories()))

    def test_list_categories_returns_list(self):
        cats = list_categories()
        assert isinstance(cats, list)
        assert len(cats) >= len(EXPECTED_CATEGORIES)

    def test_get_mapping_returns_all_categories(self):
        mapping = get_mapping()
        assert isinstance(mapping, dict)
        assert EXPECTED_CATEGORIES.issubset(set(mapping.keys()))

    def test_get_category_returns_nonempty_list(self):
        for cat in EXPECTED_CATEGORIES:
            entries = get_category(cat)
            assert isinstance(entries, list), f"Category {cat!r} should be a list"
            assert len(entries) >= 1, f"Category {cat!r} should have at least one entry"

    def test_get_category_raises_on_unknown(self):
        with pytest.raises(KeyError, match="Unknown category"):
            get_category("nonexistent_category")


class TestFunctionEntries:
    """Validate the content of every FunctionEntry."""

    @staticmethod
    def all_entries():
        for entries in MATH_TOOLS_MAP.values():
            yield from entries

    def test_all_entries_are_function_entry_instances(self):
        for entry in self.all_entries():
            assert isinstance(entry, FunctionEntry)

    def test_module_non_empty(self):
        for entry in self.all_entries():
            assert entry.module, f"{entry.function}: module must not be empty"

    def test_function_non_empty(self):
        for entry in self.all_entries():
            assert entry.function, f"function name must not be empty"

    def test_mathematical_basis_non_empty(self):
        for entry in self.all_entries():
            assert entry.mathematical_basis, (
                f"{entry.qualified_name}: mathematical_basis must not be empty"
            )

    def test_role_non_empty(self):
        for entry in self.all_entries():
            assert entry.role, f"{entry.qualified_name}: role must not be empty"

    def test_justification_non_empty(self):
        for entry in self.all_entries():
            assert entry.justification, (
                f"{entry.qualified_name}: justification must not be empty"
            )

    def test_qualified_name_contains_module_and_function(self):
        for entry in self.all_entries():
            qn = entry.qualified_name
            assert entry.module in qn
            assert entry.function in qn

    def test_no_duplicate_qualified_names(self):
        names = [e.qualified_name for e in self.all_entries()]
        assert len(names) == len(set(names)), "Duplicate qualified_name entries found"


class TestFunctionsExistInCodebase:
    """
    For every FunctionEntry, verify that the referenced module can be imported
    and that the stated function (method) exists on the stated class (or at
    module level if class_name is empty).
    """

    @staticmethod
    def all_entries():
        for entries in MATH_TOOLS_MAP.values():
            yield from entries

    def test_referenced_functions_exist(self):
        missing = []
        for entry in self.all_entries():
            try:
                mod = importlib.import_module(entry.module)
            except ImportError as exc:
                missing.append(f"Cannot import {entry.module}: {exc}")
                continue

            if entry.class_name:
                cls = getattr(mod, entry.class_name, None)
                if cls is None:
                    missing.append(
                        f"Class {entry.class_name} not found in {entry.module}"
                    )
                    continue
                if not hasattr(cls, entry.function):
                    missing.append(
                        f"Method {entry.class_name}.{entry.function} "
                        f"not found in {entry.module}"
                    )
            else:
                if not hasattr(mod, entry.function):
                    missing.append(
                        f"Function {entry.function} not found in {entry.module}"
                    )

        assert not missing, "Missing referenced symbols:\n" + "\n".join(missing)


class TestDescribe:
    def test_describe_all_returns_string(self):
        report = describe()
        assert isinstance(report, str)
        assert len(report) > 0

    def test_describe_contains_category_titles(self):
        report = describe()
        # Each category key should appear in some form in the report
        for cat in EXPECTED_CATEGORIES:
            # Category is rendered as title-cased words
            title_words = cat.replace("_", " ").title()
            assert title_words in report, (
                f"Expected {title_words!r} in describe() output"
            )

    def test_describe_contains_function_names(self):
        report = describe()
        for entry in (e for entries in MATH_TOOLS_MAP.values() for e in entries):
            assert entry.function in report, (
                f"Expected function name {entry.function!r} in describe() output"
            )

    def test_describe_single_category(self):
        cat = "spectral_analysis_fft"
        report = describe(cat)
        # Should contain all qualified names from the requested category
        for entry in get_category(cat):
            assert entry.qualified_name in report, (
                f"Expected {entry.qualified_name!r} in single-category describe() output"
            )
        # Should NOT contain qualified names from an unrelated category
        for entry in get_category("dimensionality_reduction"):
            assert entry.qualified_name not in report, (
                f"Unexpected {entry.qualified_name!r} in single-category describe() output"
            )

    def test_describe_unknown_category_raises(self):
        with pytest.raises(KeyError):
            describe("nonexistent")


class TestMinimumCoverage:
    """
    Sanity-check that each category covers the most important functions
    documented in README / PROJECT_GUIDE.
    """

    def test_spectral_fft_includes_spectral_analysis(self):
        fns = {e.function for e in get_category("spectral_analysis_fft")}
        assert "spectral_analysis" in fns

    def test_statistical_features_includes_trajectory_dynamics(self):
        fns = {e.function for e in get_category("statistical_features")}
        assert "calculate_trajectory_dynamics" in fns

    def test_autocorrelation_includes_acf(self):
        fns = {e.function for e in get_category("autocorrelation_lag")}
        assert "autocorrelation_analysis" in fns

    def test_dimensionality_reduction_includes_svd_compress(self):
        fns = {e.function for e in get_category("dimensionality_reduction")}
        assert "svd_compress" in fns

    def test_changepoint_includes_hmm(self):
        fns = {e.function for e in get_category("changepoint_state_detection")}
        assert "_discover_phases_hmm" in fns

    def test_evaluation_includes_fdr_correction(self):
        fns = {e.function for e in get_category("evaluation_validation")}
        assert "_apply_fdr_correction" in fns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
