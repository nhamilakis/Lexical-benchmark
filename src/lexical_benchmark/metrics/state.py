#!/usr/bin/env python
"""State management for CDI cumulative word counts."""

import json
import typing as t
from pathlib import Path


class CDIStateManager:
    """Manage CDI cumulative word counts across months and processing runs."""

    def __init__(self) -> None:
        """Initialize empty state manager."""
        self.state: dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]] = {}

    def get_previous_words(self, dataset: str, chunk: str, model: str, temp: float) -> dict[str, float]:
        """Get previous word counts for a specific configuration."""
        try:
            temp_str = str(temp)
            return self.state.get(dataset, {}).get(chunk, {}).get(model, {}).get(temp_str, {})
        except (KeyError, AttributeError):
            return {}

    def update_words(self, dataset: str, chunk: str, model: str, temp: float, counts: dict[str, float]) -> None:
        """Update word counts for a specific configuration."""
        temp_str = str(temp)

        # Initialize nested dictionaries if they don't exist
        if dataset not in self.state:
            self.state[dataset] = {}
        if chunk not in self.state[dataset]:
            self.state[dataset][chunk] = {}
        if model not in self.state[dataset][chunk]:
            self.state[dataset][chunk][model] = {}
        if temp_str not in self.state[dataset][chunk][model]:
            self.state[dataset][chunk][model][temp_str] = {}

        # Update counts (merge with existing)
        existing_counts = self.state[dataset][chunk][model][temp_str]
        for word, count in counts.items():
            existing_counts[word] = existing_counts.get(word, 0.0) + count

    def clear_state(self) -> None:
        """Clear all state data."""
        self.state = {}

    def save_state(self, path: Path) -> None:
        """Save state to JSON file."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            print(f"Error saving state to {path}: {e}")

    def load_state(self, path: Path) -> None:
        """Load state from JSON file."""
        try:
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    self.state = json.load(f)
            else:
                print(f"State file {path} not found, starting with empty state")
                self.state = {}
        except Exception as e:
            print(f"Error loading state from {path}: {e}")
            self.state = {}

    def get_state_summary(self) -> dict[str, t.Any]:
        """Get summary statistics of current state."""
        summary = {"datasets": list(self.state.keys()), "total_configurations": 0, "configurations_by_dataset": {}}

        for dataset, dataset_data in self.state.items():
            config_count = 0
            for chunk_data in dataset_data.values():
                for model_data in chunk_data.values():
                    config_count += len(model_data)

            summary["configurations_by_dataset"][dataset] = config_count
            summary["total_configurations"] += config_count

        return summary


class WordDictManager:
    """Compatibility wrapper for the old WordDictManager interface."""

    def __init__(self, dataset: str, chunk: str | int, model_type: str, temp: str | float) -> None:
        """Initialize with configuration parameters."""
        self.dataset = str(dataset)
        self.chunk = str(chunk)
        self.model_type = str(model_type)
        self.temp = str(temp)

    def load_word_dict(self, cdi_month_dict: dict) -> dict[str, float]:
        """Load word dictionary from legacy CDI month dict format."""
        try:
            return cdi_month_dict.get(self.dataset, {}).get(self.chunk, {}).get(self.model_type, {}).get(self.temp, {})
        except (KeyError, AttributeError):
            return {}

    def write_word_dict(self, previous_words: dict[str, float], cdi_month_dict: dict) -> dict:
        """Write word dictionary to legacy CDI month dict format."""
        # Initialize nested structure if needed
        if not isinstance(cdi_month_dict, dict):
            cdi_month_dict = {}

        if self.dataset not in cdi_month_dict:
            cdi_month_dict[self.dataset] = {}
        if self.chunk not in cdi_month_dict[self.dataset]:
            cdi_month_dict[self.dataset][self.chunk] = {}
        if self.model_type not in cdi_month_dict[self.dataset][self.chunk]:
            cdi_month_dict[self.dataset][self.chunk][self.model_type] = {}

        # Store the words
        cdi_month_dict[self.dataset][self.chunk][self.model_type][self.temp] = previous_words

        return cdi_month_dict
