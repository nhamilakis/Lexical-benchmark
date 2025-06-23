#!/usr/bin/env python

import sys
import typing as t

from clypi import Command, Positional
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from lexical_benchmark import datasets, lb_types
from lexical_benchmark.dataloaders import generation_loaders
from lexical_benchmark.train import checkpoint_utils

console = Console()


class CheckPointView(Command):
    """Preview & check a generation checkpoint."""

    dataset_name: Positional[lb_types.TRAINABLE_DATASETS]
    lang: Positional[str]
    split: Positional[int]
    chunk: Positional[int]
    model_type: Positional[lb_types.MODEL_TYPE]
    temperature: Positional[float]

    action: t.Literal["preview", "status"] = "status"
    raw_text: bool = False

    def is_final(self) -> bool:
        """Check if checkpoint has final."""
        return generation_loaders.GenerationCheckpointLoader(
            lang=self.lang,
            split=self.split,
            chunk=self.chunk,
            model_type=self.model_type,
            temperature=self.temperature,
            dt_cfg=datasets.get_config(self.dataset_name),
        ).is_finished()

    def get_cpt(self) -> checkpoint_utils.GenerationCheckpoint | None:
        """Load the checkpoint."""
        item = generation_loaders.GenerationCheckpointLoader(
            lang=self.lang,
            split=self.split,
            chunk=self.chunk,
            model_type=self.model_type,
            temperature=self.temperature,
            dt_cfg=datasets.get_config(self.dataset_name),
        )
        if not item.exists():
            print("No checkpoint for that entry exists !!", file=sys.stderr)
            sys.exit(1)

        if item.is_finished():
            return item.load_final()
        print("[WARN] No final checkpoint !! Loading intermediate", file=sys.stderr)

        if item.has_intermediate():
            return item.load_intermediate()

        print("[ERROR] Given checkpoint dir was empty !!", file=sys.stderr)
        sys.exit(1)

    def _create_stats_content(self, checkpoint: checkpoint_utils.GenerationCheckpoint) -> str:
        """Create statistics content for the stats panel."""
        total_items = len(checkpoint.gen_items)
        total_target = sum(item["target_count"] for item in checkpoint.gen_items.values())
        total_current = sum(item["current_count"] for item in checkpoint.gen_items.values())

        completion_rate = (total_current / total_target * 100) if total_target > 0 else 0

        # Create progress indicator
        progress_bar = "█" * int(completion_rate / 5) + "░" * (20 - int(completion_rate / 5))

        stats = [
            f"Total Items: [bold cyan]{total_items}[/bold cyan]",
            f"Target Tokens: [bold green]{total_target:,}[/bold green]",
            f"Current Tokens: [bold yellow]{total_current:,}[/bold yellow]",
            f"Progress: [bold]{completion_rate:.1f}%[/bold] [{progress_bar}]",
            f"Temperature: [bold magenta]{checkpoint.temperature}[/bold magenta]",
        ]

        return "\n".join(stats)

    def _create_mini_progress_bar(self, percentage: float, width: int = 8) -> str:
        """Create a mini progress bar for table display."""
        filled = int(percentage / 100 * width)
        bar = "█" * filled + "░" * (width - filled)
        return f"[green]{bar}[/green]" if percentage >= 100 else f"[orange4]{bar}[/orange4]"

    def _create_items_table(self, gen_items: dict) -> Table:
        """Create detailed table of generation items."""
        table = Table(show_header=True, header_style="bold blue")

        # Add columns
        table.add_column("Estimation Type", style="cyan", width=15)
        table.add_column("Month", style="magenta", width=8)
        table.add_column("Target", style="green", justify="right", width=10)
        table.add_column("Current", style="yellow", justify="right", width=10)
        table.add_column("Progress", style="white", width=12)
        table.add_column("Status", style="bold", width=12)

        # Sort items by estimation type and month for consistent display
        sorted_items = sorted(gen_items.items(), key=lambda x: (str(x[0][0]), x[0][1]))

        for (estimation_type, month), gen_struct in sorted_items:
            target = gen_struct["target_count"]
            current = gen_struct["current_count"]

            # Calculate progress
            progress_pct = (current / target * 100) if target > 0 else 0
            progress_bar = self._create_mini_progress_bar(progress_pct)
            progress_color = "green" if progress_pct >= 100 else "orange4"

            # Status styling
            if current >= target:
                status = "[bold green]COMPLETE[/bold green]"
            elif current > 0:
                status = "[bold yellow]PARTIAL[/bold yellow]"
            else:
                status = "[bold red]PENDING[/bold red]"

            table.add_row(
                str(estimation_type),
                str(month),
                f"{target:,}",
                f"{current:,}",
                f"{progress_bar} [{progress_color}]{progress_pct:.1f}%[/{progress_color}]",
                status,
            )

        return table

    def show_status(self, title: str, checkpoint: checkpoint_utils.GenerationCheckpoint) -> None:
        """Show checkpoint status."""
        layout = Layout()
        # Create main sections
        layout.split_column(Layout(name="header", size=3), Layout(name="stats", size=5), Layout(name="table"))
        # Header with status
        status_color = "green" if self.is_final() else "yellow"
        status_text = "COMPLETED" if self.is_final() else "IN PROGRESS"
        header_panel = Panel(
            f"[bold]{title}[/bold] - [bold {status_color}]{status_text}[/bold {status_color}]", style="blue"
        )
        layout["header"].update(header_panel)

        # Statistics panel
        stats_content = self._create_stats_content(checkpoint)
        stats_panel = Panel(stats_content, title="Statistics", style="cyan")
        layout["stats"].update(stats_panel)

        # Items table
        items_table = self._create_items_table(checkpoint.gen_items)
        table_panel = Panel(items_table, title="Generation Items", style="magenta")
        layout["table"].update(table_panel)

        table_height = (len(checkpoint.gen_items) * 2) + 6  # +6 for header & tail
        layout["table"].size = table_height

        console.print(layout)

    def get_title(self) -> str:
        """Build title."""
        return (
            f"Checkpoint {self.dataset_name}-{self.lang}"
            f"[{self.split:02}/{self.chunk:02}]({self.model_type}){{{self.temperature}}}"
        )

    @staticmethod
    def process_text(text: str) -> str:
        """Make text readable."""
        word_list = text.split("|")
        clean_text = [
            word.replace(" ", "").replace(".", " . \n").replace("?", " ? \n").replace(",", " , ") for word in word_list
        ]
        return " ".join(clean_text)

    def _show_text_page(self, texts: list[str], start_idx: int, current_page: int, total_pages: int) -> None:  # noqa: ARG002
        """Display a page of text content."""
        for i, text in enumerate(texts, start_idx + 1):
            display_text = text if self.raw_text else self.process_text(text)

            text_panel = Panel(
                display_text, title=f"[bold]Line {i}[/bold]", style="green" if len(text) <= 500 else "yellow"
            )
            console.print(text_panel)
            console.print()  # Add spacing

    def _show_item_header(self, checkpoint: checkpoint_utils.GenerationCheckpoint, item_key: tuple, item: dict) -> None:
        """Show header information for selected item."""
        estimation_type, month = item_key
        target = item["target_count"]
        current = item["current_count"]
        progress_pct = (current / target * 100) if target > 0 else 0

        # Create progress bar
        progress_bar = self._create_mini_progress_bar(progress_pct, 20)

        header_content = [
            f"[bold cyan]Estimation Type:[/bold cyan] {estimation_type}",
            f"[bold magenta]Month:[/bold magenta] {month}",
            f"[bold green]Target:[/bold green] {target:,} tokens",
            f"[bold yellow]Current:[/bold yellow] {current:,} tokens",
            f"[bold white]Progress:[/bold white] {progress_bar} {progress_pct:.1f}%",
            f"[bold blue]Temperature:[/bold blue] {checkpoint.temperature}",
            f"[bold white]Total Texts:[/bold white] {len(item['text'])}",
        ]

        header_panel = Panel(
            Align.center("\n".join(header_content)), title="[bold]Generation Item Preview[/bold]", style="blue"
        )

        console.print(header_panel)

    def _preview_item_text(self, checkpoint: checkpoint_utils.GenerationCheckpoint, item_key: tuple) -> None:
        """Preview text content of selected item with pagination."""
        item = checkpoint.gen_items[item_key]
        texts = item["text"]

        if not texts:
            console.print("[yellow]No text content available for this item.[/yellow]")
            return

        # Pagination settings
        items_per_page = 5
        total_pages = (len(texts) + items_per_page - 1) // items_per_page
        current_page = 1

        while True:
            # Clear screen and show header
            console.clear()
            self._show_item_header(checkpoint, item_key, item)

            # Calculate page boundaries
            start_idx = (current_page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, len(texts))
            page_texts = texts[start_idx:end_idx]

            # Show current page content
            self._show_text_page(page_texts, start_idx, current_page, total_pages)

            # Navigation prompt
            if total_pages > 1:
                nav_options = []
                if current_page > 1:
                    nav_options.extend(["p", "prev"])
                if current_page < total_pages:
                    nav_options.extend(["n", "next"])
                nav_options.extend(["q", "quit"])

                action = Prompt.ask(f"\nPage {current_page}/{total_pages}", choices=nav_options, default="q").lower()

                if action in ["p", "prev"] and current_page > 1:
                    current_page -= 1
                elif action in ["n", "next"] and current_page < total_pages:
                    current_page += 1
                elif action in ["q", "quit"]:
                    break
            else:
                Prompt.ask("\nPress Enter to continue", default="")
                break

    def interactive_item_preview(self, checkpoint: checkpoint_utils.GenerationCheckpoint) -> None:
        """Interactive item selection and text preview with pagination.

        Raises:
            KeyboardInterrupt: If user cancels selection
            ValueError: If invalid selection is made

        """
        if not checkpoint.gen_items:
            console.print("[red]No generation items found![/red]")
            return

        # Create options list for selection
        options = {}
        for i, (key, item) in enumerate(checkpoint.gen_items.items(), 1):
            estimation_type, month = key
            target = item["target_count"]
            current = item["current_count"]
            progress = (current / target * 100) if target > 0 else 0

            display_text = f"{estimation_type} - Month {month} ({current}/{target} - {progress:.1f}%)"
            options[str(i)] = (key, display_text)

        # Show options
        console.print("\n[bold cyan]Available Generation Items:[/bold cyan]")
        for num, (_, display) in options.items():
            console.print(f"[yellow]{num}.[/yellow] {display}")

        # Get user selection
        while True:
            try:
                choice = Prompt.ask("\nSelect item to preview", choices=list(options.keys()), default="1")
                selected_key, _ = options[choice]
                break
            except KeyboardInterrupt:
                console.print("\n[yellow]Selection cancelled.[/yellow]")
                return

        # Preview selected item
        self._preview_item_text(checkpoint, selected_key)

    async def run(self) -> None:
        """Run command."""
        checkpoint = self.get_cpt()

        match self.action:
            case "status":
                self.show_status(title=self.get_title(), checkpoint=checkpoint)
            case "preview":
                self.interactive_item_preview(checkpoint)


if __name__ == "__main__":
    cmd = CheckPointView.parse()
    cmd.start()
