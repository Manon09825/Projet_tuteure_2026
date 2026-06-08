from pathlib import Path
from typing import Any
from sklearn.metrics import cohen_kappa_score
import click


class InterAnnotatorAgreement:
    def __init__(self, annotations_directory: str):
        self.annotations_directory = Path(annotations_directory)
        self.annotators, self.annotations = self._load_annotations()

    def _load_annotations(self):
        annotators: list[str] = []
        annotations: list[list[str]] = []
        for conll in self.annotations_directory.glob("*.conll"):
            with open(conll, "r") as file:
                annotators.append(conll.stem)
                content = file.read()
                annotations.append(self._clean_conll(content))

        return annotators, annotations

    def _clean_conll(self, content: str) -> list[str]:
        content = content.split("\n")
        annotations = []
        for line in content:
            if line.endswith("ALTE"):
                annotations.append("ALTE")
            elif line.endswith("ALTC"):
                annotations.append("ALTC")
            elif line.endswith("ALTS"):
                annotations.append("ALTS")
            elif line.endswith("héméronyme"):
                annotations.append("héméronyme")
            elif line.endswith("NONALT"):
                annotations.append("NONALT")
            elif line.endswith("O"):
                annotations.append("O")
            else:
                annotations.append("O")
        return annotations

    def _compute_cohen_kappa(self, y1: list[str], y2: list[str]) -> float:
        if len(y1) != len(y2):
            raise ValueError("Annotations must have the same length")
        else:
            print('ok')
        return cohen_kappa_score(y1, y2)

    def __call__(self, *args: Any, **kwds: Any) -> None:
        # Prepare results file
        results_path = self.annotations_directory / "inter_annotator_agreement_results.txt"
        
        # Initialize results file
        with open(results_path, 'w') as txtfile:
            txtfile.write("INTER-ANNOTATOR AGREEMENT RESULTS\n")
            txtfile.write("=" * 50 + "\n\n")
            
            for i in range(len(self.annotations)):
                for j in range(i + 1, len(self.annotations)):
                    annotator1 = self.annotators[i]
                    annotator2 = self.annotators[j]
                    kappa = self._compute_cohen_kappa(
                        self.annotations[i], self.annotations[j]
                    )
                    
                    # Print to console
                    print(f"Cohen's Kappa between {annotator1} and {annotator2}: {kappa:.4f}")
                    
                    # Write to file
                    txtfile.write(f"ANNOTATOR PAIR: {annotator1} vs {annotator2}\n")
                    txtfile.write(f"Cohen's Kappa: {kappa:.4f}\n")
                    txtfile.write("-" * 40 + "\n\n")
        
        print(f"\nResults saved to:")
        print(f"- {results_path}")


@click.command()
@click.argument("annotations_directory", type=click.Path(exists=True, file_okay=False))
def main(annotations_directory: str):
    """Calculate inter-annotator agreement using Cohen's Kappa."""
    agreement = InterAnnotatorAgreement(annotations_directory)
    agreement()


if __name__ == "__main__":
    main()
