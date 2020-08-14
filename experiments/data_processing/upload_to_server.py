from pathlib import Path
import pandas as pd
import shutil


def main(corpus_path, gfi_csv, ngfi_csv):
    """
    I only exist bc. @hv10 needed to not train on his machine and wanted to copy the relevant data to a netshare.

    :param corpus_path:
    :param gfi_csv:
    :param ngfi_csv:
    :return:
    """
    corpus_path = Path(corpus_path).resolve()
    gfis = pd.read_csv(gfi_csv)
    ngfis = pd.read_csv(ngfi_csv)

    datapoints = list(gfis["name"]) + list(ngfis["name"])
    l = len(datapoints)
    for i, dp in enumerate(datapoints):
        print(f"{i:4d}/{l:4d}", end="\r")
        destination = Path("~/ssh_dhd/corpus").expanduser().resolve() / dp
        destination.parent.mkdir(exist_ok=True, parents=True)
        if not destination.exists():
            shutil.copy(corpus_path / dp, destination)
    print()


if __name__ == "__main__":
    main(
        Path(__file__).parent.parent / "corpus",
        Path(__file__).parent.parent / "notebooks/df_gfi_1000.csv",
        Path(__file__).parent.parent / "notebooks/df_ngfi_1000.csv",
    )
