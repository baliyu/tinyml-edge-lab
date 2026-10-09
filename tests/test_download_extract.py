import zipfile
from tinyml_lab.download import extract_uci_har


def test_extracts_nested_zip(tmp_path):
    inner = tmp_path / "UCI HAR Dataset.zip"
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("UCI HAR Dataset/train/y_train.txt", "1\n")
        z.writestr("UCI HAR Dataset/test/y_test.txt", "1\n")
    outer = tmp_path / "outer.zip"
    with zipfile.ZipFile(outer, "w") as z:
        z.write(inner, "UCI HAR Dataset.zip")
    dest = tmp_path / "raw"
    extract_uci_har(outer, dest)
    assert (dest / "UCI HAR Dataset" / "train" / "y_train.txt").exists()
    assert (dest / "UCI HAR Dataset" / "test" / "y_test.txt").exists()
