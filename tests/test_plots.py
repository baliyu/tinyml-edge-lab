from tinyml_lab.har_baseline import run
from tinyml_lab.plots import build_report_figures, confusion_matrix_fig

PNG = b"\x89PNG\r\n\x1a\n"
EXPECTED = {"confusion_matrix.png", "per_class_recall.png", "depth_sweep.png", "class_examples.png", "top_features.png"}


def test_all_figures_are_created_and_are_real_png_files(fake_uci_root, tmp_path):
    results = tmp_path / "results"
    run(fake_uci_root, depths=(2, 4), out_dir=results, chosen_depth=4)
    names = build_report_figures(fake_uci_root, results_dir=results)
    assert set(names) == EXPECTED
    for n in EXPECTED:
        data = (results / "figures" / n).read_bytes()
        assert data[:8] == PNG
        assert len(data) > 3000


def test_confusion_matrix_survives_a_class_that_never_occurs(tmp_path):
    confusion_matrix_fig([[0, 0], [1, 1]], ["a", "b"], tmp_path / "cm.png")
    assert (tmp_path / "cm.png").read_bytes()[:8] == PNG
