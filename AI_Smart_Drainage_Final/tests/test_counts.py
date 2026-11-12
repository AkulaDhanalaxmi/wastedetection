from app.classification import Prediction
from app.counts import SessionCounts


def observe_three_frames(counts: SessionCounts, key: str, material: str) -> None:
    for _ in range(3):
        counts.observe(key, material, 0.8, b"crop" if material == "Plastic" else None)


def test_unique_material_and_plastic_subcategory_counts():
    counts = SessionCounts()
    observe_three_frames(counts, "plastic-1", "Plastic")
    observe_three_frames(counts, "metal-1", "Metal")
    observe_three_frames(counts, "organic-1", "Organic")

    pending = counts.snapshot()
    assert pending["total"] == 3
    assert pending["plastic_total"] == pending["plastic_manual_review"] == 1
    assert pending["metal_total"] == 1
    assert pending["organic_total"] == 1

    counts.apply_prediction(
        "plastic-1",
        Prediction("recyclable", "Clean rigid plastic.", 0.72, 4.1),
    )
    updated = counts.snapshot()
    assert updated["total"] == 3
    assert updated["plastic_total"] == 1
    assert updated["plastic_recyclable"] == 1
    assert updated["plastic_manual_review"] == 0
    assert (
        updated["plastic_recyclable"]
        + updated["plastic_non_recyclable"]
        + updated["plastic_manual_review"]
        == updated["plastic_total"]
    )
    assert updated["total"] == (
        updated["plastic_total"] + updated["metal_total"] + updated["organic_total"]
    )


def test_uncertain_prediction_can_be_updated_without_double_counting():
    counts = SessionCounts()
    observe_three_frames(counts, "plastic-1", "Plastic")
    counts.apply_prediction(
        "plastic-1",
        Prediction("uncertain", "Visible evidence is unclear.", 0.42, 3.0),
    )
    before = counts.snapshot()
    assert before["plastic_manual_review"] == 1
    assert before["total"] == 1
    counts.apply_prediction(
        "plastic-1",
        Prediction("non_recyclable", "Flexible film is visible.", 0.68, 3.4),
    )
    after = counts.snapshot()
    assert after["plastic_non_recyclable"] == 1
    assert after["plastic_manual_review"] == 0
    assert after["plastic_total"] == after["total"] == 1


def test_reset_increments_epoch_to_isolate_pending_old_results():
    counts = SessionCounts()
    observe_three_frames(counts, "plastic-1", "Plastic")
    first_epoch = counts.epoch
    counts.reset()
    assert counts.epoch == first_epoch + 1
    assert counts.snapshot()["total"] == 0
    assert not counts.apply_prediction(
        "plastic-1",
        Prediction("recyclable", "A result from before reset.", 0.8, 1.0),
    )


def test_upload_detection_can_be_counted_from_one_image():
    counts = SessionCounts()
    item, should_classify = counts.observe(
        "upload-1",
        "Plastic",
        0.9,
        b"crop",
        confirm_immediately=True,
    )
    assert item is not None
    assert should_classify
    assert counts.snapshot()["plastic_total"] == 1
