from transformers import AutoProcessor

from src.training.sft import make_labels


def test_make_labels_sets_english_transcribe_prefix():
    processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
    labels = make_labels(
        processor,
        "HELLO WORLD",
        "cpu",
        {"language": "en", "task": "transcribe", "predict_timestamps": False},
    )
    ids = labels[0].tolist()
    assert ids[:4] == [50258, 50259, 50360, 50364]
    assert ids[-1] == 50257
    assert -100 not in ids
