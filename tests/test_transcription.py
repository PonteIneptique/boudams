from unittest import TestCase
from boudams.tagger import BoudamsTagger


tagger = BoudamsTagger.load("./tests/boudams.tar", device="cpu")


class TestTranscription(TestCase):
    def test_transcription(self):
        output = tagger.annotate_text("avecabonentendem̅tabonzenseignemenz.")
        self.assertEqual(
            list(output), ['avec a bon entendem̅t \uf158 ', 'a bonz enseignemenz . ']
        )
