import unittest

from sft.data import build_chunk_record, build_inference_record, extract_s1_text


class TinyTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) + 1 for character in text]


class SFTDataTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TinyTokenizer()

    def test_extracts_s1_1_fields(self):
        extracted = extract_s1_text(
            {
                "question": "What is 1 + 1?",
                "deepseek_thinking_trajectory": "Add the integers.",
                "deepseek_attempt": "2",
            }
        )
        self.assertEqual(extracted[0], "What is 1 + 1?")
        self.assertIn("Final answer", extracted[1])
        self.assertTrue(extracted[1].endswith("2"))

    def test_training_masks_are_disjoint_and_fixed_length(self):
        record = build_chunk_record(
            tokenizer=self.tokenizer,
            question="Q",
            response_ids=[10, 11, 12, 0],
            chunk_index=0,
            max_length=32,
            target_length=8,
            max_question_tokens=8,
        )
        self.assertEqual(len(record["input_ids"]), 32)
        self.assertEqual(len(record["condition_mask"]), 32)
        self.assertEqual(len(record["loss_mask"]), 32)
        self.assertFalse(
            any(c and loss for c, loss in zip(record["condition_mask"], record["loss_mask"]))
        )
        supervised = [
            token
            for token, enabled in zip(record["input_ids"], record["loss_mask"])
            if enabled
        ]
        self.assertEqual(supervised, [10, 11, 12, 0])

    def test_inference_opens_exact_target_block(self):
        record = build_inference_record(
            tokenizer=self.tokenizer,
            question="Q",
            history_ids=[20, 21],
            max_length=32,
            target_length=8,
            max_question_tokens=8,
        )
        self.assertEqual(sum(not item for item in record["condition_mask"]), 8)
        self.assertEqual(len(record["input_ids"]), 32)


if __name__ == "__main__":
    unittest.main()
