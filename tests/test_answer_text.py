import unittest

from rag_app.answer_text import plain_answer


class AnswerTextTests(unittest.TestCase):
    def test_chinese_references_and_markdown(self):
        text = (
            "### 回答\n\n1. **保留答案**[1, 2]。\n"
            "2. *继续说明*，见[说明](https://example.com)。\n\n"
            "---\n**参考资料：**\n[1] D:\\data\\book.txt"
        )
        self.assertEqual(plain_answer(text), "回答\n\n• 保留答案。\n• 继续说明，见说明。")

    def test_list_markers_survive_cleaning_and_repeat_formatting(self):
        text = "普通段落。\n\n- **第一项**\n* 第二项\n3. 第三项\n四、第四项"
        expected = "普通段落。\n\n• 第一项\n• 第二项\n• 第三项\n• 第四项"
        self.assertEqual(plain_answer(text), expected)
        self.assertEqual(plain_answer(expected), expected)

    def test_preserves_content_punctuation_and_ordinary_source_words(self):
        text = "持续 2–3 周，温度 -5°C，2 * 3 = 6。\n文件 AI_agent_RAG。\n来源不同会影响结果。"
        self.assertEqual(plain_answer(text), text)


if __name__ == "__main__":
    unittest.main()
