import unittest

from tagger import Tagger

class TestTagPackage(unittest.TestCase):
    postagger = Tagger()    

    def pos_tag(self, text):
        return self.postagger.tag(text)  

    def test_pos_tag(self):
        with self.assertRaises(TypeError):
            self.pos_tag(None) # neither list nor string
        with self.assertRaises(TypeError):
            self.pos_tag([]) # empty list
        with self.assertRaises(TypeError):
            self.pos_tag('') # empty string
        with self.assertRaises(TypeError):
            self.pos_tag(['นายกรัฐมนตรี', {'key':'value'},0.1,'ห้าม']) # list of not strings/list
        with self.assertRaises(TypeError):
            self.pos_tag([['ฉัน','รัก','คุณ'], [0, 1, 'ครับ']]) # list of list of not strings
        self.assertEqual(
            self.pos_tag("กิน"),
            ["VERB"],
        )

        self.assertEqual(
            self.pos_tag(["สัตว์", "วิ่ง", "เร็ว"]),
            ["NOUN", "VERB", "ADV"],
        )

        self.assertEqual(
            self.pos_tag([['ผู้ช่วยเลขา', 'และ', 'กรรมการ'], ['สอบถาม', 'ประธาน']]),
            [['NOUN', 'SCONJ', 'NOUN'], ['VERB', 'NOUN']]
        )
        
        self.assertEqual(
            len(self.pos_tag(["การ", "ประชุม", "วิชาการ", "ครั้ง", "ที่ 1"])), 5
        )


        
if __name__ == "__main__":
    TestTagPackage().test_pos_tag()
